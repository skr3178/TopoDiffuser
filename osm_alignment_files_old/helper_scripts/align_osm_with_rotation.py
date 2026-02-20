#!/usr/bin/env python3
"""
Align OSM with trajectory using rotation correction.

Uses trajectory heading to properly rotate OSM roads.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def load_poses(pose_file: str) -> np.ndarray:
    """Load KITTI poses."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses: np.ndarray) -> np.ndarray:
    """Extract (x, z) trajectory."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def compute_trajectory_heading(trajectory, num_points=50):
    """Compute average heading of trajectory at start."""
    # Use first N points to determine heading
    points = trajectory[:num_points]
    
    # Fit line to get heading
    dx = points[-1, 0] - points[0, 0]
    dy = points[-1, 1] - points[0, 1]
    
    heading = np.arctan2(dy, dx)
    return heading


def align_with_rotation(seq: str = "00", data_root: str = "data"):
    """Align OSM with proper rotation correction."""
    
    # Map sequence to raw data
    seq_to_raw = {
        '00': '2011_10_03_drive_0027_sync',
        '05': '2011_09_30_drive_0018_sync',
        '08': '2011_09_30_drive_0028_sync',
    }
    
    raw_folder = seq_to_raw.get(seq)
    if not raw_folder:
        print(f"Sequence {seq} not in mapping")
        return
    
    raw_data_root = Path(data_root) / "raw_data"
    
    # Find OXTS
    oxts_dir = None
    for date_prefix in ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']:
        candidate = raw_data_root / raw_folder / date_prefix / raw_folder / "oxts" / "data"
        if candidate.exists():
            oxts_dir = candidate
            break
    
    if not oxts_dir:
        print(f"OXTS not found")
        return
    
    # Load data
    print(f"Loading OXTS from: {oxts_dir}")
    oxts_data = load_oxts_data(oxts_dir)
    gps_lats = oxts_data[:, 0]
    gps_lons = oxts_data[:, 1]
    
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses[:len(oxts_data)])
    
    print(f"GPS trajectory: {len(gps_lats)} points")
    print(f"Trajectory: {len(trajectory)} points")
    
    # Load OSM
    osm_edges_file = Path(data_root) / "osm" / f"{seq}_edges.npy"
    osm_edges = np.load(osm_edges_file)
    print(f"OSM edges: {len(osm_edges)} points")
    
    # Filter OSM to trajectory area + margin
    margin = 0.001
    lat_min, lat_max = gps_lats.min() - margin, gps_lats.max() + margin
    lon_min, lon_max = gps_lons.min() - margin, gps_lons.max() + margin
    
    mask = (
        (osm_edges[:, 0] >= lat_min) & (osm_edges[:, 0] <= lat_max) &
        (osm_edges[:, 1] >= lon_min) & (osm_edges[:, 1] <= lon_max)
    )
    osm_filtered = osm_edges[mask]
    print(f"OSM filtered: {len(osm_filtered)} points")
    
    # Convert to UTM
    print("Converting to UTM...")
    utm_trajectory = np.array([latlon_to_utm(lat, lon) for lat, lon in zip(gps_lats, gps_lons)])
    utm_osm = np.array([latlon_to_utm(lat, lon) for lat, lon in zip(osm_filtered[:, 0], osm_filtered[:, 1])])
    
    # === ALIGNMENT WITH ROTATION ===
    
    # Step 1: Offset to align origins
    offset_east = utm_trajectory[0, 0] - trajectory[0, 0]
    offset_north = utm_trajectory[0, 1] - trajectory[0, 1]
    
    # Step 2: Compute rotation needed
    # Get trajectory heading from first 50 points
    traj_heading = compute_trajectory_heading(trajectory, num_points=50)
    
    # Get GPS/UTM heading from first 50 points
    gps_heading = compute_trajectory_heading(utm_trajectory - [offset_east, offset_north], num_points=50)
    
    # Rotation needed
    rotation = traj_heading - gps_heading
    
    print(f"\n=== Alignment Parameters ===")
    print(f"Offset: east={offset_east:.1f}, north={offset_north:.1f}")
    print(f"Trajectory heading: {np.degrees(traj_heading):.2f}°")
    print(f"GPS heading: {np.degrees(gps_heading):.2f}°")
    print(f"Rotation needed: {np.degrees(rotation):.2f}°")
    
    # Apply transformation
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    
    osm_aligned = []
    for east, north in utm_osm:
        # Offset
        x = east - offset_east
        y = north - offset_north
        
        # Rotate
        x_rot = x * cos_r - y * sin_r
        y_rot = x * sin_r + y * cos_r
        
        osm_aligned.append([x_rot, y_rot])
    
    osm_aligned = np.array(osm_aligned)
    
    # Compute alignment error
    errors = []
    for tx, ty in trajectory[::10]:  # Sample every 10th point
        dists = np.sqrt((osm_aligned[:, 0] - tx)**2 + (osm_aligned[:, 1] - ty)**2)
        errors.append(np.min(dists))
    
    print(f"\n=== Alignment Error ===")
    print(f"Mean: {np.mean(errors):.2f}m")
    print(f"Median: {np.median(errors):.2f}m")
    print(f"Max: {np.max(errors):.2f}m")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Before and after alignment
    
    # Before: Offset only
    ax = axes[0, 0]
    osm_offset_only = np.array([[e - offset_east, n - offset_north] for e, n in utm_osm])
    ax.scatter(osm_offset_only[:, 0], osm_offset_only[:, 1], s=1, c='blue', alpha=0.5, label='OSM (offset only)')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'g*', markersize=20)
    ax.set_title('BEFORE: Offset Only (No Rotation)', fontsize=11, fontweight='bold', color='red')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # After: With rotation
    ax = axes[0, 1]
    ax.scatter(osm_aligned[:, 0], osm_aligned[:, 1], s=1, c='blue', alpha=0.5, label='OSM (aligned)')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'g*', markersize=20)
    ax.set_title(f'AFTER: With Rotation ({np.degrees(rotation):.1f}°)', fontsize=11, fontweight='bold', color='green')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Error histogram
    ax = axes[0, 2]
    ax.hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f}m')
    ax.axvline(x=np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.1f}m')
    ax.set_xlabel('Distance to Nearest OSM Road (m)')
    ax.set_ylabel('Count')
    ax.set_title('Alignment Error Distribution', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: Zoomed views at 3 locations
    
    frame_indices = [len(trajectory) // 4, len(trajectory) // 2, 3 * len(trajectory) // 4]
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = axes[1, idx]
        
        ego_x, ego_y = trajectory[frame_idx]
        
        # Filter to zoom area
        zoom_radius = 50
        dists = np.sqrt((osm_aligned[:, 0] - ego_x)**2 + (osm_aligned[:, 1] - ego_y)**2)
        nearby_mask = dists <= zoom_radius
        osm_nearby = osm_aligned[nearby_mask]
        
        # Plot
        ax.scatter(osm_nearby[:, 0], osm_nearby[:, 1], s=10, c='blue', alpha=0.6, label='OSM roads')
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
        ax.plot(ego_x, ego_y, 'g*', markersize=25, markeredgecolor='black', markeredgewidth=2, label='Ego')
        
        # BEV range box
        bev_x = [ego_x - 20, ego_x + 20, ego_x + 20, ego_x - 20, ego_x - 20]
        bev_y = [ego_y - 10, ego_y - 10, ego_y + 30, ego_y + 30, ego_y - 10]
        ax.plot(bev_x, bev_y, 'g--', linewidth=2, label='BEV range')
        
        ax.set_xlim(ego_x - 60, ego_x + 60)
        ax.set_ylim(ego_y - 60, ego_y + 60)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Frame {frame_idx}: OSM + Trajectory', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.suptitle(f'Sequence {seq}: OSM Alignment with Rotation Correction\n' +
                f'Error: Mean={np.mean(errors):.1f}m, Median={np.median(errors):.1f}m',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    output_file = f'osm_aligned_rotation_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to: {output_file}")
    
    # Save aligned OSM
    output_npy = f'osm_aligned_rotation_seq{seq}.npy'
    np.save(output_npy, osm_aligned)
    print(f"✅ Saved aligned OSM to: {output_npy}")
    
    return osm_aligned


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Align OSM with rotation correction")
    parser.add_argument("--seq", type=str, default="00", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root")
    
    args = parser.parse_args()
    
    align_with_rotation(args.seq, args.data_root)
    print("\n🎉 Done!")
