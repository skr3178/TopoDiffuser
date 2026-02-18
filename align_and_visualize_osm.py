#!/usr/bin/env python3
"""
Align OSM roads with KITTI local frame using OXTS GPS data.

This script:
1. Loads OXTS GPS data for a sequence
2. Computes GPS → Local frame transformation
3. Transforms OSM road network to local frame
4. Visualizes aligned OSM + trajectory
5. Extracts OSM route around ego vehicle (5 past + 15 future keyframes)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, compute_gps_to_local_transform, latlon_to_utm


def load_poses(pose_file: str) -> np.ndarray:
    """Load KITTI poses from file."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses: np.ndarray) -> np.ndarray:
    """Extract (x, z) trajectory from poses."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def transform_osm_to_local(osm_edges, transform):
    """Transform OSM edges from GPS to local frame with rotation."""
    local_edges = []
    
    # Get rotation parameters
    yaw_offset = transform['yaw_offset']
    cos_yaw = np.cos(-yaw_offset)
    sin_yaw = np.sin(-yaw_offset)
    
    for edge in osm_edges:
        local_edge = []
        for lat, lon in edge:
            # Convert to UTM
            east, north = latlon_to_utm(lat, lon)
            # Apply offset
            x = east - transform['offset_east']
            y = north - transform['offset_north']
            # Apply rotation
            x_rot = x * cos_yaw - y * sin_yaw
            y_rot = x * sin_yaw + y * cos_yaw
            local_edge.append((x_rot, y_rot))
        local_edges.append(local_edge)
    
    return local_edges


def extract_osm_route_around_ego(osm_edges_local, ego_pose, past_dist=10.0, future_dist=30.0):
    """
    Extract OSM road segments within a certain distance of ego vehicle.
    
    Args:
        osm_edges_local: OSM edges in local frame
        ego_pose: (x, y, yaw) ego position
        past_dist: Distance behind ego to include (meters)
        future_dist: Distance ahead of ego to include (meters)
    
    Returns:
        List of road segments near ego
    """
    ego_x, ego_y, ego_yaw = ego_pose
    
    # Define search box
    x_min = ego_x - 20  # 20m left/right
    x_max = ego_x + 20
    y_min = ego_y - past_dist  # behind
    y_max = ego_y + future_dist  # ahead
    
    nearby_edges = []
    
    for edge in osm_edges_local:
        # Check if any point of edge is within search box
        in_range = False
        for x, y in edge:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                in_range = True
                break
        
        if in_range:
            nearby_edges.append(edge)
    
    return nearby_edges


def visualize_aligned_osm(seq: str, data_root: str = "data"):
    """Align and visualize OSM + trajectory for a sequence."""
    
    # Paths
    raw_data_root = Path(data_root) / "raw_data"
    
    # Map sequence to raw data folder
    seq_to_raw = {
        '00': '2011_10_03_drive_0027_sync',
        '01': '2011_10_03_drive_0042_sync',
        '02': '2011_10_03_drive_0034_sync',
        '05': '2011_09_30_drive_0018_sync',
        '07': '2011_09_30_drive_0027_sync',
        '08': '2011_09_30_drive_0028_sync',
        '09': '2011_09_30_drive_0033_sync',
        '10': '2011_09_30_drive_0034_sync',
    }
    
    if seq not in seq_to_raw:
        print(f"Sequence {seq} raw data mapping not known")
        return
    
    raw_folder = seq_to_raw[seq]
    oxts_dir = raw_data_root / raw_folder / "2011_" / raw_folder / "oxts" / "data"
    
    # Alternative path structure
    if not oxts_dir.exists():
        # Try different date prefix
        for date_prefix in ['09_26', '09_28', '09_29', '09_30', '10_03']:
            oxts_dir = raw_data_root / raw_folder / f"2011_{date_prefix}" / raw_folder / "oxts" / "data"
            if oxts_dir.exists():
                break
    
    if not oxts_dir.exists():
        print(f"OXTS data not found for sequence {seq} at {oxts_dir}")
        return
    
    print(f"Loading OXTS data from: {oxts_dir}")
    oxts_data = load_oxts_data(oxts_dir)
    print(f"Loaded {len(oxts_data)} OXTS frames")
    
    # Load poses
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(pose_file)
    print(f"Loaded {len(poses)} poses")
    
    # Match lengths (OXTS and poses may differ slightly)
    min_len = min(len(oxts_data), len(poses))
    oxts_data = oxts_data[:min_len]
    poses = poses[:min_len]
    
    # Compute transformation
    print("Computing GPS to local frame transformation...")
    transform = compute_gps_to_local_transform(oxts_data, poses)
    print(f"Transform: offset=({transform['offset_east']:.1f}, {transform['offset_north']:.1f}), "
          f"scale={transform['scale']:.6f}, yaw_offset={np.degrees(transform['yaw_offset']):.2f}°")
    
    # Load OSM edges
    osm_edges_file = Path(data_root) / "osm" / f"{seq}_edges.npy"
    if osm_edges_file.exists():
        osm_edges = np.load(osm_edges_file)
        # Convert to list of edges (each edge is list of points)
        # The edges file is a flat array of [lat, lon] points
        # Split into segments (assuming consecutive points form edges)
        osm_edge_list = []
        for i in range(0, len(osm_edges) - 1, 2):
            edge = [
                (osm_edges[i, 0], osm_edges[i, 1]),
                (osm_edges[i+1, 0], osm_edges[i+1, 1])
            ]
            osm_edge_list.append(edge)
    else:
        # Use polylines
        polylines_file = Path(data_root) / "osm_polylines" / f"{seq}_polylines.pkl"
        if polylines_file.exists():
            with open(polylines_file, 'rb') as f:
                osm_edge_list = pickle.load(f)
        else:
            print(f"OSM data not found for sequence {seq}")
            return
    
    print(f"Loaded {len(osm_edge_list)} OSM road segments")
    
    # Transform OSM to local frame
    print("Transforming OSM to local frame...")
    osm_local = transform_osm_to_local(osm_edge_list, transform)
    
    # Get trajectory
    trajectory = extract_trajectory(poses)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Full trajectory + all OSM roads
    ax = axes[0]
    
    # Plot OSM roads (sample for visibility)
    sample_rate = max(1, len(osm_local) // 5000)
    for i, edge in enumerate(osm_local[::sample_rate]):
        if len(edge) >= 2:
            xs = [p[0] for p in edge]
            ys = [p[1] for p in edge]
            ax.plot(xs, ys, 'b-', linewidth=0.5, alpha=0.4)
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=20, label='End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Sequence {seq}: OSM Aligned with Trajectory\n({len(osm_local)} road segments)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Right: Zoomed view at specific frame with OSM route
    ax = axes[1]
    
    # Pick a frame in the middle
    frame_idx = len(trajectory) // 2
    ego_pose = (trajectory[frame_idx, 0], trajectory[frame_idx, 1], 0)
    
    # Extract OSM route around ego
    osm_route = extract_osm_route_around_ego(osm_local, ego_pose, past_dist=10, future_dist=30)
    
    # Plot full trajectory faintly
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1, alpha=0.3)
    
    # Plot OSM route around ego
    for edge in osm_route:
        if len(edge) >= 2:
            xs = [p[0] for p in edge]
            ys = [p[1] for p in edge]
            ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
    
    # Plot ego position
    ax.plot(ego_pose[0], ego_pose[1], 'g*', markersize=25, label='Ego', markeredgecolor='black', markeredgewidth=2)
    
    # Draw BEV range box (-20 to +20 x, -10 to +30 y)
    bev_x = [ego_pose[0] - 20, ego_pose[0] + 20, ego_pose[0] + 20, ego_pose[0] - 20, ego_pose[0] - 20]
    bev_y = [ego_pose[1] - 10, ego_pose[1] - 10, ego_pose[1] + 30, ego_pose[1] + 30, ego_pose[1] - 10]
    ax.plot(bev_x, bev_y, 'g--', linewidth=2, label='BEV Range (40m x 40m)')
    
    # Zoom to area around ego
    ax.set_xlim(ego_pose[0] - 50, ego_pose[0] + 50)
    ax.set_ylim(ego_pose[1] - 50, ego_pose[1] + 50)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'OSM Route Around Ego (Frame {frame_idx})\n{len(osm_route)} road segments in local area', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    output_file = f'osm_aligned_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")
    
    # Save alignment data
    alignment_data = {
        'sequence': seq,
        'transform': transform,
        'num_osm_segments': len(osm_local),
        'num_frames': len(trajectory)
    }
    
    import json
    with open(f'osm_alignment_seq{seq}.json', 'w') as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() 
                  for k, v in alignment_data.items()}, f, indent=2)
    print(f"Saved alignment data to: osm_alignment_seq{seq}.json")
    
    return transform, osm_local


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Align OSM with KITTI trajectory")
    parser.add_argument("--seq", type=str, default="05", 
                       help="Sequence number (05 has good OXTS data)")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Data root directory")
    
    args = parser.parse_args()
    
    visualize_aligned_osm(args.seq, args.data_root)
    print("\n✅ Done!")
