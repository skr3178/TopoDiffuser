#!/usr/bin/env python3
"""
Properly align OSM with KITTI by filtering roads to trajectory area first.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


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


def align_osm_properly(seq: str = "00", data_root: str = "data"):
    """Align OSM with trajectory by filtering to relevant area first."""
    
    # Map sequence to raw data folder
    seq_to_raw = {
        '00': '2011_10_03_drive_0027_sync',
        '05': '2011_09_30_drive_0018_sync',
        '08': '2011_09_30_drive_0028_sync',
    }
    
    if seq not in seq_to_raw:
        print(f"Sequence {seq} not in mapping")
        return
    
    raw_folder = seq_to_raw[seq]
    raw_data_root = Path(data_root) / "raw_data"
    
    # Find OXTS directory
    oxts_dir = None
    for date_prefix in ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']:
        candidate = raw_data_root / raw_folder / date_prefix / raw_folder / "oxts" / "data"
        if candidate.exists():
            oxts_dir = candidate
            break
    
    if not oxts_dir:
        print(f"OXTS directory not found")
        return
    
    # Load OXTS data
    print(f"Loading OXTS from: {oxts_dir}")
    oxts_data = load_oxts_data(oxts_dir)
    
    # Extract GPS trajectory
    gps_lats = oxts_data[:, 0]
    gps_lons = oxts_data[:, 1]
    
    print(f"GPS trajectory: {len(gps_lats)} points")
    print(f"GPS lat range: {gps_lats.min():.6f} to {gps_lats.max():.6f}")
    print(f"GPS lon range: {gps_lons.min():.6f} to {gps_lons.max():.6f}")
    
    # Load poses
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses[:len(oxts_data)])
    
    # Load OSM edges
    osm_edges_file = Path(data_root) / "osm" / f"{seq}_edges.npy"
    osm_edges = np.load(osm_edges_file)
    
    print(f"OSM edges (before filter): {len(osm_edges)} points")
    
    # Filter OSM edges to GPS bounding box + margin
    margin = 0.001  # ~100m margin in degrees
    lat_min, lat_max = gps_lats.min() - margin, gps_lats.max() + margin
    lon_min, lon_max = gps_lons.min() - margin, gps_lons.max() + margin
    
    mask = (
        (osm_edges[:, 0] >= lat_min) & (osm_edges[:, 0] <= lat_max) &
        (osm_edges[:, 1] >= lon_min) & (osm_edges[:, 1] <= lon_max)
    )
    osm_filtered = osm_edges[mask]
    
    print(f"OSM edges (after filter): {len(osm_filtered)} points")
    
    # Convert GPS trajectory to UTM
    print("Converting to UTM...")
    utm_trajectory = np.array([latlon_to_utm(lat, lon) for lat, lon in zip(gps_lats, gps_lons)])
    utm_osm = np.array([latlon_to_utm(lat, lon) for lat, lon in zip(osm_filtered[:, 0], osm_filtered[:, 1])])
    
    # Compute simple offset alignment (no rotation needed if both in UTM)
    offset_east = utm_trajectory[0, 0] - trajectory[0, 0]
    offset_north = utm_trajectory[0, 1] - trajectory[0, 1]
    
    print(f"UTM offset: east={offset_east:.1f}, north={offset_north:.1f}")
    
    # Transform OSM to local frame
    osm_local = np.array([
        [utm_osm[i, 0] - offset_east, utm_osm[i, 1] - offset_north]
        for i in range(len(utm_osm))
    ])
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Left: GPS coordinates
    ax = axes[0]
    ax.scatter(osm_filtered[:, 1], osm_filtered[:, 0], s=1, c='blue', alpha=0.5, label='OSM Roads (filtered)')
    ax.plot(gps_lons, gps_lats, 'r-', linewidth=2, label='GPS Trajectory')
    ax.scatter(gps_lons[0], gps_lats[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(gps_lons[-1], gps_lats[-1], c='red', s=150, marker='*', label='End', zorder=5)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'Sequence {seq}: Filtered OSM + GPS', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Center: UTM coordinates
    ax = axes[1]
    ax.scatter(utm_osm[:, 0], utm_osm[:, 1], s=1, c='blue', alpha=0.5, label='OSM (UTM)')
    ax.plot(utm_trajectory[:, 0], utm_trajectory[:, 1], 'r-', linewidth=2, label='Trajectory (UTM)')
    ax.scatter(utm_trajectory[0, 0], utm_trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(utm_trajectory[-1, 0], utm_trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    ax.set_xlabel('UTM Easting (m)', fontsize=11)
    ax.set_ylabel('UTM Northing (m)', fontsize=11)
    ax.set_title(f'UTM Coordinates (Before Alignment)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Right: Aligned local frame
    ax = axes[2]
    ax.scatter(osm_local[:, 0], osm_local[:, 1], s=1, c='blue', alpha=0.5, label='OSM (aligned)')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title(f'Aligned: OSM + Trajectory (Local Frame)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    output_file = f'osm_aligned_properly_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")
    
    # Save the filtered and aligned OSM data
    alignment_data = {
        'osm_local': osm_local,
        'offset_east': offset_east,
        'offset_north': offset_north,
        'gps_bounds': {
            'lat_min': float(lat_min), 'lat_max': float(lat_max),
            'lon_min': float(lon_min), 'lon_max': float(lon_max)
        }
    }
    
    output_npy = f'osm_aligned_seq{seq}.npy'
    np.save(output_npy, osm_local)
    print(f"Saved aligned OSM to: {output_npy}")
    
    return alignment_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Properly align OSM with trajectory")
    parser.add_argument("--seq", type=str, default="00", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root")
    
    args = parser.parse_args()
    
    align_osm_properly(args.seq, args.data_root)
    print("\n✅ Done!")
