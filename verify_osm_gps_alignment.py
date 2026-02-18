#!/usr/bin/env python3
"""
Verify OSM and trajectory alignment in GPS coordinates before transformation.

This shows both the OSM roads and the GPS trajectory from OXTS to verify they overlap.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def verify_gps_overlap(seq: str = "00", data_root: str = "data"):
    """Verify that OSM roads and OXTS GPS trajectory overlap in GPS space."""
    
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
        print(f"OXTS directory not found for sequence {seq}")
        return
    
    # Load OXTS data
    print(f"Loading OXTS from: {oxts_dir}")
    oxts_data = load_oxts_data(oxts_dir)
    
    # Extract GPS coordinates from OXTS
    gps_lats = oxts_data[:, 0]
    gps_lons = oxts_data[:, 1]
    
    print(f"GPS trajectory: {len(gps_lats)} points")
    print(f"GPS lat range: {gps_lats.min():.6f} to {gps_lats.max():.6f}")
    print(f"GPS lon range: {gps_lons.min():.6f} to {gps_lons.max():.6f}")
    
    # Load OSM edges
    osm_edges_file = Path(data_root) / "osm" / f"{seq}_edges.npy"
    if osm_edges_file.exists():
        osm_edges = np.load(osm_edges_file)
        print(f"OSM edges: {len(osm_edges)} points")
        print(f"OSM lat range: {osm_edges[:,0].min():.6f} to {osm_edges[:,0].max():.6f}")
        print(f"OSM lon range: {osm_edges[:,1].min():.6f} to {osm_edges[:,1].max():.6f}")
    else:
        print(f"OSM edges file not found: {osm_edges_file}")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: GPS coordinates
    ax = axes[0]
    ax.scatter(osm_edges[:, 1], osm_edges[:, 0], s=0.1, c='blue', alpha=0.3, label='OSM Roads')
    ax.plot(gps_lons, gps_lats, 'r-', linewidth=2, label='GPS Trajectory')
    ax.scatter(gps_lons[0], gps_lats[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(gps_lons[-1], gps_lats[-1], c='red', s=150, marker='*', label='End', zorder=5)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Sequence {seq}: OSM + GPS Trajectory (GPS Coordinates)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Right: UTM coordinates
    ax = axes[1]
    
    # Convert to UTM
    print("Converting to UTM...")
    utm_trajectory = np.array([latlon_to_utm(lat, lon) for lat, lon in zip(gps_lats, gps_lons)])
    utm_osm = np.array([latlon_to_utm(lat, lon) for lat, lon in zip(osm_edges[:, 0], osm_edges[:, 1])])
    
    ax.scatter(utm_osm[:, 0], utm_osm[:, 1], s=0.1, c='blue', alpha=0.3, label='OSM Roads')
    ax.plot(utm_trajectory[:, 0], utm_trajectory[:, 1], 'r-', linewidth=2, label='UTM Trajectory')
    ax.scatter(utm_trajectory[0, 0], utm_trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(utm_trajectory[-1, 0], utm_trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    
    ax.set_xlabel('UTM Easting (m)', fontsize=12)
    ax.set_ylabel('UTM Northing (m)', fontsize=12)
    ax.set_title(f'Sequence {seq}: OSM + Trajectory (UTM Coordinates)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    
    output_file = f'osm_gps_verify_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify OSM and GPS trajectory overlap")
    parser.add_argument("--seq", type=str, default="00", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root")
    
    args = parser.parse_args()
    
    verify_gps_overlap(args.seq, args.data_root)
    print("\n✅ Done!")
