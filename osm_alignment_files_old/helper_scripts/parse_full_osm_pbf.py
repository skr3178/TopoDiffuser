#!/usr/bin/env python3
"""
Parse full Karlsruhe OSM PBF to extract road polylines.

Uses pyrosm to get actual road geometries (not just edge endpoints).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
import pandas as pd  # Needed for pyrosm

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
    points = trajectory[:num_points]
    dx = points[-1, 0] - points[0, 0]
    dy = points[-1, 1] - points[0, 1]
    return np.arctan2(dy, dx)


def parse_osm_pbf(pbf_file: str, bounding_box: tuple = None):
    """
    Parse OSM PBF file and extract road polylines.
    
    Args:
        pbf_file: Path to .osm.pbf file
        bounding_box: (min_lat, min_lon, max_lat, max_lon) to filter
    
    Returns:
        road_polylines: List of [(lat, lon), ...] for each road segment
    """
    from pyrosm import OSM
    
    print(f"Loading OSM from: {pbf_file}")
    # Load full OSM - pyrosm's bounding_box has issues
    osm = OSM(pbf_file)
    
    # Get driving network (roads)
    print("Extracting driving network...")
    network = osm.get_network(network_type="driving")
    
    print(f"Found {len(network)} road segments")
    print(f"Columns: {network.columns.tolist()}")
    
    # Filter by bounding box if provided
    if bounding_box is not None:
        min_lat, min_lon, max_lat, max_lon = bounding_box
        
        def is_in_bbox(geom):
            if geom is None:
                return False
            bounds = geom.bounds  # (minx, miny, maxx, maxy) = (min_lon, min_lat, max_lon, max_lat)
            return not (bounds[2] < min_lon or bounds[0] > max_lon or 
                       bounds[3] < min_lat or bounds[1] > max_lat)
        
        network = network[network.geometry.apply(is_in_bbox)]
        print(f"After bounding box filter: {len(network)} road segments")
    
    # Extract polylines
    road_polylines = []
    
    for idx, row in network.iterrows():
        # Get geometry - can be LineString or MultiLineString
        geom = row['geometry']
        
        if geom is None:
            continue
        
        # Handle different geometry types
        if geom.geom_type == 'LineString':
            # Extract coordinates
            coords = list(geom.coords)
            if len(coords) >= 2:
                # Convert to (lat, lon) format
                polyline = [(lat, lon) for lon, lat in coords]
                road_polylines.append(polyline)
        
        elif geom.geom_type == 'MultiLineString':
            # Handle multi-part geometries
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    polyline = [(lat, lon) for lon, lat in coords]
                    road_polylines.append(polyline)
    
    print(f"Extracted {len(road_polylines)} road polylines")
    
    # Also get some metadata
    road_types = network.get('highway', pd.Series(['unknown'] * len(network))).value_counts()
    print(f"\nRoad types:")
    print(road_types.head(10))
    
    return road_polylines


def transform_polylines_to_local(polylines, transform):
    """Transform polylines from GPS to local frame."""
    cos_r = np.cos(transform['rotation'])
    sin_r = np.sin(transform['rotation'])
    
    local_polylines = []
    
    for polyline in polylines:
        local_polyline = []
        for lat, lon in polyline:
            # Convert to UTM
            east, north = latlon_to_utm(lat, lon)
            
            # Offset
            x = east - transform['offset_east']
            y = north - transform['offset_north']
            
            # Rotate
            x_rot = x * cos_r - y * sin_r
            y_rot = x * sin_r + y * cos_r
            
            local_polyline.append((x_rot, y_rot))
        
        local_polylines.append(local_polyline)
    
    return local_polylines


def filter_polylines_to_area(polylines, center, radius):
    """Filter polylines to those within radius of center."""
    cx, cy = center
    filtered = []
    
    for polyline in polylines:
        # Check if any point is within radius
        for x, y in polyline:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist <= radius:
                filtered.append(polyline)
                break
    
    return filtered


def parse_and_align_osm(seq: str = "00", data_root: str = "data"):
    """Parse full OSM PBF and align with trajectory."""
    
    # Map sequence to raw data (KITTI sequence to raw folder mapping)
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
    
    raw_folder = seq_to_raw.get(seq)
    if not raw_folder:
        print(f"Sequence {seq} not in mapping")
        return
    
    raw_data_root = Path(data_root) / "raw_data"
    
    # Find OXTS - check multiple possible path structures
    oxts_dir = None
    
    # Try paths with date prefix first
    for date_prefix in ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']:
        candidate = raw_data_root / raw_folder / date_prefix / raw_folder / "oxts" / "data"
        if candidate.exists():
            oxts_dir = candidate
            break
    
    # Try alternative path without date folder
    if not oxts_dir:
        candidate = raw_data_root / raw_folder / "oxts" / "data"
        if candidate.exists():
            oxts_dir = candidate
    
    if not oxts_dir:
        print(f"OXTS not found for sequence {seq}")
        return
    
    # Load GPS data
    print(f"Loading OXTS from: {oxts_dir}")
    oxts_data = load_oxts_data(oxts_dir)
    gps_lats = oxts_data[:, 0]
    gps_lons = oxts_data[:, 1]
    
    # Get bounding box with margin
    margin = 0.002  # ~200m
    bbox = (
        gps_lats.min() - margin,  # min_lat
        gps_lons.min() - margin,  # min_lon
        gps_lats.max() + margin,  # max_lat
        gps_lons.max() + margin   # max_lon
    )
    
    print(f"\nBounding box: {bbox}")
    print(f"  Lat: {bbox[0]:.6f} to {bbox[2]:.6f}")
    print(f"  Lon: {bbox[1]:.6f} to {bbox[3]:.6f}")
    
    # Parse OSM PBF
    pbf_file = Path(data_root) / "osm" / "karlsruhe.osm.pbf"
    if not pbf_file.exists():
        print(f"PBF file not found: {pbf_file}")
        return
    
    road_polylines = parse_osm_pbf(str(pbf_file), bounding_box=bbox)
    
    # Load trajectory
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses[:len(oxts_data)])
    
    # Compute alignment transform
    print("\nComputing alignment...")
    utm_trajectory = np.array([latlon_to_utm(lat, lon) for lat, lon in zip(gps_lats, gps_lons)])
    
    offset_east = utm_trajectory[0, 0] - trajectory[0, 0]
    offset_north = utm_trajectory[0, 1] - trajectory[0, 1]
    
    traj_heading = compute_trajectory_heading(trajectory, num_points=50)
    gps_heading = compute_trajectory_heading(utm_trajectory - [offset_east, offset_north], num_points=50)
    rotation = traj_heading - gps_heading
    
    transform = {
        'offset_east': offset_east,
        'offset_north': offset_north,
        'rotation': rotation
    }
    
    print(f"Offset: east={offset_east:.1f}, north={offset_north:.1f}")
    print(f"Rotation: {np.degrees(rotation):.2f}°")
    
    # Transform polylines to local frame
    print(f"\nTransforming {len(road_polylines)} polylines to local frame...")
    local_polylines = transform_polylines_to_local(road_polylines, transform)
    
    # Filter to trajectory area + margin
    print("Filtering to trajectory area...")
    traj_center = np.mean(trajectory, axis=0)
    search_radius = np.max(np.linalg.norm(trajectory - traj_center, axis=1)) + 100
    
    filtered_polylines = filter_polylines_to_area(local_polylines, traj_center, search_radius)
    print(f"Filtered to {len(filtered_polylines)} polylines in trajectory area")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Left: OSM only
    ax = axes[0]
    for polyline in filtered_polylines:
        xs = [p[0] for p in polyline]
        ys = [p[1] for p in polyline]
        ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.6)
    ax.set_title(f'OSM Roads Only\n{len(filtered_polylines)} polylines', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Center: Trajectory only
    ax = axes[1]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    ax.set_title('Trajectory Only', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Right: Overlay
    ax = axes[2]
    for idx, polyline in enumerate(filtered_polylines):
        xs = [p[0] for p in polyline]
        ys = [p[1] for p in polyline]
        ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.5, label='OSM' if idx == 0 else '')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5, alpha=0.8, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', zorder=5)
    ax.set_title('OVERLAY: OSM + Trajectory', fontsize=12, fontweight='bold', color='darkgreen')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    output_file = f'osm_pbf_aligned_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_file}")
    
    # Save aligned polylines
    output_pkl = f'osm_polylines_aligned_seq{seq}.pkl'
    with open(output_pkl, 'wb') as f:
        pickle.dump(filtered_polylines, f)
    print(f"✅ Saved aligned polylines to: {output_pkl}")
    
    # Also save as numpy array for easy loading
    # Flatten all points
    all_points = []
    for polyline in filtered_polylines:
        all_points.extend(polyline)
    all_points = np.array(all_points)
    output_npy = f'osm_points_aligned_seq{seq}.npy'
    np.save(output_npy, all_points)
    print(f"✅ Saved points to: {output_npy}")
    
    return filtered_polylines, transform


def parse_and_align_osm_trajectory_only(seq: str = "03", data_root: str = "data"):
    """
    Parse OSM PBF for sequences without OXTS data.
    Uses trajectory extent to define search area (no GPS alignment).
    """
    print(f"Processing sequence {seq} (trajectory-only mode, no OXTS)")
    
    # Load trajectory
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    if not pose_file.exists():
        print(f"Pose file not found: {pose_file}")
        return
    
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses)
    
    print(f"Loaded trajectory: {len(trajectory)} points")
    print(f"Trajectory bounds: X=[{trajectory[:,0].min():.1f}, {trajectory[:,0].max():.1f}], Y=[{trajectory[:,1].min():.1f}, {trajectory[:,1].max():.1f}]")
    
    # Estimate geographic bounds from trajectory
    # Use approximate conversion: Karlsruhe is roughly at lat=49.0, lon=8.4
    # This is a rough estimate - trajectory-only mode won't have perfect alignment
    
    # Get trajectory extent
    x_min, x_max = trajectory[:,0].min(), trajectory[:,0].max()
    y_min, y_max = trajectory[:,1].min(), trajectory[:,1].max()
    
    # Add margin
    margin = 100  # meters
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    # Approximate conversion from local to lat/lon
    # Karlsruhe: 1 degree lat ~ 111km, 1 degree lon ~ 71km at 49N
    # This is very approximate - for sequences without GPS, we just get roads in the general area
    
    # Center of trajectory in local coords
    cx = (trajectory[:,0].min() + trajectory[:,0].max()) / 2
    cy = (trajectory[:,1].min() + trajectory[:,1].max()) / 2
    
    # Approximate center lat/lon for Karlsruhe
    center_lat, center_lon = 49.0, 8.4
    
    # Convert local extent to lat/lon (very approximate)
    lat_extent = (y_max - y_min) / 111000  # degrees
    lon_extent = (x_max - x_min) / (111000 * np.cos(np.radians(49)))  # degrees
    
    bbox = (
        center_lat - lat_extent/2,  # min_lat
        center_lon - lon_extent/2,  # min_lon
        center_lat + lat_extent/2,  # max_lat
        center_lon + lon_extent/2   # max_lon
    )
    
    print(f"\nEstimated bounding box: {bbox}")
    print(f"  Lat: {bbox[0]:.6f} to {bbox[2]:.6f}")
    print(f"  Lon: {bbox[1]:.6f} to {bbox[3]:.6f}")
    
    # Parse OSM PBF
    pbf_file = Path(data_root) / "osm" / "karlsruhe.osm.pbf"
    if not pbf_file.exists():
        print(f"PBF file not found: {pbf_file}")
        return
    
    road_polylines = parse_osm_pbf(str(pbf_file), bounding_box=bbox)
    
    # For trajectory-only mode, we don't have GPS alignment
    # So we just use the raw OSM coordinates transformed to local frame
    # This assumes OSM is already roughly aligned (which it won't be perfectly)
    
    # Simple transformation: center the OSM data on trajectory center
    # This is a placeholder - real alignment would need manual tuning
    
    print("\n⚠️  Trajectory-only mode: OSM data not GPS-aligned!")
    print("   Using approximate positioning.")
    
    # Transform to local frame (simple centering)
    # Convert OSM lat/lon to approximate local coordinates
    local_polylines = []
    
    for polyline in road_polylines:
        local_polyline = []
        for lat, lon in polyline:
            # Very rough conversion - for display only
            # This won't be accurate without proper GPS alignment
            x = (lon - center_lon) * 111000 * np.cos(np.radians(49))
            y = (lat - center_lat) * 111000
            local_polyline.append((x + cx, y + cy))
        local_polylines.append(np.array(local_polyline))
    
    # Filter to trajectory area
    traj_center = np.mean(trajectory, axis=0)
    search_radius = np.max(np.linalg.norm(trajectory - traj_center, axis=1)) + 100
    
    filtered_polylines = filter_polylines_to_area(local_polylines, traj_center, search_radius)
    print(f"Filtered to {len(filtered_polylines)} polylines in trajectory area")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Left: OSM only
    ax = axes[0]
    for polyline in filtered_polylines:
        xs = [p[0] for p in polyline]
        ys = [p[1] for p in polyline]
        ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.6)
    ax.set_title(f'OSM Roads Only\n{len(filtered_polylines)} polylines', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Center: Trajectory only
    ax = axes[1]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    ax.set_title('Trajectory Only', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Right: Overlay
    ax = axes[2]
    for polyline in filtered_polylines:
        xs = [p[0] for p in polyline]
        ys = [p[1] for p in polyline]
        ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.5, label='OSM' if polyline == filtered_polylines[0] else '')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5, alpha=0.8, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', zorder=5)
    ax.set_title('OVERLAY: OSM + Trajectory (UNALIGNED)', fontsize=12, fontweight='bold', color='orange')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    output_file = f'osm_pbf_aligned_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_file}")
    
    # Save polylines
    output_pkl = f'osm_polylines_aligned_seq{seq}.pkl'
    with open(output_pkl, 'wb') as f:
        pickle.dump(filtered_polylines, f)
    print(f"✅ Saved polylines to: {output_pkl}")
    
    # Save points
    all_points = []
    for polyline in filtered_polylines:
        all_points.extend(polyline)
    all_points = np.array(all_points)
    output_npy = f'osm_points_aligned_seq{seq}.npy'
    np.save(output_npy, all_points)
    print(f"✅ Saved points to: {output_npy}")
    
    return filtered_polylines, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse full OSM PBF and align with trajectory")
    parser.add_argument("--seq", type=str, default="00", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root")
    parser.add_argument("--trajectory-only", action="store_true", help="Use trajectory-only mode (no OXTS)")
    
    args = parser.parse_args()
    
    # Sequences without OXTS data: 03, 04, 06
    if args.trajectory_only or args.seq in ['03', '04', '06']:
        parse_and_align_osm_trajectory_only(args.seq, args.data_root)
    else:
        parse_and_align_osm(args.seq, args.data_root)
    print("\n🎉 Done!")
