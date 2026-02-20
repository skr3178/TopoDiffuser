#!/usr/bin/env python3
"""
Extract OSM road polylines for specific KITTI sequences from PBF file.
"""

import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def load_poses(pose_file):
    """Load KITTI poses."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses):
    """Extract (x, z) trajectory."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def get_bounding_box_from_trajectory(trajectory, oxts_data, frame_offset, margin_m=1000):
    """
    Get lat/lon bounding box from trajectory with margin.
    
    Returns (min_lat, min_lon, max_lat, max_lon)
    """
    # Get GPS reference from OXTS
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)
    
    # Offset to align trajectory[0] with (0,0)
    offset_east = east0 - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]
    
    # Convert trajectory to UTM
    traj_utm = np.array([
        [trajectory[i, 0] + offset_east, trajectory[i, 1] + offset_north]
        for i in range(len(trajectory))
    ])
    
    # Get UTM bounds with margin
    east_min = traj_utm[:, 0].min() - margin_m
    east_max = traj_utm[:, 0].max() + margin_m
    north_min = traj_utm[:, 1].min() - margin_m
    north_max = traj_utm[:, 1].max() + margin_m
    
    # Simple inverse UTM (approximate) - convert back to lat/lon
    # This is a rough approximation using the local UTM zone
    zone = 32  # Karlsruhe is in UTM zone 32
    
    def utm_to_latlon_approx(east, north, zone=32):
        """Approximate UTM to lat/lon conversion (good enough for bbox)."""
        # Reference point
        lat0, lon0 = 49.0, 8.4  # Karlsruhe approximate
        east0, north0 = latlon_to_utm(lat0, lon0)
        
        # Approximate conversion (1 degree ~ 111km)
        dlat = (north - north0) / 111000.0
        dlon = (east - east0) / (111000.0 * np.cos(np.radians(lat0)))
        
        return lat0 + dlat, lon0 + dlon
    
    lat_min, lon_min = utm_to_latlon_approx(east_min, north_min)
    lat_max, lon_max = utm_to_latlon_approx(east_max, north_max)
    
    # Ensure correct ordering
    min_lat, max_lat = min(lat_min, lat_max), max(lat_min, lat_max)
    min_lon, max_lon = min(lon_min, lon_max), max(lon_min, lon_max)
    
    return (min_lat, min_lon, max_lat, max_lon)


def parse_osm_pbf(pbf_file, bounding_box=None):
    """
    Parse OSM PBF file and extract road polylines.
    """
    from pyrosm import OSM
    
    print(f"Loading OSM from: {pbf_file}")
    
    if bounding_box:
        # Bounding box passed to constructor: (min_lon, min_lat, max_lon, max_lat)
        min_lat, min_lon, max_lat, max_lon = bounding_box
        osm = OSM(pbf_file, bounding_box=[min_lon, min_lat, max_lon, max_lat])
        print(f"Using bounding box: lat=[{min_lat:.4f}, {max_lat:.4f}], lon=[{min_lon:.4f}, {max_lon:.4f}]")
    else:
        osm = OSM(pbf_file)
    
    print("Extracting driving network...")
    network = osm.get_network(network_type="driving")
    
    print(f"Found {len(network)} road segments")
    
    # Extract polylines
    road_polylines = []
    
    for idx, row in network.iterrows():
        geom = row['geometry']
        
        if geom is None:
            continue
        
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                polyline = [(lat, lon) for lon, lat in coords]
                road_polylines.append(polyline)
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    polyline = [(lat, lon) for lon, lat in coords]
                    road_polylines.append(polyline)
    
    print(f"Extracted {len(road_polylines)} polylines")
    return road_polylines


def extract_for_sequence(seq, pbf_file, oxts_dir, pose_file, frame_offset):
    """Extract OSM polylines for a specific sequence."""
    print(f"\n{'='*60}")
    print(f"Seq {seq}: Extracting OSM polylines")
    print(f"{'='*60}")
    
    # Load trajectory
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    print(f"  x range: [{trajectory[:,0].min():.1f}, {trajectory[:,0].max():.1f}]")
    print(f"  z range: [{trajectory[:,1].min():.1f}, {trajectory[:,1].max():.1f}]")
    
    # Load OXTS data
    try:
        oxts_data = load_oxts_data(oxts_dir)
        print(f"OXTS: {len(oxts_data)} frames")
    except Exception as e:
        print(f"❌ Error loading OXTS: {e}")
        return None
    
    # Get bounding box
    bbox = get_bounding_box_from_trajectory(trajectory, oxts_data, frame_offset, margin_m=500)
    print(f"Bounding box (lat/lon): {bbox}")
    print(f"  lat: [{bbox[0]:.4f}, {bbox[2]:.4f}]")
    print(f"  lon: [{bbox[1]:.4f}, {bbox[3]:.4f}]")
    
    # Extract OSM
    polylines = parse_osm_pbf(pbf_file, bbox)
    
    # Save
    output_file = f'osm_polylines_latlon_seq{seq}_regbez.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(polylines, f)
    print(f"✓ Saved: {output_file} ({len(polylines)} polylines)")
    
    return polylines


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['00', '05', '09', '10'])
    parser.add_argument('--pbf', default='data/osm/karlsruhe-regbez.osm.pbf')
    args = parser.parse_args()
    
    # Sequence configurations
    SEQ_CONFIG = {
        '00': {
            'raw': '2011_10_03_drive_0027_sync',
            'date': '2011_10_03',
            'frame_offset': 3346
        },
        '01': {
            'raw': '2011_10_03_drive_0027_sync',
            'date': '2011_10_03',
            'frame_offset': 1850  # Discovered from heading match
        },
        '05': {
            'raw': '2011_09_30_drive_0018_sync',
            'date': '2011_09_30',
            'frame_offset': 46
        },
        '09': {
            'raw': '2011_09_30_drive_0033_sync',
            'date': '2011_09_30',
            'frame_offset': 1497
        },
        '10': {
            'raw': '2011_09_30_drive_0034_sync',
            'date': '2011_09_30',
            'frame_offset': 0
        }
    }
    
    for seq in args.seqs:
        if seq not in SEQ_CONFIG:
            print(f"❌ Unknown sequence: {seq}")
            continue
        
        cfg = SEQ_CONFIG[seq]
        oxts_dir = f"data/raw_data/{cfg['raw']}/{cfg['date']}/{cfg['raw']}/oxts/data"
        pose_file = f'data/kitti/poses/{seq}.txt'
        
        extract_for_sequence(seq, args.pbf, oxts_dir, pose_file, cfg['frame_offset'])
