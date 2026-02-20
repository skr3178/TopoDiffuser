#!/usr/bin/env python3
"""
Extract road polylines from Germany OSM file for all KITTI sequences.
This extracts once and saves to a file that can be reused.
"""

import numpy as np
import pickle
import sys
from pathlib import Path
from pyrosm import OSM

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def load_poses(pose_file):
    """Load KITTI poses."""
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def extract_trajectory(poses):
    """Extract (x, z) trajectory."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def find_oxts_dir(raw_folder, data_root='data'):
    """Find OXTS data directory."""
    root = Path(data_root) / 'raw_data'
    for date in ['2011_10_03', '2011_09_30', '2011_09_29', '2011_09_28', '2011_09_26']:
        candidate = root / raw_folder / date / raw_folder / 'oxts' / 'data'
        if candidate.exists():
            return candidate
    candidate = root / raw_folder / 'oxts' / 'data'
    if candidate.exists():
        return candidate
    return None


def get_combined_bounding_box(sequences, data_root='data'):
    """Get a combined bounding box for all sequences."""
    SEQ_TO_RAW = {
        '00': '2011_10_03_drive_0027_sync',
        '01': '2011_10_03_drive_0042_sync',
        '02': '2011_10_03_drive_0034_sync',
        '07': '2011_09_30_drive_0027_sync',
        '08': '2011_09_30_drive_0028_sync',
        '09': '2011_09_30_drive_0033_sync',
        '10': '2011_09_30_drive_0034_sync',
    }
    SEQ_FRAME_OFFSET = {
        '00': 3346, '01': 0, '02': 57, '07': 42,
        '08': 252, '09': 1497, '10': 0,
    }
    
    all_lats = []
    all_lons = []
    
    for seq in sequences:
        print(f"  Getting bbox for seq {seq}...")
        pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
        trajectory = extract_trajectory(load_poses(pose_file))
        
        raw_folder = SEQ_TO_RAW[seq]
        frame_offset = SEQ_FRAME_OFFSET.get(seq, 0)
        oxts_dir = find_oxts_dir(raw_folder, data_root)
        
        if oxts_dir is None:
            print(f"    ⚠️  OXTS not found, skipping")
            continue
            
        oxts_data = load_oxts_data(str(oxts_dir))
        
        # Get reference frame
        ref_frame = min(frame_offset, len(oxts_data) - 1)
        lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
        east0, north0 = latlon_to_utm(lat0, lon0)
        
        # Offset to align
        offset_east = east0 - trajectory[0, 0]
        offset_north = north0 - trajectory[0, 1]
        
        # Convert trajectory to UTM
        traj_utm = np.array([
            [trajectory[i, 0] + offset_east, trajectory[i, 1] + offset_north]
            for i in range(len(trajectory))
        ])
        
        # Get bounds with 1000m margin
        margin = 1000
        east_min = traj_utm[:, 0].min() - margin
        east_max = traj_utm[:, 0].max() + margin
        north_min = traj_utm[:, 1].min() - margin
        north_max = traj_utm[:, 1].max() + margin
        
        # Approximate UTM to lat/lon
        def utm_to_latlon_approx(east, north):
            lat_ref, lon_ref = 49.0, 8.4
            east_ref, north_ref = latlon_to_utm(lat_ref, lon_ref)
            dlat = (north - north_ref) / 111000.0
            dlon = (east - east_ref) / (111000.0 * np.cos(np.radians(lat_ref)))
            return lat_ref + dlat, lon_ref + dlon
        
        lat_min, lon_min = utm_to_latlon_approx(east_min, north_min)
        lat_max, lon_max = utm_to_latlon_approx(east_max, north_max)
        
        all_lats.extend([lat_min, lat_max])
        all_lons.extend([lon_min, lon_max])
        print(f"    lat=[{lat_min:.4f}, {lat_max:.4f}], lon=[{lon_min:.4f}, {lon_max:.4f}]")
    
    combined_bbox = (min(all_lats), min(all_lons), max(all_lats), max(all_lons))
    print(f"\nCombined bounding box:")
    print(f"  lat=[{combined_bbox[0]:.4f}, {combined_bbox[2]:.4f}]")
    print(f"  lon=[{combined_bbox[1]:.4f}, {combined_bbox[3]:.4f}]")
    
    return combined_bbox


def extract_roads(pbf_file, bounding_box, output_file):
    """Extract road polylines from OSM PBF."""
    print(f"\nLoading OSM from: {pbf_file}")
    print(f"This may take a few minutes for large files...")
    
    min_lat, min_lon, max_lat, max_lon = bounding_box
    osm = OSM(pbf_file, bounding_box=[min_lon, min_lat, max_lon, max_lat])
    
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
                # Store as (lat, lon) tuples
                polyline = [(lat, lon) for lon, lat in coords]
                road_polylines.append(polyline)
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    polyline = [(lat, lon) for lon, lat in coords]
                    road_polylines.append(polyline)
    
    print(f"Extracted {len(road_polylines)} polylines")
    
    # Save
    with open(output_file, 'wb') as f:
        pickle.dump({
            'polylines': road_polylines,
            'bbox': bounding_box
        }, f)
    print(f"Saved to: {output_file}")
    
    return road_polylines


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbf', default='data/osm/germany-latest.osm.pbf')
    parser.add_argument('--output', default='data/osm/germany_roads_karlsruhe_area.pkl')
    parser.add_argument('--seqs', nargs='+', default=['00', '01', '02', '07', '08', '09', '10'])
    args = parser.parse_args()
    
    print("="*70)
    print("Extract Road Polylines from Germany OSM")
    print("="*70)
    
    # Get combined bounding box for all sequences
    print("\nCalculating combined bounding box...")
    bbox = get_combined_bounding_box(args.seqs)
    
    # Extract roads
    roads = extract_roads(args.pbf, bbox, args.output)
    
    print("\n" + "="*70)
    print("Extraction complete!")
    print(f"Total polylines: {len(roads)}")
    print("="*70)


if __name__ == '__main__':
    main()
