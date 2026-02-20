#!/usr/bin/env python3
"""
Find optimal rotation angle for each sequence by testing different angles.
"""

import numpy as np
import pickle
import sys
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def find_optimal_rotation(seq, latlon_pkl, oxts_dir, pose_file, frame_offset, anchor='start'):
    """Find optimal rotation angle for a sequence."""
    print(f"\n{'='*60}")
    print(f"Seq {seq} - Finding optimal rotation (anchor={anchor})")
    print(f"{'='*60}")
    
    # Load trajectory
    trajectory = np.array([[p[0,3], p[2,3]] for p in load_poses(pose_file)])
    
    # Load OXTS and compute offset
    oxts_data = load_oxts_data(oxts_dir)
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)
    offset_east = east0 - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]
    
    # Load and convert OSM
    with open(latlon_pkl, 'rb') as f:
        latlon_polylines = pickle.load(f)
    
    local_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        local_polylines.append(pts)
    
    # Filter to trajectory area
    xmin, xmax = trajectory[:,0].min()-400, trajectory[:,0].max()+400
    ymin, ymax = trajectory[:,1].min()-400, trajectory[:,1].max()+400
    local_polylines = [pl for pl in local_polylines 
                       if np.any((pl[:,0]>=xmin)&(pl[:,0]<=xmax)&
                                 (pl[:,1]>=ymin)&(pl[:,1]<=ymax))]
    
    print(f"Trajectory: {len(trajectory)} frames")
    print(f"OSM polylines: {len(local_polylines)} (after filtering)")
    
    # Sample OSM points
    all_pts = np.vstack(local_polylines)
    
    # Test angles
    pivot = trajectory[0] if anchor == 'start' else trajectory[-1]
    all_pts_c = all_pts - pivot
    
    print(f"\nTesting angles from 0° to 180° (anchor={anchor}):")
    print("-" * 60)
    
    results = []
    for angle_deg in range(0, 181, 3):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        xr = all_pts_c[:, 0] * c - all_pts_c[:, 1] * s + pivot[0]
        yr = all_pts_c[:, 0] * s + all_pts_c[:, 1] * c + pivot[1]
        rotated_pts = np.column_stack([xr, yr])
        
        tree = cKDTree(rotated_pts)
        dists, _ = tree.query(trajectory, k=1)
        
        mean_err = dists.mean()
        max_err = dists.max()
        start_err = dists[0]
        end_err = dists[-1]
        
        results.append((angle_deg, mean_err, max_err, start_err, end_err))
    
    # Sort by mean error
    results.sort(key=lambda x: x[1])
    
    print("Top 5 angles by mean error:")
    for angle, mean_err, max_err, start_err, end_err in results[:5]:
        print(f"  {angle:3d}°: mean={mean_err:.2f}m, max={max_err:.2f}m, start={start_err:.2f}m, end={end_err:.2f}m")
    
    # Best angle
    best = results[0]
    print(f"\n✓ Best: {best[0]}° with mean error {best[1]:.2f}m")
    
    return best[0], best[1]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['00', '05', '09', '10'])
    args = parser.parse_args()
    
    # Configurations
    SEQ_CONFIG = {
        '00': {
            'raw': '2011_10_03_drive_0027_sync',
            'date': '2011_10_03',
            'frame_offset': 3346,
            'anchor': 'start'
        },
        '02': {
            'raw': '2011_10_03_drive_0034_sync',
            'date': '2011_10_03',
            'frame_offset': 57,
            'anchor': 'start'
        },
        '05': {
            'raw': '2011_09_30_drive_0018_sync',
            'date': '2011_09_30',
            'frame_offset': 46,
            'anchor': 'start'
        },
        '07': {
            'raw': '2011_09_30_drive_0027_sync',
            'date': '2011_09_30',
            'frame_offset': 42,
            'anchor': 'end'
        },
        '08': {
            'raw': '2011_09_30_drive_0028_sync',
            'date': '2011_09_30',
            'frame_offset': 252,
            'anchor': 'start'
        },
        '09': {
            'raw': '2011_09_30_drive_0033_sync',
            'date': '2011_09_30',
            'frame_offset': 1497,
            'anchor': 'start'
        },
        '10': {
            'raw': '2011_09_30_drive_0034_sync',
            'date': '2011_09_30',
            'frame_offset': 0,
            'anchor': 'start'
        },
    }
    
    all_results = {}
    
    for seq in args.seqs:
        if seq not in SEQ_CONFIG:
            print(f"❌ Unknown sequence: {seq}")
            continue
        
        cfg = SEQ_CONFIG[seq]
        latlon_pkl = f'osm_polylines_latlon_seq{seq}_regbez.pkl'
        oxts_dir = f"data/raw_data/{cfg['raw']}/{cfg['date']}/{cfg['raw']}/oxts/data"
        pose_file = f'data/kitti/poses/{seq}.txt'
        
        if not Path(latlon_pkl).exists():
            print(f"❌ File not found: {latlon_pkl}")
            continue
        
        angle, error = find_optimal_rotation(
            seq, latlon_pkl, oxts_dir, pose_file, 
            cfg['frame_offset'], cfg['anchor']
        )
        
        all_results[seq] = {'angle': angle, 'error': error, 'anchor': cfg['anchor']}
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Update SEQ_ROTATION_HINT in align_and_viz_bestfit.py:")
    print("="*60)
    for seq, result in all_results.items():
        print(f"    '{seq}': ({result['angle']}.0, 'force'),  # anchor={result['anchor']}, error={result['error']:.1f}m")
