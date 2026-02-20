#!/usr/bin/env python3
"""
Find correct OXTS frame offsets for all sequences by matching heading.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm

# KITTI odometry to raw drive mapping
SEQ_TO_RAW = {
    '00': '2011_10_03_drive_0027_sync',
    '01': '2011_10_03_drive_0042_sync',
    '02': '2011_10_03_drive_0034_sync',
    '03': '2011_09_26_drive_0067_sync',  # Different date
    '04': '2011_09_30_drive_0016_sync',
    '05': '2011_09_30_drive_0018_sync',
    '06': '2011_09_30_drive_0020_sync',
    '07': '2011_09_30_drive_0027_sync',
    '08': '2011_09_30_drive_0028_sync',
    '09': '2011_09_30_drive_0033_sync',
    '10': '2011_09_30_drive_0034_sync',  # Note: seq 02 also uses drive_0034
}

SEQ_TO_DATE = {
    '00': '2011_10_03',
    '01': '2011_10_03',
    '02': '2011_10_03',
    '03': '2011_09_26',
    '04': '2011_09_30',
    '05': '2011_09_30',
    '06': '2011_09_30',
    '07': '2011_09_30',
    '08': '2011_09_30',
    '09': '2011_09_30',
    '10': '2011_09_30',
}


def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def find_oxts_dir(raw_folder, date):
    """Find OXTS data directory."""
    root = Path('data/raw_data')
    # Try standard structure
    candidate = root / raw_folder / date / raw_folder / 'oxts' / 'data'
    if candidate.exists():
        return candidate
    # Try alternative structure
    candidate = root / raw_folder / 'oxts' / 'data'
    if candidate.exists():
        return candidate
    return None


def find_correct_frame(seq):
    """Find OXTS frame matching trajectory heading."""
    print(f"\n{'='*60}")
    print(f"Seq {seq}")
    print(f"{'='*60}")
    
    raw_folder = SEQ_TO_RAW[seq]
    date = SEQ_TO_DATE[seq]
    
    # Load trajectory
    trajectory = np.array([[p[0,3], p[2,3]] for p in load_poses(f'data/kitti/poses/{seq}.txt')])
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Compute trajectory heading from first 50 frames
    N = min(50, len(trajectory)-1)
    dx = trajectory[N, 0] - trajectory[0, 0]
    dz = trajectory[N, 1] - trajectory[0, 1]
    traj_heading = np.arctan2(dz, dx)
    print(f"Trajectory start heading: {np.degrees(traj_heading):.1f}°")
    
    # Find OXTS directory
    oxts_dir = find_oxts_dir(raw_folder, date)
    if oxts_dir is None:
        print(f"❌ OXTS directory not found for {raw_folder}")
        return None
    
    # Load OXTS data
    try:
        oxts_data = load_oxts_data(str(oxts_dir))
    except Exception as e:
        print(f"❌ Error loading OXTS: {e}")
        return None
    
    print(f"OXTS: {len(oxts_data)} frames from {oxts_dir}")
    
    # Search for matching heading
    best_frame = 0
    best_diff = float('inf')
    
    for frame in range(min(len(oxts_data), len(trajectory))):
        oxts_yaw = oxts_data[frame, 5]  # Yaw column
        # Normalize angle difference to [-pi, pi]
        diff = abs(np.arctan2(np.sin(oxts_yaw - traj_heading), 
                              np.cos(oxts_yaw - traj_heading)))
        if diff < best_diff:
            best_diff = diff
            best_frame = frame
    
    # Get GPS at best frame
    lat, lon = oxts_data[best_frame, 0], oxts_data[best_frame, 1]
    east, north = latlon_to_utm(lat, lon)
    
    print(f"Best match: OXTS frame {best_frame}")
    print(f"  Heading diff: {np.degrees(best_diff):.1f}°")
    print(f"  OXTS yaw: {np.degrees(oxts_data[best_frame, 5]):.1f}°")
    print(f"  GPS: lat={lat:.6f}, lon={lon:.6f}")
    print(f"  UTM: east={east:.1f}, north={north:.1f}")
    
    return {
        'seq': seq,
        'raw_folder': raw_folder,
        'date': date,
        'frame_offset': best_frame,
        'heading_diff': np.degrees(best_diff),
        'traj_heading': np.degrees(traj_heading),
        'oxts_yaw': np.degrees(oxts_data[best_frame, 5]),
        'oxts_count': len(oxts_data),
        'traj_count': len(trajectory)
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['00', '02', '05', '07', '09', '10'])
    args = parser.parse_args()
    
    results = []
    for seq in args.seqs:
        result = find_correct_frame(seq)
        if result:
            results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Update these in align_and_viz_bestfit.py:")
    print("="*60)
    print("\nSEQ_TO_RAW = {")
    for r in results:
        print(f"    '{r['seq']}': '{r['raw_folder']}',")
    print("}")
    
    print("\nSEQ_FRAME_OFFSET = {")
    for r in results:
        print(f"    '{r['seq']}': {r['frame_offset']},  # heading_diff={r['heading_diff']:.1f}°")
    print("}")
