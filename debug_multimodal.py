#!/usr/bin/env python3
"""
Debug script for multimodal pipeline.

Tests OSM alignment, history trajectory, and generates visualizations.
"""

import sys
import numpy as np
from pathlib import Path
import pickle
import argparse

sys.path.insert(0, 'models')
sys.path.insert(0, 'utils')

from utils.bev_multimodal import (
    latlon_to_utm,
    compute_gps_to_local_transform,
    lidar_to_bev,
    trajectory_to_bev,
    osm_to_bev,
    create_multimodal_input,
    evaluate_osm_alignment,
    evaluate_trajectory_alignment,
    visualize_multimodal_bev
)


def test_gps_alignment(sequence='00', data_root='data/kitti', raw_data_root='data/raw_data'):
    """Test GPS to local frame alignment for a sequence."""
    print("=" * 70)
    print(f"GPS Alignment Test - Sequence {sequence}")
    print("=" * 70)
    
    # Mapping from odometry to raw data
    ODOMETRY_TO_RAW = {
        '00': ('2011_10_03', '0027'),
        '01': ('2011_10_03', '0042'),
        '02': ('2011_10_03', '0034'),
        '03': ('2011_09_26', '0067'),
        '04': ('2011_09_30', '0016'),
        '05': ('2011_09_30', '0018'),
        '06': ('2011_09_30', '0020'),
        '07': ('2011_09_30', '0027'),
        '08': ('2011_09_30', '0028'),
        '09': ('2011_09_30', '0033'),
        '10': ('2011_09_30', '0034'),
    }
    
    if sequence not in ODOMETRY_TO_RAW:
        print(f"Unknown sequence: {sequence}")
        return
    
    date, drive = ODOMETRY_TO_RAW[sequence]
    raw_path = Path(raw_data_root) / f"{date}_drive_{drive}_sync" / date / f"{date}_drive_{drive}_sync"
    
    # Load OXTS data
    oxts_dir = raw_path / "oxts" / "data"
    if not oxts_dir.exists():
        print(f"❌ OXTS data not found at {oxts_dir}")
        return
    
    print(f"\nLoading OXTS data from {oxts_dir}...")
    oxts_files = sorted(oxts_dir.glob("*.txt"))
    oxts_data = []
    for f in oxts_files[:100]:  # Load first 100 frames
        values = np.loadtxt(f)
        oxts_data.append(values)
    oxts_data = np.array(oxts_data)
    print(f"✓ Loaded {len(oxts_data)} OXTS frames")
    
    # Load poses
    poses_path = Path(data_root) / 'poses' / f'{sequence}.txt'
    if not poses_path.exists():
        print(f"❌ Poses not found at {poses_path}")
        return
    
    poses = np.loadtxt(poses_path)
    poses = poses[:len(oxts_data)]  # Match lengths
    print(f"✓ Loaded {len(poses)} poses")
    
    # Compute transform
    print("\nComputing GPS to local transform...")
    transform = compute_gps_to_local_transform(oxts_data, poses)
    
    print(f"\nAlignment Results:")
    print(f"  Offset East:  {transform['offset_east']:.2f} m")
    print(f"  Offset North: {transform['offset_north']:.2f} m")
    print(f"  Mean Error:   {transform['mean_error']:.2f} m")
    print(f"  Max Error:    {transform['max_error']:.2f} m")
    print(f"  Std Error:    {transform['std_error']:.2f} m")
    
    if transform['mean_error'] < 1.0:
        print(f"  ✅ Excellent alignment")
    elif transform['mean_error'] < 5.0:
        print(f"  ⚠️  Acceptable alignment")
    else:
        print(f"  ❌ Poor alignment - check GPS data")
    
    return transform


def test_multimodal_bev(sequence='00', frame_idx=100, 
                        data_root='data/kitti',
                        raw_data_root='data/raw_data',
                        viz_dir='viz/debug'):
    """Test multimodal BEV generation for a single frame."""
    print("\n" + "=" * 70)
    print(f"Multimodal BEV Test - Seq {sequence}, Frame {frame_idx}")
    print("=" * 70)
    
    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    
    # Load LiDAR
    lidar_path = Path(data_root) / 'sequences' / sequence / 'velodyne' / f'{frame_idx:06d}.bin'
    if lidar_path.exists():
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        lidar_bev = lidar_to_bev(points)
        print(f"✓ LiDAR BEV: {lidar_bev.shape}")
    else:
        print(f"❌ LiDAR not found: {lidar_path}")
        lidar_bev = np.zeros((3, 300, 400), dtype=np.float32)
    
    # Load poses
    poses_path = Path(data_root) / 'poses' / f'{sequence}.txt'
    poses = np.loadtxt(poses_path)
    current_pose = poses[frame_idx].reshape(3, 4)
    
    # Load history
    past_poses = poses[max(0, frame_idx-50):frame_idx+1]
    trajectory = []
    for pose in past_poses:
        pose_mat = pose.reshape(3, 4)
        trajectory.append([pose_mat[0, 3], pose_mat[1, 3]])
    trajectory = np.array(trajectory)
    
    history_bev, history_debug = trajectory_to_bev(
        trajectory, current_pose, past_poses, debug=True
    )
    print(f"✓ History BEV: {history_bev.shape}")
    print(f"  Waypoints: {history_debug['num_waypoints']}")
    print(f"  In bounds: {history_debug['waypoints_in_bounds']}")
    print(f"  Coverage: {history_debug['coverage_percent']:.2f}%")
    
    # Load future trajectory for evaluation
    future_poses = poses[frame_idx:frame_idx+80]
    future_traj = []
    for pose in future_poses[::10]:  # Sample every 10 frames
        pose_mat = pose.reshape(3, 4)
        # Transform to ego frame
        R = current_pose[:, :3]
        t = current_pose[:, 3]
        pos = pose_mat[:, 3]
        ego_pos = R.T @ (pos - t)
        future_traj.append(ego_pos[:2])
    future_traj = np.array(future_traj[:8])  # First 8 waypoints
    
    # Evaluate trajectory continuity
    traj_metrics = evaluate_trajectory_alignment(history_bev, future_traj, current_pose)
    print(f"\n  Trajectory Alignment:")
    print(f"    Continuity error: {traj_metrics['continuity_error']:.2f}m")
    print(f"    Is continuous: {traj_metrics['is_continuous']}")
    print(f"    Mean curvature: {traj_metrics['mean_curvature']:.3f} rad")
    
    # Create placeholder OSM (would load real OSM data here)
    osm_bev = np.zeros((1, 300, 400), dtype=np.float32)
    print(f"✓ OSM BEV: {osm_bev.shape} (placeholder)")
    
    # Create multimodal input
    multimodal_input, input_debug = create_multimodal_input(
        lidar_bev, history_bev, osm_bev, debug=True
    )
    print(f"\n✓ Multimodal input: {multimodal_input.shape}")
    print(f"  LiDAR coverage: {input_debug['lidar_coverage']*100:.2f}%")
    print(f"  History coverage: {input_debug['history_coverage']*100:.2f}%")
    print(f"  OSM coverage: {input_debug['osm_coverage']*100:.2f}%")
    
    # Visualize
    save_path = Path(viz_dir) / f"{sequence}_{frame_idx:06d}_debug.png"
    visualize_multimodal_bev(
        lidar_bev, history_bev, osm_bev,
        save_path=str(save_path),
        metrics=traj_metrics
    )
    print(f"\n✓ Visualization saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Debug multimodal pipeline'
    )
    parser.add_argument('--sequence', type=str, default='00',
                       help='Sequence to test')
    parser.add_argument('--frame', type=int, default=100,
                       help='Frame index to test')
    parser.add_argument('--data_root', type=str, default='data/kitti',
                       help='Path to KITTI data')
    parser.add_argument('--raw_data_root', type=str, default='data/raw_data',
                       help='Path to KITTI raw data')
    parser.add_argument('--viz_dir', type=str, default='viz/debug',
                       help='Visualization output directory')
    parser.add_argument('--test_gps', action='store_true',
                       help='Test GPS alignment')
    parser.add_argument('--test_bev', action='store_true',
                       help='Test multimodal BEV generation')
    parser.add_argument('--test_all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.test_all or args.test_gps:
        test_gps_alignment(args.sequence, args.data_root, args.raw_data_root)
    
    if args.test_all or args.test_bev:
        test_multimodal_bev(
            args.sequence, args.frame,
            args.data_root, args.raw_data_root,
            args.viz_dir
        )
    
    if not (args.test_gps or args.test_bev or args.test_all):
        print("Usage:")
        print("  python debug_multimodal.py --test_all")
        print("  python debug_multimodal.py --test_gps --sequence 00")
        print("  python debug_multimodal.py --test_bev --sequence 00 --frame 100")


if __name__ == "__main__":
    main()
