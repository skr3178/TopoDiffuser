#!/usr/bin/env python3
"""
Refine OSM alignment using grid search on rotation.

The initial alignment may have rotational errors. This script searches
for the rotation that minimizes the distance between trajectory and OSM roads.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree

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


def rotate_polylines_around_center(polylines, center, angle):
    """Rotate polylines around a center point."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cx, cy = center
    
    rotated = []
    for polyline in polylines:
        # Translate to origin
        x = polyline[:, 0] - cx
        y = polyline[:, 1] - cy
        
        # Rotate
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        
        # Translate back
        rotated.append(np.column_stack([x_rot + cx, y_rot + cy]))
    
    return rotated


def compute_alignment_error(trajectory, polylines):
    """
    Compute mean distance from trajectory points to nearest OSM polyline.
    """
    # Build KD-tree from all polyline points
    all_points = []
    for pl in polylines:
        if len(pl) > 0:
            all_points.extend(pl)
    
    if len(all_points) == 0:
        return float('inf')
    
    all_points = np.array(all_points)
    tree = cKDTree(all_points)
    
    # Query distance for each trajectory point
    distances, _ = tree.query(trajectory, k=1)
    
    return np.mean(distances)


def find_best_rotation(trajectory, polylines, 
                       rotation_range=(-45, 45), n_samples=91):
    """
    Find best rotation by searching over a range.
    
    Rotates polylines around the trajectory center.
    
    Returns:
        best_rotation: Rotation in radians
        best_error: Alignment error with best rotation
        rotations_deg: Array of tested rotations
        errors: Array of errors for each rotation
    """
    rotations_deg = np.linspace(rotation_range[0], rotation_range[1], n_samples)
    traj_center = np.mean(trajectory, axis=0)
    
    errors = []
    for rot_deg in rotations_deg:
        rot_rad = np.radians(rot_deg)
        
        # Rotate polylines around trajectory center
        rotated = rotate_polylines_around_center(polylines, traj_center, rot_rad)
        
        # Compute error
        error = compute_alignment_error(trajectory, rotated)
        errors.append(error)
    
    errors = np.array(errors)
    best_idx = np.argmin(errors)
    best_rotation = np.radians(rotations_deg[best_idx])
    best_error = errors[best_idx]
    
    return best_rotation, best_error, rotations_deg, errors


def refine_alignment(seq: str, data_root: str = "data"):
    """Refine OSM alignment for a sequence."""
    
    # Map sequence to raw data
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
    
    # Find OXTS
    oxts_dir = None
    for date_prefix in ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']:
        candidate = raw_data_root / raw_folder / date_prefix / raw_folder / "oxts" / "data"
        if candidate.exists():
            oxts_dir = candidate
            break
    
    if not oxts_dir:
        candidate = raw_data_root / raw_folder / "oxts" / "data"
        if candidate.exists():
            oxts_dir = candidate
    
    if not oxts_dir:
        print(f"OXTS not found for sequence {seq}")
        return
    
    print(f"\n{'='*60}")
    print(f"Refining alignment for sequence {seq}")
    print(f"{'='*60}")
    
    # Load data
    oxts_data = load_oxts_data(oxts_dir)
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses[:len(oxts_data)])
    
    # Load OSM polylines (already transformed to local coords)
    pkl_file = Path(f"osm_polylines_aligned_seq{seq}.pkl")
    if not pkl_file.exists():
        print(f"OSM polylines not found: {pkl_file}")
        return
    
    with open(pkl_file, 'rb') as f:
        polylines = pickle.load(f)
    
    # Convert to numpy arrays
    polylines = [np.array(pl) for pl in polylines]
    
    print(f"Loaded {len(polylines)} polylines")
    print(f"Trajectory: {len(trajectory)} points")
    
    # Compute initial error
    initial_error = compute_alignment_error(trajectory, polylines)
    print(f"Initial alignment error: {initial_error:.2f}m")
    
    # Find best rotation
    print("\nSearching for best rotation...")
    best_rotation, best_error, rotations_deg, errors = find_best_rotation(
        trajectory, polylines,
        rotation_range=(-45, 45), n_samples=181
    )
    
    print(f"Best rotation adjustment: {np.degrees(best_rotation):.2f}°")
    print(f"Improved error: {best_error:.2f}m (was {initial_error:.2f}m)")
    
    # Apply best rotation
    traj_center = np.mean(trajectory, axis=0)
    refined_polylines = rotate_polylines_around_center(polylines, traj_center, best_rotation)
    
    # Filter to trajectory area
    search_radius = np.max(np.linalg.norm(trajectory - traj_center, axis=1)) + 100
    
    filtered_polylines = []
    for polyline in refined_polylines:
        distances = np.sqrt((polyline[:, 0] - traj_center[0])**2 + (polyline[:, 1] - traj_center[1])**2)
        if np.any(distances <= search_radius):
            filtered_polylines.append(polyline)
    
    print(f"Filtered to {len(filtered_polylines)} polylines")
    
    # Visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top left: Error vs rotation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rotations_deg, errors, 'b-', linewidth=2)
    ax1.axvline(np.degrees(best_rotation), color='r', linestyle='--', 
                label=f'Best: {np.degrees(best_rotation):.1f}°')
    ax1.set_xlabel('Rotation Adjustment (degrees)')
    ax1.set_ylabel('Mean Distance Error (m)')
    ax1.set_title('Rotation Search', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top middle: OSM only
    ax2 = fig.add_subplot(gs[0, 1])
    for polyline in filtered_polylines:
        ax2.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=0.8, alpha=0.6)
    ax2.set_title(f'OSM Roads ({len(filtered_polylines)} polylines)', fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Top right: Trajectory only
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax3.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    ax3.set_title('Trajectory', fontweight='bold')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Bottom row: Full overlay with zoom
    ax4 = fig.add_subplot(gs[1, :])
    for idx, polyline in enumerate(filtered_polylines):
        ax4.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1.0, alpha=0.5, 
                label='OSM' if idx == 0 else '')
    ax4.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5, alpha=0.9, label='Trajectory')
    ax4.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=150, marker='o', zorder=5, label='Start')
    ax4.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=200, marker='*', zorder=5, label='End')
    
    # Add some trajectory points to show alignment
    step = max(1, len(trajectory) // 20)
    ax4.scatter(trajectory[::step, 0], trajectory[::step, 1], c='orange', s=30, alpha=0.5, zorder=4)
    
    improvement = initial_error - best_error
    ax4.set_title(f'REFINED: Seq {seq} | Rotation: {np.degrees(best_rotation):.1f}° | '
                  f'Error: {best_error:.1f}m (↓{improvement:.1f}m)', 
                  fontsize=14, fontweight='bold', color='darkgreen')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    plt.savefig(f'osm_pbf_aligned_seq{seq}_refined.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved refined visualization to: osm_pbf_aligned_seq{seq}_refined.png")
    
    # Save refined polylines
    output_pkl = f'osm_polylines_aligned_seq{seq}_refined.pkl'
    with open(output_pkl, 'wb') as f:
        pickle.dump(filtered_polylines, f)
    print(f"✅ Saved refined polylines to: {output_pkl}")
    
    # Save transform
    transform = {
        'rotation_adjustment': best_rotation,
        'center': traj_center,
        'initial_error': initial_error,
        'refined_error': best_error
    }
    
    output_transform = f'osm_transform_seq{seq}.pkl'
    with open(output_transform, 'wb') as f:
        pickle.dump(transform, f)
    print(f"✅ Saved transform to: {output_transform}")
    
    return best_rotation, best_error


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine OSM alignment using grid search")
    parser.add_argument("--seq", type=str, default="01", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root")
    
    args = parser.parse_args()
    
    refine_alignment(args.seq, args.data_root)
    print("\n🎉 Done!")
