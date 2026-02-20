#!/usr/bin/env python3
"""
Refine OSM alignment using Multi-Point Heading Alignment + Combined Metric.

Approaches:
1. Multi-Point Heading: Sample multiple points along trajectory, match headings
2. Combined Metric: error = 0.5 * mean_error + 0.5 * max_error
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

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


def compute_local_heading(points, idx, window=10):
    """Compute heading at trajectory point idx using local window."""
    start = max(0, idx - window)
    end = min(len(points), idx + window + 1)
    if end - start < 2:
        return 0.0
    
    local_points = points[start:end]
    dx = local_points[-1, 0] - local_points[0, 0]
    dy = local_points[-1, 1] - local_points[0, 1]
    return np.arctan2(dy, dx)


def sample_trajectory_points(trajectory, n_samples=8):
    """Sample n points evenly along trajectory."""
    indices = np.linspace(0, len(trajectory) - 1, n_samples, dtype=int)
    points = trajectory[indices]
    headings = [compute_local_heading(trajectory, idx) for idx in indices]
    return indices, points, np.array(headings)


def find_osm_segments_near_point(polylines, point, radius=30.0):
    """Find OSM polylines near a point and their headings."""
    nearby_segments = []
    
    for polyline in polylines:
        # Check if any point in polyline is within radius
        distances = np.linalg.norm(polyline - point, axis=1)
        if np.any(distances <= radius):
            # Compute heading of this polyline segment
            if len(polyline) >= 2:
                # Find closest point on polyline to our point
                closest_idx = np.argmin(distances)
                # Compute local heading around that point
                start = max(0, closest_idx - 2)
                end = min(len(polyline), closest_idx + 3)
                if end - start >= 2:
                    local_seg = polyline[start:end]
                    dx = local_seg[-1, 0] - local_seg[0, 0]
                    dy = local_seg[-1, 1] - local_seg[0, 1]
                    heading = np.arctan2(dy, dx)
                    nearby_segments.append({
                        'polyline': polyline,
                        'heading': heading,
                        'closest_point': polyline[closest_idx],
                        'distance': distances[closest_idx]
                    })
    
    return nearby_segments


def angle_difference(a1, a2):
    """Compute smallest angle difference (handles wraparound)."""
    diff = a1 - a2
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return abs(diff)


def compute_combined_error(trajectory, polylines, weight_mean=0.5, weight_max=0.5):
    """Compute combined mean + max error."""
    all_points = np.vstack(polylines)
    if len(all_points) == 0:
        return float('inf')
    
    tree = cKDTree(all_points)
    distances, _ = tree.query(trajectory, k=1)
    
    mean_err = np.mean(distances)
    max_err = np.max(distances)
    
    return weight_mean * mean_err + weight_max * max_err


def apply_similarity_transform(polylines, scale, rotation, translate):
    """Apply similarity transform to polylines."""
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    transformed = []
    
    for polyline in polylines:
        # Scale
        scaled = polyline * scale
        # Rotate
        x_rot = scaled[:, 0] * cos_r - scaled[:, 1] * sin_r
        y_rot = scaled[:, 0] * sin_r + scaled[:, 1] * cos_r
        rotated = np.column_stack([x_rot, y_rot])
        # Translate
        translated = rotated + translate
        transformed.append(translated)
    
    return transformed


def objective_function(params, trajectory, polylines_base, sample_points, sample_headings, 
                       traj_start, traj_end):
    """
    Objective function for optimization.
    params = [scale, rotation, tx, ty]
    """
    scale = params[0]
    rotation = params[1]
    translate = np.array([params[2], params[3]])
    
    # Apply transform
    transformed = apply_similarity_transform(polylines_base, scale, rotation, translate)
    
    # Combined metric: mean + max
    combined_error = compute_combined_error(trajectory, transformed, 0.5, 0.5)
    
    # Heading alignment penalty
    heading_penalty = 0.0
    for i, (sp, sh) in enumerate(zip(sample_points, sample_headings)):
        # Find nearby OSM segments
        nearby = find_osm_segments_near_point(transformed, sp, radius=30.0)
        if nearby:
            # Find best matching heading
            best_heading_diff = min([angle_difference(sh, seg['heading']) for seg in nearby])
            heading_penalty += best_heading_diff * 10.0  # Weight heading match
    
    # Anchor constraints (start/end should be close)
    all_points = np.vstack(transformed)
    tree = cKDTree(all_points)
    dist_start, _ = tree.query(traj_start)
    dist_end, _ = tree.query(traj_end)
    anchor_penalty = (dist_start + dist_end) * 2.0  # Heavy penalty for start/end mismatch
    
    return combined_error + heading_penalty + anchor_penalty


def refine_multiheading(seq='01', data_root='data'):
    """Refine alignment using multi-point heading + combined metric."""
    
    print(f"\n{'='*70}")
    print(f"Multi-Point Heading Refinement for Sequence {seq}")
    print(f"{'='*70}")
    
    # Load trajectory
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses)
    traj_start = trajectory[0]
    traj_end = trajectory[-1]
    
    # Load refined polylines as base
    with open(f'osm_polylines_aligned_seq{seq}_refined.pkl', 'rb') as f:
        polylines_base = pickle.load(f)
    
    print(f"Trajectory: {len(trajectory)} points")
    print(f"OSM polylines: {len(polylines_base)}")
    
    # Sample trajectory points
    sample_indices, sample_points, sample_headings = sample_trajectory_points(trajectory, n_samples=10)
    print(f"\nSampled {len(sample_points)} points for heading alignment")
    print(f"Sample headings (deg): {np.degrees(sample_headings)}")
    
    # Initial parameters from previous refinement
    initial_scale = 0.9887
    initial_rotation = np.radians(-0.25)
    initial_translate = np.array([0.0, 0.0])  # Will be computed to align start
    
    # Adjust initial translate to align start point
    all_base = np.vstack(polylines_base)
    tree_base = cKDTree(all_base)
    _, idx_start = tree_base.query(traj_start)
    osm_start = all_base[idx_start]
    initial_translate = traj_start - osm_start * initial_scale
    
    initial_params = [initial_scale, initial_rotation, initial_translate[0], initial_translate[1]]
    
    print(f"\nInitial params: scale={initial_scale:.4f}, rot={np.degrees(initial_rotation):.2f}°")
    print(f"Initial translate: {initial_translate}")
    
    # Evaluate initial error
    initial_transformed = apply_similarity_transform(
        polylines_base, initial_scale, initial_rotation, initial_translate
    )
    initial_error = compute_combined_error(trajectory, initial_transformed, 0.5, 0.5)
    print(f"Initial combined error: {initial_error:.2f}m")
    
    # Optimize
    print("\nOptimizing...")
    bounds = [
        (0.95, 1.05),      # scale
        (np.radians(-10), np.radians(10)),  # rotation
        (-50, 50),         # tx
        (-50, 50)          # ty
    ]
    
    result = minimize(
        objective_function,
        initial_params,
        args=(trajectory, polylines_base, sample_points, sample_headings, traj_start, traj_end),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100, 'disp': True}
    )
    
    # Extract optimal parameters
    opt_scale = result.x[0]
    opt_rotation = result.x[1]
    opt_translate = np.array([result.x[2], result.x[3]])
    
    print(f"\nOptimized params:")
    print(f"  Scale: {opt_scale:.6f}")
    print(f"  Rotation: {np.degrees(opt_rotation):.2f}°")
    print(f"  Translate: {opt_translate}")
    print(f"  Final objective: {result.fun:.2f}")
    
    # Apply final transform
    final_polylines = apply_similarity_transform(
        polylines_base, opt_scale, opt_rotation, opt_translate
    )
    
    # Evaluate final errors
    all_final = np.vstack(final_polylines)
    tree_final = cKDTree(all_final)
    distances, _ = tree_final.query(trajectory, k=1)
    
    dist_start, _ = tree_final.query(traj_start)
    dist_end, _ = tree_final.query(traj_end)
    
    print(f"\nFinal alignment:")
    print(f"  Start error: {dist_start:.2f}m")
    print(f"  End error: {dist_end:.2f}m")
    print(f"  Mean error: {np.mean(distances):.2f}m")
    print(f"  Max error: {np.max(distances):.2f}m")
    print(f"  Combined (0.5*mean + 0.5*max): {0.5*np.mean(distances) + 0.5*np.max(distances):.2f}m")
    
    # Save
    with open(f'osm_polylines_aligned_seq{seq}_multi.pkl', 'wb') as f:
        pickle.dump(final_polylines, f)
    print(f"\n✅ Saved: osm_polylines_aligned_seq{seq}_multi.pkl")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Full view
    ax = axes[0, 0]
    for polyline in final_polylines:
        ax.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=0.8, alpha=0.5)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax.scatter(sample_points[:, 0], sample_points[:, 1], c='orange', s=50, zorder=5, label='Sample points')
    ax.scatter(traj_start[0], traj_start[1], c='green', s=200, marker='o', zorder=5)
    ax.scatter(traj_end[0], traj_end[1], c='red', s=250, marker='*', zorder=5)
    ax.set_title(f'Full View - Multi-Point Heading Aligned', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Top-right: Start zoom
    ax = axes[0, 1]
    for polyline in final_polylines:
        ax.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax.scatter(traj_start[0], traj_start[1], c='green', s=300, marker='o', zorder=5, label='Start')
    ax.set_xlim(traj_start[0] - 100, traj_start[0] + 100)
    ax.set_ylim(traj_start[1] - 100, traj_start[1] + 100)
    ax.set_title('Start Region (Zoom)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: End zoom
    ax = axes[1, 0]
    for polyline in final_polylines:
        ax.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax.scatter(traj_end[0], traj_end[1], c='red', s=300, marker='*', zorder=5, label='End')
    ax.set_xlim(traj_end[0] - 100, traj_end[0] + 100)
    ax.set_ylim(traj_end[1] - 100, traj_end[1] + 100)
    ax.set_title('End Region (Zoom)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Error histogram
    ax = axes[1, 1]
    ax.hist(distances, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.1f}m')
    ax.axvline(np.max(distances), color='orange', linestyle='--', linewidth=2, label=f'Max: {np.max(distances):.1f}m')
    ax.set_xlabel('Distance Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'osm_pbf_aligned_seq{seq}_multiheading.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: osm_pbf_aligned_seq{seq}_multiheading.png")
    
    return final_polylines


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, default="01")
    parser.add_argument("--data_root", type=str, default="data")
    
    args = parser.parse_args()
    refine_multiheading(args.seq, args.data_root)
    print("\n🎉 Done!")
