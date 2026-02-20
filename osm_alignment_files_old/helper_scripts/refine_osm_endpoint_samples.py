#!/usr/bin/env python3
"""
Align OSM using end point + sample points along trajectory.

Approach:
- Use end point as anchor (seems accurate)
- Use multiple sample points along trajectory to match OSM
- Do NOT force start point alignment (known to be inaccurate)
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
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses):
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def sample_trajectory_evenly(trajectory, n_samples=15):
    """Sample n points evenly spaced along trajectory by arc length."""
    # Compute cumulative distance
    diffs = np.diff(trajectory, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]
    
    # Sample at regular intervals
    sample_dists = np.linspace(0, total_length, n_samples)
    indices = []
    for sd in sample_dists:
        idx = np.argmin(np.abs(cum_dists - sd))
        indices.append(idx)
    
    return np.array(indices), trajectory[indices]


def compute_error_metrics(trajectory, polylines):
    """Compute mean and max error."""
    all_osm = np.vstack(polylines)
    if len(all_osm) == 0:
        return float('inf'), float('inf')
    
    tree = cKDTree(all_osm)
    distances, _ = tree.query(trajectory, k=1)
    return np.mean(distances), np.max(distances)


def apply_similarity_transform(polylines, scale, rotation, translate):
    """Apply similarity transform to polylines."""
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    transformed = []
    
    for polyline in polylines:
        scaled = polyline * scale
        x_rot = scaled[:, 0] * cos_r - scaled[:, 1] * sin_r
        y_rot = scaled[:, 0] * sin_r + scaled[:, 1] * cos_r
        rotated = np.column_stack([x_rot, y_rot])
        transformed.append(rotated + translate)
    
    return transformed


def objective_endpoint_samples(params, trajectory, polylines_base, 
                                sample_points, traj_end, weight_end=2.0):
    """
    Objective: minimize distance from sample points to OSM,
    with strong weight on end point accuracy.
    """
    scale, rotation, tx, ty = params
    translate = np.array([tx, ty])
    
    # Transform OSM
    transformed = apply_similarity_transform(polylines_base, scale, rotation, translate)
    all_osm = np.vstack(transformed)
    tree = cKDTree(all_osm)
    
    # Error at sample points
    dists_samples, _ = tree.query(sample_points, k=1)
    sample_error = np.mean(dists_samples)
    
    # Strong penalty for end point misalignment
    dist_end, _ = tree.query(traj_end)
    end_error = dist_end * weight_end  # Heavy weight on end accuracy
    
    # Combined metric
    return sample_error + end_error


def align_endpoint_samples(seq='01', data_root='data', n_samples=15):
    """Align using end point + sample points."""
    
    print(f"\n{'='*70}")
    print(f"End Point + Samples Alignment for Sequence {seq}")
    print(f"{'='*70}")
    
    # Load data
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    traj_end = trajectory[-1]
    
    with open(f'osm_polylines_aligned_seq{seq}_refined.pkl', 'rb') as f:
        polylines_base = pickle.load(f)
    
    print(f"Trajectory: {len(trajectory)} points")
    print(f"OSM polylines: {len(polylines_base)}")
    
    # Sample points evenly along trajectory (excluding start)
    sample_indices, sample_points = sample_trajectory_evenly(trajectory, n_samples)
    # Remove first sample (near start) as it's inaccurate
    sample_indices = sample_indices[1:]
    sample_points = sample_points[1:]
    
    print(f"\nUsing {len(sample_points)} sample points (excluding start)")
    print(f"End point: {traj_end}")
    
    # Initial guess from refined version
    initial_params = [0.9887, np.radians(-0.25), 26.0, 15.0]
    
    # Evaluate initial
    initial_error = objective_endpoint_samples(
        initial_params, trajectory, polylines_base, sample_points, traj_end
    )
    print(f"Initial objective: {initial_error:.2f}")
    
    # Optimize with bounds
    bounds = [
        (0.90, 1.05),      # scale
        (np.radians(-20), np.radians(20)),  # rotation
        (-50, 100),        # tx
        (-50, 100)         # ty
    ]
    
    print("\nOptimizing...")
    result = minimize(
        objective_endpoint_samples,
        initial_params,
        args=(trajectory, polylines_base, sample_points, traj_end),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200}
    )
    
    opt_scale, opt_rotation, opt_tx, opt_ty = result.x
    opt_translate = np.array([opt_tx, opt_ty])
    
    print(f"\nOptimized parameters:")
    print(f"  Scale: {opt_scale:.6f}")
    print(f"  Rotation: {np.degrees(opt_rotation):.2f}°")
    print(f"  Translate: ({opt_tx:.2f}, {opt_ty:.2f})")
    
    # Apply final transform
    final_polylines = apply_similarity_transform(
        polylines_base, opt_scale, opt_rotation, opt_translate
    )
    
    # Evaluate
    all_final = np.vstack(final_polylines)
    tree_final = cKDTree(all_final)
    
    dists_all, _ = tree_final.query(trajectory, k=1)
    dist_end, _ = tree_final.query(traj_end)
    dist_start, _ = tree_final.query(trajectory[0])
    
    print(f"\nFinal alignment:")
    print(f"  Start error: {dist_start:.2f}m (expected to be higher)")
    print(f"  End error: {dist_end:.2f}m")
    print(f"  Mean error (all): {np.mean(dists_all):.2f}m")
    print(f"  Max error: {np.max(dists_all):.2f}m")
    
    # Sample point errors
    dists_samples, _ = tree_final.query(sample_points, k=1)
    print(f"  Mean error (samples): {np.mean(dists_samples):.2f}m")
    
    # Save
    with open(f'osm_polylines_aligned_seq{seq}_endpoint.pkl', 'wb') as f:
        pickle.dump(final_polylines, f)
    print(f"\n✅ Saved: osm_polylines_aligned_seq{seq}_endpoint.pkl")
    
    # Visualize
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main overlay
    ax1 = fig.add_subplot(gs[:, :2])
    for polyline in final_polylines:
        ax1.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=0.8, alpha=0.5, label='OSM' if polyline is final_polylines[0] else '')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5, label='Trajectory')
    ax1.scatter(sample_points[:, 0], sample_points[:, 1], c='orange', s=80, marker='o', 
               zorder=5, label=f'Sample points ({len(sample_points)})', edgecolors='red', linewidths=1)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5, label='Start')
    ax1.scatter(traj_end[0], traj_end[1], c='red', s=400, marker='*', zorder=5, label='End')
    ax1.set_title(f'Seq {seq} - End Point + Samples Aligned\n'
                  f'End error: {dist_end:.2f}m | Sample mean: {np.mean(dists_samples):.2f}m | '
                  f'Overall mean: {np.mean(dists_all):.2f}m',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Start zoom
    ax2 = fig.add_subplot(gs[0, 2])
    for polyline in final_polylines:
        ax2.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=400, marker='o', zorder=5)
    ax2.scatter(sample_points[0, 0], sample_points[0, 1], c='orange', s=100, marker='o', zorder=5, edgecolors='red')
    ax2.set_xlim(trajectory[0, 0] - 50, trajectory[0, 0] + 50)
    ax2.set_ylim(trajectory[0, 1] - 50, trajectory[0, 1] + 50)
    ax2.set_title(f'Start (error: {dist_start:.1f}m)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # End zoom
    ax3 = fig.add_subplot(gs[1, 2])
    for polyline in final_polylines:
        ax3.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax3.scatter(traj_end[0], traj_end[1], c='red', s=400, marker='*', zorder=5)
    ax3.scatter(sample_points[-1, 0], sample_points[-1, 1], c='orange', s=100, marker='o', zorder=5, edgecolors='red')
    ax3.set_xlim(traj_end[0] - 50, traj_end[0] + 50)
    ax3.set_ylim(traj_end[1] - 50, traj_end[1] + 50)
    ax3.set_title(f'End (error: {dist_end:.1f}m)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(f'osm_pbf_aligned_seq{seq}_endpoint.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: osm_pbf_aligned_seq{seq}_endpoint.png")
    
    return final_polylines


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, default="01")
    parser.add_argument("--n-samples", type=int, default=15)
    
    args = parser.parse_args()
    align_endpoint_samples(args.seq, n_samples=args.n_samples)
    print("\n🎉 Done!")
