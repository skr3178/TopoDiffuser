#!/usr/bin/env python3
"""
Best-fit OSM alignment - minimize total error across all trajectory points.

Approach:
- Sample many points along trajectory (fine granularity)
- Optimize similarity transform to minimize sum of distances to OSM
- No special constraints on start or end points
- Pure minimization of accumulated error
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


def sample_trajectory_dense(trajectory, spacing=10.0):
    """Sample points along trajectory at approximately spacing meters."""
    # Compute cumulative distance
    diffs = np.diff(trajectory, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]
    
    # Sample at regular intervals
    n_samples = int(total_length / spacing) + 1
    sample_dists = np.linspace(0, total_length, n_samples)
    
    indices = []
    for sd in sample_dists:
        idx = np.argmin(np.abs(cum_dists - sd))
        indices.append(idx)
    
    # Remove duplicates
    indices = sorted(list(set(indices)))
    
    return np.array(indices), trajectory[indices]


def compute_total_error(trajectory_sample, polylines, scale, rotation, translate):
    """Compute sum of distances from sample points to OSM."""
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    
    # Transform all OSM points
    all_osm = np.vstack(polylines)
    scaled = all_osm * scale
    x_rot = scaled[:, 0] * cos_r - scaled[:, 1] * sin_r
    y_rot = scaled[:, 0] * sin_r + scaled[:, 1] * cos_r
    transformed = np.column_stack([x_rot, y_rot]) + translate
    
    # Query distances
    tree = cKDTree(transformed)
    distances, _ = tree.query(trajectory_sample, k=1)
    
    return np.sum(distances), np.mean(distances), np.max(distances)


def bestfit_alignment(seq='01', data_root='data', spacing=10.0):
    """Align OSM using best-fit across all trajectory points."""
    
    print(f"\n{'='*70}")
    print(f"Best-Fit Alignment for Sequence {seq}")
    print(f"Sample spacing: {spacing}m")
    print(f"{'='*70}")
    
    # Load data
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    
    with open(f'osm_polylines_aligned_seq{seq}_refined.pkl', 'rb') as f:
        polylines_base = pickle.load(f)
    # Convert to numpy arrays if needed
    polylines_base = [np.array(pl) if not isinstance(pl, np.ndarray) else pl for pl in polylines_base]
    
    print(f"Trajectory: {len(trajectory)} points")
    print(f"OSM polylines: {len(polylines_base)}")
    
    # Dense sampling
    sample_indices, sample_points = sample_trajectory_dense(trajectory, spacing)
    print(f"Sampled {len(sample_points)} points at ~{spacing}m spacing")
    
    # Objective function
    def objective(params):
        scale, rotation, tx, ty = params
        translate = np.array([tx, ty])
        total_err, mean_err, max_err = compute_total_error(
            sample_points, polylines_base, scale, rotation, translate
        )
        # Combined: total sum + weight on max to control outliers
        return total_err + max_err * 5.0
    
    # Initial guess (from refined)
    initial_params = [0.9887, np.radians(-0.25), 26.0, 15.0]
    
    print("\nInitial evaluation...")
    init_total, init_mean, init_max = compute_total_error(
        sample_points, polylines_base, *initial_params[:2], np.array(initial_params[2:])
    )
    print(f"Initial: total={init_total:.1f}m, mean={init_mean:.2f}m, max={init_max:.2f}m")
    
    # Optimize
    print("\nOptimizing...")
    bounds = [
        (0.90, 1.05),      # scale
        (np.radians(-30), np.radians(30)),  # rotation
        (-100, 100),       # tx
        (-100, 100)        # ty
    ]
    
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'disp': False}
    )
    
    opt_scale, opt_rotation, opt_tx, opt_ty = result.x
    opt_translate = np.array([opt_tx, opt_ty])
    
    print(f"\nOptimized parameters:")
    print(f"  Scale: {opt_scale:.6f}")
    print(f"  Rotation: {np.degrees(opt_rotation):.2f}°")
    print(f"  Translate: ({opt_tx:.2f}, {opt_ty:.2f})")
    
    # Apply transform
    cos_r, sin_r = np.cos(opt_rotation), np.sin(opt_rotation)
    final_polylines = []
    for polyline in polylines_base:
        scaled = polyline * opt_scale
        x_rot = scaled[:, 0] * cos_r - scaled[:, 1] * sin_r
        y_rot = scaled[:, 0] * sin_r + scaled[:, 1] * cos_r
        final_polylines.append(np.column_stack([x_rot, y_rot]) + opt_translate)
    
    # Evaluate on FULL trajectory
    all_final = np.vstack(final_polylines)
    tree_final = cKDTree(all_final)
    dists_all, _ = tree_final.query(trajectory, k=1)
    
    dist_start, _ = tree_final.query(trajectory[0])
    dist_end, _ = tree_final.query(trajectory[-1])
    
    print(f"\nFinal alignment (all {len(trajectory)} points):")
    print(f"  Start error: {dist_start:.2f}m")
    print(f"  End error: {dist_end:.2f}m")
    print(f"  Mean error: {np.mean(dists_all):.2f}m")
    print(f"  Max error: {np.max(dists_all):.2f}m")
    print(f"  Total accumulated: {np.sum(dists_all):.1f}m")
    
    # Evaluate on samples
    dists_samples, _ = tree_final.query(sample_points, k=1)
    print(f"\nSample points ({len(sample_points)}):")
    print(f"  Mean: {np.mean(dists_samples):.2f}m")
    print(f"  Max: {np.max(dists_samples):.2f}m")
    
    # Save
    with open(f'osm_polylines_aligned_seq{seq}_bestfit.pkl', 'wb') as f:
        pickle.dump(final_polylines, f)
    print(f"\n✅ Saved: osm_polylines_aligned_seq{seq}_bestfit.pkl")
    
    # Visualize
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Main overlay with sample points
    ax1 = fig.add_subplot(gs[:, :2])
    for polyline in final_polylines:
        ax1.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=0.8, alpha=0.5, label='OSM' if polyline is final_polylines[0] else '')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    # Show every 5th sample point to avoid clutter
    ax1.scatter(sample_points[::5, 0], sample_points[::5, 1], 
               c='orange', s=30, alpha=0.6, zorder=4, label=f'Samples ({len(sample_points)})')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5, label='End')
    ax1.set_title(f'Seq {seq} - Best-Fit Alignment\n'
                  f'Mean: {np.mean(dists_all):.2f}m | Max: {np.max(dists_all):.2f}m | '
                  f'Total: {np.sum(dists_all):.0f}m',
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
    ax2.set_xlim(trajectory[0, 0] - 50, trajectory[0, 0] + 50)
    ax2.set_ylim(trajectory[0, 1] - 50, trajectory[0, 1] + 50)
    ax2.set_title(f'Start (error: {dist_start:.1f}m)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # End zoom
    ax3 = fig.add_subplot(gs[1, 2])
    for polyline in final_polylines:
        ax3.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax3.set_xlim(trajectory[-1, 0] - 50, trajectory[-1, 0] + 50)
    ax3.set_ylim(trajectory[-1, 1] - 50, trajectory[-1, 1] + 50)
    ax3.set_title(f'End (error: {dist_end:.1f}m)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(f'osm_pbf_aligned_seq{seq}_bestfit.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: osm_pbf_aligned_seq{seq}_bestfit.png")
    
    return final_polylines


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, default="01")
    parser.add_argument("--spacing", type=float, default=10.0)
    
    args = parser.parse_args()
    bestfit_alignment(args.seq, spacing=args.spacing)
    print("\n🎉 Done!")
