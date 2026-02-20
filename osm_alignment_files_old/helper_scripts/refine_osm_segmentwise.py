#!/usr/bin/env python3
"""
Segment-wise OSM alignment.

Divide trajectory into segments, compute optimal transform for each segment,
ensuring continuity at segment boundaries.
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


def create_segments(trajectory, segment_length=200.0):
    """Split trajectory into segments of approximately segment_length meters."""
    segments = []
    current_seg = [0]
    current_dist = 0.0
    
    for i in range(1, len(trajectory)):
        dist = np.linalg.norm(trajectory[i] - trajectory[i-1])
        current_dist += dist
        
        if current_dist >= segment_length or i == len(trajectory) - 1:
            current_seg.append(i)
            segments.append((current_seg[0], current_seg[-1]))
            current_seg = [i]
            current_dist = 0.0
        else:
            current_seg.append(i)
    
    return segments


def compute_segment_error(trajectory_seg, polylines, transform):
    """Compute error for a trajectory segment."""
    scale, rotation, tx, ty = transform
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    translate = np.array([tx, ty])
    
    # Transform trajectory points (inverse: we move OSM to match trajectory)
    # Actually, we need to transform OSM to match trajectory
    all_osm = np.vstack(polylines)
    scaled = all_osm * scale
    rotated = np.column_stack([
        scaled[:, 0] * cos_r - scaled[:, 1] * sin_r,
        scaled[:, 0] * sin_r + scaled[:, 1] * cos_r
    ])
    transformed = rotated + translate
    
    tree = cKDTree(transformed)
    distances, _ = tree.query(trajectory_seg, k=1)
    
    return np.mean(distances), np.max(distances)


def find_best_transform_for_segment(trajectory_seg, polylines_base, 
                                    initial_guess, bounds_factor=0.1):
    """Find best similarity transform for a trajectory segment."""
    
    def objective(params):
        mean_err, max_err = compute_segment_error(trajectory_seg, polylines_base, params)
        return 0.5 * mean_err + 0.5 * max_err  # Combined metric
    
    # Bounds around initial guess
    bounds = [
        (initial_guess[0] * (1 - bounds_factor), initial_guess[0] * (1 + bounds_factor)),  # scale
        (initial_guess[1] - 0.2, initial_guess[1] + 0.2),  # rotation (±11°)
        (initial_guess[2] - 30, initial_guess[2] + 30),    # tx
        (initial_guess[3] - 30, initial_guess[3] + 30)     # ty
    ]
    
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 50})
    
    return result.x


def apply_transform_to_polylines(polylines, transform):
    """Apply similarity transform to polylines."""
    scale, rotation, tx, ty = transform
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    translate = np.array([tx, ty])
    
    transformed = []
    for polyline in polylines:
        scaled = polyline * scale
        rotated = np.column_stack([
            scaled[:, 0] * cos_r - scaled[:, 1] * sin_r,
            scaled[:, 0] * sin_r + scaled[:, 1] * cos_r
        ])
        transformed.append(rotated + translate)
    
    return transformed


def blend_transforms(transform1, transform2, alpha):
    """Blend two transforms with weight alpha (0 = transform1, 1 = transform2)."""
    return [
        transform1[0] * (1 - alpha) + transform2[0] * alpha,  # scale
        transform1[1] * (1 - alpha) + transform2[1] * alpha,  # rotation
        transform1[2] * (1 - alpha) + transform2[2] * alpha,  # tx
        transform1[3] * (1 - alpha) + transform2[3] * alpha   # ty
    ]


def segment_wise_alignment(seq='01', data_root='data', segment_length=200.0):
    """Align OSM using segment-wise transforms."""
    
    print(f"\n{'='*70}")
    print(f"Segment-wise Alignment for Sequence {seq}")
    print(f"Segment length: {segment_length}m")
    print(f"{'='*70}")
    
    # Load data
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    
    with open(f'osm_polylines_aligned_seq{seq}_refined.pkl', 'rb') as f:
        polylines_base = pickle.load(f)
    
    print(f"Trajectory: {len(trajectory)} points")
    print(f"OSM polylines: {len(polylines_base)}")
    
    # Create segments
    segments = create_segments(trajectory, segment_length)
    print(f"Created {len(segments)} segments")
    
    # Initial global transform (from refined version)
    # scale=0.9887, rotation=-0.25°, translate based on start alignment
    all_base = np.vstack(polylines_base)
    tree_base = cKDTree(all_base)
    _, idx_start = tree_base.query(trajectory[0])
    osm_start = all_base[idx_start]
    initial_transform = [0.9887, np.radians(-0.25), 
                        trajectory[0, 0] - osm_start[0] * 0.9887,
                        trajectory[0, 1] - osm_start[1] * 0.9887]
    
    # Compute transform for each segment
    segment_transforms = []
    
    for i, (start_idx, end_idx) in enumerate(segments):
        traj_seg = trajectory[start_idx:end_idx+1]
        
        # Use previous segment's transform as initial guess (continuity)
        if i == 0:
            init_guess = initial_transform
        else:
            init_guess = segment_transforms[-1]
        
        print(f"\nSegment {i+1}/{len(segments)}: frames {start_idx}-{end_idx} "
              f"({len(traj_seg)} points)")
        
        # Find best transform
        best_transform = find_best_transform_for_segment(
            traj_seg, polylines_base, init_guess, bounds_factor=0.05
        )
        
        mean_err, max_err = compute_segment_error(traj_seg, polylines_base, best_transform)
        print(f"  Transform: scale={best_transform[0]:.4f}, "
              f"rot={np.degrees(best_transform[1]):.2f}°, "
              f"translate=({best_transform[2]:.1f}, {best_transform[3]:.1f})")
        print(f"  Error: mean={mean_err:.2f}m, max={max_err:.2f}m")
        
        segment_transforms.append(best_transform)
    
    # Now apply transforms with blending at segment boundaries
    print(f"\n{'='*70}")
    print("Applying transforms with boundary blending...")
    
    # For each trajectory point, find which segment it belongs to
    # and apply appropriate transform (with blending near boundaries)
    
    final_polylines_per_segment = []
    
    for i, transform in enumerate(segment_transforms):
        start_idx, end_idx = segments[i]
        
        # For segment i, use transform i, but blend at edges
        # We'll create a weighted combination of neighboring transforms
        
        if i == 0:
            # First segment: use its own transform
            effective_transform = transform
        else:
            # Blend with previous segment at the boundary
            prev_transform = segment_transforms[i-1]
            # Use average of previous and current
            effective_transform = blend_transforms(prev_transform, transform, 0.5)
        
        # Apply transform to all OSM data
        transformed = apply_transform_to_polylines(polylines_base, effective_transform)
        final_polylines_per_segment.append(transformed)
    
    # For visualization, we'll use the last segment's transform as the "final"
    # But for actual BEV generation, we'd use segment-specific transforms
    
    # Let's create a visualization showing improvement
    # Use segment-wise approach: for each trajectory point, use nearest segment's transform
    
    print("\nComputing per-point transformed OSM...")
    
    # Find best alignment overall using segment information
    # Simplification: use the transform that minimizes error for each region
    
    # Actually, let's use a different approach:
    # Transform the entire OSM for each segment, then combine
    
    all_errors = []
    for i, (transform, (start_idx, end_idx)) in enumerate(zip(segment_transforms, segments)):
        traj_seg = trajectory[start_idx:end_idx+1]
        mean_err, max_err = compute_segment_error(traj_seg, polylines_base, transform)
        all_errors.append((mean_err, max_err))
    
    avg_mean = np.mean([e[0] for e in all_errors])
    avg_max = np.mean([e[1] for e in all_errors])
    
    print(f"\nSegment-wise error summary:")
    print(f"  Average mean error: {avg_mean:.2f}m")
    print(f"  Average max error: {avg_max:.2f}m")
    
    # For final output, use the best single transform that works across all segments
    # Or use the median transform
    median_transform = [
        np.median([t[0] for t in segment_transforms]),
        np.median([t[1] for t in segment_transforms]),
        np.median([t[2] for t in segment_transforms]),
        np.median([t[3] for t in segment_transforms])
    ]
    
    print(f"\nMedian transform across segments:")
    print(f"  scale={median_transform[0]:.4f}, rot={np.degrees(median_transform[1]):.2f}°")
    
    # Apply median transform
    final_polylines = apply_transform_to_polylines(polylines_base, median_transform)
    
    # Evaluate
    all_final = np.vstack(final_polylines)
    tree_final = cKDTree(all_final)
    distances, _ = tree_final.query(trajectory, k=1)
    
    print(f"\nFinal alignment (median transform):")
    print(f"  Mean error: {np.mean(distances):.2f}m")
    print(f"  Max error: {np.max(distances):.2f}m")
    
    # Save
    with open(f'osm_polylines_aligned_seq{seq}_segment.pkl', 'wb') as f:
        pickle.dump(final_polylines, f)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Full view with segment markers
    ax = axes[0, 0]
    for polyline in final_polylines:
        ax.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=0.8, alpha=0.5)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    
    # Mark segment boundaries
    for i, (start_idx, end_idx) in enumerate(segments):
        mid_idx = (start_idx + end_idx) // 2
        ax.scatter(trajectory[mid_idx, 0], trajectory[mid_idx, 1], 
                  c='orange', s=100, marker='o', zorder=5)
    
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=200, marker='o', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=250, marker='*', zorder=5)
    ax.set_title(f'Segment-wise Aligned ({len(segments)} segments)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Start zoom
    ax = axes[0, 1]
    for polyline in final_polylines:
        ax.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5)
    ax.set_xlim(trajectory[0, 0] - 100, trajectory[0, 0] + 100)
    ax.set_ylim(trajectory[0, 1] - 100, trajectory[0, 1] + 100)
    ax.set_title('Start Region', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # End zoom
    ax = axes[1, 0]
    for polyline in final_polylines:
        ax.plot(polyline[:, 0], polyline[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=300, marker='*', zorder=5)
    ax.set_xlim(trajectory[-1, 0] - 100, trajectory[-1, 0] + 100)
    ax.set_ylim(trajectory[-1, 1] - 100, trajectory[-1, 1] + 100)
    ax.set_title('End Region', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Error by segment
    ax = axes[1, 1]
    segment_means = [e[0] for e in all_errors]
    segment_maxs = [e[1] for e in all_errors]
    x = np.arange(len(segments))
    width = 0.35
    ax.bar(x - width/2, segment_means, width, label='Mean', alpha=0.7)
    ax.bar(x + width/2, segment_maxs, width, label='Max', alpha=0.7)
    ax.set_xlabel('Segment')
    ax.set_ylabel('Error (m)')
    ax.set_title('Per-Segment Error', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'osm_pbf_aligned_seq{seq}_segment.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: osm_pbf_aligned_seq{seq}_segment.png")
    
    return final_polylines, segment_transforms


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, default="01")
    parser.add_argument("--segment-length", type=float, default=200.0)
    
    args = parser.parse_args()
    segment_wise_alignment(args.seq, segment_length=args.segment_length)
    print("\n🎉 Done!")
