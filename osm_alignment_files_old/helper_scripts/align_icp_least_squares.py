#!/usr/bin/env python3
"""
Segment-wise least-squares alignment using ICP (Iterative Closest Point).
Matches trajectory points to nearest OSM road segments and solves for optimal
rigid transformation using robust least-squares.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


SEQ_TO_RAW = {
    '02': '2011_10_03_drive_0034_sync',
    '07': '2011_09_30_drive_0027_sync',
    '08': '2011_09_30_drive_0028_sync',
}

SEQ_FRAME_OFFSET = {
    '02': 0,
    '07': 0,
    '08': 252,  # Corrected: frame 252 matches heading
}


def load_poses(pose_file):
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
    root = Path(data_root) / 'raw_data'
    for date in ['2011_10_03', '2011_09_30', '2011_09_29', '2011_09_28', '2011_09_26']:
        candidate = root / raw_folder / date / raw_folder / 'oxts' / 'data'
        if candidate.exists():
            return candidate
    return None


def gps_offset_polylines(latlon_polylines, oxts_data, trajectory, frame_offset=0):
    """Convert lat/lon polylines to local frame using GPS offset."""
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)

    offset_east = east0 - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]

    raw_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        raw_polylines.append(pts)

    return raw_polylines


def sample_osm_points(polylines, spacing=5.0):
    """Sample points along OSM polylines at regular spacing."""
    sampled = []
    for pl in polylines:
        if len(pl) < 2:
            continue
        # Compute cumulative distance along polyline
        diffs = np.diff(pl, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cumdist = np.concatenate([[0], np.cumsum(dists)])
        total = cumdist[-1]
        if total < spacing:
            continue
        # Sample at regular intervals
        n_samp = max(int(total / spacing) + 1, 2)
        sample_dists = np.linspace(0, total, n_samp)
        # Interpolate
        sampled_pts = np.zeros((n_samp, 2))
        for i, sd in enumerate(sample_dists):
            idx = np.searchsorted(cumdist, sd)
            if idx >= len(pl):
                idx = len(pl) - 1
            if idx == 0:
                sampled_pts[i] = pl[0]
            else:
                t = (sd - cumdist[idx-1]) / (cumdist[idx] - cumdist[idx-1])
                sampled_pts[i] = pl[idx-1] + t * (pl[idx] - pl[idx-1])
        sampled.append(sampled_pts)
    return np.vstack(sampled) if sampled else np.array([])


def find_nearest_points(source_pts, target_pts, max_dist=50.0):
    """Find nearest target point for each source point."""
    tree = cKDTree(target_pts)
    dists, indices = tree.query(source_pts, k=1, distance_upper_bound=max_dist)
    # Filter out unmatched points (distance_upper_bound returns inf)
    valid = dists < max_dist
    return valid, indices[valid], dists[valid]


def solve_rigid_transform(source_pts, target_pts):
    """
    Solve for optimal rigid transformation (R, t) that minimizes
    sum ||R*source + t - target||^2 using least squares (Kabsch algorithm).
    """
    # Center the points
    source_center = np.mean(source_pts, axis=0)
    target_center = np.mean(target_pts, axis=0)
    
    source_centered = source_pts - source_center
    target_centered = target_pts - target_center
    
    # Compute covariance matrix
    H = source_centered.T @ target_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_center - R @ source_center
    
    return R, t


def icp_alignment(source_pts, target_pts, max_iters=100, tol=1e-6, max_dist=50.0):
    """
    Iterative Closest Point alignment.
    Returns optimal R, t and convergence info.
    """
    # Initial guess: identity
    R_total = np.eye(2)
    t_total = np.zeros(2)
    
    prev_error = float('inf')
    
    for iteration in range(max_iters):
        # Transform source points
        transformed = (R_total @ source_pts.T).T + t_total
        
        # Find correspondences
        valid, indices, dists = find_nearest_points(transformed, target_pts, max_dist)
        
        if valid.sum() < 10:
            print(f"  ICP iteration {iteration}: too few matches ({valid.sum()})")
            break
        
        matched_source = transformed[valid]
        matched_target = target_pts[indices]
        
        # Compute error
        error = np.mean(dists ** 2)
        
        if iteration % 10 == 0 or abs(prev_error - error) < tol:
            print(f"  ICP iteration {iteration}: error={error:.4f}, matches={valid.sum()}")
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
        
        # Solve for incremental transformation
        R_inc, t_inc = solve_rigid_transform(matched_source, matched_target)
        
        # Update total transformation
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc
    
    return R_total, t_total, prev_error


def segment_wise_icp(trajectory, osm_polylines, segment_length=100.0):
    """
    Apply ICP to overlapping segments of the trajectory.
    This handles local deformations better than global ICP.
    """
    # Sample OSM points
    osm_pts = sample_osm_points(osm_polylines, spacing=2.0)
    print(f"  Sampled {len(osm_pts)} OSM points")
    
    # Compute trajectory segment indices
    diffs = np.diff(trajectory, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumdist = np.concatenate([[0], np.cumsum(dists)])
    total_length = cumdist[-1]
    
    # Create overlapping segments
    n_segments = max(int(total_length / segment_length), 1)
    segment_step = total_length / n_segments
    
    print(f"  Trajectory length: {total_length:.1f}m, {n_segments} segments")
    
    # Global ICP first (coarse alignment)
    print("  Running global ICP (coarse)...")
    R_global, t_global, error_global = icp_alignment(
        trajectory, osm_pts, max_iters=50, max_dist=100.0
    )
    print(f"  Global ICP error: {error_global:.4f}")
    
    # Apply global transformation
    aligned_trajectory = (R_global @ trajectory.T).T + t_global
    
    # Check if we need segment-wise refinement
    if error_global > 10.0 and n_segments > 1:
        print("  Running segment-wise refinement...")
        
        # For each segment, compute local refinement
        segment_transforms = []
        
        for i in range(n_segments):
            seg_start = i * segment_step
            seg_end = min((i + 1) * segment_step + segment_length/2, total_length)
            
            # Find trajectory points in this segment
            seg_mask = (cumdist >= seg_start) & (cumdist <= seg_end)
            if seg_mask.sum() < 20:
                continue
            
            seg_pts = aligned_trajectory[seg_mask]
            
            # Find local ICP
            R_seg, t_seg, err_seg = icp_alignment(
                seg_pts, osm_pts, max_iters=30, max_dist=30.0
            )
            
            print(f"    Segment {i}: {seg_mask.sum()} pts, error={err_seg:.4f}")
            segment_transforms.append({
                'start_dist': seg_start,
                'end_dist': seg_end,
                'indices': np.where(seg_mask)[0],
                'R': R_seg,
                't': t_seg
            })
        
        # Apply segment transforms with smooth blending
        refined_trajectory = aligned_trajectory.copy()
        for seg in segment_transforms:
            idx = seg['indices']
            # Apply segment transform
            seg_transformed = (seg['R'] @ aligned_trajectory[idx].T).T + seg['t']
            refined_trajectory[idx] = seg_transformed
        
        return refined_trajectory, R_global, t_global, segment_transforms
    
    return aligned_trajectory, R_global, t_global, []


def apply_transform_to_polylines(polylines, R, t):
    """Apply rigid transformation to all polylines."""
    transformed = []
    for pl in polylines:
        pl_t = (R @ pl.T).T + t
        transformed.append(pl_t)
    return transformed


def visualize(seq, polylines, trajectory, aligned_trajectory):
    """Visualize alignment results."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Sample OSM points for visualization
    osm_pts = sample_osm_points(polylines, spacing=5.0)
    
    # Compute errors
    tree = cKDTree(osm_pts)
    dists_orig, _ = tree.query(trajectory, k=1)
    dists_aligned, _ = tree.query(aligned_trajectory, k=1)
    
    # Main plot - original
    ax1 = fig.add_subplot(gs[0, :2])
    for pl in polylines:
        ax1.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.4)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2, label='Original Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax1.set_title(f'Seq {seq} - BEFORE (mean error: {np.mean(dists_orig):.2f}m)', fontweight='bold')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Main plot - aligned
    ax2 = fig.add_subplot(gs[1, :2])
    for pl in polylines:
        ax2.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.4)
    ax2.plot(aligned_trajectory[:, 0], aligned_trajectory[:, 1], 'g-', lw=2, label='Aligned Trajectory')
    ax2.scatter(aligned_trajectory[0, 0], aligned_trajectory[0, 1], c='green', s=300, marker='o', zorder=5)
    ax2.scatter(aligned_trajectory[-1, 0], aligned_trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax2.set_title(f'Seq {seq} - AFTER ICP (mean error: {np.mean(dists_aligned):.2f}m)', fontweight='bold')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # Error comparison
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.hist(dists_orig, bins=50, alpha=0.5, label=f'Before: {np.mean(dists_orig):.1f}m', color='red')
    ax3.hist(dists_aligned, bins=50, alpha=0.5, label=f'After: {np.mean(dists_aligned):.1f}m', color='green')
    ax3.set_xlabel('Distance to nearest road (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    out = f'osm_pbf_aligned_seq{seq}_icp.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    
    return np.mean(dists_orig), np.mean(dists_aligned)


def process(seq, data_root='data'):
    print(f"\n{'='*65}")
    print(f"Seq {seq}: ICP Least-Squares Alignment")
    print(f"{'='*65}")
    
    # Load lat/lon polylines
    latlon_pkl = f'osm_polylines_latlon_seq{seq}_regbez.pkl'
    with open(latlon_pkl, 'rb') as f:
        latlon_polylines = pickle.load(f)
    latlon_polylines = [np.array(pl) for pl in latlon_polylines]
    print(f"Loaded {len(latlon_polylines)} polylines from {latlon_pkl}")
    
    # Load trajectory
    trajectory = extract_trajectory(load_poses(f'{data_root}/kitti/poses/{seq}.txt'))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Load OXTS GPS
    raw_folder = SEQ_TO_RAW[seq]
    frame_offset = SEQ_FRAME_OFFSET.get(seq, 0)
    oxts_dir = find_oxts_dir(raw_folder, data_root)
    if oxts_dir is None:
        raise FileNotFoundError(f"OXTS not found for seq {seq}")
    oxts_data = load_oxts_data(str(oxts_dir))
    print(f"OXTS: {len(oxts_data)} frames")
    
    # GPS offset alignment (initial guess)
    print("\nStep 1: GPS offset alignment")
    local_polylines = gps_offset_polylines(
        latlon_polylines, oxts_data, trajectory, frame_offset)
    
    # ICP alignment
    print("\nStep 2: ICP alignment")
    aligned_trajectory, R, t, segment_transforms = segment_wise_icp(
        trajectory, local_polylines, segment_length=200.0)
    
    print(f"\nStep 3: Applying transform to OSM polylines")
    # We want to transform OSM to match trajectory, so apply inverse
    R_inv = R.T
    t_inv = -R.T @ t
    aligned_polylines = apply_transform_to_polylines(local_polylines, R_inv, t_inv)
    
    # Save aligned polylines
    out_pkl = f'osm_polylines_aligned_seq{seq}_icp.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(aligned_polylines, f)
    print(f"  Saved: {out_pkl}")
    
    # Save transform
    transform = {
        'R': R,
        't': t,
        'R_inv': R_inv,
        't_inv': t_inv,
        'segment_transforms': segment_transforms
    }
    with open(f'osm_transform_seq{seq}_icp.pkl', 'wb') as f:
        pickle.dump(transform, f)
    print(f"  Saved: osm_transform_seq{seq}_icp.pkl")
    
    # Visualize
    print("\nStep 4: Visualizing")
    err_before, err_after = visualize(seq, aligned_polylines, trajectory, aligned_trajectory)
    print(f"  Before: {err_before:.2f}m, After: {err_after:.2f}m")
    
    return err_before, err_after


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['08'])
    parser.add_argument('--data_root', default='data')
    args = parser.parse_args()

    for seq in args.seqs:
        process(seq, args.data_root)

    print('\nAll done.')
