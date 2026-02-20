#!/usr/bin/env python3
"""
Densify already-aligned OSM polylines.

This script takes the correctly-aligned (but sparse) polylines from the 
bestfit files and densifies them by interpolating points every N meters.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree

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


def densify_polyline(polyline, spacing=2.0):
    """
    Densify a polyline by interpolating points every `spacing` meters.
    Points are already in local KITTI frame (meters).
    """
    if len(polyline) < 2:
        return np.array(polyline)
    
    polyline = np.array(polyline)
    
    # Calculate cumulative distances
    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dists[-1]
    
    if total_dist < spacing:
        return polyline
    
    # Generate new samples at regular spacing
    n_samples = max(int(total_dist / spacing) + 1, 2)
    new_dists = np.linspace(0, total_dist, n_samples)
    
    # Interpolate
    new_points = []
    for d in new_dists:
        idx = np.searchsorted(cum_dists, d)
        if idx >= len(polyline):
            idx = len(polyline) - 1
        if idx == 0:
            new_points.append(polyline[0])
        else:
            t = (d - cum_dists[idx-1]) / (cum_dists[idx] - cum_dists[idx-1] + 1e-10)
            pt = polyline[idx-1] + t * (polyline[idx] - polyline[idx-1])
            new_points.append(pt)
    
    return np.array(new_points)


def densify_polylines(polylines, spacing=2.0):
    """Densify all polylines."""
    densified = []
    for pl in polylines:
        dpl = densify_polyline(pl, spacing)
        if len(dpl) >= 2:
            densified.append(dpl)
    return densified


def visualize(seq, polylines, trajectory, output_dir='osm_aligned_final'):
    """Create visualization."""
    all_osm = np.vstack(polylines)
    tree = cKDTree(all_osm)
    dists, _ = tree.query(trajectory, k=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Main view
    ax1 = axes[0]
    for i, pl in enumerate(polylines):
        ax1.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.4)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='s', zorder=5)
    ax1.set_title(f'Seq {seq} - Mean Error: {np.mean(dists):.2f}m, Polylines: {len(polylines)}')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Zoomed view
    ax2 = axes[1]
    for pl in polylines:
        ax2.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.8, alpha=0.5)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5)
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='s', zorder=5)
    margin = 100
    ax2.set_xlim(trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin)
    ax2.set_ylim(trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin)
    ax2.set_title(f'Seq {seq} - Zoomed View')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    Path(output_dir).mkdir(exist_ok=True)
    out_png = f'{output_dir}/osm_aligned_seq{seq}.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    return np.mean(dists)


def process_sequence(seq, spacing=2.0, data_root='data', output_dir='osm_aligned_final'):
    """Densify already-aligned polylines for a sequence."""
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Densifying aligned polylines")
    print(f"{'='*70}")
    
    # Load already-aligned (but sparse) polylines
    aligned_pkl = f'osm_polylines_aligned_seq{seq}_bestfit.pkl'
    try:
        with open(aligned_pkl, 'rb') as f:
            sparse_polylines = pickle.load(f)
        sparse_polylines = [np.array(pl) for pl in sparse_polylines]
        print(f"Loaded {len(sparse_polylines)} sparse polylines from {aligned_pkl}")
    except FileNotFoundError:
        print(f"  ⚠️  {aligned_pkl} not found, skipping...")
        return None
    
    # Load trajectory
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Densify polylines (already in local frame)
    print(f"Densifying polylines (spacing={spacing}m)...")
    densified_polylines = densify_polylines(sparse_polylines, spacing)
    
    total_pts = sum(len(p) for p in densified_polylines)
    avg_pts = total_pts / len(densified_polylines) if densified_polylines else 0
    print(f"  After densification: {len(densified_polylines)} polylines, {total_pts} points")
    print(f"  Avg points per polyline: {avg_pts:.1f}")
    
    # Save densified polylines
    Path(output_dir).mkdir(exist_ok=True)
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(densified_polylines, f)
    print(f"Saved: {out_pkl}")
    
    # Also copy the transform file
    transform_src = f'osm_transform_seq{seq}_bestfit_new.pkl'
    transform_dst = f'{output_dir}/osm_transform_seq{seq}.pkl'
    try:
        with open(transform_src, 'rb') as f:
            transform = pickle.load(f)
        with open(transform_dst, 'wb') as f:
            pickle.dump(transform, f)
        print(f"Copied transform: {transform_src} -> {transform_dst}")
    except FileNotFoundError:
        print(f"  ⚠️  Transform file not found: {transform_src}")
    
    # Visualize
    print("Generating visualization...")
    mean_err = visualize(seq, densified_polylines, trajectory, output_dir)
    
    print(f"\n✓ Complete: {len(densified_polylines)} polylines, {total_pts} points, {mean_err:.2f}m error")
    
    return {
        'seq': seq,
        'polylines': len(densified_polylines),
        'points': total_pts,
        'mean_error': mean_err
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['00', '01', '02', '05', '07', '08', '09', '10'])
    parser.add_argument('--output', default='osm_aligned_final')
    parser.add_argument('--spacing', type=float, default=2.0)
    args = parser.parse_args()
    
    print("="*70)
    print("Densify Already-Aligned OSM Polylines")
    print("="*70)
    print(f"Spacing: {args.spacing}m")
    
    results = []
    for seq in args.seqs:
        result = process_sequence(seq, args.spacing, output_dir=args.output)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Seq':>6} {'Polylines':>12} {'Points':>12} {'Avg/Poly':>10} {'Mean Err':>10}")
    print("-"*60)
    for r in results:
        avg = r['points'] / r['polylines'] if r['polylines'] > 0 else 0
        print(f"{r['seq']:>6} {r['polylines']:>12} {r['points']:>12} {avg:>10.1f} {r['mean_error']:>10.2f}m")
    print("="*70)


if __name__ == '__main__':
    main()
