#!/usr/bin/env python3
"""
Densify already-aligned OSM polylines from bestfit files.

This script takes the aligned polylines (already in local frame) from
osm_polylines_aligned_seq{seq}_bestfit.pkl and densifies them by
interpolating points every 2 meters along each polyline.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree

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
    """Densify a polyline by interpolating points every `spacing` meters."""
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
    for i, pl in enumerate(polylines):
        dpl = densify_polyline(pl, spacing)
        if len(dpl) >= 2:
            densified.append(dpl)
        
        if (i + 1) % 1000 == 0:
            print(f"  Densified {i+1}/{len(polylines)} polylines...")
    
    return densified


def visualize(seq, polylines, trajectory, output_dir='osm_aligned_final'):
    """Create visualization."""
    all_osm = np.vstack(polylines)
    tree = cKDTree(all_osm)
    dists, _ = tree.query(trajectory, k=1)
    dist_start, _ = tree.query(trajectory[0])
    dist_end, _ = tree.query(trajectory[-1])

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Main overlay
    ax1 = fig.add_subplot(gs[:, :2])
    for i, pl in enumerate(polylines):
        ax1.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.4)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5, label='Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax1.set_title(f'Seq {seq} - Dense Alignment\nMean: {np.mean(dists):.2f}m | {len(polylines)} polylines | {sum(len(p) for p in polylines)} points')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Start zoom
    ax2 = fig.add_subplot(gs[0, 2])
    for pl in polylines:
        ax2.plot(pl[:, 0], pl[:, 1], 'b-', lw=1, alpha=0.6)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=400, marker='o', zorder=5)
    ax2.set_xlim(trajectory[0, 0] - 60, trajectory[0, 0] + 60)
    ax2.set_ylim(trajectory[0, 1] - 60, trajectory[0, 1] + 60)
    ax2.set_title(f'Start ({dist_start:.1f}m)')
    ax2.grid(True, alpha=0.3)

    # End zoom
    ax3 = fig.add_subplot(gs[1, 2])
    for pl in polylines:
        ax3.plot(pl[:, 0], pl[:, 1], 'b-', lw=1, alpha=0.6)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax3.set_xlim(trajectory[-1, 0] - 60, trajectory[-1, 0] + 60)
    ax3.set_ylim(trajectory[-1, 1] - 60, trajectory[-1, 1] + 60)
    ax3.set_title(f'End ({dist_end:.1f}m)')
    ax3.grid(True, alpha=0.3)

    Path(output_dir).mkdir(exist_ok=True)
    out_png = f'{output_dir}/osm_aligned_seq{seq}.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    return np.mean(dists), dist_start, dist_end


def process_sequence(seq, data_root='data', output_dir='osm_aligned_final'):
    """Process a sequence by densifying its bestfit-aligned polylines."""
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Densifying BestFit-Aligned Polylines")
    print(f"{'='*70}")
    
    # Load bestfit-aligned polylines (already in local frame)
    bestfit_pkl = f'osm_polylines_aligned_seq{seq}_bestfit.pkl'
    try:
        with open(bestfit_pkl, 'rb') as f:
            polylines = pickle.load(f)
        print(f"Loaded {len(polylines)} polylines from {bestfit_pkl}")
    except FileNotFoundError:
        print(f"  ⚠️  {bestfit_pkl} not found, skipping...")
        return None
    
    # Load trajectory
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Densify
    print(f"Densifying polylines (spacing=2.0m)...")
    densified = densify_polylines(polylines, spacing=2.0)
    
    total_pts = sum(len(p) for p in densified)
    print(f"Final: {len(densified)} polylines, {total_pts} points, {total_pts/len(densified):.1f} avg points/polyline")
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(densified, f)
    print(f"Saved: {out_pkl}")
    
    # Copy transform
    transform_file = f'osm_transform_seq{seq}_bestfit_new.pkl'
    if Path(transform_file).exists():
        with open(transform_file, 'rb') as f:
            transform = pickle.load(f)
        with open(f'{output_dir}/osm_transform_seq{seq}.pkl', 'wb') as f:
            pickle.dump(transform, f)
        print(f"Copied transform: {transform_file}")
    
    # Visualize
    print("Generating visualization...")
    mean_err, start_err, end_err = visualize(seq, densified, trajectory, output_dir)
    
    print(f"\n✓ Complete: mean={mean_err:.2f}m, start={start_err:.2f}m, end={end_err:.2f}m")
    
    return {
        'seq': seq,
        'polylines': len(densified),
        'points': total_pts,
        'mean_error': mean_err,
        'start_error': start_err,
        'end_error': end_err
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', 
                        default=['00', '01', '02', '05', '07', '08', '09', '10'])
    parser.add_argument('--output', default='osm_aligned_final')
    args = parser.parse_args()
    
    print("="*70)
    print("Densify BestFit-Aligned Polylines")
    print("="*70)
    
    results = []
    for seq in args.seqs:
        try:
            result = process_sequence(seq, output_dir=args.output)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing seq {seq}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"Seq {r['seq']}: {r['polylines']} polylines, {r['points']} points, "
              f"mean={r['mean_error']:.2f}m, start={r['start_error']:.2f}m, end={r['end_error']:.2f}m")


if __name__ == '__main__':
    main()
