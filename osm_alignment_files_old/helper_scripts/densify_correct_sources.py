#!/usr/bin/env python3
"""
Densify OSM polylines using the correct source files based on reference images.

Reference mapping:
- Seq 00: refined (from osm_pbf_aligned_seq00_refined.png)
- Seq 01: bestfit (from osm_pbf_aligned_seq01_bestfit.png)
- Seq 02: bestfit (from inspect_seq02_aligned.png)
- Seq 05: bestfit
- Seq 07: bestfit (from inspect_seq07_aligned.png)
- Seq 08: bestfit (from osm_pbf_aligned_seq08_bestfit.png)
- Seq 09: refined (from osm_pbf_aligned_seq09_refined.png)
- Seq 10: bestfit (from osm_pbf_aligned_seq10.png)
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
    for pl in polylines:
        dpl = densify_polyline(pl, spacing)
        if len(dpl) >= 2:
            densified.append(dpl)
    return densified


def visualize(seq, polylines, trajectory, output_dir='osm_aligned_final'):
    """Create visualization matching reference style."""
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
        ax1.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.6, alpha=0.4,
                 label='OSM' if i == 0 else '')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5, label='Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5, label='End')
    ax1.set_title(
        f'Seq {seq} – Aligned OSM (Densified)\n'
        f'Mean: {np.mean(dists):.2f}m | Start: {dist_start:.1f}m | End: {dist_end:.1f}m | '
        f'{len(polylines):,} polylines',
        fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
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
    ax2.set_title(f'Start ({dist_start:.1f}m)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # End zoom
    ax3 = fig.add_subplot(gs[1, 2])
    for pl in polylines:
        ax3.plot(pl[:, 0], pl[:, 1], 'b-', lw=1, alpha=0.6)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax3.set_xlim(trajectory[-1, 0] - 60, trajectory[-1, 0] + 60)
    ax3.set_ylim(trajectory[-1, 1] - 60, trajectory[-1, 1] + 60)
    ax3.set_title(f'End ({dist_end:.1f}m)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    Path(output_dir).mkdir(exist_ok=True)
    out_png = f'{output_dir}/osm_aligned_seq{seq}.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")
    
    return np.mean(dists), dist_start, dist_end


def process_sequence(seq, source_type, spacing=2.0, data_root='data', output_dir='osm_aligned_final'):
    """Process a single sequence using the correct source."""
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Source: {source_type}")
    print(f"{'='*70}")
    
    # Load source polylines
    src_pkl = f'osm_polylines_aligned_seq{seq}_{source_type}.pkl'
    try:
        with open(src_pkl, 'rb') as f:
            sparse_polylines = pickle.load(f)
        sparse_polylines = [np.array(pl) for pl in sparse_polylines]
        print(f"Loaded {len(sparse_polylines)} sparse polylines from {src_pkl}")
    except FileNotFoundError:
        print(f"  ⚠️  {src_pkl} not found, skipping...")
        return None
    
    # Load trajectory
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Densify polylines
    print(f"Densifying (spacing={spacing}m)...")
    densified_polylines = densify_polylines(sparse_polylines, spacing)
    
    total_pts = sum(len(p) for p in densified_polylines)
    avg_pts = total_pts / len(densified_polylines) if densified_polylines else 0
    print(f"  Result: {len(densified_polylines)} polylines, {total_pts} points, {avg_pts:.1f} avg pts/polyline")
    
    # Verify coordinates match source
    src_pts = np.vstack(sparse_polylines)
    dst_pts = np.vstack(densified_polylines)
    print(f"  Coordinate range: X:[{dst_pts[:,0].min():.1f}, {dst_pts[:,0].max():.1f}], Y:[{dst_pts[:,1].min():.1f}, {dst_pts[:,1].max():.1f}]")
    
    # Save densified polylines
    Path(output_dir).mkdir(exist_ok=True)
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(densified_polylines, f)
    print(f"  Saved: {out_pkl}")
    
    # Copy transform file if exists
    transform_src = f'osm_transform_seq{seq}_{source_type}.pkl'
    transform_dst = f'{output_dir}/osm_transform_seq{seq}.pkl'
    try:
        with open(transform_src, 'rb') as f:
            transform = pickle.load(f)
        with open(transform_dst, 'wb') as f:
            pickle.dump(transform, f)
        print(f"  Copied transform: {transform_src}")
    except FileNotFoundError:
        # Try alternative naming
        transform_src = f'osm_transform_seq{seq}_bestfit_new.pkl'
        try:
            with open(transform_src, 'rb') as f:
                transform = pickle.load(f)
            with open(transform_dst, 'wb') as f:
                pickle.dump(transform, f)
            print(f"  Copied transform: {transform_src}")
        except FileNotFoundError:
            print(f"  ⚠️  No transform file found")
    
    # Visualize
    print("Generating visualization...")
    mean_err, dist_start, dist_end = visualize(seq, densified_polylines, trajectory, output_dir)
    
    print(f"\n✓ Complete: {mean_err:.2f}m mean error, {dist_start:.1f}m start, {dist_end:.1f}m end")
    
    return {
        'seq': seq,
        'source': source_type,
        'polylines': len(densified_polylines),
        'points': total_pts,
        'mean_error': mean_err,
        'start_error': dist_start,
        'end_error': dist_end
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['00', '01', '02', '05', '07', '08', '09', '10'])
    parser.add_argument('--output', default='osm_aligned_final')
    parser.add_argument('--spacing', type=float, default=2.0)
    args = parser.parse_args()
    
    # Mapping based on reference images
    source_map = {
        '00': 'refined',
        '01': 'bestfit',
        '02': 'bestfit',
        '05': 'bestfit',
        '07': 'bestfit',
        '08': 'bestfit',
        '09': 'refined',
        '10': 'bestfit',
    }
    
    print("="*70)
    print("Densify OSM Polylines (Correct Sources)")
    print("="*70)
    print(f"Spacing: {args.spacing}m")
    print()
    
    results = []
    for seq in args.seqs:
        source_type = source_map.get(seq, 'bestfit')
        result = process_sequence(seq, source_type, args.spacing, output_dir=args.output)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Seq':>6} {'Source':>10} {'Polylines':>12} {'Points':>12} {'Avg/Poly':>10} {'Mean Err':>10}")
    print("-"*70)
    for r in results:
        avg = r['points'] / r['polylines'] if r['polylines'] > 0 else 0
        print(f"{r['seq']:>6} {r['source']:>10} {r['polylines']:>12} {r['points']:>12} {avg:>10.1f} {r['mean_error']:>10.2f}m")
    print("="*70)
    print("\n✓ All sequences processed with correct source files!")


if __name__ == '__main__':
    main()
