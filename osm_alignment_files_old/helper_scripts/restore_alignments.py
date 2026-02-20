#!/usr/bin/env python3
"""
Restore OSM alignments using the verified source files.
Based on precompute_bev_5ch.py VERIFIED_VARIANTS mapping.
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
    for i, pl in enumerate(polylines):
        dpl = densify_polyline(pl, spacing)
        if len(dpl) >= 2:
            densified.append(dpl)
        
        if (i + 1) % 1000 == 0:
            print(f"  Densified {i+1}/{len(polylines)} polylines...")
    
    return densified


# VERIFIED_VARIANTS from precompute_bev_5ch.py - these were verified visually
VERIFIED_VARIANTS = {
    '00': 'refined',   # osm_pbf_aligned_seq00_refined.png
    '01': 'bestfit',   # osm_pbf_aligned_seq01_bestfit.png
    '02': 'bestfit',   # inspect_seq02_aligned.png shows good fit
    '05': 'bestfit',   # osm_pbf_aligned_seq05_bestfit.png
    '07': 'bestfit',   # inspect_seq07_aligned.png shows good fit
    '08': 'bestfit',   # osm_pbf_aligned_seq08_bestfit.png
    '09': 'refined',   # osm_pbf_aligned_seq09_refined.png
    '10': 'bestfit',   # osm_pbf_aligned_seq10_bestfit.png
}


def process_sequence(seq, data_root='data', output_dir='osm_aligned_final', spacing=2.0):
    """Process a sequence using its verified source."""
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Restoring from Verified Source")
    print(f"{'='*70}")
    
    # Get verified variant
    variant = VERIFIED_VARIANTS.get(seq, 'bestfit')
    source_file = f'osm_polylines_aligned_seq{seq}_{variant}.pkl'
    
    print(f"Source: {source_file}")
    
    # Load source polylines (already aligned)
    with open(source_file, 'rb') as f:
        polylines = pickle.load(f)
    
    print(f"Loaded {len(polylines)} polylines")
    
    # Load trajectory
    trajectory = extract_trajectory(load_poses(f'{data_root}/kitti/poses/{seq}.txt'))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Check alignment before densification
    all_pts = np.vstack(polylines)
    tree = cKDTree(all_pts)
    dists_before, _ = tree.query(trajectory, k=1)
    print(f"Source alignment: mean={dists_before.mean():.2f}m")
    
    # Densify
    print(f"Densifying with {spacing}m spacing...")
    densified = densify_polylines(polylines, spacing=spacing)
    total_pts = sum(len(p) for p in densified)
    print(f"Final: {len(densified)} polylines, {total_pts} points")
    
    # Compute error metrics
    all_pts = np.vstack(densified)
    tree = cKDTree(all_pts)
    dists, _ = tree.query(trajectory, k=1)
    dist_start, _ = tree.query(trajectory[0])
    dist_end, _ = tree.query(trajectory[-1])
    
    print(f"Error: mean={dists.mean():.2f}m, start={dist_start:.2f}m, end={dist_end:.2f}m")
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(densified, f)
    print(f"Saved: {out_pkl}")
    
    # Copy transform if available
    transform_file = f'osm_transform_seq{seq}_bestfit_new.pkl'
    if Path(transform_file).exists():
        with open(transform_file, 'rb') as f:
            transform = pickle.load(f)
        with open(f'{output_dir}/osm_transform_seq{seq}.pkl', 'wb') as f:
            pickle.dump(transform, f)
    
    # Visualize
    print("Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Seq {seq} - Dense Alignment (source: {variant})\n'
                 f'Mean: {dists.mean():.2f}m | Start: {dist_start:.2f}m | End: {dist_end:.2f}m',
                 fontsize=14, fontweight='bold')
    
    # OSM only
    ax = axes[0]
    for pl in densified[:500]:
        ax.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.7, alpha=0.5)
    ax.set_title('OSM Roads (sample)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Trajectory only
    ax = axes[1]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', lw=2)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='lime', s=200, marker='o', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=300, marker='*', zorder=5)
    ax.set_title('Trajectory')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Overlay
    ax = axes[2]
    for pl in densified[:500]:
        ax.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.7, alpha=0.4)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'g-', lw=2)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='lime', s=200, marker='o', zorder=5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=300, marker='*', zorder=5)
    ax.set_title('Overlay')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/osm_aligned_seq{seq}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Seq {seq} complete!")
    
    return {
        'seq': seq,
        'source': variant,
        'polylines': len(densified),
        'points': total_pts,
        'mean_error': float(dists.mean()),
        'start_error': float(dist_start),
        'end_error': float(dist_end),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Restore OSM alignments from verified sources')
    parser.add_argument('--seqs', nargs='+', default=list(VERIFIED_VARIANTS.keys()),
                        help='Sequences to process')
    parser.add_argument('--output', default='osm_aligned_final',
                        help='Output directory')
    parser.add_argument('--spacing', type=float, default=2.0,
                        help='Point spacing in meters')
    args = parser.parse_args()
    
    print("="*70)
    print("RESTORE OSM ALIGNMENTS FROM VERIFIED SOURCES")
    print("="*70)
    print(f"Processing sequences: {', '.join(args.seqs)}")
    print(f"Output directory: {args.output}")
    print(f"Spacing: {args.spacing}m")
    
    results = []
    for seq in args.seqs:
        if seq not in VERIFIED_VARIANTS:
            print(f"\n⚠️ Seq {seq} not in verified variants, skipping")
            continue
        
        try:
            result = process_sequence(seq, output_dir=args.output, spacing=args.spacing)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing seq {seq}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Seq':>4} | {'Source':>8} | {'Polylines':>9} | {'Points':>8} | {'Mean':>7} | {'Start':>7} | {'End':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['seq']:>4} | {r['source']:>8} | {r['polylines']:>9,} | {r['points']:>8,} | "
              f"{r['mean_error']:>7.2f} | {r['start_error']:>7.2f} | {r['end_error']:>7.2f}")
    
    print("\n✓ All sequences restored!")


if __name__ == '__main__':
    main()
