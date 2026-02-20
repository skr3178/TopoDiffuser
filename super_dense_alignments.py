#!/usr/bin/env python3
"""
Create SUPER DENSE OSM alignments using largebbox files.

This uses the full Germany OSM extracts (20k-30k polylines) with proper alignment.
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
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def extract_trajectory(poses):
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def densify_polyline(polyline, spacing=2.0):
    """Densify a polyline by interpolating points every `spacing` meters."""
    if len(polyline) < 2:
        return np.array(polyline)
    
    polyline = np.array(polyline)
    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dists[-1]
    
    if total_dist < spacing:
        return polyline
    
    n_samples = max(int(total_dist / spacing) + 1, 2)
    new_dists = np.linspace(0, total_dist, n_samples)
    
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
        if (i + 1) % 5000 == 0:
            print(f"  Densified {i+1}/{len(polylines)}...")
    return densified


# Configuration for largebbox processing
# Using frame_offset=0 with optimized rotations
SEQ_CONFIG = {
    '02': {
        'raw_folder': '2011_10_03_drive_0034_sync',
        'date': '2011_10_03',
        'frame_offset': 0,
        'rotation_deg': 36.0,
        'anchor': 'start',
        'margin': 500,
    },
    '07': {
        'raw_folder': '2011_09_30_drive_0027_sync',
        'date': '2011_09_30',
        'frame_offset': 0,
        'rotation_deg': 122.0,
        'anchor': 'start',
        'margin': 500,
    },
    '08': {
        'raw_folder': '2011_09_30_drive_0028_sync',
        'date': '2011_09_30',
        'frame_offset': 0,
        'rotation_deg': 80.0,  # Optimized
        'anchor': 'start',
        'margin': 500,
    },
}


def process_sequence(seq, data_root='data', output_dir='osm_aligned_final', spacing=2.0):
    """Process largebbox file for a sequence."""
    print(f"\n{'='*70}")
    print(f"Seq {seq} - SUPER DENSE (largebbox)")
    print(f"{'='*70}")
    
    cfg = SEQ_CONFIG.get(seq)
    if not cfg:
        print(f"No largebbox config for seq {seq}, skipping")
        return None
    
    trajectory = extract_trajectory(load_poses(f'{data_root}/kitti/poses/{seq}.txt'))
    print(f"Trajectory: {len(trajectory)} frames")
    print(f"Config: frame_offset={cfg['frame_offset']}, rotation={cfg['rotation_deg']}°")
    
    # Load largebbox lat/lon file
    largebbox_file = f'osm_polylines_latlon_seq{seq}_largebbox.pkl'
    print(f"Loading {largebbox_file}...")
    
    with open(largebbox_file, 'rb') as f:
        latlon_polylines = pickle.load(f)
    
    print(f"Loaded {len(latlon_polylines)} lat/lon polylines")
    
    # Load OXTS
    oxts_dir = f'{data_root}/raw_data/{cfg["raw_folder"]}/{cfg["date"]}/{cfg["raw_folder"]}/oxts/data'
    oxts = load_oxts_data(oxts_dir)
    
    # GPS offset
    ref = min(cfg['frame_offset'], len(oxts) - 1)
    lat0, lon0 = oxts[ref, 0], oxts[ref, 1]
    e0, n0 = latlon_to_utm(lat0, lon0)
    offset_east = e0 - trajectory[0, 0]
    offset_north = n0 - trajectory[0, 1]
    print(f"GPS offset (frame {ref}): ({offset_east:.2f}, {offset_north:.2f})")
    
    # Convert to local
    print("Converting lat/lon to local frame...")
    local_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        local_polylines.append(pts)
    
    # Apply rotation
    print(f"Applying rotation: {cfg['rotation_deg']}°...")
    r = np.radians(cfg['rotation_deg'])
    c, s = np.cos(r), np.sin(r)
    px, py = trajectory[0] if cfg['anchor'] == 'start' else trajectory[-1]
    
    rotated = []
    for pl in local_polylines:
        xc = pl[:, 0] - px
        yc = pl[:, 1] - py
        xr = xc * c - yc * s + px
        yr = xc * s + yc * c + py
        rotated.append(np.column_stack([xr, yr]))
    
    # Filter to trajectory area (larger margin for super density)
    print(f"Filtering to trajectory area (margin={cfg['margin']}m)...")
    xmin, xmax = trajectory[:, 0].min() - cfg['margin'], trajectory[:, 0].max() + cfg['margin']
    ymin, ymax = trajectory[:, 1].min() - cfg['margin'], trajectory[:, 1].max() + cfg['margin']
    
    filtered = []
    for pl in rotated:
        if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                  (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
            filtered.append(pl)
    
    print(f"After filtering: {len(filtered)} polylines")
    
    # Densify
    print(f"Densifying (spacing={spacing}m)...")
    densified = densify_polylines(filtered, spacing=spacing)
    total_pts = sum(len(p) for p in densified)
    print(f"Final: {len(densified)} polylines, {total_pts} points")
    
    # Compute error
    all_pts = np.vstack(densified)
    tree = cKDTree(all_pts)
    dists, _ = tree.query(trajectory, k=1)
    d_start, _ = tree.query(trajectory[0])
    d_end, _ = tree.query(trajectory[-1])
    
    print(f"Error: mean={dists.mean():.2f}m, start={d_start:.2f}m, end={d_end:.2f}m")
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(densified, f)
    print(f"Saved: {out_pkl}")
    
    # Visualize (sample for speed)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Seq {seq} - SUPER DENSE (largebbox)\n'
                 f'{len(densified)} polylines, {total_pts} points\n'
                 f'Mean: {dists.mean():.2f}m | Start: {d_start:.2f}m | End: {d_end:.2f}m',
                 fontsize=14, fontweight='bold')
    
    sample = densified[::max(1, len(densified)//500)]  # Sample for display
    for pl in sample:
        axes[0].plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.4)
    axes[0].set_title('OSM (sample)')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(trajectory[:, 0], trajectory[:, 1], 'g-', lw=2)
    axes[1].scatter(trajectory[0, 0], trajectory[0, 1], c='lime', s=200, marker='o', zorder=5)
    axes[1].scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=300, marker='*', zorder=5)
    axes[1].set_title('Trajectory')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    for pl in sample:
        axes[2].plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.3)
    axes[2].plot(trajectory[:, 0], trajectory[:, 1], 'g-', lw=2)
    axes[2].scatter(trajectory[0, 0], trajectory[0, 1], c='lime', s=200, marker='o', zorder=5)
    axes[2].scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=300, marker='*', zorder=5)
    axes[2].set_title('Overlay')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/osm_aligned_seq{seq}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Seq {seq} complete!")
    
    return {
        'seq': seq,
        'polylines': len(densified),
        'points': total_pts,
        'mean_error': float(dists.mean()),
        'start_error': float(d_start),
        'end_error': float(d_end),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['02', '07', '08'])
    parser.add_argument('--output', default='osm_aligned_final')
    parser.add_argument('--spacing', type=float, default=2.0)
    args = parser.parse_args()
    
    print("="*70)
    print("SUPER DENSE OSM ALIGNMENTS (LARGEBBOX)")
    print("="*70)
    
    results = []
    for seq in args.seqs:
        try:
            result = process_sequence(seq, output_dir=args.output, spacing=args.spacing)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Seq':>4} | {'Polylines':>10} | {'Points':>10} | {'Mean':>7} | {'Start':>7} | {'End':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['seq']:>4} | {r['polylines']:>10,} | {r['points']:>10,} | "
              f"{r['mean_error']:>7.2f} | {r['start_error']:>7.2f} | {r['end_error']:>7.2f}")
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
