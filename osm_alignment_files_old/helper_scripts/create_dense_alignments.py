#!/usr/bin/env python3
"""
Unified script to create dense OSM alignments for all KITTI sequences.

Each sequence has specific parameters determined through visual inspection
and optimization. This script uses a centralized configuration to ensure
consistency.
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


# ═════════════════════════════════════════════════════════════════════════════
# CENTRALIZED SEQUENCE CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
#
# Each sequence has unique parameters determined through visual inspection
# and grid search optimization. These must match the parameters used in
# align_and_viz_bestfit.py
#
SEQ_CONFIG = {
    '00': {
        'raw_folder': '2011_10_03_drive_0027_sync',
        'date': '2011_10_03',
        'frame_offset': 0,          # Use 0 instead of 3346
        'rotation_deg': 93.0,       # Coarse rotation for alignment
        'anchor': 'start',
        'source': 'regbez',         # Use regbez (refined lat/lon not available)
    },
    '01': {
        'raw_folder': '2011_10_03_drive_0027_sync',
        'date': '2011_10_03',
        'frame_offset': 0,
        'rotation_deg': 0,
        'anchor': 'start',
        'source': 'bestfit',        # No lat/lon file, use pre-aligned
    },
    '02': {
        'raw_folder': '2011_10_03_drive_0034_sync',
        'date': '2011_10_03',
        'frame_offset': 0,          # CRITICAL: 0 works better than 57
        'rotation_deg': 36.0,
        'anchor': 'start',
        'source': 'regbez',
    },
    '05': {
        'raw_folder': '2011_09_30_drive_0018_sync',
        'date': '2011_09_30',
        'frame_offset': 0,          # Use 0 instead of 46
        'rotation_deg': 120.0,      # Optimized for frame_offset=0
        'anchor': 'start',
        'source': 'regbez',
    },
    '07': {
        'raw_folder': '2011_09_30_drive_0027_sync',
        'date': '2011_09_30',
        'frame_offset': 0,          # Use 0 instead of 42
        'rotation_deg': 122.0,      # Optimized for frame_offset=0
        'anchor': 'start',
        'source': 'regbez',
    },
    '08': {
        'raw_folder': '2011_09_30_drive_0028_sync',
        'date': '2011_09_30',
        'frame_offset': 0,          # Use 0 instead of 252
        'rotation_deg': 80.0,       # Optimized for frame_offset=0
        'anchor': 'start',
        'source': 'bestfit',        # Bestfit already works well
    },
    '09': {
        'raw_folder': '2011_09_30_drive_0033_sync',
        'date': '2011_09_30',
        'frame_offset': 0,          # Use 0 instead of 1497
        'rotation_deg': 118.0,      # Optimized
        'anchor': 'start',
        'source': 'regbez',         # Use regbez (refined lat/lon not available)
    },
    '10': {
        'raw_folder': '2011_09_30_drive_0034_sync',
        'date': '2011_09_30',
        'frame_offset': 0,
        'rotation_deg': 70.0,       # Optimized
        'anchor': 'start',
        'source': 'regbez',
    },
}


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


def process_sequence(seq, data_root='data', output_dir='osm_aligned_final', spacing=2.0):
    """
    Process a single sequence using its configuration.
    
    Two modes:
    1. source='bestfit': Load pre-aligned polylines and densify
    2. source='regbez'/'refined'/'largebbox': Load lat/lon, apply transforms, densify
    """
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Dense Alignment")
    print(f"{'='*70}")
    
    cfg = SEQ_CONFIG[seq]
    print(f"Config: frame_offset={cfg['frame_offset']}, rotation={cfg['rotation_deg']}°, "
          f"anchor={cfg['anchor']}, source={cfg['source']}")
    
    # Load trajectory
    trajectory = extract_trajectory(load_poses(f'{data_root}/kitti/poses/{seq}.txt'))
    print(f"Trajectory: {len(trajectory)} frames")
    
    if cfg['source'] == 'bestfit':
        # Mode 1: Use pre-aligned bestfit file
        bestfit_file = f'osm_polylines_aligned_seq{seq}_bestfit.pkl'
        print(f"Loading pre-aligned polylines from {bestfit_file}")
        
        with open(bestfit_file, 'rb') as f:
            polylines = pickle.load(f)
        
        print(f"Loaded {len(polylines)} polylines")
        
    else:
        # Mode 2: Load lat/lon and apply transforms
        source_file = f'osm_polylines_latlon_seq{seq}_{cfg["source"]}.pkl'
        print(f"Loading lat/lon polylines from {source_file}")
        
        with open(source_file, 'rb') as f:
            latlon_polylines = pickle.load(f)
        print(f"Loaded {len(latlon_polylines)} lat/lon polylines")
        
        # Load OXTS data
        oxts_dir = f'{data_root}/raw_data/{cfg["raw_folder"]}/{cfg["date"]}/{cfg["raw_folder"]}/oxts/data'
        oxts = load_oxts_data(oxts_dir)
        
        # Compute GPS offset
        ref_frame = min(cfg['frame_offset'], len(oxts) - 1)
        lat0, lon0 = oxts[ref_frame, 0], oxts[ref_frame, 1]
        east0, north0 = latlon_to_utm(lat0, lon0)
        offset_east = east0 - trajectory[0, 0]
        offset_north = north0 - trajectory[0, 1]
        
        print(f"GPS offset (frame {ref_frame}): ({offset_east:.2f}, {offset_north:.2f})")
        
        # Convert to local frame
        local_polylines = []
        for polyline in latlon_polylines:
            pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                             latlon_to_utm(lat, lon)[1] - offset_north]
                            for lat, lon in polyline])
            local_polylines.append(pts)
        
        # Apply rotation
        if cfg['rotation_deg'] != 0:
            print(f"Applying rotation: {cfg['rotation_deg']}° around {cfg['anchor']}")
            r = np.radians(cfg['rotation_deg'])
            c, s = np.cos(r), np.sin(r)
            
            if cfg['anchor'] == 'start':
                px, py = trajectory[0]
            else:
                px, py = trajectory[-1]
            
            polylines = []
            for pl in local_polylines:
                xc = pl[:, 0] - px
                yc = pl[:, 1] - py
                xr = xc * c - yc * s + px
                yr = xc * s + yc * c + py
                polylines.append(np.column_stack([xr, yr]))
        else:
            polylines = local_polylines
        
        # Filter to trajectory area
        margin = 400
        xmin, xmax = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
        ymin, ymax = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin
        
        filtered = []
        for pl in polylines:
            if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                      (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
                filtered.append(pl)
        
        polylines = filtered
        print(f"After filtering: {len(polylines)} polylines")
    
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
    
    # Save configuration for reference
    cfg_out = {
        'seq': seq,
        'frame_offset': cfg['frame_offset'],
        'rotation_deg': cfg['rotation_deg'],
        'anchor': cfg['anchor'],
        'source': cfg['source'],
        'spacing': spacing,
        'mean_error': float(dists.mean()),
        'start_error': float(dist_start),
        'end_error': float(dist_end),
    }
    with open(f'{output_dir}/osm_config_seq{seq}.pkl', 'wb') as f:
        pickle.dump(cfg_out, f)
    
    # Visualize
    print("Generating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Seq {seq} - Dense Alignment (offset={cfg["frame_offset"]}, rot={cfg["rotation_deg"]}°)\n'
                 f'Mean: {dists.mean():.2f}m | Start: {dist_start:.2f}m | End: {dist_end:.2f}m',
                 fontsize=14, fontweight='bold')
    
    # OSM only
    ax = axes[0]
    for pl in densified[:500]:  # Limit for display speed
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
        'polylines': len(densified),
        'points': total_pts,
        'mean_error': float(dists.mean()),
        'start_error': float(dist_start),
        'end_error': float(dist_end),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Create dense OSM alignments')
    parser.add_argument('--seqs', nargs='+', default=list(SEQ_CONFIG.keys()),
                        help='Sequences to process')
    parser.add_argument('--output', default='osm_aligned_final',
                        help='Output directory')
    parser.add_argument('--spacing', type=float, default=2.0,
                        help='Point spacing in meters')
    args = parser.parse_args()
    
    print("="*70)
    print("CREATE DENSE OSM ALIGNMENTS")
    print("="*70)
    print(f"Processing sequences: {', '.join(args.seqs)}")
    print(f"Output directory: {args.output}")
    print(f"Spacing: {args.spacing}m")
    
    results = []
    for seq in args.seqs:
        if seq not in SEQ_CONFIG:
            print(f"\n⚠️ Seq {seq} not in configuration, skipping")
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
    print(f"{'Seq':>4} | {'Polylines':>9} | {'Points':>8} | {'Mean':>7} | {'Start':>7} | {'End':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['seq']:>4} | {r['polylines']:>9,} | {r['points']:>8,} | "
              f"{r['mean_error']:>7.2f} | {r['start_error']:>7.2f} | {r['end_error']:>7.2f}")
    
    print("\n✓ All sequences processed!")


if __name__ == '__main__':
    main()
