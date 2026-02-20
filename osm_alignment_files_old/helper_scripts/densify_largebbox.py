#!/usr/bin/env python3
"""
Densify existing largebbox polylines using the correct transforms.
This uses the pre-extracted largebbox files which already have more polylines.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm

# Frame offsets must match align_and_viz_bestfit.py for consistency
# These are the raw-drive frame indices that correspond to odometry pose frame 0
SEQ_FRAME_OFFSET = {
    '00': 3346,
    '02': 57,    # CRITICAL: was 0, must be 57 for correct GPS alignment
    '05': 46,
    '07': 42,
    '08': 252,
    '09': 1497,
    '10': 0,
}


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
        
        # Progress update every 1000 polylines
        if (i + 1) % 1000 == 0:
            print(f"  Densified {i+1}/{len(polylines)} polylines...")
    
    return densified


def filter_to_trajectory_area(polylines, trajectory, margin=300):
    """Keep polylines within margin of trajectory."""
    xmin, xmax = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
    ymin, ymax = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin

    filtered = []
    for pl in polylines:
        if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                  (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
            filtered.append(pl)
    return filtered


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
    
    return np.mean(dists)


def process_sequence(seq, data_root='data', output_dir='osm_aligned_final'):
    """Process using largebbox file with correct transform."""
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Densifying LargeBBox Polylines")
    print(f"{'='*70}")
    
    # Load largebbox polylines (already in lat/lon)
    largebbox_pkl = f'osm_polylines_latlon_seq{seq}_largebbox.pkl'
    try:
        with open(largebbox_pkl, 'rb') as f:
            latlon_polylines = pickle.load(f)
        print(f"Loaded {len(latlon_polylines)} lat/lon polylines from {largebbox_pkl}")
    except FileNotFoundError:
        print(f"  ⚠️  {largebbox_pkl} not found, skipping...")
        return None
    
    # Load trajectory
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Load transform
    transform_file = f'osm_transform_seq{seq}_bestfit_new.pkl'
    try:
        with open(transform_file, 'rb') as f:
            transform = pickle.load(f)
        print(f"Loaded transform from {transform_file}")
    except FileNotFoundError:
        print(f"  ⚠️  Transform not found, skipping...")
        return None
    
    # Extract transform parameters
    offset_east = transform['gps']['offset_east']
    offset_north = transform['gps']['offset_north']
    anchor = transform['anchor']
    pivot = np.array(transform['pivot'])
    coarse_rot = np.radians(transform['coarse_rot_deg'])
    fine_rot = transform['bestfit']['rotation']
    fine_trans = transform['bestfit']['translation']
    
    # Convert lat/lon to local frame
    print("Converting lat/lon to local frame...")
    local_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        local_polylines.append(pts)
    
    # Apply coarse rotation
    print(f"Applying coarse rotation ({np.degrees(coarse_rot):.1f}°)...")
    c, s = np.cos(coarse_rot), np.sin(coarse_rot)
    rotated = []
    for pl in local_polylines:
        xc = pl[:, 0] - pivot[0]
        yc = pl[:, 1] - pivot[1]
        xr = xc * c - yc * s + pivot[0]
        yr = xc * s + yc * c + pivot[1]
        rotated.append(np.column_stack([xr, yr]))
    
    # Apply fine rotation and translation
    print(f"Applying fine-tuning (rot={np.degrees(fine_rot):.2f}°, T={fine_trans})...")
    c, s = np.cos(fine_rot), np.sin(fine_rot)
    traj_center = trajectory.mean(axis=0)
    
    final_polylines = []
    for pl in rotated:
        xc = pl[:, 0] - traj_center[0]
        yc = pl[:, 1] - traj_center[1]
        xr = xc * c - yc * s + traj_center[0] + fine_trans[0]
        yr = xc * s + yc * c + traj_center[1] + fine_trans[1]
        final_polylines.append(np.column_stack([xr, yr]))
    
    # Filter to trajectory area
    print("Filtering to trajectory area...")
    final_polylines = filter_to_trajectory_area(final_polylines, trajectory, margin=500)
    print(f"After filtering: {len(final_polylines)} polylines")
    
    # Densify
    print("Densifying polylines (spacing=2.0m)...")
    densified_polylines = densify_polylines(final_polylines, spacing=2.0)
    
    total_pts = sum(len(p) for p in densified_polylines)
    print(f"Final: {len(densified_polylines)} polylines, {total_pts} points, {total_pts/len(densified_polylines):.1f} avg")
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(densified_polylines, f)
    print(f"Saved: {out_pkl}")
    
    # Copy transform
    with open(f'{output_dir}/osm_transform_seq{seq}.pkl', 'wb') as f:
        pickle.dump(transform, f)
    
    # Visualize
    print("Generating visualization...")
    mean_err = visualize(seq, densified_polylines, trajectory, output_dir)
    
    print(f"\n✓ Complete: {mean_err:.2f}m mean error")
    
    return {
        'seq': seq,
        'polylines': len(densified_polylines),
        'points': total_pts,
        'mean_error': mean_err
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['02', '07'])
    parser.add_argument('--output', default='osm_aligned_final')
    args = parser.parse_args()
    
    print("="*70)
    print("Densify LargeBBox Polylines")
    print("="*70)
    
    results = []
    for seq in args.seqs:
        try:
            result = process_sequence(seq, output_dir=args.output)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"Seq {r['seq']}: {r['polylines']} polylines, {r['points']} points, {r['mean_error']:.2f}m error")


if __name__ == '__main__':
    main()
