#!/usr/bin/env python3
"""
Densify OSM polylines and align for all KITTI sequences.

This script:
1. Loads pre-extracted OSM polylines
2. Densifies them by interpolating points every N meters
3. Aligns to KITTI trajectory for each sequence
4. Saves densified aligned polylines
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


# Sequence configurations
SEQ_TO_RAW = {
    '00': '2011_10_03_drive_0027_sync',
    '01': '2011_10_03_drive_0042_sync',
    '02': '2011_10_03_drive_0034_sync',
    '05': '2011_09_30_drive_0018_sync',
    '07': '2011_09_30_drive_0027_sync',
    '08': '2011_09_30_drive_0028_sync',
    '09': '2011_09_30_drive_0033_sync',
    '10': '2011_09_30_drive_0034_sync',
}

SEQ_FRAME_OFFSET = {
    '00': 3346, '01': 23, '02': 57, '05': 46,
    '07': 42, '08': 252, '09': 1497, '10': 0,
}

SEQ_ANCHOR = {
    '00': 'start', '01': 'start', '02': 'start', '05': 'start',
    '07': 'end', '08': 'start', '09': 'start', '10': 'start',
}

SEQ_ROTATION_HINT = {
    '00': 93.0, '01': 0.0, '02': 36.0, '05': 129.0,
    '07': 123.0, '08': 84.0, '09': 111.0, '10': 42.0,
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


def find_oxts_dir(raw_folder, data_root='data'):
    """Find OXTS data directory."""
    root = Path(data_root) / 'raw_data'
    # Try with date subfolder first
    for date in ['2011_10_03', '2011_09_30', '2011_09_29', '2011_09_28', '2011_09_26']:
        candidate = root / raw_folder / date / raw_folder / 'oxts' / 'data'
        if candidate.exists():
            return candidate
    # Try without date subfolder (e.g., seq 01)
    candidate = root / raw_folder / 'oxts' / 'data'
    if candidate.exists():
        return candidate
    return None


def latlon_to_meters(polyline):
    """Convert lat/lon polyline to meters using simple equirectangular projection."""
    # Reference point (approximate center of Karlsruhe)
    lat_ref = 49.0
    lon_ref = 8.4
    
    # Conversion factors
    meters_per_lat = 111000.0
    meters_per_lon = 111000.0 * np.cos(np.radians(lat_ref))
    
    polyline = np.array(polyline)
    x = (polyline[:, 1] - lon_ref) * meters_per_lon  # lon -> x (east)
    y = (polyline[:, 0] - lat_ref) * meters_per_lat  # lat -> y (north)
    
    return np.column_stack([x, y])


def meters_to_latlon(polyline_meters, lat_ref=49.0, lon_ref=8.4):
    """Convert meters back to lat/lon."""
    meters_per_lat = 111000.0
    meters_per_lon = 111000.0 * np.cos(np.radians(lat_ref))
    
    lat = polyline_meters[:, 1] / meters_per_lat + lat_ref
    lon = polyline_meters[:, 0] / meters_per_lon + lon_ref
    
    return np.column_stack([lat, lon])


def densify_polyline(polyline, spacing=2.0):
    """
    Densify a polyline by interpolating points every `spacing` meters.
    
    Args:
        polyline: Nx2 array of (lat, lon) points
        spacing: Desired spacing between points in meters
    
    Returns:
        Mx2 array with densified (lat, lon) points
    """
    if len(polyline) < 2:
        return np.array(polyline)
    
    # Convert to meters for accurate distance calculation
    polyline_m = latlon_to_meters(polyline)
    
    # Calculate cumulative distances in meters
    diffs = np.diff(polyline_m, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dists[-1]
    
    if total_dist < spacing:
        return np.array(polyline)
    
    # Generate new samples at regular spacing
    n_samples = max(int(total_dist / spacing) + 1, 2)
    new_dists = np.linspace(0, total_dist, n_samples)
    
    # Interpolate in meter space
    new_points_m = []
    for d in new_dists:
        idx = np.searchsorted(cum_dists, d)
        if idx >= len(polyline_m):
            idx = len(polyline_m) - 1
        if idx == 0:
            new_points_m.append(polyline_m[0])
        else:
            t = (d - cum_dists[idx-1]) / (cum_dists[idx] - cum_dists[idx-1] + 1e-10)
            pt = polyline_m[idx-1] + t * (polyline_m[idx] - polyline_m[idx-1])
            new_points_m.append(pt)
    
    new_points_m = np.array(new_points_m)
    
    # Convert back to lat/lon
    return meters_to_latlon(new_points_m)


def densify_polylines(polylines, spacing=2.0):
    """Densify all polylines."""
    densified = []
    for pl in polylines:
        dpl = densify_polyline(pl, spacing)
        if len(dpl) >= 2:
            densified.append(dpl)
    return densified


def convert_to_local_frame(latlon_polylines, oxts_data, trajectory, frame_offset=0):
    """Convert lat/lon polylines to local KITTI frame."""
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)

    offset_east = east0 - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]

    local_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        local_polylines.append(pts)

    return local_polylines, (offset_east, offset_north)


def filter_by_bbox(polylines, trajectory, margin=200):
    """Keep polylines within margin of trajectory."""
    xmin, xmax = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
    ymin, ymax = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin

    filtered = []
    for pl in polylines:
        if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                  (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
            filtered.append(pl)
    return filtered


def optimize_alignment(local_polylines, trajectory, anchor='start', hint_deg=0.0):
    """Find optimal rotation using grid search."""
    pivot = trajectory[0] if anchor == 'start' else trajectory[-1]
    
    all_osm = np.vstack(local_polylines)
    all_osm_c = all_osm - pivot

    # Grid search around hint
    best_err, best_angle = float('inf'), np.radians(hint_deg)
    angles = np.linspace(np.radians(hint_deg - 30), np.radians(hint_deg + 30), 360)
    
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        xr = all_osm_c[:, 0] * c - all_osm_c[:, 1] * s + pivot[0]
        yr = all_osm_c[:, 0] * s + all_osm_c[:, 1] * c + pivot[1]
        tree = cKDTree(np.column_stack([xr, yr]))
        d, _ = tree.query(trajectory, k=1)
        err = d.mean()
        if err < best_err:
            best_err = err
            best_angle = angle

    # Apply best rotation
    c, s = np.cos(best_angle), np.sin(best_angle)
    rotated = []
    for pl in local_polylines:
        xc = pl[:, 0] - pivot[0]
        yc = pl[:, 1] - pivot[1]
        xr = xc * c - yc * s + pivot[0]
        yr = xc * s + yc * c + pivot[1]
        rotated.append(np.column_stack([xr, yr]))
    
    return rotated, best_angle, pivot


def bestfit_optimize(local_polylines, trajectory, spacing=5.0):
    """Fine-tune alignment with translation + small rotation."""
    diffs = np.diff(trajectory, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    n_samp = max(int(total / spacing) + 1, 2)
    sample_dists = np.linspace(0, total, n_samp)
    indices = sorted(set(int(np.argmin(np.abs(cum - sd))) for sd in sample_dists))
    sample_pts = trajectory[indices]

    all_osm = np.vstack(local_polylines)
    traj_center = trajectory.mean(axis=0)

    def cost(params):
        rot, tx, ty = params
        c, s = np.cos(rot), np.sin(rot)
        xc = all_osm[:, 0] - traj_center[0]
        yc = all_osm[:, 1] - traj_center[1]
        xr = xc * c - yc * s + traj_center[0] + tx
        yr = xc * s + yc * c + traj_center[1] + ty
        tree = cKDTree(np.column_stack([xr, yr]))
        d, _ = tree.query(sample_pts, k=1)
        return d.sum() + d.max() * 3.0

    x0 = [0.0, 0.0, 0.0]
    bounds = [(np.radians(-30), np.radians(30)), (-200, 200), (-200, 200)]

    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'disp': False})
    rot, tx, ty = res.x

    c, s = np.cos(rot), np.sin(rot)
    final_polylines = []
    for pl in local_polylines:
        xc = pl[:, 0] - traj_center[0]
        yc = pl[:, 1] - traj_center[1]
        xr = xc * c - yc * s + traj_center[0] + tx
        yr = xc * s + yc * c + traj_center[1] + ty
        final_polylines.append(np.column_stack([xr, yr]))

    return final_polylines, (rot, tx, ty)


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


def process_sequence(seq, latlon_polylines, data_root='data', output_dir='osm_aligned_final', densify_spacing=2.0):
    """Process a single sequence."""
    print(f"\n{'='*70}")
    print(f"Seq {seq}")
    print(f"{'='*70}")
    
    # Load trajectory
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Load OXTS
    raw_folder = SEQ_TO_RAW[seq]
    frame_offset = SEQ_FRAME_OFFSET.get(seq, 0)
    oxts_dir = find_oxts_dir(raw_folder, data_root)
    
    if oxts_dir is None:
        print(f"  ⚠️  OXTS not found, skipping")
        return None
    
    oxts_data = load_oxts_data(str(oxts_dir))
    print(f"OXTS: {len(oxts_data)} frames (offset={frame_offset})")
    
    # Densify polylines
    print(f"Densifying polylines (spacing={densify_spacing}m)...")
    densified_polylines = densify_polylines(latlon_polylines, densify_spacing)
    total_pts = sum(len(p) for p in densified_polylines)
    print(f"  {len(densified_polylines)} polylines, {total_pts} points, {total_pts/len(densified_polylines):.1f} avg")
    
    # Convert to local frame
    print("Converting to local frame...")
    local_polylines, gps_offset = convert_to_local_frame(
        densified_polylines, oxts_data, trajectory, frame_offset)
    
    # Filter to trajectory area
    local_polylines = filter_by_bbox(local_polylines, trajectory, margin=800)
    print(f"  After filtering: {len(local_polylines)} polylines")
    
    # Optimize rotation
    anchor = SEQ_ANCHOR.get(seq, 'start')
    hint_deg = SEQ_ROTATION_HINT.get(seq, 0.0)
    print(f"Optimizing rotation (anchor={anchor}, hint={hint_deg}°)...")
    rotated, best_rot, pivot = optimize_alignment(local_polylines, trajectory, anchor, hint_deg)
    print(f"  Best rotation: {np.degrees(best_rot):.2f}°")
    
    # Refilter
    rotated = filter_by_bbox(rotated, trajectory, margin=300)
    print(f"  After rotation: {len(rotated)} polylines")
    
    # Bestfit
    print("Fine-tuning alignment...")
    final_polylines, bf_adjust = bestfit_optimize(rotated, trajectory)
    print(f"  Adjustment: rot={np.degrees(bf_adjust[0]):.3f}°, T=({bf_adjust[1]:.2f},{bf_adjust[2]:.2f})")
    
    # Save
    Path(output_dir).mkdir(exist_ok=True)
    
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(final_polylines, f)
    
    transform_file = f'{output_dir}/osm_transform_seq{seq}.pkl'
    with open(transform_file, 'wb') as f:
        pickle.dump({
            'gps_offset': gps_offset,
            'anchor': anchor,
            'pivot': pivot.tolist(),
            'coarse_rot_deg': float(np.degrees(best_rot)),
            'bestfit_rot_deg': float(np.degrees(bf_adjust[0])),
            'bestfit_translation': [float(bf_adjust[1]), float(bf_adjust[2])]
        }, f)
    
    # Visualize
    mean_err = visualize(seq, final_polylines, trajectory, output_dir)
    
    # Summary
    final_pts = sum(len(p) for p in final_polylines)
    print(f"\n✓ Complete: {len(final_polylines)} polylines, {final_pts} points, {mean_err:.2f}m error")
    
    return {
        'seq': seq,
        'polylines': len(final_polylines),
        'points': final_pts,
        'mean_error': mean_err
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['00', '01', '02', '07', '08', '09', '10'])
    parser.add_argument('--roads', default='data/osm/karlsruhe_all_roads.pkl')
    parser.add_argument('--output', default='osm_aligned_final')
    parser.add_argument('--spacing', type=float, default=2.0)
    args = parser.parse_args()
    
    print("="*70)
    print("Densify and Align OSM Polylines")
    print("="*70)
    
    # Load road polylines
    print(f"\nLoading roads from: {args.roads}")
    with open(args.roads, 'rb') as f:
        latlon_polylines = pickle.load(f)
    print(f"Loaded {len(latlon_polylines)} polylines")
    
    # Process each sequence
    results = []
    for seq in args.seqs:
        try:
            result = process_sequence(seq, latlon_polylines, output_dir=args.output, densify_spacing=args.spacing)
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
    print(f"{'Seq':>6} {'Polylines':>12} {'Points':>12} {'Avg/Poly':>10} {'Mean Err':>10}")
    print("-"*60)
    for r in results:
        avg = r['points'] / r['polylines'] if r['polylines'] > 0 else 0
        print(f"{r['seq']:>6} {r['polylines']:>12} {r['points']:>12} {avg:>10.1f} {r['mean_error']:>10.2f}m")
    print("="*70)
    print("\n✓ All done!")


if __name__ == '__main__':
    main()
