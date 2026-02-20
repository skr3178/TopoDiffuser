#!/usr/bin/env python3
"""
Recreation script for OSM alignments with DENSIFIED polylines.

This script:
1. Extracts OSM road polylines from the full Germany OSM file
2. Densifies polylines by interpolating points every ~2 meters
3. Applies GPS-based offset + rotation alignment
4. Saves aligned polylines in KITTI local frame

Target sequences: 00, 01, 02, 07, 08, 09, 10
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


# Map KITTI odometry sequences to raw drive folders
SEQ_TO_RAW = {
    '00': '2011_10_03_drive_0027_sync',
    '01': '2011_10_03_drive_0042_sync',
    '02': '2011_10_03_drive_0034_sync',
    '07': '2011_09_30_drive_0027_sync',
    '08': '2011_09_30_drive_0028_sync',
    '09': '2011_09_30_drive_0033_sync',
    '10': '2011_09_30_drive_0034_sync',
}

# Frame offsets - OXTS frame that corresponds to odometry pose 0
# Found by matching trajectory heading with OXTS yaw
SEQ_FRAME_OFFSET = {
    '00': 3346,
    '01': 0,      # Need to determine
    '02': 57,
    '07': 42,
    '08': 252,
    '09': 1497,
    '10': 0,
}

# Anchor point for rotation
SEQ_ANCHOR = {
    '00': 'start',
    '01': 'start',  # Need to verify
    '02': 'start',
    '07': 'end',
    '08': 'start',
    '09': 'start',
    '10': 'start',
}

# Rotation hints (determined by previous optimization)
SEQ_ROTATION_HINT = {
    '00': 93.0,
    '01': 0.0,      # Need to determine
    '02': 36.0,
    '07': 123.0,
    '08': 84.0,
    '09': 111.0,
    '10': 42.0,
}

# Densification spacing (meters)
DENSIFY_SPACING = 2.0


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
    for date in ['2011_10_03', '2011_09_30', '2011_09_29', '2011_09_28', '2011_09_26']:
        candidate = root / raw_folder / date / raw_folder / 'oxts' / 'data'
        if candidate.exists():
            return candidate
    candidate = root / raw_folder / 'oxts' / 'data'
    if candidate.exists():
        return candidate
    return None


def densify_polyline(polyline, spacing=DENSIFY_SPACING):
    """
    Densify a polyline by interpolating points every `spacing` meters.
    
    Args:
        polyline: Nx2 array of (x, y) or (lat, lon) points
        spacing: Desired spacing between points in meters
    
    Returns:
        Mx2 array with densified points
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
        # Segment too short, return original
        return polyline
    
    # Generate new samples at regular spacing
    n_samples = max(int(total_dist / spacing) + 1, 2)
    new_dists = np.linspace(0, total_dist, n_samples)
    
    # Interpolate
    new_points = []
    for d in new_dists:
        # Find segment containing this distance
        idx = np.searchsorted(cum_dists, d)
        if idx >= len(polyline):
            idx = len(polyline) - 1
        if idx == 0:
            new_points.append(polyline[0])
        else:
            # Linear interpolation within segment
            t = (d - cum_dists[idx-1]) / (cum_dists[idx] - cum_dists[idx-1] + 1e-10)
            pt = polyline[idx-1] + t * (polyline[idx] - polyline[idx-1])
            new_points.append(pt)
    
    return np.array(new_points)


def densify_polylines(polylines, spacing=DENSIFY_SPACING):
    """Densify all polylines."""
    densified = []
    for pl in polylines:
        dpl = densify_polyline(pl, spacing)
        if len(dpl) >= 2:
            densified.append(dpl)
    return densified


def get_bounding_box_from_trajectory(trajectory, oxts_data, frame_offset, margin_m=1000):
    """Get lat/lon bounding box from trajectory with margin."""
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)
    
    offset_east = east0 - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]
    
    traj_utm = np.array([
        [trajectory[i, 0] + offset_east, trajectory[i, 1] + offset_north]
        for i in range(len(trajectory))
    ])
    
    east_min = traj_utm[:, 0].min() - margin_m
    east_max = traj_utm[:, 0].max() + margin_m
    north_min = traj_utm[:, 1].min() - margin_m
    north_max = traj_utm[:, 1].max() + margin_m
    
    # Approximate UTM to lat/lon conversion
    def utm_to_latlon_approx(east, north, zone=32):
        lat0, lon0 = 49.0, 8.4
        east0, north0 = latlon_to_utm(lat0, lon0)
        dlat = (north - north0) / 111000.0
        dlon = (east - east0) / (111000.0 * np.cos(np.radians(lat0)))
        return lat0 + dlat, lon0 + dlon
    
    lat_min, lon_min = utm_to_latlon_approx(east_min, north_min)
    lat_max, lon_max = utm_to_latlon_approx(east_max, north_max)
    
    return (min(lat_min, lat_max), min(lon_min, lon_max), 
            max(lat_min, lat_max), max(lon_min, lon_max))


def parse_osm_pbf(pbf_file, bounding_box=None):
    """Parse OSM PBF and extract road polylines."""
    from pyrosm import OSM
    
    print(f"Loading OSM from: {pbf_file}")
    
    if bounding_box:
        min_lat, min_lon, max_lat, max_lon = bounding_box
        osm = OSM(pbf_file, bounding_box=[min_lon, min_lat, max_lon, max_lat])
        print(f"Using bounding box: lat=[{min_lat:.4f}, {max_lat:.4f}], lon=[{min_lon:.4f}, {max_lon:.4f}]")
    else:
        osm = OSM(pbf_file)
    
    print("Extracting driving network...")
    network = osm.get_network(network_type="driving")
    
    print(f"Found {len(network)} road segments")
    
    road_polylines = []
    for idx, row in network.iterrows():
        geom = row['geometry']
        if geom is None:
            continue
        
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                polyline = [(lat, lon) for lon, lat in coords]
                road_polylines.append(polyline)
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    polyline = [(lat, lon) for lon, lat in coords]
                    road_polylines.append(polyline)
    
    print(f"Extracted {len(road_polylines)} raw polylines")
    return road_polylines


def gps_offset_polylines(latlon_polylines, oxts_data, trajectory, frame_offset=0):
    """Convert lat/lon polylines to local KITTI frame."""
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)

    offset_east = east0 - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]

    print(f"  OXTS ref frame: {ref_frame}")
    print(f"  UTM offset: ({offset_east:.1f}, {offset_north:.1f})")

    local_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        local_polylines.append(pts)

    return local_polylines, {'offset_east': offset_east, 'offset_north': offset_north}


def rotate_polylines(polylines, angle_deg, pivot):
    """Rotate polylines around a pivot point."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    
    rotated = []
    for pl in polylines:
        xc = pl[:, 0] - pivot[0]
        yc = pl[:, 1] - pivot[1]
        xr = xc * c - yc * s + pivot[0]
        yr = xc * s + yc * c + pivot[1]
        rotated.append(np.column_stack([xr, yr]))
    
    return rotated


def filter_to_trajectory_area(polylines, trajectory, margin=200):
    """Keep polylines within margin of trajectory."""
    xmin, xmax = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
    ymin, ymax = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin

    filtered = []
    for pl in polylines:
        if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                  (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
            filtered.append(pl)
    return filtered


def optimize_rotation(local_polylines, trajectory, anchor='start', hint_deg=0.0):
    """Find optimal rotation using grid search + fine-tuning."""
    if anchor == 'start':
        pivot = trajectory[0].copy()
    else:
        pivot = trajectory[-1].copy()
    
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

    print(f"  Best rotation: {np.degrees(best_angle):.2f}°  mean error: {best_err:.2f} m")
    
    return rotate_polylines(local_polylines, np.degrees(best_angle), pivot), best_angle, pivot


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

    print(f"  Fine-tuning with {len(sample_pts)} sample points...")
    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'disp': False})
    rot, tx, ty = res.x
    print(f"  Fine adjustment: rot={np.degrees(rot):.3f}°  T=({tx:.2f},{ty:.2f})")

    c, s = np.cos(rot), np.sin(rot)
    final_polylines = []
    for pl in local_polylines:
        xc = pl[:, 0] - traj_center[0]
        yc = pl[:, 1] - traj_center[1]
        xr = xc * c - yc * s + traj_center[0] + tx
        yr = xc * s + yc * c + traj_center[1] + ty
        final_polylines.append(np.column_stack([xr, yr]))

    return final_polylines, {'rotation': rot, 'translation': [tx, ty]}


def visualize(seq, polylines, trajectory, output_dir='osm_aligned_final'):
    """Create visualization of aligned polylines."""
    all_osm = np.vstack(polylines)
    tree = cKDTree(all_osm)
    dists, _ = tree.query(trajectory, k=1)
    dist_start, _ = tree.query(trajectory[0])
    dist_end, _ = tree.query(trajectory[-1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Main view
    ax1 = axes[0]
    for i, pl in enumerate(polylines):
        ax1.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.4,
                 label='OSM' if i == 0 else '')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5, label='Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='s', zorder=5)
    ax1.set_title(
        f'Seq {seq} - OSM Alignment (Densified)\n'
        f'Mean Error: {np.mean(dists):.2f}m | Polylines: {len(polylines)} | '
        f'Points: {sum(len(p) for p in polylines)}',
        fontsize=12, fontweight='bold'
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Zoomed view around trajectory
    ax2 = axes[1]
    for pl in polylines:
        ax2.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.8, alpha=0.5)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5, label='Start')
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='s', zorder=5, label='End')
    
    # Set zoom limits around trajectory
    margin = 100
    ax2.set_xlim(trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin)
    ax2.set_ylim(trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin)
    ax2.set_title('Seq {seq} - Zoomed View', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    out_png = f'{output_dir}/osm_aligned_seq{seq}.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")
    
    return np.mean(dists), np.max(dists)


def process_sequence(seq, pbf_file, data_root='data', output_dir='osm_aligned_final'):
    """Process a single sequence: extract, densify, align, save."""
    print(f"\n{'='*70}")
    print(f"Processing Seq {seq}")
    print(f"{'='*70}")
    
    # Load trajectory
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Load OXTS data
    raw_folder = SEQ_TO_RAW[seq]
    frame_offset = SEQ_FRAME_OFFSET.get(seq, 0)
    oxts_dir = find_oxts_dir(raw_folder, data_root)
    
    if oxts_dir is None:
        print(f"  ⚠️  OXTS not found for seq {seq}, skipping...")
        return None
    
    oxts_data = load_oxts_data(str(oxts_dir))
    print(f"OXTS: {len(oxts_data)} frames (offset={frame_offset})")
    
    # Get bounding box
    bbox = get_bounding_box_from_trajectory(trajectory, oxts_data, frame_offset, margin_m=800)
    print(f"Bounding box: lat=[{bbox[0]:.4f}, {bbox[2]:.4f}], lon=[{bbox[1]:.4f}, {bbox[3]:.4f}]")
    
    # Extract OSM polylines
    latlon_polylines = parse_osm_pbf(pbf_file, bbox)
    print(f"Raw polylines: {len(latlon_polylines)}")
    
    # DENSIFY polylines
    print(f"Densifying polylines (spacing={DENSIFY_SPACING}m)...")
    latlon_polylines = densify_polylines(latlon_polylines, DENSIFY_SPACING)
    total_points = sum(len(p) for p in latlon_polylines)
    print(f"  After densification: {len(latlon_polylines)} polylines, {total_points} total points")
    print(f"  Avg points per polyline: {total_points / len(latlon_polylines):.1f}")
    
    # Convert to local frame
    local_polylines, gps_transform = gps_offset_polylines(
        latlon_polylines, oxts_data, trajectory, frame_offset)
    
    # Filter to trajectory area
    local_polylines = filter_to_trajectory_area(local_polylines, trajectory, margin=800)
    print(f"  After filtering: {len(local_polylines)} polylines")
    
    # Optimize rotation
    anchor = SEQ_ANCHOR.get(seq, 'start')
    hint_deg = SEQ_ROTATION_HINT.get(seq, 0.0)
    print(f"\nOptimizing rotation (anchor={anchor}, hint={hint_deg}°)...")
    rotated, best_rot, pivot = optimize_rotation(local_polylines, trajectory, anchor, hint_deg)
    
    # Refilter after rotation
    rotated = filter_to_trajectory_area(rotated, trajectory, margin=300)
    print(f"  After rotation filter: {len(rotated)} polylines")
    
    # Bestfit fine-tuning
    print("\nFine-tuning alignment...")
    final_polylines, bf_transform = bestfit_optimize(rotated, trajectory)
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(final_polylines, f)
    print(f"\nSaved: {out_pkl}")
    
    # Save transform
    transform_file = f'{output_dir}/osm_transform_seq{seq}.pkl'
    with open(transform_file, 'wb') as f:
        pickle.dump({
            'gps': gps_transform,
            'anchor': anchor,
            'pivot': pivot.tolist(),
            'coarse_rot_rad': float(best_rot),
            'coarse_rot_deg': float(np.degrees(best_rot)),
            'bestfit': bf_transform
        }, f)
    print(f"Saved: {transform_file}")
    
    # Visualize
    print("\nGenerating visualization...")
    mean_err, max_err = visualize(seq, final_polylines, trajectory, output_dir)
    
    # Summary
    final_points = sum(len(p) for p in final_polylines)
    print(f"\n✓ Seq {seq} complete:")
    print(f"  Polylines: {len(final_polylines)}")
    print(f"  Total points: {final_points}")
    print(f"  Avg points/polyline: {final_points / len(final_polylines):.1f}")
    print(f"  Mean error: {mean_err:.2f}m")
    print(f"  Max error: {max_err:.2f}m")
    
    return {
        'seq': seq,
        'polylines': len(final_polylines),
        'points': final_points,
        'mean_error': mean_err,
        'max_error': max_err
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Recreate OSM alignments with densified polylines')
    parser.add_argument('--seqs', nargs='+', default=['00', '01', '02', '07', '08', '09', '10'])
    parser.add_argument('--pbf', default='data/osm/germany-latest.osm.pbf')
    parser.add_argument('--output', default='osm_aligned_final')
    parser.add_argument('--spacing', type=float, default=2.0, help='Densification spacing in meters')
    parser.add_argument('--data_root', default='data')
    args = parser.parse_args()
    
    global DENSIFY_SPACING
    DENSIFY_SPACING = args.spacing
    
    print("="*70)
    print("OSM Alignment Recreation with Densified Polylines")
    print("="*70)
    print(f"Source PBF: {args.pbf}")
    print(f"Output dir: {args.output}")
    print(f"Densification spacing: {DENSIFY_SPACING}m")
    print(f"Sequences: {args.seqs}")
    
    results = []
    for seq in args.seqs:
        try:
            result = process_sequence(seq, args.pbf, args.data_root, args.output)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing seq {seq}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Seq':>6} {'Polylines':>12} {'Points':>12} {'Avg/Poly':>10} {'Mean Err':>10}")
    print("-"*60)
    for r in results:
        avg_points = r['points'] / r['polylines'] if r['polylines'] > 0 else 0
        print(f"{r['seq']:>6} {r['polylines']:>12} {r['points']:>12} {avg_points:>10.1f} {r['mean_error']:>10.2f}m")
    print("="*70)
    print("\n✓ Recreation complete!")


if __name__ == '__main__':
    main()
