#!/usr/bin/env python3
"""
Full pipeline: latlon OSM polylines → local vehicle frame → bestfit alignment → visualization.

For sequences 02, 07, 08 using the new high-density Germany OSM extract.
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


SEQ_TO_RAW = {
    '00': '2011_10_03_drive_0027_sync',
    '02': '2011_10_03_drive_0034_sync',
    '05': '2011_09_30_drive_0018_sync',
    '07': '2011_09_30_drive_0027_sync',
    '08': '2011_09_30_drive_0028_sync',
    '09': '2011_09_30_drive_0033_sync',
    '10': '2011_09_30_drive_0034_sync',
}

# KITTI odometry sequences are subsets of raw drives.
# This is the raw-drive frame index that corresponds to odometry pose frame 0.
SEQ_FRAME_OFFSET = {
    '00': 3346,  # drive_0027 - frame 3346 matches heading
    '02': 57,    # drive_0034 - frame 57 matches heading
    '05': 46,    # drive_0018 - frame 46 matches heading
    '07': 42,    # drive_0027 - frame 42 matches heading
    '08': 252,   # drive_0028 - frame 252 matches heading
    '09': 1497,  # drive_0033 - frame 1497 matches heading
    '10': 0,     # drive_0034 - frame 0 (best match, heading diff 34.8°)
}

# Anchor point for rotation search (confirmed visually):
#   'start' → rotate around trajectory[0]  (start is already aligned)
#   'end'   → rotate around trajectory[-1] (end is already aligned)
SEQ_ANCHOR = {
    '00': 'start',  # to be determined
    '02': 'start',
    '05': 'start',  # to be determined
    '07': 'end',
    '08': 'start',
    '09': 'start',  # to be determined
    '10': 'start',  # to be determined
}

# Optional coarse rotation hint (degrees). When set, the grid search is restricted
# to a ±30° window around this value instead of the full ±180°. Use when a
# visual inspection has already identified the approximate correct rotation.
# When set to a tuple (hint, 'force'), the coarse rotation is FIXED to that value.
SEQ_ROTATION_HINT = {
    '00': (93.0, 'force'),   # optimized: +93° with START anchor
    '02': (36.0, 'force'),   # optimized: +36° with START anchor  
    '05': (129.0, 'force'),  # optimized: +129° with START anchor
    '07': (123.0, 'force'),  # optimized: +123° with END anchor
    '08': (84.0, 'force'),   # optimized: +84° with START anchor
    '09': (111.0, 'force'),  # optimized: +111° with START anchor
    '10': (42.0, 'force'),   # optimized: +42° with START anchor
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def extract_trajectory(poses):
    """Extract (x, z) trajectory — KITTI convention."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def find_oxts_dir(raw_folder, data_root='data'):
    root = Path(data_root) / 'raw_data'
    for date in ['2011_10_03', '2011_09_30', '2011_09_29', '2011_09_28', '2011_09_26']:
        candidate = root / raw_folder / date / raw_folder / 'oxts' / 'data'
        if candidate.exists():
            return candidate
    candidate = root / raw_folder / 'oxts' / 'data'
    if candidate.exists():
        return candidate
    return None


def compute_heading(pts, n=50):
    """Compute heading from first n points."""
    pts = pts[:n]
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    return np.arctan2(dy, dx)


# ── Step 1: GPS-based offset-only alignment ───────────────────────────────────

def gps_offset_polylines(latlon_polylines, oxts_data, trajectory, frame_offset=0):
    """
    Convert lat/lon polylines to local KITTI frame using only the UTM translation.
    No heading rotation here — that is handled by the coarse grid search below.

    frame_offset: index into oxts_data that corresponds to pose frame 0.
    """
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)

    offset_east  = east0  - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]

    print(f"  OXTS ref frame: {ref_frame}  lat={lat0:.6f} lon={lon0:.6f}")
    print(f"  UTM offset: ({offset_east:.1f}, {offset_north:.1f})")

    raw_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        raw_polylines.append(pts)

    return raw_polylines, {'offset_east': offset_east, 'offset_north': offset_north}


# ── Step 2: Coarse rotation grid search ───────────────────────────────────────

def coarse_rotation_search(local_polylines, trajectory, anchor='start', n_angles=720,
                           hint_deg=None, force_angle=None):
    """
    Grid search over -180° to +180° to find the rotation that minimises
    mean trajectory-to-road distance.

    anchor: 'start' → rotate around trajectory[0]  (start already aligned)
            'end'   → rotate around trajectory[-1]  (end already aligned)
    hint_deg: hint for the rotation angle (±30° search window)
    force_angle: if True, use hint_deg exactly without grid search
    """
    if anchor == 'start':
        pivot = trajectory[0].copy()
        print(f"  Anchor: START point {pivot}")
    else:
        pivot = trajectory[-1].copy()
        print(f"  Anchor: END point {pivot}")

    all_pts = np.vstack(local_polylines)
    all_pts_c = all_pts - pivot  # centred on pivot

    if force_angle and hint_deg is not None:
        # Skip grid search, use the hint directly
        best_angle = np.radians(hint_deg)
        c, s = np.cos(best_angle), np.sin(best_angle)
        xr = all_pts_c[:, 0] * c - all_pts_c[:, 1] * s + pivot[0]
        yr = all_pts_c[:, 0] * s + all_pts_c[:, 1] * c + pivot[1]
        tree = cKDTree(np.column_stack([xr, yr]))
        d, _ = tree.query(trajectory, k=1)
        best_err = d.mean()
        print(f"  Using FORCED rotation: {hint_deg:.1f}°  mean error: {best_err:.2f} m")
        
        c, s = np.cos(best_angle), np.sin(best_angle)
        rotated = []
        for pl in local_polylines:
            xc, yc = pl[:, 0] - pivot[0], pl[:, 1] - pivot[1]
            xr = xc * c - yc * s + pivot[0]
            yr = xc * s + yc * c + pivot[1]
            rotated.append(np.column_stack([xr, yr]))
        return rotated, best_angle, best_err, pivot

    if hint_deg is not None:
        # Narrow search: ±30° around the user-provided hint
        centre = np.radians(hint_deg)
        angles = np.linspace(centre - np.radians(30), centre + np.radians(30), n_angles)
        print(f"  Using rotation hint {hint_deg:.1f}° → searching {hint_deg-30:.1f}° to {hint_deg+30:.1f}°")
    else:
        angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    best_err, best_angle = np.inf, 0.0

    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        xr = all_pts_c[:, 0] * c - all_pts_c[:, 1] * s + pivot[0]
        yr = all_pts_c[:, 0] * s + all_pts_c[:, 1] * c + pivot[1]
        tree = cKDTree(np.column_stack([xr, yr]))
        d, _ = tree.query(trajectory, k=1)
        err = d.mean()
        if err < best_err:
            best_err = err
            best_angle = angle

    # Fine search ±5° around best with 0.05° steps
    fine_angles = np.arange(best_angle - np.radians(5), best_angle + np.radians(5),
                             np.radians(0.05))
    for angle in fine_angles:
        c, s = np.cos(angle), np.sin(angle)
        xr = all_pts_c[:, 0] * c - all_pts_c[:, 1] * s + pivot[0]
        yr = all_pts_c[:, 0] * s + all_pts_c[:, 1] * c + pivot[1]
        tree = cKDTree(np.column_stack([xr, yr]))
        d, _ = tree.query(trajectory, k=1)
        err = d.mean()
        if err < best_err:
            best_err = err
            best_angle = angle

    print(f"  Best rotation: {np.degrees(best_angle):.2f}°  mean error: {best_err:.2f} m")

    c, s = np.cos(best_angle), np.sin(best_angle)
    rotated = []
    for pl in local_polylines:
        xc, yc = pl[:, 0] - pivot[0], pl[:, 1] - pivot[1]
        xr = xc * c - yc * s + pivot[0]
        yr = xc * s + yc * c + pivot[1]
        rotated.append(np.column_stack([xr, yr]))

    return rotated, best_angle, best_err, pivot


def filter_to_trajectory_area(polylines, trajectory, margin=200):
    """Keep polylines with at least one point within margin m of the trajectory bounding box."""
    xmin, xmax = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
    ymin, ymax = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin

    filtered = []
    for pl in polylines:
        if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                  (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
            filtered.append(pl)
    return filtered


# ── Step 3: Bestfit fine-tuning (translation + small rotation, scale=1) ───────

def bestfit_optimize(local_polylines, trajectory, spacing=5.0):
    """
    Fine-tune alignment: optimise translation + small rotation only (scale fixed at 1).
    Called after the coarse rotation grid search.
    """
    diffs  = np.diff(trajectory, axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    cum    = np.concatenate([[0], np.cumsum(dists)])
    total  = cum[-1]
    n_samp = max(int(total / spacing) + 1, 2)
    sample_dists = np.linspace(0, total, n_samp)
    indices = sorted(set(int(np.argmin(np.abs(cum - sd))) for sd in sample_dists))
    sample_pts = trajectory[indices]

    all_osm = np.vstack(local_polylines)
    traj_center = trajectory.mean(axis=0)

    def cost(params):
        rot, tx, ty = params
        c, s = np.cos(rot), np.sin(rot)
        # rotate around trajectory centre
        xc = all_osm[:, 0] - traj_center[0]
        yc = all_osm[:, 1] - traj_center[1]
        xr = xc * c - yc * s + traj_center[0] + tx
        yr = xc * s + yc * c + traj_center[1] + ty
        tree = cKDTree(np.column_stack([xr, yr]))
        d, _ = tree.query(sample_pts, k=1)
        return d.sum() + d.max() * 3.0

    x0 = [0.0, 0.0, 0.0]
    bounds = [(np.radians(-30), np.radians(30)), (-200, 200), (-200, 200)]

    print(f"  Fine-tuning over {len(sample_pts)} sample points …")
    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 500, 'disp': False})
    rot, tx, ty = res.x
    print(f"  Fine rot={np.degrees(rot):.3f}°  T=({tx:.2f},{ty:.2f})")

    c, s = np.cos(rot), np.sin(rot)
    final_polylines = []
    for pl in local_polylines:
        xc = pl[:, 0] - traj_center[0]
        yc = pl[:, 1] - traj_center[1]
        xr = xc * c - yc * s + traj_center[0] + tx
        yr = xc * s + yc * c + traj_center[1] + ty
        final_polylines.append(np.column_stack([xr, yr]))

    return final_polylines, {'rotation': rot, 'translation': [tx, ty]}


# ── Step 3: Visualisation ─────────────────────────────────────────────────────

def visualize(seq, polylines, trajectory):
    all_osm = np.vstack(polylines)
    tree = cKDTree(all_osm)
    dists, _ = tree.query(trajectory, k=1)
    dist_start, _ = tree.query(trajectory[0])
    dist_end,   _ = tree.query(trajectory[-1])

    fig = plt.figure(figsize=(18, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ── Main overlay ──
    ax1 = fig.add_subplot(gs[:, :2])
    for i, pl in enumerate(polylines):
        ax1.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.6, alpha=0.35,
                 label='OSM' if i == 0 else '')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2, label='Trajectory')
    ax1.scatter(trajectory[0, 0],  trajectory[0, 1],  c='green', s=300, marker='o',  zorder=5, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red',   s=400, marker='*',  zorder=5, label='End')
    ax1.set_title(
        f'Seq {seq} – Best-Fit OSM Alignment (Germany regional extract)\n'
        f'Mean: {np.mean(dists):.2f} m  |  Max: {np.max(dists):.2f} m  |  '
        f'Total: {np.sum(dists):.0f} m  |  {len(polylines):,} polylines',
        fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # ── Start zoom ──
    ax2 = fig.add_subplot(gs[0, 2])
    for pl in polylines:
        ax2.plot(pl[:, 0], pl[:, 1], 'b-', lw=1, alpha=0.55)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=350, marker='o', zorder=5)
    ax2.set_xlim(trajectory[0, 0] - 80, trajectory[0, 0] + 80)
    ax2.set_ylim(trajectory[0, 1] - 80, trajectory[0, 1] + 80)
    ax2.set_title(f'Start (error: {dist_start:.1f} m)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # ── End zoom ──
    ax3 = fig.add_subplot(gs[1, 2])
    for pl in polylines:
        ax3.plot(pl[:, 0], pl[:, 1], 'b-', lw=1, alpha=0.55)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax3.set_xlim(trajectory[-1, 0] - 80, trajectory[-1, 0] + 80)
    ax3.set_ylim(trajectory[-1, 1] - 80, trajectory[-1, 1] + 80)
    ax3.set_title(f'End (error: {dist_end:.1f} m)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    out = f'osm_pbf_aligned_seq{seq}_bestfit.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")
    return np.mean(dists), np.max(dists)


# ── Main ─────────────────────────────────────────────────────────────────────

def process(seq, data_root='data'):
    print(f"\n{'='*65}")
    print(f"Seq {seq}: Germany OSM best-fit alignment")
    print(f"{'='*65}")

    # Load lat/lon polylines (new high-density data)
    latlon_pkl = f'osm_polylines_latlon_seq{seq}_regbez.pkl'
    with open(latlon_pkl, 'rb') as f:
        latlon_polylines = pickle.load(f)
    latlon_polylines = [np.array(pl) for pl in latlon_polylines]
    print(f"Loaded {len(latlon_polylines)} polylines from {latlon_pkl}")

    # Load trajectory
    trajectory = extract_trajectory(load_poses(f'{data_root}/kitti/poses/{seq}.txt'))
    print(f"Trajectory: {len(trajectory)} frames")

    # Load OXTS GPS
    raw_folder   = SEQ_TO_RAW[seq]
    frame_offset = SEQ_FRAME_OFFSET.get(seq, 0)
    oxts_dir     = find_oxts_dir(raw_folder, data_root)
    if oxts_dir is None:
        raise FileNotFoundError(f"OXTS not found for seq {seq}")
    oxts_data = load_oxts_data(str(oxts_dir))
    print(f"OXTS: {len(oxts_data)} frames from {oxts_dir}  (frame_offset={frame_offset})")

    # Step 1: UTM offset → local frame (no rotation yet)
    print("\nStep 1: GPS offset alignment")
    raw_local, gps_transform = gps_offset_polylines(
        latlon_polylines, oxts_data, trajectory, frame_offset)

    # Filter to trajectory bounding box + margin (before rotation, use large margin)
    before = len(raw_local)
    raw_local = filter_to_trajectory_area(raw_local, trajectory, margin=800)
    print(f"  Filtered {before} → {len(raw_local)} polylines (800 m margin)")

    # Step 2: Coarse rotation grid search anchored at the aligned point
    anchor = SEQ_ANCHOR.get(seq, 'start')
    hint_cfg = SEQ_ROTATION_HINT.get(seq, None)
    hint_deg, force_angle = None, False
    if isinstance(hint_cfg, tuple):
        hint_deg, mode = hint_cfg
        force_angle = (mode == 'force')
    elif hint_cfg is not None:
        hint_deg = hint_cfg
    print(f"\nStep 2: Coarse rotation grid search (anchor={anchor})")
    rotated, best_rot, rot_err, pivot = coarse_rotation_search(
        raw_local, trajectory, anchor=anchor, hint_deg=hint_deg, force_angle=force_angle)

    # Re-filter after rotation (tighter margin)
    rotated = filter_to_trajectory_area(rotated, trajectory, margin=300)
    print(f"  After rotation filter: {len(rotated)} polylines")

    # Step 3: Bestfit fine-tuning
    print("\nStep 3: Bestfit fine-tuning (translation + ±15° rot, scale=1)")
    final_polylines, bf_transform = bestfit_optimize(rotated, trajectory)

    # Save polylines
    out_pkl = f'osm_polylines_aligned_seq{seq}_bestfit.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(final_polylines, f)
    print(f"  Saved: {out_pkl}")

    # Save combined transform
    with open(f'osm_transform_seq{seq}_bestfit_new.pkl', 'wb') as f:
        pickle.dump({'gps': gps_transform, 'anchor': anchor,
                     'pivot': pivot.tolist(), 'coarse_rot_deg': np.degrees(best_rot),
                     'bestfit': bf_transform}, f)

    # Step 4: Visualise
    print("\nStep 4: Visualising")
    mean_err, max_err = visualize(seq, final_polylines, trajectory)
    print(f"  Final: mean={mean_err:.2f} m  max={max_err:.2f} m")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['02', '07', '08'])
    parser.add_argument('--data_root', default='data')
    args = parser.parse_args()

    for seq in args.seqs:
        process(seq, args.data_root)

    print('\nAll done.')
