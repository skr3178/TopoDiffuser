#!/usr/bin/env python3
"""
Precompute Paper-Split Dataset for TopoDiffuser.

Produces 5-channel BEV maps (float16, [5, 300, 400]) and future trajectories
for the exact train/test split cited in the paper:

  Train: seqs 00, 02, 05, 07  →  3,860 samples
  Test:  seq  08              →  1,391 samples
         seq  09              →    530 samples
         seq  10              →    349 samples

Channels:
  0-2  LiDAR  (height, intensity, density)
  3    OSM topometric map (pre-aligned polylines, ego-frame binary raster)

BEV grid: 300×400 px @ 0.1 m/px → x∈[-20,20] m, y∈[-10,30] m

Usage:
  conda run -n nuscenes python precompute_paper_split.py
  conda run -n nuscenes python precompute_paper_split.py --dry_run
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
POSES_DIR    = Path('/media/skr/storage/self_driving/CoPilot4D/data/kitti/'
                    'kitti-devkit-odom/ground_truth/poses')
LIDAR_DIR    = Path('data/kitti/sequences')
CACHE_3CH    = Path('data/kitti/bev_cache')          # may be empty for some seqs
OSM_DIR      = Path('osm_aligned_final')
OUT_DIR      = Path('data/paper_split')

# ---------------------------------------------------------------------------
# Calibration: Velodyne → Camera  (4×4, row-major)
# ---------------------------------------------------------------------------
TR_03 = np.array([
     4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
    -7.210626507497e-03,  8.081198471645e-03, -9.999413164504e-01, -5.403422615991e-02,
     9.999738645903e-01,  4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
     0, 0, 0, 1], dtype=np.float32).reshape(4, 4)

TR_30 = np.array([
    -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
    -6.481465826011e-03,  8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
     9.999773098563e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
     0, 0, 0, 1], dtype=np.float32).reshape(4, 4)

SEQ_TR = {s: TR_03 for s in ('00','01','02','03','04')}
SEQ_TR.update({s: TR_30 for s in ('05','06','07','08','09','10')})

# ---------------------------------------------------------------------------
# Paper split
# ---------------------------------------------------------------------------
TRAIN_SEQS  = ['00', '02', '05', '07']
TEST_SEQS   = ['08', '09', '10']
TEST_TARGET = {'08': 1391, '09': 530, '10': 349}
TRAIN_TOTAL = 3860

# ---------------------------------------------------------------------------
# BEV parameters
# ---------------------------------------------------------------------------
H, W        = 300, 400
RES         = 0.1            # m / pixel
X_RANGE     = (-20.0, 20.0)  # lateral
Y_RANGE     = (-10.0, 30.0)  # forward

# Trajectory / history
NUM_FUTURE      = 8
WAYPOINT_SPACING = 2.0   # m
PAST_FRAMES     = 50

OSM_MAX_DIST    = 60.0   # m  — include OSM polylines whose centre is within this radius


# ===========================================================================
# Helper: draw thick line on 2-D canvas (Bresenham + dilation)
# ===========================================================================
def draw_line(canvas, p1, p2, width=2):
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    dx, dy  = abs(x2 - x1), abs(y2 - y1)
    sx      = 1 if x1 < x2 else -1
    sy      = 1 if y1 < y2 else -1
    err     = dx - dy
    pts, cx, cy = [], x1, y1
    for _ in range(max(dx, dy) + 2):
        pts.append((cx, cy))
        if cx == x2 and cy == y2:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; cx += sx
        if e2 <  dx: err += dx; cy += sy
    r = width // 2
    for (px, py) in pts:
        for ddy in range(-r, r + 1):
            for ddx in range(-r, r + 1):
                nx, ny = px + ddx, py + ddy
                if 0 <= nx < W and 0 <= ny < H:
                    canvas[ny, nx] = 1.0
    return canvas


def to_px(x, y):
    """World-ego (x=lateral, y=forward) → pixel (col, row)."""
    return (x - X_RANGE[0]) / RES, (y - Y_RANGE[0]) / RES


# ===========================================================================
# Channel 0-2: LiDAR BEV
# ===========================================================================
def lidar_bev_from_cache(seq, frame_idx):
    p = CACHE_3CH / seq / f'{frame_idx:06d}.npy'
    if p.exists():
        return np.load(str(p))
    return None


def lidar_bev_from_raw(seq, frame_idx, Tr4):
    bin_path = LIDAR_DIR / seq / 'velodyne' / f'{frame_idx:06d}.bin'
    if not bin_path.exists():
        return np.zeros((3, H, W), dtype=np.float32)

    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    # No Velodyne z-filter — matches 3ch BEVRasterizer which used only XY
    # bounds and saw ~100% of scan points.  Filtering height here caused the
    # 5ch model to underperform by excluding the full scene geometry.

    pts_h = np.hstack([pts[:, :3], np.ones((len(pts), 1), dtype=np.float32)])
    pts_c = (Tr4 @ pts_h.T).T[:, :3]      # camera frame

    # ego-BEV: x=cam_x (lateral), y=cam_z (forward), height=-cam_y (up)
    bx  = pts_c[:, 0]
    by  = pts_c[:, 2]
    bz  = -pts_c[:, 1]
    rfl = pts[:, 3]

    mask = ((bx >= X_RANGE[0]) & (bx < X_RANGE[1]) &
            (by >= Y_RANGE[0]) & (by < Y_RANGE[1]))
    if mask.sum() == 0:
        return np.zeros((3, H, W), dtype=np.float32)

    bx, by, bz, rfl = bx[mask], by[mask], bz[mask], rfl[mask]
    px = np.clip(((bx - X_RANGE[0]) / RES).astype(np.int32), 0, W - 1)
    py = np.clip(((by - Y_RANGE[0]) / RES).astype(np.int32), 0, H - 1)

    bev = np.zeros((3, H, W), dtype=np.float32)
    cnt = np.zeros((H, W),    dtype=np.float32)

    # ch0 Height: init to -inf so negative heights (ground ≈ -1.73m) are
    # captured correctly by maximum.at; empty cells reset to 0 then
    # normalised identically to 3ch BEVRasterizer → empty = 0.429.
    bev[0, :, :] = -np.inf
    np.maximum.at(bev[0], (py, px), bz)
    np.add.at(bev[1],     (py, px), rfl)
    np.add.at(cnt,        (py, px), 1.0)

    # Reset truly-empty cells to 0 before normalisation (mirrors 3ch).
    bev[0][bev[0] == -np.inf] = 0.0

    # ch0: normalise with z_range=(-3, 4) — same as 3ch BEVRasterizer.
    # Applied to ALL cells (including empty-reset-to-0), so empty cells
    # become (0-(-3))/(4-(-3)) = 0.429, identical to 3ch behaviour.
    Z_MIN, Z_MAX = -3.0, 4.0
    bev[0] = np.clip((bev[0] - Z_MIN) / (Z_MAX - Z_MIN), 0.0, 1.0)

    # ch1: mean intensity per cell.
    # KITTI reflectance is already float32 in [0, 1] — do NOT divide by 255.
    # The 3ch BEVRasterizer divided by 255 (max_intensity=255 config) which
    # was a bug carried over from uint8 datasets; 5ch fixes it here.
    pos = cnt > 0
    bev[1][pos] /= cnt[pos]
    bev[1]       = np.clip(bev[1], 0.0, 1.0)

    # ch2: log-normalised density → [0,1].
    bev[2] = np.clip(np.log1p(cnt) / np.log1p(128), 0.0, 1.0)

    return bev


def load_lidar_bev(seq, frame_idx, Tr4):
    # Always use raw LiDAR — the 3ch BEV cache has a height normalisation bug
    # where empty cells get value 0.43 instead of 0, corrupting ch0 for all
    # sequences that had a cache hit (primarily seq00 and seq02).
    return lidar_bev_from_raw(seq, frame_idx, Tr4)


# ===========================================================================
# Channel 3: History BEV
# ===========================================================================
def compute_history_bev(poses, frame_idx):
    bev = np.zeros((H, W), dtype=np.float32)
    pm  = poses[frame_idx].reshape(3, 4)
    R, t = pm[:, :3], pm[:, 3]

    start      = max(0, frame_idx - PAST_FRAMES)
    past_poses = poses[start: frame_idx + 1]
    if len(past_poses) < 2:
        return bev[np.newaxis]

    pxs = []
    for p in past_poses:
        pm2  = p.reshape(3, 4)
        pe   = R.T @ (pm2[:, 3] - t)
        col, row = to_px(pe[0], pe[2])
        if 0 <= col < W and 0 <= row < H:
            pxs.append((col, row))

    for i in range(len(pxs) - 1):
        bev = draw_line(bev, pxs[i], pxs[i + 1], width=2)

    return bev[np.newaxis]


# ===========================================================================
# Channel 4: OSM BEV
# ===========================================================================
def compute_osm_bev(osm_polylines, poses, frame_idx):
    bev = np.zeros((H, W), dtype=np.float32)
    if not osm_polylines:
        return bev[np.newaxis]

    pm = poses[frame_idx].reshape(3, 4)
    R, t = pm[:, :3], pm[:, 3]
    cur_world = np.array([pm[0, 3], pm[2, 3]])   # (tx, tz) world

    for poly in osm_polylines:
        if len(poly) < 2:
            continue
        # poly is [N, 2] in KITTI world frame (tx, tz)
        centre = poly.mean(axis=0)
        if np.linalg.norm(centre - cur_world) > OSM_MAX_DIST:
            continue

        pxs = []
        for wp in poly:
            pt3d = np.array([wp[0], 0.0, wp[1]])
            pe   = R.T @ (pt3d - t)
            col, row = to_px(pe[0], pe[2])
            if 0 <= col < W and 0 <= row < H:
                pxs.append((col, row))

        for i in range(len(pxs) - 1):
            bev = draw_line(bev, pxs[i], pxs[i + 1], width=3)

    return bev[np.newaxis]


# ===========================================================================
# Road mask  [1, 37, 50]  — rasterised future trajectory
# ===========================================================================
MASK_H, MASK_W = 37, 50

def compute_road_mask(trajectory: np.ndarray) -> np.ndarray:
    """
    Rasterise 8-waypoint future trajectory into a [1, 37, 50] binary mask.
    Uses Bresenham line + 2-pixel dilation as road-width proxy.

    Args:
        trajectory: [8, 2] array of (lateral, forward) waypoints in ego frame

    Returns:
        mask: float16 [1, 37, 50]
    """
    from scipy.ndimage import binary_dilation

    res_x = (X_RANGE[1] - X_RANGE[0]) / MASK_W
    res_y = (Y_RANGE[1] - Y_RANGE[0]) / MASK_H
    mask  = np.zeros((MASK_H, MASK_W), dtype=np.float32)

    def w2p(pt):
        px = int((pt[0] - X_RANGE[0]) / res_x)
        py = int((pt[1] - Y_RANGE[0]) / res_y)
        return (np.clip(px, 0, MASK_W - 1), np.clip(py, 0, MASK_H - 1))

    pts = [w2p(trajectory[i]) for i in range(len(trajectory))]
    for i in range(len(pts) - 1):
        x,  y  = pts[i]
        x2, y2 = pts[i + 1]
        dx, dy = abs(x2 - x), abs(y2 - y)
        sx = 1 if x < x2 else -1
        sy = 1 if y < y2 else -1
        err = dx - dy
        while True:
            mask[y, x] = 1.0
            if (x, y) == (x2, y2):
                break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 <  dx: err += dx; y += sy

    mask = binary_dilation(mask > 0.5, iterations=2).astype(np.float16)
    return mask[np.newaxis]   # [1, 37, 50]


# ===========================================================================
# Ground-truth future trajectory
# ===========================================================================
def compute_trajectory(poses, frame_idx):
    pm   = poses[frame_idx].reshape(3, 4)
    R, t = pm[:, :3], pm[:, 3]
    traj, prev, acc = [], pm[:, 3].copy(), 0.0

    for i in range(1, len(poses) - frame_idx):
        pm2  = poses[frame_idx + i].reshape(3, 4)
        pw   = pm2[:, 3]
        acc += np.linalg.norm(pw - prev)
        prev = pw
        if acc >= WAYPOINT_SPACING:
            pe = R.T @ (pw - t)
            traj.append([pe[0], pe[2]])    # (lateral, forward)
            acc = 0.0
            if len(traj) >= NUM_FUTURE:
                break

    while len(traj) < NUM_FUTURE:
        traj.append(traj[-1] if traj else [0.0, 0.0])

    return np.array(traj[:NUM_FUTURE], dtype=np.float32)


# ===========================================================================
# Main processing
# ===========================================================================
def load_osm(seq):
    p = OSM_DIR / f'osm_polylines_aligned_seq{seq}.pkl'
    if not p.exists():
        warnings.warn(f'No OSM file for seq {seq}: {p}')
        return []
    with open(p, 'rb') as f:
        polys = pickle.load(f)
    print(f'    OSM: {len(polys)} polylines')
    return polys


def select_frames(n_usable, target):
    """Return exactly `target` frame indices spanning [0, n_usable-1]."""
    return np.round(np.linspace(0, n_usable - 1, target)).astype(int)


def process_split(name, seq_targets, out_dir, dry_run=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for seq, target in seq_targets.items():
        print(f'\n  ── seq {seq}  ({target} samples) ──')
        poses    = np.loadtxt(str(POSES_DIR / f'{seq}.txt'))
        n_usable = len(poses) - NUM_FUTURE - 1
        idxs     = select_frames(n_usable, target)
        Tr4      = SEQ_TR[seq]
        osm      = load_osm(seq)

        seq_dir = out_dir / seq
        seq_dir.mkdir(exist_ok=True)

        cache_hits = 0
        for frame_idx in tqdm(idxs, desc=f'    seq{seq}'):
            frame_idx = int(frame_idx)
            npy_path  = seq_dir / f'{frame_idx:06d}.npy'

            if not dry_run:
                lidar   = load_lidar_bev(seq, frame_idx, Tr4)
                osm_bev = compute_osm_bev(osm, poses, frame_idx)
                bev4    = np.concatenate([lidar, osm_bev], axis=0)
                np.save(str(npy_path), bev4.astype(np.float16))

                cache_p = CACHE_3CH / seq / f'{frame_idx:06d}.npy'
                if cache_p.exists():
                    cache_hits += 1

            traj      = compute_trajectory(poses, frame_idx)
            road_mask = compute_road_mask(traj)
            all_meta.append({
                'seq':        seq,
                'frame_idx':  frame_idx,
                'npy_path':   str(npy_path),
                'trajectory': traj,
                'road_mask':  road_mask,   # float16 [1, 37, 50]
            })

        if not dry_run:
            pct = 100 * cache_hits / len(idxs)
            print(f'    LiDAR cache coverage: {cache_hits}/{len(idxs)} ({pct:.0f}%)')

    meta_path = out_dir.parent / f'{name}_meta.pkl'
    with open(str(meta_path), 'wb') as f:
        pickle.dump(all_meta, f)
    print(f'\n  → Saved {len(all_meta)} samples  ({meta_path})')
    return all_meta


def compute_train_targets():
    """Proportional split: scale each sequence to hit TRAIN_TOTAL exactly."""
    totals = {}
    for seq in TRAIN_SEQS:
        poses = np.loadtxt(str(POSES_DIR / f'{seq}.txt'))
        totals[seq] = len(poses) - NUM_FUTURE - 1

    grand = sum(totals.values())
    counts = {seq: int(n * TRAIN_TOTAL / grand) for seq, n in totals.items()}

    # Distribute remaining samples (due to floor) to largest remainders
    remainder = TRAIN_TOTAL - sum(counts.values())
    fractional = sorted(
        TRAIN_SEQS,
        key=lambda s: (totals[s] * TRAIN_TOTAL / grand) - counts[s],
        reverse=True
    )
    for i in range(remainder):
        counts[fractional[i]] += 1

    print('Train targets (proportional):')
    for seq, n in counts.items():
        print(f'  seq {seq}: {totals[seq]} usable → {n} samples '
              f'(stride {totals[seq]/n:.2f})')
    print(f'  Total: {sum(counts.values())}')
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                        help='Only compute meta/trajectories, skip BEV')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--test_only',  action='store_true')
    args = parser.parse_args()

    print('=' * 65)
    print('  TopoDiffuser – Precompute Paper-Split Dataset')
    print('=' * 65)
    if args.dry_run:
        print('  [DRY RUN – BEV computation skipped]')

    train_targets = compute_train_targets()

    # ── Training split ──────────────────────────────────────────────────────
    if not args.test_only:
        print('\n── TRAINING ──────────────────────────────────────────────────')
        train_meta = process_split(
            'train', train_targets,
            OUT_DIR / 'train',
            dry_run=args.dry_run
        )

    # ── Test split ───────────────────────────────────────────────────────────
    if not args.train_only:
        print('\n── TEST ───────────────────────────────────────────────────────')
        test_meta = process_split(
            'test', TEST_TARGET,
            OUT_DIR / 'test',
            dry_run=args.dry_run
        )

    # ── Summary ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('  Done!')
    if not args.test_only:
        print(f'  Train samples: {len(train_meta)}')
    if not args.train_only:
        print(f'  Test  samples: {len(test_meta)}')
        for seq in TEST_SEQS:
            n = sum(1 for m in test_meta if m["seq"] == seq)
            print(f'    seq {seq}: {n}')
    print(f'  Output: {OUT_DIR}/')
    print('=' * 65)


if __name__ == '__main__':
    main()
