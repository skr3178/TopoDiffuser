#!/usr/bin/env python3
"""
Three-modality visualisation for all KITTI sequences with OSM aligned maps.
Produces paper_three_modalities_seq{XX}.png for each sequence.
"""

import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE        = '/media/skr/storage/self_driving/TopoDiffuser'
KITTI_BASE  = f'{BASE}/data/kitti'
OSM_DIR     = f'{BASE}/osm_aligned_final'
OUT_DIR     = BASE

SEQS = ['01', '02', '05', '07', '08', '09', '10']

# Two Tr calibrations (velodyne → camera0)
# seq 00,01,02 → date 2011_10_03
TR_03 = [
     4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
    -7.210626507497e-03,  8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
     9.999738645903e-01,  4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
]
# seq 05,07,08,09,10 → date 2011_09_30
TR_30 = [
    -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
    -6.481465826011e-03,  8.051860151134e-03, -9.999467081081e-01, -7.337429464231e-02,
     9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
]

SEQ_TR = {
    '01': TR_03, '02': TR_03,
    '05': TR_30, '07': TR_30, '08': TR_30, '09': TR_30, '10': TR_30,
}

LIDAR_CMAP = LinearSegmentedColormap.from_list(
    'lidar', ['white', '#1a1a6e', '#0e6b9e', '#f0a500', '#ff4500'], N=256)


# ── helpers ───────────────────────────────────────────────────────────────────
def make_tr4(tr_vals):
    T = np.eye(4)
    T[:3] = np.array(tr_vals).reshape(3, 4)
    return T


def load_poses(seq):
    raw = np.loadtxt(f'{KITTI_BASE}/poses/{seq}.txt')
    N   = len(raw)
    P   = np.tile(np.eye(4), (N, 1, 1))
    P[:, :3, :] = raw.reshape(N, 3, 4)
    tx = raw[:, 3]
    tz = raw[:, 11]
    return P, tx, tz


def build_lidar_density(seq, poses, Tr4, step=5, z_min=-1.5, z_max=0.5):
    velo_dir = f'{KITTI_BASE}/sequences/{seq}/velodyne'
    N = len(poses)
    wx_all, wz_all = [], []
    for idx in range(0, N, step):
        fpath = os.path.join(velo_dir, f'{idx:06d}.bin')
        if not os.path.exists(fpath):
            continue
        pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 4)
        mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
        pts  = pts[mask, :3].astype(np.float64)
        if len(pts) == 0:
            continue
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        T  = poses[idx] @ Tr4
        w  = (T @ pts_h.T).T
        wx_all.append(w[:, 0])
        wz_all.append(w[:, 2])
    return np.concatenate(wx_all), np.concatenate(wz_all)


def make_histogram(wx, wz, x_lo, x_hi, y_lo, y_hi, bins_x=800):
    bins_y = max(int(bins_x * (y_hi - y_lo) / (x_hi - x_lo)), 2)
    H, _, _ = np.histogram2d(wx, wz, bins=[bins_x, bins_y],
                              range=[[x_lo, x_hi], [y_lo, y_hi]])
    H = H.T
    H_log = np.log1p(H)
    pos = H_log[H_log > 0]
    if len(pos):
        H_log = np.clip(H_log, 0, np.percentile(pos, 98))
    return H_log


def render(seq, tx, tz, wx, wz, osm_lines, out_path):
    pad  = 40
    x_lo = min(tx.min(), wx.min()) - pad
    x_hi = max(tx.max(), wx.max()) + pad
    y_lo = min(tz.min(), wz.min()) - pad
    y_hi = max(tz.max(), wz.max()) + pad

    H_log = make_histogram(wx, wz, x_lo, x_hi, y_lo, y_hi)

    data_w  = x_hi - x_lo
    data_h  = y_hi - y_lo
    panel_w = 5.5
    fig_h   = panel_w * data_h / data_w + 1.4
    fig, axes = plt.subplots(1, 3,
                             figsize=(panel_w * 3 + 0.6, fig_h),
                             facecolor='white',
                             gridspec_kw={'wspace': 0.08})

    lw_traj, lw_osm = 2.0, 0.8
    for ax in axes:
        ax.set_xlim(x_lo, x_hi);  ax.set_ylim(y_lo, y_hi)
        ax.tick_params(labelsize=8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5);  sp.set_color('#cccccc')
        ax.grid(True, linewidth=0.3, color='#e0e0e0', zorder=0)

    # (a) LiDAR density
    axes[0].set_facecolor('white')
    axes[0].imshow(H_log, origin='lower',
                   extent=[x_lo, x_hi, y_lo, y_hi],
                   cmap=LIDAR_CMAP, interpolation='bilinear',
                   aspect='auto', vmin=0, zorder=1)
    axes[0].set_title('LiDAR Point Density', fontsize=13, fontweight='bold', pad=8)
    axes[0].set_xlabel('X (m)', fontsize=9);  axes[0].set_ylabel('Y (m)', fontsize=9)
    axes[0].text(0.02, 0.98, '(a)', transform=axes[0].transAxes,
                 fontsize=11, fontweight='bold', va='top', color='#333333')

    # (b) History trajectory
    axes[1].set_facecolor('white')
    axes[1].plot(tx, tz, color='#1a6e1a', linewidth=lw_traj,
                 zorder=2, solid_capstyle='round')
    axes[1].scatter(tx[0],  tz[0],  color='#2ecc40', s=100, zorder=5,
                    marker='o', linewidths=1.0, edgecolors='#1a6e1a', label='Start')
    axes[1].scatter(tx[-1], tz[-1], color='#e74c3c', s=200, zorder=5,
                    marker='*', linewidths=0.8, edgecolors='#8b0000', label='End')
    axes[1].set_title('History Trajectory', fontsize=13, fontweight='bold', pad=8)
    axes[1].set_xlabel('X (m)', fontsize=9)
    axes[1].legend(loc='lower left', fontsize=8, framealpha=0.88,
                   handlelength=1.2, borderpad=0.5)
    axes[1].text(0.02, 0.98, '(b)', transform=axes[1].transAxes,
                 fontsize=11, fontweight='bold', va='top', color='#333333')

    # (c) Topometric map
    axes[2].set_facecolor('white')
    for poly in osm_lines:
        if len(poly) >= 2:
            axes[2].plot(poly[:, 0], poly[:, 1],
                         color='#4a6fa5', linewidth=lw_osm,
                         alpha=0.75, solid_capstyle='round', zorder=2)
    axes[2].plot(tx, tz, color='#1a6e1a', linewidth=lw_traj - 0.3,
                 alpha=0.75, zorder=3, solid_capstyle='round')
    axes[2].scatter(tx[0],  tz[0],  color='#2ecc40', s=100, zorder=5,
                    marker='o', linewidths=1.0, edgecolors='#1a6e1a')
    axes[2].scatter(tx[-1], tz[-1], color='#e74c3c', s=200, zorder=5,
                    marker='*', linewidths=0.8, edgecolors='#8b0000')
    axes[2].set_title('Topometric Map (OSM)', fontsize=13, fontweight='bold', pad=8)
    axes[2].set_xlabel('X (m)', fontsize=9)
    axes[2].text(0.02, 0.98, '(c)', transform=axes[2].transAxes,
                 fontsize=11, fontweight='bold', va='top', color='#333333')

    fig.suptitle(f'KITTI Odometry Seq {seq} – Three Input Modalities',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08, wspace=0.08)
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── main loop ─────────────────────────────────────────────────────────────────
for seq in SEQS:
    print(f'\n{"="*60}')
    print(f'Seq {seq}')
    print(f'{"="*60}')

    Tr4 = make_tr4(SEQ_TR[seq])

    print('  Loading poses...')
    poses, tx, tz = load_poses(seq)
    print(f'  {len(poses)} frames | X=[{tx.min():.0f},{tx.max():.0f}] Z=[{tz.min():.0f},{tz.max():.0f}]')

    print('  Building LiDAR density (every 5th frame)...')
    wx, wz = build_lidar_density(seq, poses, Tr4, step=5)
    print(f'  {len(wx):,} points')

    print('  Loading OSM polylines...')
    with open(f'{OSM_DIR}/osm_polylines_aligned_seq{seq}.pkl', 'rb') as f:
        osm_lines = pickle.load(f)
    print(f'  {len(osm_lines)} polylines')

    out = f'{OUT_DIR}/paper_three_modalities_seq{seq}.png'
    print('  Rendering...')
    render(seq, tx, tz, wx, wz, osm_lines, out)

print('\n✓ All done!')
