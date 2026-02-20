"""
Side-by-side visualization of 3 modalities for KITTI seq 00:
  1. LiDAR density map (accumulated top-down point cloud)
  2. History trajectory (full start-to-finish ego path)
  3. Topometric map (aligned OSM polylines)

All panels share the same coordinate frame:
  X = KITTI odometry lateral (TX)
  Y = KITTI odometry forward (TZ)
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── paths ────────────────────────────────────────────────────────────────────
BASE = '/media/skr/storage/self_driving/TopoDiffuser'
VELODYNE_DIR = f'{BASE}/data/kitti/sequences/00/velodyne'
POSES_FILE   = f'{BASE}/data/kitti/poses/00.txt'
OSM_POLY     = f'{BASE}/osm_aligned_final/osm_polylines_aligned_seq00.pkl'
OUT_FILE     = f'{BASE}/paper_three_modalities_seq00.png'

# ── calibration: velodyne → camera0 ──────────────────────────────────────────
TR_VALS = [
    4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
   -7.210626507497e-03,  8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
    9.999738645903e-01,  4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
]
Tr = np.eye(4, dtype=np.float64)
Tr[:3] = np.array(TR_VALS).reshape(3, 4)

# ── load poses ────────────────────────────────────────────────────────────────
print('Loading poses...')
raw_poses = np.loadtxt(POSES_FILE)           # (N, 12)
N = len(raw_poses)
poses = np.tile(np.eye(4), (N, 1, 1))        # (N, 4, 4)
poses[:, :3, :] = raw_poses.reshape(N, 3, 4)

tx = raw_poses[:, 3]    # world X
tz = raw_poses[:, 11]   # world Z (forward)

# ── 1. LIDAR DENSITY MAP ─────────────────────────────────────────────────────
print('Building LiDAR density map (every 5th frame)...')
all_wx, all_wz = [], []
step = 5                          # sample every 5th frame
z_min, z_max = -1.5, 0.5         # keep near-ground points (vehicle height ~2m)

for idx in range(0, N, step):
    fpath = os.path.join(VELODYNE_DIR, f'{idx:06d}.bin')
    if not os.path.exists(fpath):
        continue
    pts = np.fromfile(fpath, dtype=np.float32).reshape(-1, 4)
    # height filter in lidar frame (z ≈ vertical in lidar = -y in camera)
    # lidar z in [-1.5, 0.5] keeps ground-plane returns
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    pts = pts[mask, :3].astype(np.float64)  # (K, 3)
    if len(pts) == 0:
        continue

    # homogeneous
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack([pts, ones])           # (K, 4)

    # lidar → camera → world
    T = poses[idx] @ Tr                      # (4, 4)
    w = (T @ pts_h.T).T                      # (K, 4)

    all_wx.append(w[:, 0])
    all_wz.append(w[:, 2])

all_wx = np.concatenate(all_wx)
all_wz = np.concatenate(all_wz)
print(f'  Total points: {len(all_wx):,}')

# ── compute shared extent ─────────────────────────────────────────────────────
# base extent on trajectory + LiDAR (tight), OSM can extend slightly beyond
with open(OSM_POLY, 'rb') as f:
    osm_lines = pickle.load(f)

pad = 40
x_lo = min(tx.min(), all_wx.min()) - pad
x_hi = max(tx.max(), all_wx.max()) + pad
y_lo = min(tz.min(), all_wz.min()) - pad
y_hi = max(tz.max(), all_wz.max()) + pad
print(f'Shared extent: X [{x_lo:.0f}, {x_hi:.0f}], Y [{y_lo:.0f}, {y_hi:.0f}]')

# ── 2D density histogram ──────────────────────────────────────────────────────
bins_x = 800
bins_y = int(bins_x * (y_hi - y_lo) / (x_hi - x_lo))
H, xedges, yedges = np.histogram2d(
    all_wx, all_wz,
    bins=[bins_x, bins_y],
    range=[[x_lo, x_hi], [y_lo, y_hi]],
)
H = H.T   # flip so y is vertical

# log-scale density, clipped for visual clarity
H_log = np.log1p(H)
H_log = np.clip(H_log, 0, np.percentile(H_log[H_log > 0], 98))

# ── figure ────────────────────────────────────────────────────────────────────
print('Rendering figure...')
# compute figure height to match coordinate aspect ratio
data_w = x_hi - x_lo
data_h = y_hi - y_lo
panel_w = 5.5   # inches per panel
fig_h = panel_w * data_h / data_w + 1.2   # +1.2 for title/xlabel
fig, axes = plt.subplots(1, 3, figsize=(panel_w * 3 + 0.6, fig_h),
                          facecolor='white',
                          gridspec_kw={'wspace': 0.08})

lw_traj = 2.0
lw_osm  = 0.8

for ax in axes:
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#cccccc')
    ax.grid(True, linewidth=0.3, color='#e0e0e0', zorder=0)

# ── Panel 1: LiDAR density ────────────────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('white')
# custom colormap: white → blue → yellow (paper-friendly on white bg)
from matplotlib.colors import LinearSegmentedColormap
lidar_cmap = LinearSegmentedColormap.from_list(
    'lidar', ['white', '#1a1a6e', '#0e6b9e', '#f0a500', '#ff4500'], N=256)
ax1.imshow(
    H_log,
    origin='lower',
    extent=[x_lo, x_hi, y_lo, y_hi],
    cmap=lidar_cmap,
    interpolation='bilinear',
    aspect='auto',
    vmin=0,
    zorder=1,
)
ax1.set_title('LiDAR Point Density', fontsize=13, fontweight='bold', pad=8)
ax1.set_xlabel('X (m)', fontsize=9)
ax1.set_ylabel('Y (m)', fontsize=9)
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', va='top', ha='left', color='#333333')

# ── Panel 2: History trajectory ────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('white')
# full ego trajectory
ax2.plot(tx, tz, color='#1a6e1a', linewidth=lw_traj, zorder=2, solid_capstyle='round')
# start / end markers
ax2.scatter(tx[0],  tz[0],  color='#2ecc40', s=100, zorder=5,
            marker='o', linewidths=1.0, edgecolors='#1a6e1a', label='Start')
ax2.scatter(tx[-1], tz[-1], color='#e74c3c', s=200, zorder=5,
            marker='*', linewidths=0.8, edgecolors='#8b0000', label='End')
ax2.set_title('History Trajectory', fontsize=13, fontweight='bold', pad=8)
ax2.set_xlabel('X (m)', fontsize=9)
ax2.legend(loc='lower left', fontsize=8, framealpha=0.88,
           handlelength=1.2, borderpad=0.5)
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', va='top', ha='left', color='#333333')

# ── Panel 3: Topometric map (OSM) ─────────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor('white')
for poly in osm_lines:
    if len(poly) >= 2:
        ax3.plot(poly[:, 0], poly[:, 1],
                 color='#4a6fa5', linewidth=lw_osm,
                 alpha=0.75, solid_capstyle='round', zorder=2)
# overlay trajectory
ax3.plot(tx, tz, color='#1a6e1a', linewidth=lw_traj - 0.3, alpha=0.75,
         zorder=3, solid_capstyle='round')
ax3.scatter(tx[0],  tz[0],  color='#2ecc40', s=100, zorder=5,
            marker='o', linewidths=1.0, edgecolors='#1a6e1a')
ax3.scatter(tx[-1], tz[-1], color='#e74c3c', s=200, zorder=5,
            marker='*', linewidths=0.8, edgecolors='#8b0000')
ax3.set_title('Topometric Map (OSM)', fontsize=13, fontweight='bold', pad=8)
ax3.set_xlabel('X (m)', fontsize=9)
ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes,
         fontsize=11, fontweight='bold', va='top', ha='left', color='#333333')

# ── global title ──────────────────────────────────────────────────────────────
fig.suptitle('KITTI Odometry Seq 00 – Three Input Modalities',
             fontsize=14, fontweight='bold', y=0.995)

fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.08, wspace=0.08)
fig.savefig(OUT_FILE, dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved → {OUT_FILE}')
