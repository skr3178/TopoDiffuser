#!/usr/bin/env python3
"""
Fix seq02 OSM alignment via fast grid-search.

Current: rot=36°, mean=7.01m (too high).
Strategy: karlsruhe_all_roads.pkl (30k polylines),
pre-convert lat/lon → local UTM once, then fast numpy grid-search.

seq02 raw folder: 2011_10_03_drive_0034_sync
"""

import sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm

DATA_ROOT  = 'data'
OUTPUT_DIR = 'osm_aligned_final'
SEQ        = '02'
RAW_FOLDER = '2011_10_03_drive_0034_sync'
DATE       = '2011_10_03'
CACHE      = '/tmp/seq02_local_polys.pkl'


def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def densify_polyline(polyline, spacing=2.0):
    polyline = np.array(polyline)
    if len(polyline) < 2:
        return polyline
    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum   = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    if total < spacing:
        return polyline
    n_samp = max(int(total / spacing) + 1, 2)
    pts = []
    for d in np.linspace(0, total, n_samp):
        idx = min(np.searchsorted(cum, d), len(polyline) - 1)
        if idx == 0:
            pts.append(polyline[0])
        else:
            t = (d - cum[idx-1]) / (cum[idx] - cum[idx-1] + 1e-10)
            pts.append(polyline[idx-1] + t * (polyline[idx] - polyline[idx-1]))
    return np.array(pts)


def rotate_polylines(local_polys, rotation_deg, pivot):
    r = np.radians(rotation_deg)
    c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s], [s, c]])
    out = []
    for pl in local_polys:
        out.append((R @ (pl - pivot).T).T + pivot)
    return out


def filter_to_bbox(polylines, trajectory, margin):
    xmin = trajectory[:, 0].min() - margin
    xmax = trajectory[:, 0].max() + margin
    ymin = trajectory[:, 1].min() - margin
    ymax = trajectory[:, 1].max() + margin
    return [pl for pl in polylines
            if ((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)).any()]


def compute_error(polylines, trajectory):
    all_pts  = np.vstack([p for p in polylines if len(p) >= 2])
    tree     = cKDTree(all_pts)
    dists, _ = tree.query(trajectory)
    return float(dists.mean()), float(tree.query(trajectory[0])[0]), \
           float(tree.query(trajectory[-1])[0])


# ── load trajectory + GPS offset ─────────────────────────────────────────────
print("Loading trajectory...")
poses      = load_poses(f'{DATA_ROOT}/kitti/poses/{SEQ}.txt')
trajectory = np.array([[p[0, 3], p[2, 3]] for p in poses])
print(f"  {len(trajectory)} frames | "
      f"X=[{trajectory[:,0].min():.1f},{trajectory[:,0].max():.1f}] "
      f"Z=[{trajectory[:,1].min():.1f},{trajectory[:,1].max():.1f}]")

print("Loading OXTS (frame_offset=0)...")
oxts_dir     = f'{DATA_ROOT}/raw_data/{RAW_FOLDER}/{DATE}/{RAW_FOLDER}/oxts/data'
oxts         = load_oxts_data(oxts_dir)
lat0, lon0   = oxts[0, 0], oxts[0, 1]
e0, n0       = latlon_to_utm(lat0, lon0)
offset_east  = e0 - trajectory[0, 0]
offset_north = n0 - trajectory[0, 1]
print(f"  GPS frame 0: lat={lat0:.6f}, lon={lon0:.6f}")

# ── pre-convert lat/lon → local XY (once, seq02 GPS offset) ──────────────────
if Path(CACHE).exists():
    print(f"\nLoading cached local polylines from {CACHE}...")
    with open(CACHE, 'rb') as f:
        local_polys = pickle.load(f)
else:
    print("\nLoading karlsruhe_all_roads (30k polylines)...")
    with open(f'{DATA_ROOT}/osm/karlsruhe_all_roads.pkl', 'rb') as f:
        raw_polys = pickle.load(f)
    print(f"  {len(raw_polys)} raw polylines")
    print("Pre-converting lat/lon → local UTM (once)...")
    local_polys = []
    for i, pl in enumerate(raw_polys):
        pts   = np.array(pl)
        east  = np.array([latlon_to_utm(la, lo)[0] for la, lo in pts]) - offset_east
        north = np.array([latlon_to_utm(la, lo)[1] for la, lo in pts]) - offset_north
        local_polys.append(np.column_stack([east, north]))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(raw_polys)}...")
    with open(CACHE, 'wb') as f:
        pickle.dump(local_polys, f)
    print(f"  Done. Cached → {CACHE}")

print(f"  {len(local_polys)} polylines in local frame")
pivot = trajectory[0]

# ── coarse grid search (0–360°, step=2°) ─────────────────────────────────────
print("\nCoarse grid search (0–360°, step=2°)...")
best_rot, best_err = 0, np.inf
results = []
for rot in range(0, 360, 2):
    filtered = filter_to_bbox(rotate_polylines(local_polys, rot, pivot),
                              trajectory, margin=300)
    if len(filtered) < 10:
        continue
    me, se, ee = compute_error(filtered, trajectory)
    results.append((rot, me, se, ee, len(filtered)))
    if me < best_err:
        best_err = me;  best_rot = rot
    if rot % 40 == 0:
        print(f"  {rot}° done, best: {best_rot}° → {best_err:.2f}m")

print(f"  Coarse best: {best_rot}° → mean={best_err:.2f}m")

# ── fine search (±10°, step=0.5°) ────────────────────────────────────────────
print(f"\nFine search around {best_rot}° (±10°, step=0.5°)...")
fine_rot, fine_err = float(best_rot), best_err
for rot in np.arange(best_rot - 10, best_rot + 10.1, 0.5):
    rot = float(rot % 360)
    filtered = filter_to_bbox(rotate_polylines(local_polys, rot, pivot),
                              trajectory, margin=300)
    if len(filtered) < 10:
        continue
    me, se, ee = compute_error(filtered, trajectory)
    if me < fine_err:
        fine_err = me;  fine_rot = rot

print(f"  Fine best: {fine_rot:.1f}° → mean={fine_err:.2f}m")

# ── ultra-fine search (±2°, step=0.1°) ───────────────────────────────────────
print(f"\nUltra-fine search around {fine_rot:.1f}° (±2°, step=0.1°)...")
uf_rot, uf_err, uf_se, uf_ee = fine_rot, fine_err, 0.0, 0.0
for rot in np.arange(fine_rot - 2, fine_rot + 2.05, 0.1):
    rot = float(rot % 360)
    filtered = filter_to_bbox(rotate_polylines(local_polys, rot, pivot),
                              trajectory, margin=300)
    if len(filtered) < 10:
        continue
    me, se, ee = compute_error(filtered, trajectory)
    if me < uf_err:
        uf_err = me;  uf_rot = rot;  uf_se = se;  uf_ee = ee

print(f"  Best: {uf_rot:.1f}° → mean={uf_err:.2f}m, start={uf_se:.2f}m, end={uf_ee:.2f}m")

# ── final output ──────────────────────────────────────────────────────────────
print(f"\nFinal alignment: rotation={uf_rot:.1f}°, margin=400m")
final_filtered = filter_to_bbox(rotate_polylines(local_polys, uf_rot, pivot),
                                trajectory, margin=400)
print(f"  {len(final_filtered)} polylines")

print("Densifying (2m spacing)...")
densified = []
for pl in final_filtered:
    d = densify_polyline(pl, spacing=2.0)
    if len(d) >= 2:
        densified.append(d)
total_pts = sum(len(p) for p in densified)
print(f"  {len(densified)} polylines, {total_pts:,} points")

me, se, ee = compute_error(densified, trajectory)
print(f"  Final error: mean={me:.2f}m, start={se:.2f}m, end={ee:.2f}m")

# ── save ──────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(exist_ok=True)
with open(f'{OUTPUT_DIR}/osm_polylines_aligned_seq{SEQ}.pkl', 'wb') as f:
    pickle.dump(densified, f)

cfg = {
    'seq': SEQ, 'frame_offset': 0, 'rotation_deg': uf_rot,
    'anchor': 'start', 'source': 'karlsruhe_all_roads', 'spacing': 2.0,
    'mean_error': me, 'start_error': se, 'end_error': ee,
}
with open(f'{OUTPUT_DIR}/osm_config_seq{SEQ}.pkl', 'wb') as f:
    pickle.dump(cfg, f)

transform = {
    'gps': {'offset_east': e0, 'offset_north': n0},
    'anchor': 'start', 'pivot': list(trajectory[0]),
    'rotation_deg': uf_rot, 'frame_offset': 0,
}
with open(f'{OUTPUT_DIR}/osm_transform_seq{SEQ}.pkl', 'wb') as f:
    pickle.dump(transform, f)

print(f"\nSaved → {OUTPUT_DIR}/osm_polylines_aligned_seq{SEQ}.pkl")

# ── visualise ─────────────────────────────────────────────────────────────────
print("Generating visualization...")
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle(f'Seq {SEQ} - karlsruhe_all_roads (rot={uf_rot:.1f}°)\n'
             f'Mean: {me:.2f}m | Start: {se:.2f}m | End: {ee:.2f}m',
             fontsize=14, fontweight='bold')

for pl in densified:
    axes[0].plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.6)
axes[0].set_title(f'OSM ({len(densified)} polylines)')
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

axes[1].plot(trajectory[:, 0], trajectory[:, 1], 'g-', lw=2)
axes[1].scatter(*trajectory[0],  c='lime', s=200, marker='o', zorder=5)
axes[1].scatter(*trajectory[-1], c='red',  s=300, marker='*', zorder=5)
axes[1].set_title('Trajectory')
axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3)

for pl in densified:
    axes[2].plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5, alpha=0.4)
axes[2].plot(trajectory[:, 0], trajectory[:, 1], 'g-', lw=2)
axes[2].scatter(*trajectory[0],  c='lime', s=200, marker='o', zorder=5)
axes[2].scatter(*trajectory[-1], c='red',  s=300, marker='*', zorder=5)
axes[2].set_title('Overlay')
axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/osm_aligned_seq{SEQ}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved → {OUTPUT_DIR}/osm_aligned_seq{SEQ}.png")

print("\nTop rotation candidates:")
results.sort(key=lambda x: x[1])
for rot, me_, se_, ee_, n in results[:8]:
    print(f"  rot={rot:5.1f}° | mean={me_:6.2f}m | start={se_:6.2f}m | "
          f"end={ee_:6.2f}m | n={n}")
print("\n✓ Done!")
