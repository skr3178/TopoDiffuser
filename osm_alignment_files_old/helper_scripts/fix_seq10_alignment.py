#!/usr/bin/env python3
"""
Fix seq10 OSM alignment - fast version.

Key optimisation: pre-convert all 30k polylines from lat/lon → local XY once,
then grid-search is just a numpy 2D rotation (no repeated UTM calls).

Source: data/osm/karlsruhe_all_roads.pkl  (30,844 polylines, already extracted)
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
SEQ        = '10'
RAW_FOLDER = '2011_09_30_drive_0034_sync'
DATE       = '2011_09_30'


# ── helpers ───────────────────────────────────────────────────────────────────
def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def densify_polyline(polyline, spacing=2.0):
    polyline = np.array(polyline, dtype=np.float64)
    if len(polyline) < 2:
        return polyline
    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum   = np.concatenate([[0], np.cumsum(dists)])
    total = cum[-1]
    if total < spacing:
        return polyline
    n_samp = max(int(total / spacing) + 1, 2)
    new_d  = np.linspace(0, total, n_samp)
    pts    = []
    for d in new_d:
        idx = min(np.searchsorted(cum, d), len(polyline) - 1)
        if idx == 0:
            pts.append(polyline[0])
        else:
            t = (d - cum[idx-1]) / (cum[idx] - cum[idx-1] + 1e-10)
            pts.append(polyline[idx-1] + t * (polyline[idx] - polyline[idx-1]))
    return np.array(pts)


def rotate_and_filter(local_polys, rotation_deg, trajectory, margin=400):
    """Apply 2D rotation and bbox filter. local_polys already in unrotated local frame."""
    r = np.radians(rotation_deg)
    c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s], [s, c]])
    px, py = trajectory[0]

    xmin = trajectory[:, 0].min() - margin
    xmax = trajectory[:, 0].max() + margin
    ymin = trajectory[:, 1].min() - margin
    ymax = trajectory[:, 1].max() + margin

    aligned = []
    for pl in local_polys:
        # rotate around start point
        centered = pl - trajectory[0]
        rot = (R @ centered.T).T + trajectory[0]
        in_box = ((rot[:, 0] >= xmin) & (rot[:, 0] <= xmax) &
                  (rot[:, 1] >= ymin) & (rot[:, 1] <= ymax))
        if in_box.any():
            aligned.append(rot)
    return aligned


def compute_error(polylines, trajectory):
    all_pts  = np.vstack([p for p in polylines if len(p) >= 2])
    tree     = cKDTree(all_pts)
    dists, _ = tree.query(trajectory)
    return float(dists.mean()), float(tree.query(trajectory[0])[0]), float(tree.query(trajectory[-1])[0])


# ── load trajectory ───────────────────────────────────────────────────────────
print("Loading trajectory...")
poses      = load_poses(f'{DATA_ROOT}/kitti/poses/{SEQ}.txt')
trajectory = np.array([[p[0, 3], p[2, 3]] for p in poses])
print(f"  {len(trajectory)} frames | "
      f"X=[{trajectory[:,0].min():.1f},{trajectory[:,0].max():.1f}] "
      f"Z=[{trajectory[:,1].min():.1f},{trajectory[:,1].max():.1f}]")

# ── GPS offset (frame_offset=0) ───────────────────────────────────────────────
print("Loading OXTS GPS (frame_offset=0)...")
oxts_dir     = f'{DATA_ROOT}/raw_data/{RAW_FOLDER}/{DATE}/{RAW_FOLDER}/oxts/data'
oxts         = load_oxts_data(oxts_dir)
lat0, lon0   = oxts[0, 0], oxts[0, 1]
e0, n0       = latlon_to_utm(lat0, lon0)
offset_east  = e0 - trajectory[0, 0]
offset_north = n0 - trajectory[0, 1]
print(f"  GPS frame 0: lat={lat0:.6f}, lon={lon0:.6f}")

# ── pre-convert lat/lon → local XY (once) ────────────────────────────────────
cache_local = '/tmp/seq10_local_polys.pkl'
if Path(cache_local).exists():
    print(f"\nLoading pre-converted local polylines from {cache_local}...")
    with open(cache_local, 'rb') as f:
        local_polys = pickle.load(f)
else:
    print("\nLoading karlsruhe_all_roads (30k polylines)...")
    with open(f'{DATA_ROOT}/osm/karlsruhe_all_roads.pkl', 'rb') as f:
        raw_polys = pickle.load(f)
    print(f"  {len(raw_polys)} raw polylines")

    print("Pre-converting lat/lon → local XY (this runs once)...")
    local_polys = []
    for i, pl in enumerate(raw_polys):
        pts = np.array(pl)   # (N, 2) lat/lon
        east  = np.array([latlon_to_utm(la, lo)[0] for la, lo in pts]) - offset_east
        north = np.array([latlon_to_utm(la, lo)[1] for la, lo in pts]) - offset_north
        local_polys.append(np.column_stack([east, north]))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(raw_polys)}...")

    print(f"  Done. Caching to {cache_local}")
    with open(cache_local, 'wb') as f:
        pickle.dump(local_polys, f)

print(f"  {len(local_polys)} local polylines ready")

# ── coarse grid search (0–360°, step=2°) ─────────────────────────────────────
print("\nCoarse grid search (0–360°, step=2°)...")
best_rot, best_err = 0, np.inf
results = []
for rot in range(0, 360, 2):
    aligned = rotate_and_filter(local_polys, rot, trajectory, margin=300)
    if len(aligned) < 10:
        continue
    me, se, ee = compute_error(aligned, trajectory)
    results.append((rot, me, se, ee, len(aligned)))
    if me < best_err:
        best_err = me;  best_rot = rot
    if rot % 60 == 0:
        print(f"  {rot}°... best so far: {best_rot}° ({best_err:.1f}m)")

print(f"  Coarse best: {best_rot}° → mean={best_err:.2f}m")

# ── fine search (±10°, step=0.5°) ────────────────────────────────────────────
print(f"\nFine search around {best_rot}° (±10°, step=0.5°)...")
fine_rot, fine_err = float(best_rot), best_err
for rot in np.arange(best_rot - 10, best_rot + 10.1, 0.5):
    rot = float(rot % 360)
    aligned = rotate_and_filter(local_polys, rot, trajectory, margin=300)
    if len(aligned) < 10:
        continue
    me, se, ee = compute_error(aligned, trajectory)
    if me < fine_err:
        fine_err = me;  fine_rot = rot

print(f"  Fine best: {fine_rot:.1f}° → mean={fine_err:.2f}m")

# ── ultra-fine search (±2°, step=0.1°) ───────────────────────────────────────
print(f"\nUltra-fine search around {fine_rot:.1f}° (±2°, step=0.1°)...")
uf_rot, uf_err, uf_se, uf_ee = fine_rot, fine_err, 0.0, 0.0
for rot in np.arange(fine_rot - 2, fine_rot + 2.05, 0.1):
    rot = float(rot % 360)
    aligned = rotate_and_filter(local_polys, rot, trajectory, margin=300)
    if len(aligned) < 10:
        continue
    me, se, ee = compute_error(aligned, trajectory)
    if me < uf_err:
        uf_err = me;  uf_rot = rot;  uf_se = se;  uf_ee = ee

print(f"  Best: {uf_rot:.1f}° → mean={uf_err:.2f}m, start={uf_se:.2f}m, end={uf_ee:.2f}m")

# ── produce final output ──────────────────────────────────────────────────────
print(f"\nFinal alignment: rotation={uf_rot:.1f}°, margin=400m")
final_aligned = rotate_and_filter(local_polys, uf_rot, trajectory, margin=400)
print(f"  {len(final_aligned)} polylines")

print("Densifying (2m spacing)...")
densified = []
for pl in final_aligned:
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
    'anchor': 'start', 'source': 'karlsruhe_all_roads',
    'spacing': 2.0, 'mean_error': me, 'start_error': se, 'end_error': ee,
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
    print(f"  rot={rot:5.1f}° | mean={me_:6.2f}m | start={se_:6.2f}m | end={ee_:6.2f}m | n={n}")

print("\n✓ Done!")
