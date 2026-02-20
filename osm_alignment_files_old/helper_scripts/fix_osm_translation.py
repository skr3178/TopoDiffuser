#!/usr/bin/env python3
"""
Fix residual translation offset between trajectory and OSM road network.

After rotation/scale alignment the trajectory is still slightly offset from
the road centerlines.  This script does a 2-D grid search for the (dx, dy)
translation applied to the OSM polylines that minimises the mean distance
between sampled trajectory points and their nearest road point.

All output goes to NEW files with a '_transfixed' suffix — existing files
are never touched.

Usage:
    python fix_osm_translation.py --seq 01
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree

sys.path.insert(0, 'utils')


# ── helpers ─────────────────────────────────────────────────────────────────

def load_poses(pose_file: str) -> np.ndarray:
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def extract_trajectory(poses: np.ndarray) -> np.ndarray:
    """Return (x, z) trajectory — same convention as the rest of the pipeline."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def mean_traj_to_road_error(trajectory, polylines):
    """Mean nearest-neighbour distance from trajectory to any road point."""
    pts = np.vstack([pl for pl in polylines if len(pl) >= 2])
    tree = cKDTree(pts)
    dists, _ = tree.query(trajectory, k=1)
    return float(np.mean(dists))


def translate_polylines(polylines, dx, dy):
    return [pl + np.array([dx, dy]) for pl in polylines]


# ── main search ─────────────────────────────────────────────────────────────

def find_best_translation(trajectory, polylines,
                          coarse_range=50, coarse_step=2,
                          fine_range=5,  fine_step=0.25):
    """
    Two-pass 2-D grid search for optimal (dx, dy) translation of OSM roads.

    Pass 1 – coarse grid over ±coarse_range metres in steps of coarse_step.
    Pass 2 – fine grid centred on the coarse best, ±fine_range in fine_step.
    """
    def grid_search(cx, cy, half, step):
        vals = np.arange(-half, half + step, step)
        best_err = float('inf')
        best_dx, best_dy = cx, cy
        err_grid = np.full((len(vals), len(vals)), np.inf)
        for i, dy_off in enumerate(vals):
            for j, dx_off in enumerate(vals):
                shifted = translate_polylines(polylines, cx + dx_off, cy + dy_off)
                e = mean_traj_to_road_error(trajectory, shifted)
                err_grid[i, j] = e
                if e < best_err:
                    best_err = e
                    best_dx, best_dy = cx + dx_off, cy + dy_off
        return best_dx, best_dy, best_err, vals, err_grid

    print("Coarse search …")
    dx1, dy1, err1, vals1, grid1 = grid_search(0, 0, coarse_range, coarse_step)
    print(f"  Coarse best: dx={dx1:+.1f} dy={dy1:+.1f}  error={err1:.2f}m")

    print("Fine search …")
    dx2, dy2, err2, vals2, grid2 = grid_search(dx1, dy1, fine_range, fine_step)
    print(f"  Fine   best: dx={dx2:+.2f} dy={dy2:+.2f}  error={err2:.2f}m")

    return dx2, dy2, err2, vals1, grid1


def fix_translation(seq: str = "01", data_root: str = "data",
                    coarse_range=50, coarse_step=2,
                    fine_range=5, fine_step=0.25):

    # ── input files (never modified) ────────────────────────────────────────
    pkl_file = Path(f"osm_polylines_aligned_seq{seq}.pkl")
    if not pkl_file.exists():
        print(f"Polylines not found: {pkl_file}")
        return

    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(str(pose_file))
    trajectory = extract_trajectory(poses)

    with open(pkl_file, 'rb') as f:
        raw_polylines = pickle.load(f)
    polylines = [np.array(pl) for pl in raw_polylines]

    # ── output file names (all NEW, no overwrites) ───────────────────────────
    out_pkl  = Path(f"osm_polylines_aligned_seq{seq}_transfixed.pkl")
    out_png  = Path(f"osm_translation_fix_seq{seq}.png")
    overlay_png = Path(f"osm_pbf_aligned_seq{seq}_transfixed.png")

    initial_err = mean_traj_to_road_error(trajectory, polylines)
    print(f"Initial error: {initial_err:.2f} m")

    best_dx, best_dy, best_err, coarse_vals, coarse_grid = find_best_translation(
        trajectory, polylines,
        coarse_range=coarse_range, coarse_step=coarse_step,
        fine_range=fine_range, fine_step=fine_step)

    improvement = initial_err - best_err
    print(f"\nBest translation: dx={best_dx:+.2f} m, dy={best_dy:+.2f} m")
    print(f"Error: {initial_err:.2f} → {best_err:.2f} m  (↓{improvement:.2f} m)")

    corrected = translate_polylines(polylines, best_dx, best_dy)

    # ── 3-panel diagnostic visualisation ─────────────────────────────────────
    step = max(1, len(trajectory) // 30)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Left: error surface (coarse grid)
    ax = axes[0]
    n = len(coarse_vals)
    extent = [coarse_vals[0], coarse_vals[-1], coarse_vals[0], coarse_vals[-1]]
    im = ax.imshow(coarse_grid, origin='lower', extent=extent,
                   aspect='auto', cmap='viridis_r')
    # Mark best point relative to the coarse origin (0,0)
    ax.scatter([best_dx], [best_dy], c='red', s=100, marker='*', zorder=5,
               label=f'Best ({best_dx:+.1f}, {best_dy:+.1f})')
    ax.axvline(0, color='w', lw=0.7, ls='--', alpha=0.6)
    ax.axhline(0, color='w', lw=0.7, ls='--', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Mean dist (m)')
    ax.set_xlabel('dx (m)')
    ax.set_ylabel('dy (m)')
    ax.set_title('Coarse translation error surface', fontweight='bold')
    ax.legend(fontsize=8)

    # Middle: BEFORE
    ax = axes[1]
    for pl in polylines:
        ax.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.7, alpha=0.45)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2, alpha=0.85,
            label='Trajectory')
    ax.scatter(trajectory[::step, 0], trajectory[::step, 1],
               c='yellow', edgecolors='k', linewidths=0.3, s=30, zorder=4,
               label='Traj samples')
    ax.set_title(f'BEFORE  (error {initial_err:.1f} m)', fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=8)
    ax.axis('equal')
    ax.grid(True, alpha=0.25)

    # Right: AFTER
    ax = axes[2]
    for pl in corrected:
        ax.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.7, alpha=0.45)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2, alpha=0.85,
            label='Trajectory')
    ax.scatter(trajectory[::step, 0], trajectory[::step, 1],
               c='yellow', edgecolors='k', linewidths=0.3, s=30, zorder=4,
               label='Traj samples')
    ax.scatter(trajectory[0, 0], trajectory[0, 1],
               c='lime', s=130, marker='o', zorder=6, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
               c='red', s=180, marker='*', zorder=6, label='End')
    ax.set_title(f'AFTER  (error {best_err:.1f} m, ↓{improvement:.1f} m)',
                 fontweight='bold', color='darkgreen')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=8)
    ax.axis('equal')
    ax.grid(True, alpha=0.25)

    fig.suptitle(f'Seq {seq} – Translation correction  '
                 f'dx={best_dx:+.2f} m  dy={best_dy:+.2f} m',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n✅ Diagnostic image:  {out_png}")

    # ── standalone overlay image ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    for pl in corrected:
        ax2.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.8, alpha=0.45)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', lw=2.5, alpha=0.9,
             label='Trajectory')
    ax2.scatter(trajectory[::step, 0], trajectory[::step, 1],
                c='yellow', edgecolors='k', linewidths=0.3, s=35, zorder=4,
                label='Traj samples')
    ax2.scatter(trajectory[0, 0], trajectory[0, 1],
                c='lime', s=150, marker='o', zorder=6, label='Start')
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1],
                c='red', s=200, marker='*', zorder=6, label='End')
    ax2.set_title(f'Seq {seq} – Final alignment  (error {best_err:.1f} m)\n'
                  f'dx={best_dx:+.2f} m  dy={best_dy:+.2f} m',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(overlay_png, dpi=150, bbox_inches='tight')
    print(f"✅ Overlay image:     {overlay_png}")

    # ── save corrected polylines to NEW file ──────────────────────────────────
    with open(out_pkl, 'wb') as f:
        pickle.dump(corrected, f)
    print(f"✅ Corrected polylines: {out_pkl}")
    print(f"\nIf satisfied, rename {out_pkl} → {pkl_file} to apply the fix.")

    return best_dx, best_dy, best_err


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix residual translation offset in OSM alignment "
                    "(writes to new files, never touches originals)")
    parser.add_argument("--seq", default="01", help="Sequence number (e.g. 01)")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--coarse-range", type=float, default=50,
                        help="Coarse search half-range in metres (default 50)")
    parser.add_argument("--coarse-step", type=float, default=2,
                        help="Coarse grid step in metres (default 2)")
    parser.add_argument("--fine-range", type=float, default=5,
                        help="Fine search half-range around coarse best (default 5)")
    parser.add_argument("--fine-step", type=float, default=0.25,
                        help="Fine grid step in metres (default 0.25)")

    args = parser.parse_args()
    fix_translation(
        args.seq, args.data_root,
        coarse_range=args.coarse_range, coarse_step=args.coarse_step,
        fine_range=args.fine_range, fine_step=args.fine_step)
    print("\nDone!")
