#!/usr/bin/env python3
"""
Generate bestfit overlay visualizations from existing bestfit pkl files.
Loads pre-computed aligned polylines and trajectory, plots the overlay.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys
from scipy.spatial import cKDTree

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def load_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses):
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def visualize_bestfit(seq, data_root='data'):
    bestfit_pkl = f'osm_polylines_aligned_seq{seq}_bestfit.pkl'
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'

    print(f"\nSeq {seq}: loading {bestfit_pkl}")
    with open(bestfit_pkl, 'rb') as f:
        polylines = pickle.load(f)
    polylines = [np.array(pl) if not isinstance(pl, np.ndarray) else pl for pl in polylines]
    print(f"  OSM polylines: {len(polylines)}")

    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"  Trajectory points: {len(trajectory)}")

    # Compute distances from each trajectory point to nearest OSM point
    all_osm = np.vstack(polylines)
    tree = cKDTree(all_osm)
    dists, _ = tree.query(trajectory, k=1)
    dist_start, _ = tree.query(trajectory[0])
    dist_end, _ = tree.query(trajectory[-1])

    print(f"  Mean error: {np.mean(dists):.2f}m  Max: {np.max(dists):.2f}m")

    # ---- Figure ----
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Main overlay
    ax1 = fig.add_subplot(gs[:, :2])
    for i, pl in enumerate(polylines):
        ax1.plot(pl[:, 0], pl[:, 1], 'b-', linewidth=0.6, alpha=0.4,
                 label='OSM' if i == 0 else '')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=300, marker='o', zorder=5, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5, label='End')
    ax1.set_title(
        f'Seq {seq} – Best-Fit OSM Alignment\n'
        f'Mean: {np.mean(dists):.2f}m  |  Max: {np.max(dists):.2f}m  |  '
        f'Total: {np.sum(dists):.0f}m  |  {len(polylines):,} polylines',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Start zoom
    ax2 = fig.add_subplot(gs[0, 2])
    for pl in polylines:
        ax2.plot(pl[:, 0], pl[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=400, marker='o', zorder=5)
    ax2.set_xlim(trajectory[0, 0] - 60, trajectory[0, 0] + 60)
    ax2.set_ylim(trajectory[0, 1] - 60, trajectory[0, 1] + 60)
    ax2.set_title(f'Start (error: {dist_start:.1f}m)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # End zoom
    ax3 = fig.add_subplot(gs[1, 2])
    for pl in polylines:
        ax3.plot(pl[:, 0], pl[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=400, marker='*', zorder=5)
    ax3.set_xlim(trajectory[-1, 0] - 60, trajectory[-1, 0] + 60)
    ax3.set_ylim(trajectory[-1, 1] - 60, trajectory[-1, 1] + 60)
    ax3.set_title(f'End (error: {dist_end:.1f}m)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    out = f'osm_pbf_aligned_seq{seq}_bestfit.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['02', '07', '08'])
    args = parser.parse_args()

    for seq in args.seqs:
        visualize_bestfit(seq)

    print('\nDone.')
