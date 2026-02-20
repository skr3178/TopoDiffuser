#!/usr/bin/env python3
"""
3-panel alignment inspection: OSM only | Trajectory only | Overlay.
Uses offset-only alignment (no rotation) so the user can see the raw mismatch
and report corrections.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm

SEQ_TO_RAW = {
    '02': ('2011_10_03_drive_0034_sync', '2011_10_03'),
    '07': ('2011_09_30_drive_0027_sync', '2011_09_30'),
    '08': ('2011_09_30_drive_0028_sync', '2011_09_30'),
}
SEQ_FRAME_OFFSET = {'02': 0, '07': 0, '08': 252}


def load_poses(f):
    poses = []
    with open(f) as fh:
        for line in fh:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def inspect(seq, data_root='data', rotation_deg=0.0, anchor='end'):
    raw_folder, date = SEQ_TO_RAW[seq]
    frame_offset = SEQ_FRAME_OFFSET[seq]

    # Load trajectory
    traj = np.array([[p[0,3], p[2,3]] for p in
                     load_poses(f'{data_root}/kitti/poses/{seq}.txt')])

    # Load GPS reference
    oxts_dir = f'{data_root}/raw_data/{raw_folder}/{date}/{raw_folder}/oxts/data'
    oxts = load_oxts_data(oxts_dir)
    ref = min(frame_offset, len(oxts)-1)
    e0, n0 = latlon_to_utm(oxts[ref, 0], oxts[ref, 1])
    off_e = e0 - traj[0, 0]
    off_n = n0 - traj[0, 1]

    # Load lat/lon polylines → local (offset only)
    with open(f'osm_polylines_latlon_seq{seq}_regbez.pkl', 'rb') as f:
        latlon_pls = pickle.load(f)

    local_pls = []
    for pl in latlon_pls:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - off_e,
                         latlon_to_utm(lat, lon)[1] - off_n]
                        for lat, lon in pl])
        local_pls.append(pts)

    # Optional rotation — always around the anchor point
    if rotation_deg != 0:
        r = np.radians(rotation_deg)
        c, s = np.cos(r), np.sin(r)
        if anchor == 'end':
            px, py = traj[-1]
        else:
            px, py = traj[0]
        rotated = []
        for pl in local_pls:
            xc, yc = pl[:,0]-px, pl[:,1]-py
            rotated.append(np.column_stack([xc*c-yc*s+px, xc*s+yc*c+py]))
        local_pls = rotated

    # Filter to trajectory bbox + 400m margin
    xmin, xmax = traj[:,0].min()-400, traj[:,0].max()+400
    ymin, ymax = traj[:,1].min()-400, traj[:,1].max()+400
    local_pls = [pl for pl in local_pls
                 if np.any((pl[:,0]>=xmin)&(pl[:,0]<=xmax)&
                           (pl[:,1]>=ymin)&(pl[:,1]<=ymax))]

    print(f'Seq {seq}: {len(local_pls)} polylines, traj len={len(traj)}')

    anchor_label = f'anchor={anchor}' if rotation_deg != 0 else 'no rotation'
    # ── 3-panel figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'Seq {seq}  –  rotation={rotation_deg:.1f}°  {anchor_label}  '
                 f'({len(local_pls)} polylines)', fontsize=14, fontweight='bold')

    # Panel 1: OSM only
    ax = axes[0]
    for pl in local_pls:
        ax.plot(pl[:,0], pl[:,1], 'b-', lw=0.7, alpha=0.5)
    ax.set_title('1. OSM roads', fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')

    # Panel 2: Trajectory only (green)
    ax = axes[1]
    ax.plot(traj[:,0], traj[:,1], color='green', lw=2)
    ax.scatter(traj[0,0],  traj[0,1],  c='lime',  s=200, marker='o', zorder=5, label='Start')
    ax.scatter(traj[-1,0], traj[-1,1], c='red',   s=300, marker='*', zorder=5, label='End')
    ax.set_title('2. Trajectory', fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')

    # Panel 3: Overlay
    ax = axes[2]
    for pl in local_pls:
        ax.plot(pl[:,0], pl[:,1], 'b-', lw=0.7, alpha=0.4)
    ax.plot(traj[:,0], traj[:,1], color='green', lw=2, label='Trajectory')
    ax.scatter(traj[0,0],  traj[0,1],  c='lime', s=200, marker='o', zorder=5, label='Start')
    ax.scatter(traj[-1,0], traj[-1,1], c='red',  s=300, marker='*', zorder=5, label='End')
    ax.set_title('3. Overlay', fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')

    plt.tight_layout()
    rot_tag = f'_rot{rotation_deg:.0f}' if rotation_deg != 0 else ''
    out = f'inspect_seq{seq}{rot_tag}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['02', '07', '08'])
    parser.add_argument('--rotation', type=float, default=0.0,
                        help='Rotate OSM by this many degrees (CCW positive) around the anchor')
    parser.add_argument('--anchor', type=str, default='end',
                        choices=['start', 'end'], help='Anchor point for rotation')
    args = parser.parse_args()
    for seq in args.seqs:
        inspect(seq, rotation_deg=args.rotation, anchor=args.anchor)
    print('Done.')
