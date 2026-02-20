#!/usr/bin/env python3
"""
Densify regbez polylines for seq 05 and 10.
"""

import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def load_poses(pose_file):
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            poses.append(np.array(vals).reshape(3, 4))
    return np.array(poses)


def extract_trajectory(poses):
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def densify_polyline(polyline, spacing=2.0):
    if len(polyline) < 2:
        return np.array(polyline)
    
    polyline = np.array(polyline)
    diffs = np.diff(polyline, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dists[-1]
    
    if total_dist < spacing:
        return polyline
    
    n_samples = max(int(total_dist / spacing) + 1, 2)
    new_dists = np.linspace(0, total_dist, n_samples)
    
    new_points = []
    for d in new_dists:
        idx = np.searchsorted(cum_dists, d)
        if idx >= len(polyline):
            idx = len(polyline) - 1
        if idx == 0:
            new_points.append(polyline[0])
        else:
            t = (d - cum_dists[idx-1]) / (cum_dists[idx] - cum_dists[idx-1] + 1e-10)
            pt = polyline[idx-1] + t * (polyline[idx] - polyline[idx-1])
            new_points.append(pt)
    
    return np.array(new_points)


def process_sequence(seq):
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Densifying Regbez Polylines")
    print(f"{'='*70}")
    
    # Load regbez polylines
    regbez_pkl = f'osm_polylines_latlon_seq{seq}_regbez.pkl'
    with open(regbez_pkl, 'rb') as f:
        latlon_polylines = pickle.load(f)
    print(f"Loaded {len(latlon_polylines)} lat/lon polylines")
    
    # Load trajectory
    pose_file = Path(f'data/kitti/poses/{seq}.txt')
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Load transform
    transform_file = f'osm_transform_seq{seq}_bestfit_new.pkl'
    with open(transform_file, 'rb') as f:
        transform = pickle.load(f)
    
    # Extract parameters
    offset_east = transform['gps']['offset_east']
    offset_north = transform['gps']['offset_north']
    pivot = np.array(transform['pivot'])
    coarse_rot = np.radians(transform['coarse_rot_deg'])
    fine_rot = transform['bestfit']['rotation']
    fine_trans = transform['bestfit']['translation']
    
    # Convert and densify
    print("Converting and densifying...")
    local_polylines = []
    for polyline in latlon_polylines:
        pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                         latlon_to_utm(lat, lon)[1] - offset_north]
                        for lat, lon in polyline])
        local_polylines.append(pts)
    
    # Apply coarse rotation
    c, s = np.cos(coarse_rot), np.sin(coarse_rot)
    rotated = []
    for pl in local_polylines:
        xc = pl[:, 0] - pivot[0]
        yc = pl[:, 1] - pivot[1]
        xr = xc * c - yc * s + pivot[0]
        yr = xc * s + yc * c + pivot[1]
        rotated.append(np.column_stack([xr, yr]))
    
    # Apply fine rotation and translation
    c, s = np.cos(fine_rot), np.sin(fine_rot)
    traj_center = trajectory.mean(axis=0)
    
    final_polylines = []
    for pl in rotated:
        xc = pl[:, 0] - traj_center[0]
        yc = pl[:, 1] - traj_center[1]
        xr = xc * c - yc * s + traj_center[0] + fine_trans[0]
        yr = xc * s + yc * c + traj_center[1] + fine_trans[1]
        final_polylines.append(np.column_stack([xr, yr]))
    
    # Filter
    margin = 500
    xmin, xmax = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
    ymin, ymax = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin
    
    filtered = []
    for pl in final_polylines:
        if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                  (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
            filtered.append(pl)
    
    print(f"After filtering: {len(filtered)} polylines")
    
    # Densify
    print("Densifying...")
    densified = []
    for i, pl in enumerate(filtered):
        dpl = densify_polyline(pl, 2.0)
        if len(dpl) >= 2:
            densified.append(dpl)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(filtered)}...")
    
    total_pts = sum(len(p) for p in densified)
    print(f"Final: {len(densified)} polylines, {total_pts} points")
    
    # Save
    output_dir = 'osm_aligned_final'
    Path(output_dir).mkdir(exist_ok=True)
    
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(densified, f)
    print(f"Saved: {out_pkl}")
    
    return len(densified), total_pts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['05', '10'])
    args = parser.parse_args()
    
    print("="*70)
    print("Densify Regbez Polylines")
    print("="*70)
    
    for seq in args.seqs:
        try:
            n, pts = process_sequence(seq)
            print(f"✓ Seq {seq}: {n} polylines, {pts} points\n")
        except Exception as e:
            print(f"❌ Seq {seq}: {e}\n")


if __name__ == '__main__':
    main()
