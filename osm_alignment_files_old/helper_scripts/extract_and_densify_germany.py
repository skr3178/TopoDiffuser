#!/usr/bin/env python3
"""
Extract and densify polylines from Germany OSM for specific sequences.
Memory-efficient approach using bounding box filtering.
"""

import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, 'utils')
from osm_alignment import load_oxts_data, latlon_to_utm


def densify_polyline(polyline, spacing=2.0):
    """Densify a polyline by interpolating points every `spacing` meters."""
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


def process_sequence(seq, pbf_file, config, data_root='data'):
    """Process a single sequence."""
    from pyrosm import OSM
    
    print(f"\n{'='*70}")
    print(f"Seq {seq} - Extracting from Germany OSM")
    print(f"{'='*70}")
    
    # Load trajectory
    def load_poses(pose_file):
        poses = []
        with open(pose_file) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                poses.append(np.array(vals).reshape(3, 4))
        return np.array(poses)
    
    def extract_trajectory(poses):
        return np.array([[p[0, 3], p[2, 3]] for p in poses])
    
    pose_file = Path(data_root) / 'kitti' / 'poses' / f'{seq}.txt'
    trajectory = extract_trajectory(load_poses(pose_file))
    print(f"Trajectory: {len(trajectory)} frames")
    
    # Load OXTS
    raw_folder = config['raw']
    frame_offset = config['frame_offset']
    
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
    
    oxts_dir = find_oxts_dir(raw_folder, data_root)
    oxts_data = load_oxts_data(str(oxts_dir))
    
    # Get bounding box
    ref_frame = min(frame_offset, len(oxts_data) - 1)
    lat0, lon0 = oxts_data[ref_frame, 0], oxts_data[ref_frame, 1]
    east0, north0 = latlon_to_utm(lat0, lon0)
    
    offset_east = east0 - trajectory[0, 0]
    offset_north = north0 - trajectory[0, 1]
    
    traj_utm = np.array([
        [trajectory[i, 0] + offset_east, trajectory[i, 1] + offset_north]
        for i in range(len(trajectory))
    ])
    
    # Larger margin for seq 05
    margin = 3000 if seq == '05' else 2000
    east_min = traj_utm[:, 0].min() - margin
    east_max = traj_utm[:, 0].max() + margin
    north_min = traj_utm[:, 1].min() - margin
    north_max = traj_utm[:, 1].max() + margin
    
    def utm_to_latlon_approx(east, north, zone=32):
        lat_ref, lon_ref = 49.0, 8.4
        east_ref, north_ref = latlon_to_utm(lat_ref, lon_ref)
        dlat = (north - north_ref) / 111000.0
        dlon = (east - east_ref) / (111000.0 * np.cos(np.radians(lat_ref)))
        return lat_ref + dlat, lon_ref + dlon
    
    lat_min, lon_min = utm_to_latlon_approx(east_min, north_min)
    lat_max, lon_max = utm_to_latlon_approx(east_max, north_max)
    
    bbox = [min(lon_min, lon_max), min(lat_min, lat_max), max(lon_min, lon_max), max(lat_min, lat_max)]
    print(f"Bounding box: {bbox}")
    
    # Extract OSM
    print("Loading OSM (this may take a minute)...")
    osm = OSM(pbf_file, bounding_box=bbox)
    
    print("Extracting driving network...")
    network = osm.get_network(network_type="driving")
    print(f"Found {len(network)} road segments")
    
    # Extract and convert
    print("Converting and densifying...")
    road_polylines = []
    for idx, row in network.iterrows():
        geom = row['geometry']
        if geom is None:
            continue
        
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                # Convert to local frame
                pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                                 latlon_to_utm(lat, lon)[1] - offset_north]
                                for lon, lat in coords])
                # Densify
                densified = densify_polyline(pts, 2.0)
                if len(densified) >= 2:
                    road_polylines.append(densified)
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                if len(coords) >= 2:
                    pts = np.array([[latlon_to_utm(lat, lon)[0] - offset_east,
                                     latlon_to_utm(lat, lon)[1] - offset_north]
                                    for lon, lat in coords])
                    densified = densify_polyline(pts, 2.0)
                    if len(densified) >= 2:
                        road_polylines.append(densified)
    
    # Apply transform if available
    transform_file = f'osm_transform_seq{seq}_bestfit_new.pkl'
    try:
        with open(transform_file, 'rb') as f:
            transform = pickle.load(f)
        
        print(f"Applying transform...")
        anchor = transform['anchor']
        pivot = np.array(transform['pivot'])
        coarse_rot = np.radians(transform['coarse_rot_deg'])
        fine_rot = transform['bestfit']['rotation']
        fine_trans = transform['bestfit']['translation']
        
        # Apply coarse rotation
        c, s = np.cos(coarse_rot), np.sin(coarse_rot)
        rotated = []
        for pl in road_polylines:
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
        
        road_polylines = final_polylines
    except FileNotFoundError:
        print(f"  No transform file, using raw alignment")
    
    # Filter to trajectory area
    margin = 500
    xmin, xmax = trajectory[:, 0].min() - margin, trajectory[:, 0].max() + margin
    ymin, ymax = trajectory[:, 1].min() - margin, trajectory[:, 1].max() + margin
    
    filtered = []
    for pl in road_polylines:
        if np.any((pl[:, 0] >= xmin) & (pl[:, 0] <= xmax) &
                  (pl[:, 1] >= ymin) & (pl[:, 1] <= ymax)):
            filtered.append(pl)
    
    total_pts = sum(len(p) for p in filtered)
    print(f"Final: {len(filtered)} polylines, {total_pts} points")
    
    # Save
    output_dir = 'osm_aligned_final'
    Path(output_dir).mkdir(exist_ok=True)
    
    out_pkl = f'{output_dir}/osm_polylines_aligned_seq{seq}.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(filtered, f)
    print(f"Saved: {out_pkl}")
    
    return len(filtered), total_pts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs', nargs='+', default=['05', '10'])
    parser.add_argument('--pbf', default='data/osm/germany-latest.osm.pbf')
    args = parser.parse_args()
    
    SEQ_CONFIG = {
        '05': {'raw': '2011_09_30_drive_0018_sync', 'frame_offset': 46},
        '10': {'raw': '2011_09_30_drive_0034_sync', 'frame_offset': 0},
    }
    
    print("="*70)
    print("Extract Dense Polylines from Germany OSM")
    print("="*70)
    
    for seq in args.seqs:
        if seq not in SEQ_CONFIG:
            print(f"⚠️ Unknown sequence: {seq}")
            continue
        
        try:
            n_poly, n_pts = process_sequence(seq, args.pbf, SEQ_CONFIG[seq])
            print(f"✓ Seq {seq}: {n_poly} polylines, {n_pts} points")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
