"""
GPS to KITTI Odometry Alignment - Fixed Version.

Uses nearest-window transformation instead of broken interpolation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle


def latlon_to_utm(lat: float, lon: float, zone: int = 32) -> Tuple[float, float]:
    """Convert latitude/longitude to UTM coordinates."""
    a = 6378137.0
    e = 0.0818191908426
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    
    k0 = 0.9996
    N = a / np.sqrt(1 - e**2 * np.sin(lat_rad)**2)
    T = np.tan(lat_rad)**2
    C = e**2 * np.cos(lat_rad)**2 / (1 - e**2)
    A = np.cos(lat_rad) * (lon_rad - lon0)
    
    M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad -
             (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*lat_rad) +
             (15*e**4/256 + 45*e**6/1024) * np.sin(4*lat_rad) -
             (35*e**6/3072) * np.sin(6*lat_rad))
    
    east = k0 * N * (A + (1 - T + C) * A**3 / 6 + 
                     (5 - 18*T + T**2 + 72*C - 58*0.006739497) * A**5 / 120)
    east += 500000
    
    north = k0 * (M + N * np.tan(lat_rad) * (A**2 / 2 + 
                  (5 - T + 9*C + 4*C**2) * A**4 / 24 +
                  (61 - 58*T + T**2 + 600*C - 330*0.006739497) * A**6 / 720))
    
    return east, north


def compute_procrustes_alignment(source_points: np.ndarray, 
                                  target_points: np.ndarray) -> Dict:
    """Compute optimal rigid transformation to align source to target."""
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    
    source_scale = np.sqrt(np.sum(source_centered**2) / len(source_points))
    target_scale = np.sqrt(np.sum(target_centered**2) / len(target_points))
    scale = target_scale / (source_scale + 1e-8)
    
    source_normalized = source_centered / (source_scale + 1e-8)
    
    H = source_normalized.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = target_mean - scale * R @ source_mean
    
    aligned = scale * (R @ source_points.T).T + t
    errors = np.linalg.norm(aligned - target_points, axis=1)
    
    return {
        'R': R,
        't': t,
        'scale': scale,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
    }


def compute_windowed_alignment_fixed(source_points: np.ndarray,
                                      target_points: np.ndarray,
                                      window_size: int = 200) -> Dict:
    """
    Compute piecewise alignment using sliding windows.
    Uses nearest window transformation for each frame.
    """
    n_frames = len(source_points)
    
    # Compute windows
    window_centers = []
    window_R = []
    window_t = []
    window_scale = []
    
    for start in range(0, n_frames, window_size // 2):  # 50% overlap
        end = min(start + window_size, n_frames)
        if end - start < 50:  # Skip small windows at end
            break
            
        center = (start + end) // 2
        
        source_window = source_points[start:end]
        target_window = target_points[start:end]
        
        alignment = compute_procrustes_alignment(source_window, target_window)
        
        window_centers.append(center)
        window_R.append(alignment['R'])
        window_t.append(alignment['t'])
        window_scale.append(alignment['scale'])
        
        print(f"  Window {start}-{end} (center {center}): error={alignment['mean_error']:.2f}m")
    
    # Assign each frame to nearest window
    window_centers = np.array(window_centers)
    frame_R = []
    frame_t = []
    frame_scale = []
    
    for i in range(n_frames):
        # Find nearest window center
        nearest_idx = np.argmin(np.abs(window_centers - i))
        frame_R.append(window_R[nearest_idx])
        frame_t.append(window_t[nearest_idx])
        frame_scale.append(window_scale[nearest_idx])
    
    # Apply transformations
    aligned_points = []
    for i in range(n_frames):
        R = frame_R[i]
        t = frame_t[i]
        scale = frame_scale[i]
        aligned = scale * (R @ source_points[i]) + t
        aligned_points.append(aligned)
    
    aligned_points = np.array(aligned_points)
    errors = np.linalg.norm(aligned_points - target_points, axis=1)
    
    return {
        'window_centers': window_centers,
        'frame_R': frame_R,
        'frame_t': frame_t,
        'frame_scale': frame_scale,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'is_windowed': True
    }


def compute_gps_alignment_fixed(sequence: str,
                                 data_root: str = 'data/kitti',
                                 raw_data_root: str = 'data/raw_data',
                                 window_size: int = 200) -> Dict:
    """Compute GPS to odometry alignment with fixed windowed approach."""
    
    ODOMETRY_TO_RAW = {
        '00': ('2011_10_03', '0027'),
        '01': ('2011_10_03', '0042'),
        '02': ('2011_10_03', '0034'),
        '03': ('2011_09_26', '0067'),
        '04': ('2011_09_30', '0016'),
        '05': ('2011_09_30', '0018'),
        '06': ('2011_09_30', '0020'),
        '07': ('2011_09_30', '0027'),
        '08': ('2011_09_30', '0028'),
        '09': ('2011_09_30', '0033'),
        '10': ('2011_09_30', '0034'),
    }
    
    date, drive = ODOMETRY_TO_RAW[sequence]
    raw_path = Path(raw_data_root) / f"{date}_drive_{drive}_sync" / date / f"{date}_drive_{drive}_sync"
    
    # Load OXTS
    oxts_dir = raw_path / "oxts" / "data"
    oxts_files = sorted(oxts_dir.glob("*.txt"))
    
    gps_coords = []
    for f in oxts_files:
        data = np.loadtxt(f)
        lat, lon = data[0], data[1]
        east, north = latlon_to_utm(lat, lon)
        gps_coords.append([east, north])
    gps_coords = np.array(gps_coords)
    
    # Load poses
    poses_path = Path(data_root) / 'poses' / f'{sequence}.txt'
    poses = np.loadtxt(poses_path)
    
    # Match lengths - sometimes OXTS and poses have different counts
    min_frames = min(len(gps_coords), len(poses))
    gps_coords = gps_coords[:min_frames]
    poses = poses[:min_frames]
    
    # Extract local coords (x, z) - NOT (x, y)!
    local_coords = []
    for pose in poses:
        pose_mat = pose.reshape(3, 4)
        local_coords.append([pose_mat[0, 3], pose_mat[2, 3]])  # (tx, tz)
    local_coords = np.array(local_coords)
    
    print(f"Computing GPS alignment for sequence {sequence}...")
    print(f"  GPS points: {len(gps_coords)}")
    print(f"  Local points: {len(local_coords)}")
    
    # Check if global alignment is good enough
    global_align = compute_procrustes_alignment(gps_coords, local_coords)
    print(f"  Global error: {global_align['mean_error']:.2f}m")
    
    if global_align['mean_error'] < 5.0:
        # Global alignment is good
        alignment = global_align
        alignment['is_windowed'] = False
        # Create per-frame arrays
        alignment['frame_R'] = [global_align['R']] * len(gps_coords)
        alignment['frame_t'] = [global_align['t']] * len(gps_coords)
        alignment['frame_scale'] = [global_align['scale']] * len(gps_coords)
    else:
        # Use windowed alignment
        print(f"  Using windowed alignment (window_size={window_size})")
        alignment = compute_windowed_alignment_fixed(gps_coords, local_coords, window_size)
    
    print(f"  Final mean error: {alignment['mean_error']:.2f}m")
    print(f"  Max error: {alignment['max_error']:.2f}m")
    
    alignment['sequence'] = sequence
    alignment['num_frames'] = len(gps_coords)
    
    return alignment


def transform_gps_to_odometry_fixed(gps_point: np.ndarray, 
                                     alignment: Dict,
                                     frame_idx: int = 0) -> np.ndarray:
    """Transform GPS point to KITTI odometry frame."""
    R = alignment['frame_R'][frame_idx]
    t = alignment['frame_t'][frame_idx]
    scale = alignment['frame_scale'][frame_idx]
    
    if gps_point.ndim == 1:
        return scale * (R @ gps_point) + t
    else:
        return scale * (R @ gps_point.T).T + t


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', nargs='+', 
                       default=['00', '02', '05', '07', '08', '09', '10'])
    parser.add_argument('--output_dir', default='data/gps_alignments')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for seq in args.sequences:
        print(f"\n{'='*70}")
        alignment = compute_gps_alignment_fixed(seq)
        
        # Save
        output_path = Path(args.output_dir) / f'{seq}_alignment.pkl'
        with open(output_path, 'wb') as f:
            # Don't save the large arrays
            alignment_to_save = {k: v for k, v in alignment.items() 
                                if k not in ['frame_R', 'frame_t', 'frame_scale']}
            alignment_to_save['frame_R'] = alignment['frame_R']
            alignment_to_save['frame_t'] = alignment['frame_t']
            alignment_to_save['frame_scale'] = alignment['frame_scale']
            pickle.dump(alignment_to_save, f)
        
        print(f"Saved to {output_path}")
        
        if alignment['mean_error'] < 2.0:
            print("✅ Excellent")
        elif alignment['mean_error'] < 5.0:
            print("✅ Good")
        else:
            print("⚠️  Poor")
