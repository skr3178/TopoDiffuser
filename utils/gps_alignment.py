"""
GPS to KITTI Odometry Alignment with Windowed Procrustes.

Handles GPS drift in long sequences (like seq 00) via piecewise alignment.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from scipy.interpolate import interp1d


def compute_procrustes_alignment(source_points: np.ndarray, 
                                  target_points: np.ndarray) -> Dict:
    """
    Compute optimal rigid transformation (R, t, scale) to align source to target.
    
    Args:
        source_points: [N, 2] source points (e.g., GPS in UTM)
        target_points: [N, 2] target points (e.g., KITTI local)
    
    Returns:
        Dictionary with R (2x2), t (2,), scale, and alignment error
    """
    # Center both point sets
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    
    # Compute scale
    source_scale = np.sqrt(np.sum(source_centered**2) / len(source_points))
    target_scale = np.sqrt(np.sum(target_centered**2) / len(target_points))
    scale = target_scale / (source_scale + 1e-8)
    
    # Normalize scale
    source_normalized = source_centered / (source_scale + 1e-8)
    
    # Compute rotation using SVD
    H = source_normalized.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = target_mean - scale * R @ source_mean
    
    # Apply transformation and compute error
    aligned = scale * (R @ source_points.T).T + t
    errors = np.linalg.norm(aligned - target_points, axis=1)
    
    return {
        'R': R,  # 2x2 rotation matrix
        't': t,  # 2D translation vector
        'scale': scale,
        'source_mean': source_mean,
        'target_mean': target_mean,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors),
        'errors': errors,
        'aligned_points': aligned
    }


def compute_windowed_alignment(source_points: np.ndarray,
                                target_points: np.ndarray,
                                window_size: int = 500,
                                step_size: int = 250) -> Dict:
    """
    Compute piecewise alignment using sliding windows.
    
    For sequences with GPS drift (like seq 00), align in overlapping windows
    and interpolate transformations between windows.
    
    Args:
        source_points: [N, 2] source points (GPS in UTM)
        target_points: [N, 2] target points (KITTI local)
        window_size: Number of frames per alignment window
        step_size: Step between window centers
    
    Returns:
        Dictionary with windowed transformations and interpolation functions
    """
    n_frames = len(source_points)
    
    # Compute window centers
    window_centers = []
    window_alignments = []
    
    for start in range(0, n_frames - window_size, step_size):
        end = min(start + window_size, n_frames)
        center = (start + end) // 2
        
        # Extract window
        source_window = source_points[start:end]
        target_window = target_points[start:end]
        
        # Compute alignment for this window
        alignment = compute_procrustes_alignment(source_window, target_window)
        
        window_centers.append(center)
        window_alignments.append(alignment)
        
        print(f"  Window {start}-{end} (center {center}): "
              f"mean_error={alignment['mean_error']:.2f}m")
    
    # Add final window if needed
    if window_centers[-1] < n_frames - window_size // 2:
        start = n_frames - window_size
        end = n_frames
        center = (start + end) // 2
        
        source_window = source_points[start:end]
        target_window = target_points[start:end]
        alignment = compute_procrustes_alignment(source_window, target_window)
        
        window_centers.append(center)
        window_alignments.append(alignment)
        
        print(f"  Window {start}-{end} (center {center}): "
              f"mean_error={alignment['mean_error']:.2f}m")
    
    # Create interpolation functions for R, t, scale
    # Convert R to angle for interpolation
    window_angles = []
    for alignment in window_alignments:
        R = alignment['R']
        angle = np.arctan2(R[1, 0], R[0, 0])
        window_angles.append(angle)
    
    window_centers = np.array(window_centers)
    window_angles = np.array(window_angles)
    window_scales = np.array([a['scale'] for a in window_alignments])
    window_t = np.array([a['t'] for a in window_alignments])
    
    # Create interpolators
    angle_interp = interp1d(window_centers, window_angles, kind='linear', 
                            fill_value=(window_angles[0], window_angles[-1]),
                            bounds_error=False)
    scale_interp = interp1d(window_centers, window_scales, kind='linear',
                           fill_value=(window_scales[0], window_scales[-1]),
                           bounds_error=False)
    t_interp = interp1d(window_centers, window_t, kind='linear', axis=0,
                       fill_value=(window_t[0], window_t[-1]),
                       bounds_error=False)
    
    # Compute per-frame transformations
    frame_R = []
    frame_t = []
    frame_scale = []
    
    for i in range(n_frames):
        angle = angle_interp(i)
        scale = scale_interp(i)
        t = t_interp(i)
        
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        frame_R.append(R)
        frame_t.append(t)
        frame_scale.append(scale)
    
    # Apply transformations and compute errors
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
        'window_alignments': window_alignments,
        'frame_R': frame_R,
        'frame_t': frame_t,
        'frame_scale': frame_scale,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors),
        'errors': errors,
        'aligned_points': aligned_points,
        'is_windowed': True
    }


def compute_gps_to_odometry_transform(sequence: str,
                                       data_root: str = 'data/kitti',
                                       raw_data_root: str = 'data/raw_data',
                                       max_frames: Optional[int] = None,
                                       use_windowed: bool = True,
                                       window_size: int = 500) -> Dict:
    """
    Compute transformation from GPS (UTM) to KITTI odometry frame.
    
    Args:
        sequence: KITTI sequence number (e.g., '00')
        data_root: Path to KITTI odometry data
        raw_data_root: Path to KITTI raw data with OXTS
        max_frames: Maximum frames to use (None for all)
        use_windowed: Use windowed alignment for long sequences with drift
        window_size: Window size for piecewise alignment
    
    Returns:
        Dictionary with transformation parameters and alignment metrics
    """
    from utils.bev_multimodal import latlon_to_utm
    
    # Mapping from odometry to raw data
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
    
    if sequence not in ODOMETRY_TO_RAW:
        raise ValueError(f"Unknown sequence: {sequence}")
    
    date, drive = ODOMETRY_TO_RAW[sequence]
    raw_path = Path(raw_data_root) / f"{date}_drive_{drive}_sync" / date / f"{date}_drive_{drive}_sync"
    
    # Load OXTS data
    oxts_dir = raw_path / "oxts" / "data"
    if not oxts_dir.exists():
        raise FileNotFoundError(f"OXTS data not found: {oxts_dir}")
    
    oxts_files = sorted(oxts_dir.glob("*.txt"))
    if max_frames:
        oxts_files = oxts_files[:max_frames]
    
    # Load GPS coordinates
    gps_coords = []
    for f in oxts_files:
        data = np.loadtxt(f)
        lat, lon = data[0], data[1]
        east, north = latlon_to_utm(lat, lon)
        gps_coords.append([east, north])
    
    gps_coords = np.array(gps_coords)
    
    # Load odometry poses
    poses_path = Path(data_root) / 'poses' / f'{sequence}.txt'
    poses = np.loadtxt(poses_path)[:len(gps_coords)]
    
    # Extract local coordinates
    # KITTI camera convention: x=right, y=DOWN, z=forward
    # Ground plane is (x, z), NOT (x, y) which is vertical
    local_coords = []
    for pose in poses:
        pose_mat = pose.reshape(3, 4)
        local_coords.append([pose_mat[0, 3], pose_mat[2, 3]])  # (tx, tz)
    
    local_coords = np.array(local_coords)
    
    # Compute alignment
    print(f"Computing GPS to odometry alignment for sequence {sequence}...")
    print(f"  GPS points: {len(gps_coords)}")
    print(f"  Local points: {len(local_coords)}")
    
    # Decide whether to use windowed alignment based on error
    if use_windowed:
        # First try global alignment to check drift
        global_alignment = compute_procrustes_alignment(gps_coords, local_coords)
        print(f"  Global alignment error: {global_alignment['mean_error']:.2f}m")
        
        if global_alignment['mean_error'] > 10.0:
            print(f"  Using windowed alignment (window_size={window_size})")
            alignment = compute_windowed_alignment(
                gps_coords, local_coords, window_size=window_size
            )
        else:
            alignment = global_alignment
            alignment['is_windowed'] = False
    else:
        alignment = compute_procrustes_alignment(gps_coords, local_coords)
        alignment['is_windowed'] = False
    
    print(f"  Final mean error: {alignment['mean_error']:.2f}m")
    print(f"  Max error: {alignment['max_error']:.2f}m")
    
    # Add metadata
    alignment['sequence'] = sequence
    alignment['num_frames'] = len(gps_coords)
    alignment['gps_coords'] = gps_coords
    alignment['local_coords'] = local_coords
    
    return alignment


def transform_gps_to_odometry(gps_point: np.ndarray, 
                               alignment: Dict,
                               frame_idx: int = 0) -> np.ndarray:
    """
    Transform GPS point (UTM) to KITTI odometry frame.
    
    Args:
        gps_point: [2] or [N, 2] (east, north) in UTM
        alignment: Alignment dictionary
        frame_idx: Frame index for windowed alignment
    
    Returns:
        point_odometry: [2] or [N, 2] in KITTI odometry frame
    """
    if alignment.get('is_windowed', False):
        # Use per-frame transformation
        R = alignment['frame_R'][frame_idx]
        t = alignment['frame_t'][frame_idx]
        scale = alignment['frame_scale'][frame_idx]
    else:
        # Use global transformation
        R = alignment['R']
        t = alignment['t']
        scale = alignment['scale']
    
    if gps_point.ndim == 1:
        return scale * (R @ gps_point) + t
    else:
        return scale * (R @ gps_point.T).T + t


def save_alignment(alignment: Dict, output_path: str):
    """Save alignment to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove large arrays for saving (can reconstruct)
    alignment_to_save = {k: v for k, v in alignment.items() 
                        if k not in ['gps_coords', 'local_coords', 'aligned_points']}
    
    with open(output_path, 'wb') as f:
        pickle.dump(alignment_to_save, f)
    print(f"Saved alignment to {output_path}")


def load_alignment(alignment_path: str) -> Dict:
    """Load alignment from file."""
    with open(alignment_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute GPS to KITTI Odometry Alignment'
    )
    parser.add_argument('--sequences', nargs='+', 
                       default=['00', '02', '05', '07', '08', '09', '10'],
                       help='Sequences to process')
    parser.add_argument('--data_root', default='data/kitti',
                       help='Path to KITTI odometry data')
    parser.add_argument('--raw_data_root', default='data/raw_data',
                       help='Path to KITTI raw data')
    parser.add_argument('--output_dir', default='data/gps_alignments',
                       help='Output directory for alignment files')
    parser.add_argument('--window_size', type=int, default=500,
                       help='Window size for piecewise alignment')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPS to KITTI Odometry Alignment")
    print("=" * 70)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for seq in args.sequences:
        print(f"\nProcessing sequence {seq}...")
        try:
            alignment = compute_gps_to_odometry_transform(
                seq, args.data_root, args.raw_data_root,
                use_windowed=True, window_size=args.window_size
            )
            
            output_path = Path(args.output_dir) / f'{seq}_alignment.pkl'
            save_alignment(alignment, output_path)
            
            if alignment['mean_error'] < 2.0:
                print(f"  ✅ Excellent alignment")
            elif alignment['mean_error'] < 5.0:
                print(f"  ✅ Good alignment")
            elif alignment['mean_error'] < 10.0:
                print(f"  ⚠️  Acceptable alignment")
            else:
                print(f"  ❌ Poor alignment")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
