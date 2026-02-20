#!/usr/bin/env python3
"""
Precompute 5-Channel BEV Cache for TopoDiffuser.

Channels:
  0-2: LiDAR (height, intensity, density) - from existing 3ch cache
  3: Trajectory history - from odometry poses
  4: OSM roads - from GPS-aligned OSM data (or trajectory proxy)

Usage:
    python precompute_bev_5ch.py --sequences 00 02 05 07 08 09 10
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, 'models')
sys.path.insert(0, 'utils')

from utils.bev_multimodal import (
    latlon_to_utm,
    lidar_to_bev,
    trajectory_to_bev,
    osm_to_bev,
    draw_line
)
from utils.osm_alignment import (
    compute_gps_to_local_transform,
    transform_osm_to_local,
    load_oxts_data
)
from models.bev_rasterization import BEVRasterizer


def load_lidar_bev(cache_dir: Path, seq: str, frame_idx: int) -> Optional[np.ndarray]:
    """Load existing 3-channel LiDAR BEV from cache."""
    bev_path = cache_dir / seq / f"{frame_idx:06d}.npy"
    if bev_path.exists():
        return np.load(bev_path)
    return None


def compute_history_bev(poses: np.ndarray, 
                        frame_idx: int,
                        past_frames: int = 50,
                        grid_size: Tuple[int, int] = (300, 400),
                        resolution: float = 0.1,
                        x_range: Tuple[float, float] = (-20, 20),
                        y_range: Tuple[float, float] = (-10, 30)) -> np.ndarray:
    """
    Compute trajectory history BEV channel.
    
    Args:
        poses: [N, 12] pose matrices
        frame_idx: Current frame index
        past_frames: Number of past poses to use
        grid_size: BEV grid size
        resolution: Meters per pixel
        x_range, y_range: Coordinate ranges
    
    Returns:
        history_bev: [1, H, W] binary mask
    """
    H, W = grid_size
    bev = np.zeros((1, H, W), dtype=np.float32)
    
    # Get current pose
    current_pose = poses[frame_idx].reshape(3, 4)
    current_R = current_pose[:, :3]
    current_t = current_pose[:, 3]
    
    # Extract past poses
    start_idx = max(0, frame_idx - past_frames)
    past_poses = poses[start_idx:frame_idx+1]
    
    if len(past_poses) < 2:
        return bev
    
    # Transform to ego frame and convert to pixels
    pixel_coords = []
    
    for pose in past_poses:
        pose_mat = pose.reshape(3, 4)
        pos_world = pose_mat[:, 3]
        
        # Transform to ego frame: p_ego = R^T @ (p_world - t)
        pos_ego = current_R.T @ (pos_world - current_t)
        # KITTI camera: x=right, y=down (vertical), z=forward
        # BEV: x=right (same), y=forward (same as camera z, both increase forward)
        # In ego frame: points ahead have positive Z, points behind have negative Z
        # In BEV: points ahead have positive Y, points behind have negative Y
        # So: BEV_Y = ego_Z (direct mapping, no negation needed)
        x, y = pos_ego[0], pos_ego[2]
        
        # Convert to pixel coordinates
        px = int((x - x_range[0]) / resolution)
        py = int((y - y_range[0]) / resolution)
        
        if 0 <= px < W and 0 <= py < H:
            pixel_coords.append((px, py))
    
    # Draw trajectory line
    for i in range(len(pixel_coords) - 1):
        pt1 = pixel_coords[i]
        pt2 = pixel_coords[i + 1]
        bev[0] = draw_line(bev[0], pt1, pt2, width=2)
    
    return bev


def split_osm_edges_to_polylines(edges_array: np.ndarray, 
                                  gap_threshold: float = 0.001) -> List[List[Tuple[float, float]]]:
    """
    Split flat OSM edges array into individual polylines.
    
    Args:
        edges_array: [N, 2] array of (lat, lon) points
        gap_threshold: Gap threshold in degrees (~100m)
    
    Returns:
        List of polylines, each [(lat, lon), ...]
    """
    if len(edges_array) == 0:
        return []
    
    polylines = []
    current_polyline = [tuple(edges_array[0])]
    
    for i in range(1, len(edges_array)):
        prev_point = edges_array[i-1]
        curr_point = edges_array[i]
        
        # Check gap
        gap = np.linalg.norm(curr_point - prev_point)
        
        if gap > gap_threshold:
            # Start new polyline
            if len(current_polyline) > 1:
                polylines.append(current_polyline)
            current_polyline = [tuple(curr_point)]
        else:
            current_polyline.append(tuple(curr_point))
    
    # Add final polyline
    if len(current_polyline) > 1:
        polylines.append(current_polyline)
    
    return polylines


def compute_osm_bev_aligned(osm_polylines: List[np.ndarray],
                            current_pose: np.ndarray,
                            grid_size: Tuple[int, int] = (300, 400),
                            resolution: float = 0.1,
                            x_range: Tuple[float, float] = (-20, 20),
                            y_range: Tuple[float, float] = (-10, 30),
                            max_distance: float = 50.0) -> np.ndarray:
    """
    Compute OSM roads BEV from PRE-ALIGNED polylines (already in KITTI coords).
    
    Args:
        osm_polylines: List of polylines, each [N, 2] array in KITTI world coordinates
        current_pose: Current ego pose [3, 4]
        grid_size: BEV grid size
        resolution: Meters per pixel
        x_range, y_range: Coordinate ranges
        max_distance: Maximum distance to include roads
        
    Returns:
        osm_bev: [1, H, W] binary mask
    """
    H, W = grid_size
    bev = np.zeros((1, H, W), dtype=np.float32)
    
    if not osm_polylines:
        return bev
    
    # Extract current position in world frame
    current_R = current_pose[:, :3]
    current_t = current_pose[:, 3]
    current_pos = current_pose[:2, 3]  # [x, z] in world frame
    
    # Process each polyline
    for polyline in osm_polylines:
        if len(polyline) < 2:
            continue
        
        # Transform polyline to ego frame
        ego_points = []
        for world_pt in polyline:
            # world_pt is [x, z] in KITTI world frame
            world_pt_3d = np.array([world_pt[0], 0, world_pt[1]])
            ego_pt = current_R.T @ (world_pt_3d - current_t)
            ego_points.append(ego_pt[:2])
        
        ego_points = np.array(ego_points)
        
        # Check if polyline is within range
        center = np.mean(ego_points, axis=0)
        distance = np.linalg.norm(center)
        
        if distance > max_distance:
            continue
        
        # Convert to pixels and draw
        pixel_coords = []
        for ego_pt in ego_points:
            px = int((ego_pt[0] - x_range[0]) / resolution)
            py = int((ego_pt[1] - y_range[0]) / resolution)
            
            if 0 <= px < W and 0 <= py < H:
                pixel_coords.append((px, py))
        
        # Draw polyline
        for i in range(len(pixel_coords) - 1):
            pt1 = pixel_coords[i]
            pt2 = pixel_coords[i + 1]
            bev[0] = draw_line(bev[0], pt1, pt2, width=3)
    
    return bev


def compute_osm_bev(osm_polylines: List[List[Tuple[float, float]]],
                    alignment: Optional[Dict],
                    current_pose: np.ndarray,
                    frame_idx: int,
                    grid_size: Tuple[int, int] = (300, 400),
                    resolution: float = 0.1,
                    x_range: Tuple[float, float] = (-20, 20),
                    y_range: Tuple[float, float] = (-10, 30),
                    max_distance: float = 50.0) -> np.ndarray:
    """
    Compute OSM roads BEV channel.
    
    Args:
        osm_polylines: List of road polylines [(lat, lon), ...]
        alignment: GPS alignment dictionary (None for proxy mode)
        current_pose: Current ego pose [3, 4]
        frame_idx: Current frame index
        grid_size: BEV grid size
        resolution: Meters per pixel
        x_range, y_range: Coordinate ranges
        max_distance: Maximum distance to include roads
    
    Returns:
        osm_bev: [1, H, W] binary mask
    """
    H, W = grid_size
    bev = np.zeros((1, H, W), dtype=np.float32)
    
    if not osm_polylines or alignment is None:
        return bev
    
    # Extract current position in world frame
    current_R = current_pose[:, :3]
    current_t = current_pose[:, 3]
    
    # Process each polyline
    for polyline in osm_polylines:
        if len(polyline) < 2:
            continue
        
        # Transform polyline: lat/lon → UTM → local → ego
        ego_points = []
        
        for lat, lon in polyline:
            # GPS → UTM
            east, north = latlon_to_utm(lat, lon)
            
            # UTM → local (using alignment)
            utm_point = np.array([east, north])
            local_point = transform_gps_to_odometry(utm_point, alignment, frame_idx)
            
            # Local → ego
            point_3d = np.array([local_point[0], local_point[1], 0])
            ego_point = current_R.T @ (point_3d - current_t)
            
            ego_points.append(ego_point[:2])
        
        # Check if polyline is within range
        ego_points = np.array(ego_points)
        center = np.mean(ego_points, axis=0)
        distance = np.linalg.norm(center)
        
        if distance > max_distance:
            continue
        
        # Convert to pixels and draw
        pixel_coords = []
        for ego_pt in ego_points:
            px = int((ego_pt[0] - x_range[0]) / resolution)
            py = int((ego_pt[1] - y_range[0]) / resolution)
            
            if 0 <= px < W and 0 <= py < H:
                pixel_coords.append((px, py))
        
        # Draw polyline
        for i in range(len(pixel_coords) - 1):
            pt1 = pixel_coords[i]
            pt2 = pixel_coords[i + 1]
            bev[0] = draw_line(bev[0], pt1, pt2, width=3)
    
    return bev


def compute_osm_proxy_from_trajectory(poses: np.ndarray,
                                      frame_idx: int,
                                      window_frames: int = 100,
                                      dilation_radius: int = 10,
                                      grid_size: Tuple[int, int] = (300, 400),
                                      resolution: float = 0.1,
                                      x_range: Tuple[float, float] = (-20, 20),
                                      y_range: Tuple[float, float] = (-10, 30)) -> np.ndarray:
    """
    Compute OSM proxy from past + future trajectory (for sequences without good GPS).
    
    Args:
        poses: [N, 12] pose matrices
        frame_idx: Current frame index
        window_frames: Number of frames to look back/ahead
        dilation_radius: Radius for binary dilation
        grid_size: BEV grid size
        resolution: Meters per pixel
        x_range, y_range: Coordinate ranges
    
    Returns:
        osm_proxy_bev: [1, H, W] binary mask
    """
    from scipy.ndimage import binary_dilation
    
    H, W = grid_size
    
    # Get trajectory window (past + future)
    start_idx = max(0, frame_idx - window_frames)
    end_idx = min(len(poses), frame_idx + window_frames)
    window_poses = poses[start_idx:end_idx]
    
    if len(window_poses) < 2:
        return np.zeros((1, H, W), dtype=np.float32)
    
    # Get current pose
    current_pose = poses[frame_idx].reshape(3, 4)
    current_R = current_pose[:, :3]
    current_t = current_pose[:, 3]
    
    # Create binary mask from trajectory
    bev = np.zeros((H, W), dtype=np.float32)
    
    for pose in window_poses:
        pose_mat = pose.reshape(3, 4)
        pos_world = pose_mat[:, 3]
        
        # Transform to ego frame
        pos_ego = current_R.T @ (pos_world - current_t)
        # Use X and Z for ground plane (Z is forward in camera frame)
        x, y = pos_ego[0], pos_ego[2]
        
        # Convert to pixel
        px = int((x - x_range[0]) / resolution)
        py = int((y - y_range[0]) / resolution)
        
        if 0 <= px < W and 0 <= py < H:
            bev[py, px] = 1.0
    
    # Dilate to create road corridor
    bev_dilated = binary_dilation(bev, iterations=dilation_radius).astype(np.float32)
    
    return bev_dilated[np.newaxis, :, :]


def load_aligned_osm_polylines(seq: str) -> Optional[List[np.ndarray]]:
    """
    Load pre-aligned OSM polylines for a sequence.
    
    Uses VERIFIED aligned files based on visual inspection:
    - seq00: refined
    - seq01: bestfit
    - seq02: bestfit (inspect_seq02_aligned.png)
    - seq05: bestfit
    - seq07: bestfit (inspect_seq07_aligned.png)
    - seq08: bestfit
    - seq09: refined
    - seq10: bestfit
    
    Args:
        seq: Sequence number (e.g., '00')
        
    Returns:
        List of polyline segments, each is [N, 2] array in KITTI odometry coords
        Returns None if file not found.
    """
    # Map of verified variants based on visual inspection
    VERIFIED_VARIANTS = {
        '00': 'refined',
        '01': 'bestfit',
        '02': 'bestfit',  # inspect_seq02_aligned.png
        '05': 'bestfit',
        '07': 'bestfit',  # inspect_seq07_aligned.png
        '08': 'bestfit',  # osm_pbf_aligned_seq08_bestfit.png
        '09': 'refined',  # osm_pbf_aligned_seq09_refined.png
        '10': 'bestfit',
    }
    
    # Get verified variant for this sequence
    verified_variant = VERIFIED_VARIANTS.get(seq, 'bestfit')
    
    # Try verified variant first, then fallbacks
    variants = [verified_variant, 'bestfit', 'refined', 'regbez', '']
    variants = list(dict.fromkeys(variants))  # Remove duplicates while preserving order
    
    for variant in variants:
        if variant:
            filename = f'osm_polylines_aligned_seq{seq}_{variant}.pkl'
        else:
            filename = f'osm_polylines_aligned_seq{seq}.pkl'
        
        filepath = Path(filename)
        if filepath.exists():
            print(f"  Loading aligned OSM polylines from {filepath}")
            try:
                with open(filepath, 'rb') as f:
                    polylines = pickle.load(f)
                
                # Convert to expected format (list of arrays)
                if isinstance(polylines, list):
                    # Check if it's a list of segments (each [2,2]) or full polylines
                    if len(polylines) > 0 and hasattr(polylines[0], 'shape'):
                        if polylines[0].shape == (2, 2):
                            # List of segments - each segment is a 2-point line
                            # Treat each segment as its own polyline for rendering
                            segment_polylines = [seg for seg in polylines]
                            print(f"  Loaded {len(segment_polylines)} segments as polylines")
                            return segment_polylines
                        else:
                            # Already in polyline format (each polyline has N points)
                            print(f"  Loaded {len(polylines)} polylines")
                            return polylines
                
                print(f"  Warning: Unexpected format, using as-is")
                return polylines
                
            except Exception as e:
                print(f"  Warning: Failed to load {filepath}: {e}")
                continue
    
    return None


def process_sequence(seq: str,
                     data_root: Path,
                     raw_data_root: Path,
                     cache_3ch_dir: Path,
                     cache_5ch_dir: Path,
                     osm_edges_dir: Path,
                     gps_alignments_dir: Path,
                     use_osm_proxy: bool = False,
                     use_aligned_osm: bool = True) -> Dict:
    """
    Process one sequence and generate 5-channel BEV cache.
    
    Args:
        seq: Sequence number (e.g., '00')
        data_root: Path to KITTI odometry data
        raw_data_root: Path to KITTI raw data
        cache_3ch_dir: Path to existing 3ch BEV cache
        cache_5ch_dir: Output path for 5ch BEV cache
        osm_edges_dir: Path to OSM edge .npy files
        gps_alignments_dir: Path to GPS alignment .pkl files
        use_osm_proxy: Use trajectory proxy instead of real OSM
        use_aligned_osm: Use pre-aligned OSM polylines (default: True)
    
    Returns:
        Statistics dictionary
    """
    print(f"\nProcessing sequence {seq}...")
    
    # Load poses
    poses_path = data_root / 'poses' / f'{seq}.txt'
    if not poses_path.exists():
        print(f"  ❌ Poses not found: {poses_path}")
        return {'error': 'poses not found'}
    
    poses = np.loadtxt(poses_path)
    num_frames = len(poses)
    print(f"  Total frames: {num_frames}")
    
    # Load OSM data if available
    osm_polylines = None
    alignment = None  # Not needed for aligned polylines
    
    if not use_osm_proxy:
        if use_aligned_osm:
            # Try to load pre-aligned OSM polylines
            osm_polylines = load_aligned_osm_polylines(seq)
        
        if osm_polylines is None:
            # Fallback to raw OSM + GPS alignment
            osm_edges_path = osm_edges_dir / f'{seq}_edges.npy'
            if osm_edges_path.exists():
                print(f"  Loading OSM edges from {osm_edges_path}")
                edges_array = np.load(osm_edges_path)
                osm_polylines = split_osm_edges_to_polylines(edges_array)
                print(f"  Split into {len(osm_polylines)} polylines")
            
            # Try to load GPS alignment
            alignment_path = gps_alignments_dir / f'{seq}_alignment.pkl'
            if alignment_path.exists():
                print(f"  Loading GPS alignment from {alignment_path}")
                try:
                    with open(alignment_path, 'rb') as f:
                        alignment = pickle.load(f)
                except Exception as e:
                    print(f"  Warning: Failed to load alignment: {e}")
                    alignment = None
                
                # Check alignment quality
                if alignment and alignment.get('mean_error', 999) > 10.0 and not alignment.get('is_windowed', False):
                    print(f"  ⚠️  Poor alignment ({alignment['mean_error']:.1f}m), switching to proxy")
                    alignment = None
                    osm_polylines = None
            else:
                print(f"  ⚠️  No GPS alignment found, using trajectory proxy")
    
    # Mode detection: 
    # - "aligned": using pre-aligned polylines (no alignment needed)
    # - "osm": using raw OSM with GPS alignment
    # - "proxy": using trajectory proxy (no OSM data)
    if osm_polylines is None:
        mode = "proxy"
    elif alignment is None and use_aligned_osm:
        mode = "aligned"
    else:
        mode = "osm"
    print(f"  Mode: {mode}")
    
    # Create output directory
    output_dir = cache_5ch_dir / seq
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    stats = {
        'num_frames': 0,
        'num_with_lidar': 0,
        'num_with_history': 0,
        'num_with_osm': 0,
    }
    
    # BEV parameters
    grid_size = (300, 400)
    resolution = 0.1
    x_range = (-20, 20)
    y_range = (-10, 30)
    
    for frame_idx in tqdm(range(num_frames), desc=f"  Seq {seq}"):
        # Load LiDAR BEV (channels 0-2)
        lidar_bev = load_lidar_bev(cache_3ch_dir, seq, frame_idx)
        
        if lidar_bev is None:
            # Create empty LiDAR BEV
            lidar_bev = np.zeros((3, *grid_size), dtype=np.float32)
        else:
            stats['num_with_lidar'] += 1
        
        # Compute history BEV (channel 3)
        history_bev = compute_history_bev(
            poses, frame_idx,
            past_frames=50,
            grid_size=grid_size,
            resolution=resolution,
            x_range=x_range,
            y_range=y_range
        )
        
        if history_bev.sum() > 0:
            stats['num_with_history'] += 1
        
        # Compute OSM BEV (channel 4)
        if osm_polylines is not None:
            current_pose = poses[frame_idx].reshape(3, 4)
            
            if alignment is None:
                # Use aligned polylines (already in KITTI coords)
                osm_bev = compute_osm_bev_aligned(
                    osm_polylines, current_pose,
                    grid_size=grid_size,
                    resolution=resolution,
                    x_range=x_range,
                    y_range=y_range
                )
            else:
                # Use raw polylines with GPS alignment
                osm_bev = compute_osm_bev(
                    osm_polylines, alignment, current_pose, frame_idx,
                    grid_size=grid_size,
                    resolution=resolution,
                    x_range=x_range,
                    y_range=y_range
                )
            if osm_bev.sum() > 0:
                stats['num_with_osm'] += 1
        else:
            # Use trajectory proxy
            osm_bev = compute_osm_proxy_from_trajectory(
                poses, frame_idx,
                window_frames=100,
                dilation_radius=10,
                grid_size=grid_size,
                resolution=resolution,
                x_range=x_range,
                y_range=y_range
            )
            if osm_bev.sum() > 0:
                stats['num_with_osm'] += 1
        
        # Concatenate to 5-channel BEV
        bev_5ch = np.concatenate([lidar_bev, history_bev, osm_bev], axis=0)
        assert bev_5ch.shape == (5, *grid_size), f"Unexpected shape: {bev_5ch.shape}"
        
        # Save
        output_path = output_dir / f"{frame_idx:06d}.npy"
        np.save(output_path, bev_5ch.astype(np.float32))
        
        stats['num_frames'] += 1
    
    print(f"  ✓ Saved {stats['num_frames']} frames to {output_dir}")
    print(f"    LiDAR coverage: {stats['num_with_lidar'] / stats['num_frames'] * 100:.1f}%")
    print(f"    History coverage: {stats['num_with_history'] / stats['num_frames'] * 100:.1f}%")
    print(f"    OSM coverage: {stats['num_with_osm'] / stats['num_frames'] * 100:.1f}%")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Precompute 5-Channel BEV Cache'
    )
    parser.add_argument('--sequences', nargs='+',
                       default=['00', '02', '05', '07', '08', '09', '10'],
                       help='Sequences to process')
    parser.add_argument('--data_root', type=str, default='data/kitti',
                       help='Path to KITTI odometry data')
    parser.add_argument('--raw_data_root', type=str, default='data/raw_data',
                       help='Path to KITTI raw data')
    parser.add_argument('--cache_3ch_dir', type=str, default='data/kitti/bev_cache',
                       help='Path to existing 3ch BEV cache')
    parser.add_argument('--cache_5ch_dir', type=str, default='data/kitti/bev_cache_5ch',
                       help='Output path for 5ch BEV cache')
    parser.add_argument('--osm_edges_dir', type=str, default='data/osm',
                       help='Path to OSM edges .npy files')
    parser.add_argument('--gps_alignments_dir', type=str, default='data/gps_alignments',
                       help='Path to GPS alignment .pkl files')
    parser.add_argument('--force_proxy', nargs='+', default=[],
                       help='Sequences to force using trajectory proxy (e.g., 00)')
    parser.add_argument('--use_aligned_osm', action='store_true', default=True,
                       help='Use pre-aligned OSM polylines (default: True)')
    parser.add_argument('--no_aligned_osm', dest='use_aligned_osm', action='store_false',
                       help='Disable use of pre-aligned OSM polylines')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Precompute 5-Channel BEV Cache")
    print("=" * 70)
    
    # Setup paths
    data_root = Path(args.data_root)
    raw_data_root = Path(args.raw_data_root)
    cache_3ch_dir = Path(args.cache_3ch_dir)
    cache_5ch_dir = Path(args.cache_5ch_dir)
    osm_edges_dir = Path(args.osm_edges_dir)
    gps_alignments_dir = Path(args.gps_alignments_dir)
    
    cache_5ch_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each sequence
    all_stats = {}
    
    for seq in args.sequences:
        use_proxy = seq in args.force_proxy
        
        stats = process_sequence(
            seq, data_root, raw_data_root,
            cache_3ch_dir, cache_5ch_dir,
            osm_edges_dir, gps_alignments_dir,
            use_osm_proxy=use_proxy,
            use_aligned_osm=args.use_aligned_osm
        )
        
        all_stats[seq] = stats
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for seq, stats in all_stats.items():
        if 'error' not in stats:
            print(f"Seq {seq}: {stats['num_frames']} frames, "
                  f"OSM coverage: {stats['num_with_osm'] / stats['num_frames'] * 100:.1f}%")
        else:
            print(f"Seq {seq}: Error - {stats['error']}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
