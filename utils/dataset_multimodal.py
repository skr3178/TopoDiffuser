"""
Full Multimodal KITTI Dataset for TopoDiffuser.

Loads LiDAR, trajectory history, and OSM map data, converts to 5-channel BEV.
Includes extensive debugging and evaluation metrics.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional

from utils.bev_multimodal import (
    lidar_to_bev,
    trajectory_to_bev,
    osm_to_bev,
    create_multimodal_input,
    compute_gps_to_local_transform,
    evaluate_osm_alignment,
    evaluate_trajectory_alignment,
    visualize_multimodal_bev
)


class KITTIFullMultimodalDataset(Dataset):
    """
    Full multimodal KITTI dataset with LiDAR + History + OSM.
    
    Returns:
        input_bev: [5, H, W] tensor - concatenated BEV input
        target_trajectory: [T_future, 2] tensor - future waypoints (x, y)
        road_mask: [1, H, W] tensor - ground truth road mask
        debug_info: dict - alignment metrics and debug data
    """
    
    def __init__(self,
                 data_root: str,
                 sequences: List[str],
                 past_frames: int = 5,
                 future_frames: int = 8,
                 spacing_meters: float = 2.0,
                 grid_size: Tuple[int, int] = (300, 400),
                 resolution: float = 0.1,
                 x_range: Tuple[float, float] = (-20, 20),
                 y_range: Tuple[float, float] = (-10, 30),
                 raw_data_root: Optional[str] = None,
                 enable_debug: bool = False,
                 viz_dir: Optional[str] = None):
        """
        Args:
            data_root: Path to KITTI odometry dataset root
            sequences: List of sequence numbers (e.g., ['00', '01'])
            past_frames: Number of past keyframes to use for history
            future_frames: Number of future waypoints to predict
            spacing_meters: Spacing between keyframes in meters
            grid_size: (H, W) BEV grid size
            resolution: Meters per pixel
            x_range: (min, max) x coordinates in ego frame
            y_range: (min, max) y coordinates in ego frame
            raw_data_root: Path to KITTI raw data with OXTS (for OSM alignment)
            enable_debug: If True, compute and return debug metrics
            viz_dir: Optional directory to save visualizations
        """
        self.data_root = Path(data_root)
        self.sequences = sequences
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.spacing_meters = spacing_meters
        self.grid_size = grid_size
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range
        self.raw_data_root = Path(raw_data_root) if raw_data_root else None
        self.enable_debug = enable_debug
        self.viz_dir = Path(viz_dir) if viz_dir else None
        
        if self.viz_dir:
            self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Load OXTS data and compute GPS transforms for each sequence
        self.gps_transforms = {}
        self.osm_data = {}
        
        if self.raw_data_root:
            self._load_osm_data()
        
        # Build index of valid samples
        self.samples = self._build_index()
        
        print(f"Loaded {len(self.samples)} samples from sequences {sequences}")
        if self.raw_data_root:
            print(f"GPS transforms computed for {len(self.gps_transforms)} sequences")
            print(f"OSM data loaded for {len(self.osm_data)} sequences")
    
    def _get_raw_sequence_path(self, seq: str) -> Optional[Path]:
        """Get path to raw data for a sequence."""
        # Mapping from odometry sequence to raw data
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
        
        if seq not in ODOMETRY_TO_RAW:
            return None
        
        date, drive = ODOMETRY_TO_RAW[seq]
        raw_path = self.raw_data_root / f"{date}_drive_{drive}_sync" / date / f"{date}_drive_{drive}_sync"
        return raw_path if raw_path.exists() else None
    
    def _load_oxts_data(self, seq: str) -> Optional[np.ndarray]:
        """Load OXTS data for a sequence."""
        raw_path = self._get_raw_sequence_path(seq)
        if not raw_path:
            return None
        
        oxts_dir = raw_path / "oxts" / "data"
        if not oxts_dir.exists():
            return None
        
        # Load all OXTS files
        oxts_files = sorted(oxts_dir.glob("*.txt"))
        if not oxts_files:
            return None
        
        data = []
        for f in oxts_files:
            values = np.loadtxt(f)
            data.append(values)
        
        return np.array(data)
    
    def _load_osm_data(self):
        """Load OSM data and compute GPS transforms for all sequences."""
        print("\nLoading OSM data and computing GPS transforms...")
        
        for seq in self.sequences:
            print(f"\n  Sequence {seq}:")
            
            # Load OXTS data
            oxts_data = self._load_oxts_data(seq)
            if oxts_data is None:
                print(f"    ⚠️  No OXTS data found")
                continue
            
            # Load poses
            poses_path = self.data_root / 'poses' / f'{seq}.txt'
            if not poses_path.exists():
                print(f"    ⚠️  No poses found")
                continue
            
            poses = np.loadtxt(poses_path)
            
            # Compute GPS transform - MATCH LENGTHS FIRST
            try:
                # Match lengths to avoid shape mismatch
                min_frames = min(len(oxts_data), len(poses))
                oxts_data_matched = oxts_data[:min_frames]
                poses_matched = poses[:min_frames]
                
                transform = compute_gps_to_local_transform(oxts_data_matched, poses_matched)
                transform['num_frames'] = min_frames  # Store matched frame count
                self.gps_transforms[seq] = transform
                print(f"    ✓ GPS transform computed ({min_frames} frames)")
                print(f"      Mean alignment error: {transform['mean_error']:.2f}m")
                print(f"      Max alignment error: {transform['max_error']:.2f}m")
            except Exception as e:
                print(f"    ⚠️  Error computing GPS transform: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Load actual OSM polylines data
            osm_polylines_path = Path('data/osm_polylines') / f'{seq}_polylines.pkl'
            if osm_polylines_path.exists():
                try:
                    with open(osm_polylines_path, 'rb') as f:
                        polylines = pickle.load(f)
                    # polylines is a list of [(lat, lon), ...] segments
                    self.osm_data[seq] = {
                        'polylines': polylines,
                        'edges': polylines,  # Use polylines as edges
                        'source': 'osm_polylines',
                        'num_polylines': len(polylines)
                    }
                    print(f"    ✓ OSM data loaded: {len(polylines)} polylines")
                except Exception as e:
                    print(f"    ⚠️  Error loading OSM data: {e}")
                    self.osm_data[seq] = {
                        'polylines': [],
                        'edges': [],
                        'source': 'error',
                        'num_polylines': 0
                    }
            else:
                print(f"    ⚠️  No OSM polylines found at {osm_polylines_path}")
                self.osm_data[seq] = {
                    'polylines': [],
                    'edges': [],
                    'source': 'missing',
                    'num_polylines': 0
                }
    
    def _build_index(self) -> List[Dict]:
        """Build index of valid (sequence, frame) pairs."""
        samples = []
        
        for seq in self.sequences:
            seq_path = self.data_root / 'sequences' / seq
            
            if not seq_path.exists():
                print(f"Warning: Sequence {seq} not found at {seq_path}")
                continue
            
            velodyne_path = seq_path / 'velodyne'
            if not velodyne_path.exists():
                continue
            
            lidar_files = sorted(velodyne_path.glob('*.bin'))
            num_frames = len(lidar_files)
            
            # Need enough frames for history and future
            min_required = self.past_frames * 10 + self.future_frames * 10 + 10
            
            for frame_idx in range(min_required, num_frames - self.future_frames * 10):
                samples.append({
                    'sequence': seq,
                    'frame_idx': frame_idx
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_lidar(self, seq: str, frame_idx: int) -> np.ndarray:
        """Load and rasterize LiDAR."""
        lidar_path = self.data_root / 'sequences' / seq / 'velodyne' / f'{frame_idx:06d}.bin'
        
        if not lidar_path.exists():
            return np.zeros((3, *self.grid_size), dtype=np.float32)
        
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        bev = lidar_to_bev(
            points,
            grid_size=self.grid_size,
            resolution=self.resolution,
            x_range=self.x_range,
            y_range=self.y_range
        )
        return bev
    
    def _load_history(self, seq: str, frame_idx: int, current_pose: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Load and rasterize trajectory history."""
        poses_path = self.data_root / 'poses' / f'{seq}.txt'
        
        if not poses_path.exists():
            bev = np.zeros((1, *self.grid_size), dtype=np.float32)
            return bev, {'num_waypoints': 0}
        
        poses = np.loadtxt(poses_path)
        
        # Extract past trajectory
        start_idx = max(0, frame_idx - self.past_frames * 10)
        past_poses = poses[start_idx:frame_idx+1]
        
        # Convert to trajectory waypoints
        # KITTI: x=right, y=down, z=forward
        # Ground plane is (x, z), NOT (x, y)
        trajectory = []
        for pose in past_poses:
            pose_mat = pose.reshape(3, 4)
            x, z = pose_mat[0, 3], pose_mat[2, 3]  # tx (right), tz (forward)
            trajectory.append([x, z])
        
        trajectory = np.array(trajectory)
        
        # Rasterize with debug
        bev, debug_info = trajectory_to_bev(
            trajectory,
            current_pose,
            past_poses,
            grid_size=self.grid_size,
            resolution=self.resolution,
            x_range=self.x_range,
            y_range=self.y_range,
            debug=self.enable_debug
        )
        
        return bev, debug_info
    
    def _load_osm(self, seq: str, current_pose: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Load and rasterize OSM roads."""
        # Check if we have OSM data and GPS transform for this sequence
        if seq not in self.osm_data or seq not in self.gps_transforms:
            bev = np.zeros((1, *self.grid_size), dtype=np.float32)
            return bev, {'num_edges': 0, 'edges_rendered': 0}
        
        osm_edges = self.osm_data[seq]['edges']
        gps_transform = self.gps_transforms[seq]
        
        if len(osm_edges) == 0:
            bev = np.zeros((1, *self.grid_size), dtype=np.float32)
            return bev, {'num_edges': 0, 'edges_rendered': 0}
        
        # Rasterize with debug
        bev, debug_info = osm_to_bev(
            osm_edges,
            current_pose,
            gps_transform,
            grid_size=self.grid_size,
            resolution=self.resolution,
            x_range=self.x_range,
            y_range=self.y_range,
            debug=self.enable_debug
        )
        
        return bev, debug_info
    
    def _load_future_trajectory(self, seq: str, frame_idx: int, current_pose: np.ndarray) -> np.ndarray:
        """Load future trajectory as prediction target."""
        poses_path = self.data_root / 'poses' / f'{seq}.txt'
        
        if not poses_path.exists():
            return np.zeros((self.future_frames, 2), dtype=np.float32)
        
        poses = np.loadtxt(poses_path)
        
        # Extract future trajectory
        future_poses = poses[frame_idx:frame_idx + self.future_frames * 10]
        
        # Sample at spacing intervals
        trajectory = []
        current_pos = poses[frame_idx].reshape(3, 4)[:, 3]
        dist_accum = 0
        
        for i in range(1, len(future_poses)):
            pose = future_poses[i].reshape(3, 4)
            pos = pose[:, 3]
            # Use (x, z) for ground plane distance, not (x, y)
            dist = np.linalg.norm([pos[0] - current_pos[0], pos[2] - current_pos[2]])
            dist_accum += dist
            
            if dist_accum >= self.spacing_meters and len(trajectory) < self.future_frames:
                # Convert to ego frame
                R = current_pose[:, :3]
                t = current_pose[:, 3]
                ego_pos = R.T @ (pos - t)
                trajectory.append(ego_pos[:2])
                dist_accum = 0
                current_pos = pos
        
        # Pad if needed
        while len(trajectory) < self.future_frames:
            if trajectory:
                trajectory.append(trajectory[-1])
            else:
                trajectory.append([0, 0])
        
        return np.array(trajectory[:self.future_frames], dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Returns:
            input_bev: [5, H, W] multimodal input
            target_trajectory: [future_frames, 2] future waypoints
            road_mask: [1, H, W] road segmentation target (from OSM or LiDAR)
            debug_info: dict (if enable_debug=True)
        """
        sample = self.samples[idx]
        seq = sample['sequence']
        frame_idx = sample['frame_idx']
        
        # Load current pose
        poses_path = self.data_root / 'poses' / f'{seq}.txt'
        if poses_path.exists():
            poses = np.loadtxt(poses_path)
            current_pose = poses[frame_idx].reshape(3, 4)
        else:
            current_pose = np.eye(3, 4)
        
        # Load modalities
        lidar_bev = self._load_lidar(seq, frame_idx)
        
        if self.enable_debug:
            history_bev, history_debug = self._load_history(seq, frame_idx, current_pose)
            osm_bev, osm_debug = self._load_osm(seq, current_pose)
        else:
            history_bev, _ = self._load_history(seq, frame_idx, current_pose)
            osm_bev, _ = self._load_osm(seq, current_pose)
            history_debug = {}
            osm_debug = {}
        
        # Load target
        target_trajectory = self._load_future_trajectory(seq, frame_idx, current_pose)
        
        # Create multimodal input
        if self.enable_debug:
            input_bev, input_debug = create_multimodal_input(
                lidar_bev, history_bev, osm_bev, debug=True
            )
        else:
            input_bev = create_multimodal_input(lidar_bev, history_bev, osm_bev)
            input_debug = {}
        
        # Convert to tensors
        input_bev = torch.from_numpy(input_bev).float()
        target_trajectory = torch.from_numpy(target_trajectory).float()
        
        # Create road mask target (from OSM if available, else LiDAR proxy)
        road_mask = torch.from_numpy(osm_bev).float()
        
        # Compile debug info
        if self.enable_debug:
            debug_info = {
                'sequence': seq,
                'frame_idx': frame_idx,
                'history': history_debug,
                'osm': osm_debug,
                'input': input_debug,
            }
            
            # Compute alignment metrics
            if osm_bev.sum() > 0:
                osm_metrics = evaluate_osm_alignment(osm_bev, lidar_bev)
                debug_info['osm_alignment'] = osm_metrics
            
            if history_bev.sum() > 0:
                traj_metrics = evaluate_trajectory_alignment(
                    history_bev, target_trajectory.numpy(), current_pose
                )
                debug_info['trajectory_alignment'] = traj_metrics
            
            # Save visualization if requested
            if self.viz_dir and idx % 100 == 0:  # Save every 100th sample
                save_path = self.viz_dir / f"{seq}_{frame_idx:06d}_multimodal.png"
                visualize_multimodal_bev(
                    lidar_bev, history_bev, osm_bev,
                    save_path=str(save_path),
                    metrics=debug_info.get('osm_alignment', {})
                )
            
            return input_bev, target_trajectory, road_mask, debug_info
        
        return input_bev, target_trajectory, road_mask


if __name__ == "__main__":
    print("=" * 70)
    print("Full Multimodal Dataset Test")
    print("=" * 70)
    
    # This would require actual data
    print("\nNote: This requires KITTI data to test properly.")
    print("Example usage:")
    print("""
    dataset = KITTIFullMultimodalDataset(
        data_root='data/kitti',
        sequences=['00'],
        raw_data_root='data/raw_data',
        enable_debug=True,
        viz_dir='viz/multimodal'
    )
    
    input_bev, target, road_mask, debug = dataset[0]
    print(f"Input shape: {input_bev.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Debug info: {debug.keys()}")
    """)
