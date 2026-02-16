"""
KITTI Dataset for TopoDiffuser.

Loads LiDAR, trajectory history, and OSM map data, converts to BEV.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    from .bev_utils import (
        lidar_to_bev,
        trajectory_to_bev,
        osm_to_bev,
        load_lidar_bin,
        poses_to_trajectory,
        create_input_tensor
    )
except ImportError:
    from bev_utils import (
        lidar_to_bev,
        trajectory_to_bev,
        osm_to_bev,
        load_lidar_bin,
        poses_to_trajectory,
        create_input_tensor
    )


class KITTITrajectoryDataset(Dataset):
    """
    KITTI dataset for multimodal trajectory prediction.
    
    Returns:
        input_bev: [5, H, W] tensor - concatenated BEV input
        target_trajectory: [T_future, 2] tensor - future waypoints (x, y)
        road_mask: [1, H, W] tensor - ground truth road mask (optional)
    """
    
    def __init__(self, 
                 data_root,
                 sequences,
                 past_frames=5,
                 future_frames=8,
                 spacing_meters=2.0,
                 grid_size=(300, 400),
                 resolution=0.1,
                 x_range=(-20, 20),
                 y_range=(-10, 30),
                 osm_root=None,
                 transform=None):
        """
        Args:
            data_root: Path to KITTI dataset root
            sequences: List of sequence numbers (e.g., ['00', '01'])
            past_frames: Number of past keyframes to use
            future_frames: Number of future waypoints to predict
            spacing_meters: Spacing between keyframes in meters
            grid_size: (H, W) BEV grid size
            resolution: Meters per pixel
            x_range: (min, max) x coordinates in ego frame
            y_range: (min, max) y coordinates in ego frame
            osm_root: Path to OSM data (if None, will use placeholder)
            transform: Optional data augmentation
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
        self.osm_root = Path(osm_root) if osm_root else None
        self.transform = transform
        
        # Build index of valid samples
        self.samples = self._build_index()
        
        print(f"Loaded {len(self.samples)} samples from sequences {sequences}")
    
    def _build_index(self):
        """Build index of valid (sequence, frame) pairs."""
        samples = []
        
        for seq in self.sequences:
            seq_path = self.data_root / 'sequences' / seq
            
            # Check if sequence exists
            if not seq_path.exists():
                print(f"Warning: Sequence {seq} not found at {seq_path}")
                continue
            
            # Count frames in sequence
            velodyne_path = seq_path / 'velodyne'
            if not velodyne_path.exists():
                continue
            
            lidar_files = sorted(velodyne_path.glob('*.bin'))
            num_frames = len(lidar_files)
            
            # Need enough frames for history and future
            min_required = self.past_frames + self.future_frames + 10
            
            for frame_idx in range(min_required, num_frames - self.future_frames):
                samples.append({
                    'sequence': seq,
                    'frame_idx': frame_idx
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = sample['sequence']
        frame_idx = sample['frame_idx']
        
        # Load LiDAR
        lidar_path = self.data_root / 'sequences' / seq / 'velodyne' / f'{frame_idx:06d}.bin'
        if lidar_path.exists():
            lidar_points = load_lidar_bin(str(lidar_path))
            lidar_bev = lidar_to_bev(
                lidar_points, 
                grid_size=self.grid_size,
                resolution=self.resolution,
                x_range=self.x_range,
                y_range=self.y_range
            )
        else:
            lidar_bev = np.zeros((3, *self.grid_size), dtype=np.float32)
        
        # Load trajectory history from poses
        poses_path = self.data_root / 'poses' / f'{seq}.txt'
        if poses_path.exists():
            poses = np.loadtxt(poses_path)
            
            # Extract past trajectory
            start_idx = max(0, frame_idx - self.past_frames * 10)
            past_poses = poses[start_idx:frame_idx+1]
            past_trajectory = poses_to_trajectory(
                past_poses, 
                num_frames=self.past_frames,
                spacing_meters=self.spacing_meters
            )
            
            # Convert to BEV
            history_bev = trajectory_to_bev(
                past_trajectory,
                grid_size=self.grid_size,
                resolution=self.resolution,
                x_range=self.x_range,
                y_range=self.y_range
            )
            
            # Extract future trajectory (target)
            future_poses = poses[frame_idx:frame_idx + self.future_frames * 10]
            future_trajectory = poses_to_trajectory(
                future_poses,
                num_frames=self.future_frames,
                spacing_meters=self.spacing_meters
            )
            
            # Convert to ego frame
            current_pose = poses[frame_idx].reshape(3, 4)
            current_R = current_pose[:, :3]
            current_t = current_pose[:, 3]
            
            future_trajectory_ego = []
            for i in range(len(future_trajectory)):
                # Convert from world to ego frame
                world_pos = np.array([future_trajectory[i][0], future_trajectory[i][1], 0])
                ego_pos = current_R.T @ (world_pos - current_t)
                future_trajectory_ego.append([ego_pos[0], ego_pos[1]])
            
            target_trajectory = np.array(future_trajectory_ego, dtype=np.float32)
            
        else:
            history_bev = np.zeros((1, *self.grid_size), dtype=np.float32)
            target_trajectory = np.zeros((self.future_frames, 2), dtype=np.float32)
        
        # Load OSM map (topometric route)
        # Currently uses trajectory as proxy - see OSM_INTEGRATION.md
        # For full OSM integration, download OXTS data and implement GPS alignment
        map_bev = history_bev.copy()
        
        # Concatenate inputs
        input_bev = create_input_tensor(lidar_bev, history_bev, map_bev)
        
        # Convert to tensors
        input_bev = torch.from_numpy(input_bev).float()
        target_trajectory = torch.from_numpy(target_trajectory).float()
        
        # Create ground truth road mask for auxiliary loss
        # In practice, this would come from OSM or annotated data
        road_mask = torch.from_numpy(map_bev).float()
        
        if self.transform:
            input_bev, target_trajectory, road_mask = self.transform(
                input_bev, target_trajectory, road_mask
            )
        
        return {
            'input_bev': input_bev,
            'target_trajectory': target_trajectory,
            'road_mask': road_mask,
            'sequence': seq,
            'frame_idx': frame_idx
        }


def get_dataloaders(data_root,
                    train_sequences=['00', '02', '05', '07'],
                    test_sequences=['08', '09', '10'],
                    batch_size=8,
                    num_workers=4,
                    **dataset_kwargs):
    """
    Create train and test dataloaders.
    
    Args:
        data_root: Path to KITTI dataset
        train_sequences: List of training sequence numbers
        test_sequences: List of test sequence numbers
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for KITTITrajectoryDataset
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = KITTITrajectoryDataset(
        data_root=data_root,
        sequences=train_sequences,
        **dataset_kwargs
    )
    
    test_dataset = KITTITrajectoryDataset(
        data_root=data_root,
        sequences=test_sequences,
        **dataset_kwargs
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    data_root = "/media/skr/storage/self_driving/TopoDiffuser/data/kitti"
    
    print("Testing KITTITrajectoryDataset...")
    
    # Create dataset for sequence 00
    dataset = KITTITrajectoryDataset(
        data_root=data_root,
        sequences=['00'],
        past_frames=5,
        future_frames=8,
        grid_size=(300, 400)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get first sample
        sample = dataset[0]
        
        print(f"\nSample keys: {sample.keys()}")
        print(f"Input BEV shape: {sample['input_bev'].shape}")
        print(f"Target trajectory shape: {sample['target_trajectory'].shape}")
        print(f"Road mask shape: {sample['road_mask'].shape}")
        print(f"Sequence: {sample['sequence']}, Frame: {sample['frame_idx']}")
        
        print("\nInput BEV channel statistics:")
        for i, name in enumerate(['Height', 'Intensity', 'Density', 'History', 'OSM']):
            channel = sample['input_bev'][i]
            print(f"  {name}: min={channel.min():.3f}, max={channel.max():.3f}, mean={channel.mean():.3f}")
        
        print("\nTarget trajectory:")
        print(f"  Shape: {sample['target_trajectory'].shape}")
        print(f"  First waypoint: {sample['target_trajectory'][0]}")
        print(f"  Last waypoint: {sample['target_trajectory'][-1]}")
    
    print("\nTest complete!")
