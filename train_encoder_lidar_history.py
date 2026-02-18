#!/usr/bin/env python3
"""
Train 4-Channel Encoder: LiDAR + Trajectory History (Modality 2)

Paper: "TopoDiffuser: A Diffusion-Based Multimodal Trajectory Prediction Model"
Section IV-B: Ablation Study - Phase 2

Channels:
  0-2: LiDAR (height, intensity, density)
  3: Trajectory history (5 keyframes, 2m spacing) - Modality 2

This corresponds to the "LiDAR + History" ablation in Table II.
Expected improvement: HD↓ 14.8% (0.27 → 0.23)

Usage:
    # From LiDAR-only checkpoint (warm start):
    python train_encoder_lidar_history.py \
        --init_from checkpoints/encoder_expanded_best.pth \
        --epochs 50
    
    # From scratch:
    python train_encoder_lidar_history.py --epochs 120
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

sys.path.insert(0, 'models')
sys.path.insert(0, 'utils')

from encoder import build_encoder


class KITTILiDARHistoryDataset(Dataset):
    """
    Dataset for 4-channel encoder: LiDAR + Trajectory History.
    
    Supports ablation: set use_history=False for LiDAR-only baseline.
    """
    
    def __init__(self, 
                 sequences=['00', '02', '05', '07'],
                 split='train',
                 data_root='data/kitti',
                 use_history=True,
                 past_frames=5,
                 future_frames=8,
                 spacing_meters=2.0,
                 grid_size=(300, 400),
                 resolution=0.1,
                 x_range=(-20, 20),
                 y_range=(-10, 30)):
        """
        Args:
            sequences: List of sequence numbers
            split: 'train', 'val', or 'all'
            data_root: Path to KITTI dataset
            use_history: Whether to use trajectory history (ablation support)
            past_frames: Number of past keyframes (paper: 5)
            future_frames: Number of future waypoints (paper: 8)
            spacing_meters: Spacing between keyframes (paper: 2.0)
        """
        self.data_root = Path(data_root)
        self.use_history = use_history
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.spacing_meters = spacing_meters
        self.grid_size = grid_size
        self.resolution = resolution
        self.x_range = x_range
        self.y_range = y_range
        
        # Build sample index
        self.samples = self._build_index(sequences, split)
        
        print(f"[{split}] Loaded {len(self.samples)} samples")
        print(f"  History enabled: {use_history}")
        print(f"  Past frames: {past_frames} @ {spacing_meters}m spacing")
        print(f"  Future frames: {future_frames}")
    
    def _build_index(self, sequences, split):
        """Build index of valid (sequence, frame) pairs."""
        samples = []
        
        for seq in sequences:
            pose_file = self.data_root / 'poses' / f'{seq}.txt'
            velodyne_dir = self.data_root / 'sequences' / seq / 'velodyne'
            
            if not pose_file.exists() or not velodyne_dir.exists():
                continue
            
            # Count available frames
            lidar_files = sorted(velodyne_dir.glob('*.bin'))
            num_frames = len(lidar_files)
            
            # Need: past_frames for history + future_frames for target
            min_required = self.past_frames * 10 + self.future_frames * 10 + 10
            
            valid_frames = range(min_required, num_frames - self.future_frames * 10)
            
            for frame_idx in valid_frames:
                samples.append({
                    'sequence': seq,
                    'frame_idx': frame_idx,
                    'lidar_path': lidar_files[frame_idx],
                    'pose_file': pose_file
                })
            
            print(f"  Sequence {seq}: {len(valid_frames)} valid frames")
        
        # Train/val split
        if split != 'all':
            n = len(samples)
            if split == 'train':
                samples = samples[:int(n * 0.8)]
            else:
                samples = samples[int(n * 0.8):]
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_lidar_bev(self, lidar_path: Path) -> np.ndarray:
        """Load and convert LiDAR to BEV (3 channels)."""
        # Load LiDAR points
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        # BEV rasterization
        H, W = self.grid_size
        bev = np.zeros((3, H, W), dtype=np.float32)
        
        # Filter points in range
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = (
            (x >= self.x_range[0]) & (x < self.x_range[1]) &
            (y >= self.y_range[0]) & (y < self.y_range[1]) &
            (z >= -3) & (z < 4)
        )
        points = points[mask]
        
        if len(points) == 0:
            return bev
        
        # Convert to pixel coordinates
        px = ((points[:, 0] - self.x_range[0]) / self.resolution).astype(int)
        py = ((points[:, 1] - self.y_range[0]) / self.resolution).astype(int)
        
        # Clip to bounds
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)
        
        # Channel 0: Max height
        for i in range(len(points)):
            bev[0, py[i], px[i]] = max(bev[0, py[i], px[i]], points[i, 2])
        
        # Channel 1: Mean intensity
        intensity_sums = {}
        intensity_counts = {}
        for i in range(len(points)):
            key = (py[i], px[i])
            if key not in intensity_sums:
                intensity_sums[key] = 0
                intensity_counts[key] = 0
            intensity_sums[key] += points[i, 3]
            intensity_counts[key] += 1
        
        for (py_idx, px_idx), total in intensity_sums.items():
            bev[1, py_idx, px_idx] = total / intensity_counts[(py_idx, px_idx)]
        
        # Channel 2: Density
        for i in range(len(points)):
            bev[2, py[i], px[i]] += 1
        
        # Normalize density
        if bev[2].max() > 0:
            bev[2] = np.clip(bev[2] / 10, 0, 1)
        
        return bev
    
    def _load_history_bev(self, poses: np.ndarray, frame_idx: int) -> np.ndarray:
        """Load and rasterize trajectory history (1 channel)."""
        H, W = self.grid_size
        
        if not self.use_history:
            return np.zeros((1, H, W), dtype=np.float32)
        
        # Extract past poses
        start_idx = max(0, frame_idx - self.past_frames * 10)
        past_poses = poses[start_idx:frame_idx+1]
        
        if len(past_poses) < 2:
            return np.zeros((1, H, W), dtype=np.float32)
        
        # Sample at spacing intervals (paper: 2m)
        trajectory = []
        positions = []
        
        for pose in past_poses:
            pose_mat = pose.reshape(3, 4)
            positions.append(pose_mat[:, 3])  # (x, y, z)
        
        positions = np.array(positions)
        
        # Sample keyframes at spacing intervals
        current_pos = positions[-1]  # Current position
        trajectory.insert(0, current_pos[:2])  # Add (x, y)
        
        dist_accum = 0
        for i in range(len(positions) - 2, -1, -1):
            dist = np.linalg.norm(positions[i] - positions[i + 1])
            dist_accum += dist
            
            if dist_accum >= self.spacing_meters and len(trajectory) < self.past_frames:
                trajectory.insert(0, positions[i][:2])
                dist_accum = 0
        
        # Pad if needed
        while len(trajectory) < self.past_frames:
            trajectory.insert(0, trajectory[0] if trajectory else np.array([0, 0]))
        
        trajectory = np.array(trajectory[-self.past_frames:])
        
        # Transform to ego frame
        current_pose_mat = poses[frame_idx].reshape(3, 4)
        current_R = current_pose_mat[:, :3]
        current_t = current_pose_mat[:, 3]
        
        trajectory_ego = []
        for pos in trajectory:
            pos_world = np.array([pos[0], pos[1], 0])
            pos_ego = current_R.T @ (pos_world - current_t)
            trajectory_ego.append([pos_ego[0], pos_ego[1]])
        
        trajectory_ego = np.array(trajectory_ego)
        
        # Rasterize to BEV binary mask
        bev = np.zeros((1, H, W), dtype=np.float32)
        
        def world_to_pixel(pt):
            px = int((pt[0] - self.x_range[0]) / self.resolution)
            py = int((pt[1] - self.y_range[0]) / self.resolution)
            return (px, py)
        
        # Draw trajectory lines
        for i in range(len(trajectory_ego) - 1):
            pt1 = world_to_pixel(trajectory_ego[i])
            pt2 = world_to_pixel(trajectory_ego[i + 1])
            
            # Bresenham's line algorithm
            x1, y1 = pt1
            x2, y2 = pt2
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            
            while True:
                # Draw with width
                for w in range(-1, 2):
                    for h in range(-1, 2):
                        px, py = x1 + w, y1 + h
                        if 0 <= px < W and 0 <= py < H:
                            bev[0, py, px] = 1.0
                
                if x1 == x2 and y1 == y2:
                    break
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
        
        return bev
    
    def _load_target_trajectory(self, poses: np.ndarray, frame_idx: int) -> np.ndarray:
        """Load future trajectory as prediction target."""
        # Extract future poses
        future_poses = poses[frame_idx:frame_idx + self.future_frames * 10]
        
        if len(future_poses) < 2:
            return np.zeros((self.future_frames, 2), dtype=np.float32)
        
        # Sample at spacing intervals
        current_pose_mat = poses[frame_idx].reshape(3, 4)
        current_pos = current_pose_mat[:, 3]
        current_R = current_pose_mat[:, :3]
        
        trajectory = []
        dist_accum = 0
        
        for i in range(1, len(future_poses)):
            pose_mat = future_poses[i].reshape(3, 4)
            pos = pose_mat[:, 3]
            
            dist = np.linalg.norm(pos - current_pos)
            dist_accum += dist
            
            if dist_accum >= self.spacing_meters and len(trajectory) < self.future_frames:
                # Convert to ego frame
                ego_pos = current_R.T @ (pos - current_pose_mat[:, 3])
                trajectory.append([ego_pos[0], ego_pos[1]])
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
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load LiDAR BEV (3 channels)
        lidar_bev = self._load_lidar_bev(sample['lidar_path'])
        
        # Load trajectory history (1 channel)
        poses = np.loadtxt(sample['pose_file'])
        frame_idx = sample['frame_idx']
        
        history_bev = self._load_history_bev(poses, frame_idx)
        
        # Concatenate: [4, H, W]
        input_bev = np.concatenate([lidar_bev, history_bev], axis=0)
        
        # Load target trajectory
        target_trajectory = self._load_target_trajectory(poses, frame_idx)
        
        # Convert to tensors
        input_bev = torch.from_numpy(input_bev).float()
        target_trajectory = torch.from_numpy(target_trajectory).float()
        
        return input_bev, target_trajectory


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = dataloader
    for batch_idx, (input_bev, target_traj) in enumerate(pbar):
        input_bev = input_bev.to(device)
        target_traj = target_traj.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(device_type='cuda', enabled=scaler.is_enabled()):
            # Forward pass
            output = model(input_bev)
            
            # For now, use simple MSE loss on conditioning vector
            # (Full diffusion loss will be added later)
            # This is a placeholder - the encoder training should use
            # the diffusion policy's loss
            
            # Flatten for loss computation
            target_flat = target_traj.view(target_traj.size(0), -1)
            
            # Simple reconstruction loss (placeholder)
            # In full implementation, this would be diffusion loss
            loss = torch.mean((output - output.mean()) ** 2)  # Dummy loss
        
        # Backward pass
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{num_batches}] Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_bev, target_traj in dataloader:
            input_bev = input_bev.to(device)
            target_traj = target_traj.to(device)
            
            with autocast(device_type='cuda'):
                output = model(input_bev)
                # Placeholder loss
                loss = torch.mean((output - output.mean()) ** 2)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train 4-channel encoder (LiDAR + History)")
    parser.add_argument("--data_root", type=str, default="data/kitti")
    parser.add_argument("--sequences", nargs="+", default=["00", "02", "05", "07"],
                        help="Training sequences")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--init_from", type=str, default=None,
                        help="Path to LiDAR-only checkpoint for warm start")
    parser.add_argument("--no_history", action="store_true",
                        help="Ablation: disable history (LiDAR-only)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Configuration: {'LiDAR-only' if args.no_history else 'LiDAR + History (4ch)'}")
    
    # Create datasets
    train_dataset = KITTILiDARHistoryDataset(
        sequences=args.sequences,
        split='train',
        data_root=args.data_root,
        use_history=not args.no_history
    )
    
    val_dataset = KITTILiDARHistoryDataset(
        sequences=args.sequences,
        split='val',
        data_root=args.data_root,
        use_history=not args.no_history
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Build model (4 channels)
    model = build_encoder(input_channels=4).to(device)
    
    # Load LiDAR-only weights for warm start
    if args.init_from and not args.no_history:
        print(f"Loading LiDAR weights from: {args.init_from}")
        checkpoint = torch.load(args.init_from, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter for encoder weights only (ignore heads if present)
        encoder_state = {k: v for k, v in state_dict.items() 
                        if k.startswith('encoder.') or not k.startswith(('road_head', 'cond_head'))}
        
        # Adapt first conv layer from 3ch to 4ch
        if 'encoder.conv1.conv.weight' in encoder_state:
            old_weight = encoder_state['encoder.conv1.conv.weight']  # [32, 3, 3, 3]
            new_weight = torch.zeros(32, 4, 3, 3, device=device)
            new_weight[:, :3, :, :] = old_weight
            # Initialize 4th channel with mean of first 3
            new_weight[:, 3, :, :] = old_weight.mean(dim=1)
            encoder_state['encoder.conv1.conv.weight'] = new_weight
            print("  Adapted conv1: 3ch -> 4ch")
        
        model.load_state_dict(encoder_state, strict=False)
        print("  Warm start complete")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # Training loop
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            suffix = "lidar_only" if args.no_history else "lidar_history"
            save_path = save_dir / f"encoder_{suffix}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'use_history': not args.no_history
            }, save_path)
            print(f"  Saved best model: {save_path}")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
