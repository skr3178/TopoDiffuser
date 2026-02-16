#!/usr/bin/env python3
"""
Full Mini Training: Encoder on Real KITTI Dataset

Trains the encoder segmentation head (Head A) using BCE loss
on real KITTI data with proper data loading and logging.

Usage:
    nohup python train_encoder_kitti.py > training.log 2>&1 &
    tail -f training.log
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
import argparse

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from models.bev_rasterization import BEVRasterizer, load_kitti_lidar
from models.encoder import build_encoder


class KITTITrajectoryDataset(Dataset):
    """KITTI Dataset for trajectory prediction."""
    
    def __init__(self, sequence='00', split='train', data_root='/media/skr/storage/self_driving/TopoDiffuser/data/kitti'):
        self.sequence = sequence
        self.data_root = data_root
        self.rasterizer = BEVRasterizer()
        
        # Paths
        self.lidar_dir = os.path.join(data_root, 'sequences', sequence, 'velodyne')
        self.pose_file = os.path.join(data_root, 'poses', f'{sequence}.txt')
        
        # Load poses
        self.poses = self._load_poses()
        
        # Determine split
        total_frames = len(self.poses)
        if split == 'train':
            self.frame_indices = list(range(0, int(total_frames * 0.8)))
        else:
            self.frame_indices = list(range(int(total_frames * 0.8), total_frames))
        
        print(f"[{split}] Loaded {len(self.frame_indices)} frames from sequence {sequence}")
    
    def _load_poses(self):
        """Load KITTI poses."""
        poses = []
        with open(self.pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(values)
        return np.array(poses)
    
    def _get_future_trajectory(self, frame_idx, num_future=8):
        """Extract future trajectory relative to current frame."""
        if frame_idx + num_future >= len(self.poses):
            return None
        
        current_pose = self.poses[frame_idx]
        current_x, current_y = current_pose[3], current_pose[7]
        
        trajectory = []
        for i in range(1, num_future + 1):
            pose = self.poses[frame_idx + i]
            x, y = pose[3], pose[7]
            trajectory.append([x - current_x, y - current_y])
        
        return np.array(trajectory, dtype=np.float32)
    
    def _create_gt_mask(self, trajectory):
        """Create ground truth road mask from trajectory."""
        H, W = 37, 50
        mask = np.zeros((1, H, W), dtype=np.float32)
        
        if trajectory is None or len(trajectory) < 2:
            return mask
        
        x_range, y_range = (-20, 20), (-10, 30)
        res_x = (x_range[1] - x_range[0]) / W
        res_y = (y_range[1] - y_range[0]) / H
        
        # Draw trajectory line
        for i in range(len(trajectory) - 1):
            pt1, pt2 = trajectory[i], trajectory[i + 1]
            
            def w2p(pt):
                px = int((pt[0] - x_range[0]) / res_x)
                py = int((pt[1] - y_range[0]) / res_y)
                return (max(0, min(W-1, px)), max(0, min(H-1, py)))
            
            p1, p2 = w2p(pt1), w2p(pt2)
            
            # Bresenham line
            dx, dy = abs(p2[0] - p1[0]), abs(p2[1] - p1[1])
            sx = 1 if p1[0] < p2[0] else -1
            sy = 1 if p1[1] < p2[1] else -1
            err = dx - dy
            
            x, y = p1[0], p1[1]
            while True:
                mask[0, y, x] = 1.0
                if (x, y) == p2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
        
        # Dilate
        from scipy.ndimage import binary_dilation
        mask[0] = binary_dilation(mask[0] > 0.5, iterations=2).astype(np.float32)
        
        return mask
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]
        
        # Load LiDAR
        lidar_path = os.path.join(self.lidar_dir, f'{frame_idx:06d}.bin')
        if not os.path.exists(lidar_path):
            # Return dummy data if file missing
            bev = np.zeros((3, 300, 400), dtype=np.float32)
            mask = np.zeros((1, 37, 50), dtype=np.float32)
            return torch.from_numpy(bev), torch.from_numpy(mask)
        
        lidar_points = load_kitti_lidar(lidar_path)
        bev = self.rasterizer.rasterize_lidar(lidar_points)
        
        # Get trajectory and create mask
        trajectory = self._get_future_trajectory(frame_idx)
        mask = self._create_gt_mask(trajectory)
        
        return torch.from_numpy(bev), torch.from_numpy(mask)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    losses = AverageMeter()
    
    start_time = time.time()
    
    for i, (bev, gt_mask) in enumerate(dataloader):
        bev = bev.to(device)
        gt_mask = gt_mask.to(device)
        
        # Forward
        optimizer.zero_grad()
        conditioning, pred_mask = model(bev)
        
        # Compute BCE loss (Head A only for now)
        loss = criterion(pred_mask, gt_mask)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Record
        losses.update(loss.item(), bev.size(0))
        
        # Print progress every 50 batches
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch [{epoch}][{i+1}/{len(dataloader)}] "
                  f"Loss: {losses.avg:.4f} "
                  f"({elapsed:.1f}s)", flush=True)
    
    return losses.avg


def validate(model, dataloader, criterion, device):
    """Validate."""
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for bev, gt_mask in dataloader:
            bev = bev.to(device)
            gt_mask = gt_mask.to(device)
            
            conditioning, pred_mask = model(bev)
            loss = criterion(pred_mask, gt_mask)
            
            losses.update(loss.item(), bev.size(0))
    
    return losses.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--save_dir', type=str, default='/media/skr/storage/self_driving/TopoDiffuser/checkpoints')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("KITTI ENCODER TRAINING")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Sequence: {args.sequence}")
    print("="*70, flush=True)
    
    # Model
    model = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    print("\nLoading datasets...")
    train_dataset = KITTITrajectoryDataset(sequence=args.sequence, split='train')
    val_dataset = KITTITrajectoryDataset(sequence=args.sequence, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70, flush=True)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print summary
        print(f"\nEpoch [{epoch}/{args.epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time:.1f}s", flush=True)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'encoder_best.pth'))
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save latest
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'encoder_latest.pth'))
        
        # Save history
        with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
