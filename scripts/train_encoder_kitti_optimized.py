#!/usr/bin/env python3
"""
Optimized Training: Maximize GPU Utilization

Uses:
- Larger batch size
- Mixed precision (FP16)
- More data loader workers
- Pin memory and non-blocking transfers
- Gradient accumulation for effective larger batches

Usage:
    nohup python train_encoder_kitti_optimized.py > training_gpu_max.log 2>&1 &
    watch -n 1 nvidia-smi
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
import json
from datetime import datetime
import argparse

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from models.bev_rasterization import BEVRasterizer, load_kitti_lidar
from models.encoder import build_encoder


class KITTITrajectoryDataset(Dataset):
    """KITTI Dataset with caching."""
    
    def __init__(self, sequence='00', split='train', data_root='/media/skr/storage/self_driving/TopoDiffuser/data/kitti', cache_size=1000):
        self.sequence = sequence
        self.data_root = data_root
        self.rasterizer = BEVRasterizer()
        self.cache = {}
        self.cache_size = cache_size
        
        self.lidar_dir = os.path.join(data_root, 'sequences', sequence, 'velodyne')
        self.pose_file = os.path.join(data_root, 'poses', f'{sequence}.txt')
        
        self.poses = self._load_poses()
        
        total_frames = len(self.poses)
        if split == 'train':
            self.frame_indices = list(range(0, int(total_frames * 0.8)))
        else:
            self.frame_indices = list(range(int(total_frames * 0.8), total_frames))
        
        print(f"[{split}] {len(self.frame_indices)} frames")
    
    def _load_poses(self):
        poses = []
        with open(self.pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(values)
        return np.array(poses)
    
    def _get_future_trajectory(self, frame_idx, num_future=8):
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
        H, W = 37, 50
        mask = np.zeros((1, H, W), dtype=np.float32)
        
        if trajectory is None or len(trajectory) < 2:
            return mask
        
        x_range, y_range = (-20, 20), (-10, 30)
        res_x = (x_range[1] - x_range[0]) / W
        res_y = (y_range[1] - y_range[0]) / H
        
        for i in range(len(trajectory) - 1):
            pt1, pt2 = trajectory[i], trajectory[i + 1]
            
            def w2p(pt):
                px = int((pt[0] - x_range[0]) / res_x)
                py = int((pt[1] - y_range[0]) / res_y)
                return (max(0, min(W-1, px)), max(0, min(H-1, py)))
            
            p1, p2 = w2p(pt1), w2p(pt2)
            
            # Simple line drawing
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
        
        # Check cache
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        
        lidar_path = os.path.join(self.lidar_dir, f'{frame_idx:06d}.bin')
        if not os.path.exists(lidar_path):
            bev = np.zeros((3, 300, 400), dtype=np.float32)
            mask = np.zeros((1, 37, 50), dtype=np.float32)
            return torch.from_numpy(bev), torch.from_numpy(mask)
        
        lidar_points = load_kitti_lidar(lidar_path)
        bev = self.rasterizer.rasterize_lidar(lidar_points)
        
        trajectory = self._get_future_trajectory(frame_idx)
        mask = self._create_gt_mask(trajectory)
        
        bev_tensor = torch.from_numpy(bev)
        mask_tensor = torch.from_numpy(mask)
        
        # Cache
        if len(self.cache) < self.cache_size:
            self.cache[frame_idx] = (bev_tensor, mask_tensor)
        
        return bev_tensor, mask_tensor


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, grad_accum_steps=1):
    """Train with mixed precision and gradient accumulation."""
    model.train()
    losses = AverageMeter()
    
    optimizer.zero_grad()
    start_time = time.time()
    
    for i, (bev, gt_mask) in enumerate(dataloader):
        bev = bev.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True)
        
        # Mixed precision forward (only model forward, not loss)
        with autocast('cuda'):
            conditioning, pred_mask = model(bev)
        
        # Loss outside autocast (BCELoss + Sigmoid incompatible with autocast)
        loss = criterion(pred_mask, gt_mask)
        loss = loss / grad_accum_steps  # Normalize for accumulation
        
        # Backward with scaling
        scaler.scale(loss).backward()
        
        # Update weights every grad_accum_steps
        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        losses.update(loss.item() * grad_accum_steps, bev.size(0))
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (i + 1) * bev.size(0) / elapsed
            print(f"  [{epoch}][{i+1}/{len(dataloader)}] "
                  f"Loss: {losses.avg:.4f} "
                  f"({samples_per_sec:.1f} samples/s)", flush=True)
    
    return losses.avg


def validate(model, dataloader, criterion, device):
    model.eval()
    losses = AverageMeter()
    
    with torch.no_grad():
        for bev, gt_mask in dataloader:
            bev = bev.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)
            
            with autocast('cuda'):
                conditioning, pred_mask = model(bev)
            loss = criterion(pred_mask, gt_mask)
            
            losses.update(loss.item(), bev.size(0))
    
    return losses.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)  # MAX GPU: 256 fits in ~5GB
    parser.add_argument('--lr', type=float, default=2e-3)  # Higher LR for massive batch
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--workers', type=int, default=8)  # More workers
    parser.add_argument('--grad_accum', type=int, default=1)  # No accum needed with 256 batch
    parser.add_argument('--save_dir', type=str, default='/media/skr/storage/self_driving/TopoDiffuser/checkpoints')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("="*70)
    print("OPTIMIZED KITTI TRAINING - GPU MAX UTILIZATION")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum: {args.grad_accum} (effective batch: {args.batch_size * args.grad_accum})")
    print(f"Workers: {args.workers}")
    print(f"Mixed precision: FP16")
    print("="*70, flush=True)
    
    # Model
    model = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data with optimized settings
    print("\nLoading datasets...")
    train_dataset = KITTITrajectoryDataset(sequence=args.sequence, split='train')
    val_dataset = KITTITrajectoryDataset(sequence=args.sequence, split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size * 2,  # Larger batch for val
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    effective_batch = args.batch_size * args.grad_accum
    print(f"Train: {len(train_loader)} batches (eff. batch: {effective_batch})")
    print(f"Val: {len(val_loader)} batches")
    
    # Training
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70, flush=True)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                 scaler, device, epoch, args.grad_accum)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch [{epoch}/{args.epochs}] {epoch_time:.1f}s:")
        print(f"  Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, os.path.join(args.save_dir, 'encoder_best.pth'))
            print(f"  ✓ Best model saved")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history
        }, os.path.join(args.save_dir, 'encoder_latest.pth'))
        
        with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best val: {best_val_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
