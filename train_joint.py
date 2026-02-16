#!/usr/bin/env python3
"""
Joint Training: Encoder + Diffusion with Combined Loss

Implements end-to-end training with:
- L_total = L_diffusion + α_road * L_road  (Equation 5)

The encoder is trained through both:
1. Road segmentation head (auxiliary task)
2. Conditioning vector → Diffusion policy (main task)

This provides multi-task learning that improves encoder representations.

Usage:
    python train_joint.py --alpha_road 0.1 --epochs 50
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
import argparse
from datetime import datetime

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from bev_rasterization import BEVRasterizer, load_kitti_lidar
from encoder import build_encoder
from diffusion import TrajectoryDiffusionModel
from denoising_network import build_denoising_network
from losses import TopoDiffuserLoss
from metrics import compute_trajectory_metrics, MetricsLogger


class KITTIJointDataset(Dataset):
    """
    KITTI Dataset for Joint Training.
    
    Returns:
        - bev: [3, 300, 400] - LiDAR BEV
        - trajectory: [8, 2] - Future trajectory (GT for diffusion)
        - road_mask: [1, 37, 50] - Road mask (GT for segmentation)
    """
    
    def __init__(self, sequences=['00'], split='train', 
                 data_root='/media/skr/storage/self_driving/TopoDiffuser/data/kitti',
                 num_future=8, waypoint_spacing=2.0):
        self.data_root = data_root
        self.rasterizer = BEVRasterizer()
        self.num_future = num_future
        self.waypoint_spacing = waypoint_spacing
        
        # Load samples
        self.samples = []
        for seq in sequences:
            self._load_sequence(seq)
        
        # Split
        if len(sequences) == 1 and split != 'all':
            n = len(self.samples)
            if split == 'train':
                self.samples = self.samples[:int(n * 0.8)]
            else:
                self.samples = self.samples[int(n * 0.8):]
        
        print(f"[{split}] {len(self.samples)} samples from {sequences}")
    
    def _load_sequence(self, sequence):
        """Load samples from sequence."""
        lidar_dir = os.path.join(self.data_root, 'sequences', sequence, 'velodyne')
        pose_file = os.path.join(self.data_root, 'poses', f'{sequence}.txt')
        
        if not os.path.exists(pose_file):
            return
        
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(np.array(values).reshape(3, 4))
        
        for frame_idx in range(len(poses) - self.num_future - 1):
            lidar_path = os.path.join(lidar_dir, f'{frame_idx:06d}.bin')
            if os.path.exists(lidar_path):
                self.samples.append({
                    'sequence': sequence,
                    'frame_idx': frame_idx,
                    'lidar_path': lidar_path,
                    'poses': poses
                })
    
    def _get_trajectory(self, poses, frame_idx):
        """Get future waypoints at 2m intervals."""
        current_pose = poses[frame_idx]
        current_x, current_y = current_pose[0, 3], current_pose[1, 3]
        
        trajectory = []
        for i in range(1, len(poses) - frame_idx):
            pose = poses[frame_idx + i]
            x, y = pose[0, 3], pose[1, 3]
            dist = np.sqrt((x - current_x)**2 + (y - current_y)**2)
            
            if dist >= self.waypoint_spacing * (len(trajectory) + 1):
                trajectory.append([x - current_x, y - current_y])
                if len(trajectory) >= self.num_future:
                    break
        
        while len(trajectory) < self.num_future:
            trajectory.append(trajectory[-1] if trajectory else [0.0, 0.0])
        
        return np.array(trajectory[:self.num_future], dtype=np.float32)
    
    def _create_road_mask(self, trajectory):
        """Create road mask from trajectory (same as encoder training)."""
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
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load LiDAR
        lidar_points = load_kitti_lidar(sample['lidar_path'])
        bev = self.rasterizer.rasterize_lidar(lidar_points)
        
        # Get trajectory
        trajectory = self._get_trajectory(sample['poses'], sample['frame_idx'])
        
        # Create road mask from trajectory
        road_mask = self._create_road_mask(trajectory)
        
        return (torch.from_numpy(bev), 
                torch.from_numpy(trajectory),
                torch.from_numpy(road_mask))


def train_epoch(encoder, diffusion_model, dataloader, criterion, optimizer, scaler, device, epoch):
    """Train one epoch with combined loss."""
    encoder.train()
    diffusion_model.train()
    
    total_loss = 0.0
    total_diff_loss = 0.0
    total_road_loss = 0.0
    num_batches = 0
    
    for batch_idx, (bev, trajectory, road_mask) in enumerate(dataloader):
        bev = bev.to(device, non_blocking=True)
        trajectory = trajectory.to(device, non_blocking=True)
        road_mask = road_mask.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            # Encoder forward: get conditioning and road segmentation
            conditioning, pred_road_mask = encoder(bev)  # [B, 512], [B, 1, 37, 50]
            
            # Diffusion forward
            # Sample timesteps and noise
            batch_size = trajectory.shape[0]
            t = diffusion_model.scheduler.sample_timesteps(batch_size)
            noise = torch.randn_like(trajectory)
            x_t, _ = diffusion_model.forward_diffusion(trajectory, t, noise)
            
            # Get timestep embeddings
            t_emb = diffusion_model.timestep_embedding(t)
            
            # Predict noise
            predicted_noise = diffusion_model.denoising_network(x_t, conditioning, t_emb)
            
            # Combined loss
            loss, loss_dict = criterion(predicted_noise, noise, pred_road_mask, road_mask)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss_dict['total']
        total_diff_loss += loss_dict['diffusion']
        total_road_loss += loss_dict['road']
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                  f"L_total={loss_dict['total']:.4f} "
                  f"(L_diff={loss_dict['diffusion']:.4f}, L_road={loss_dict['road']:.4f})")
    
    return {
        'total': total_loss / num_batches,
        'diffusion': total_diff_loss / num_batches,
        'road': total_road_loss / num_batches
    }


@torch.no_grad()
def validate(encoder, diffusion_model, dataloader, criterion, device):
    """Validation with combined loss and metrics."""
    encoder.eval()
    diffusion_model.eval()
    
    total_loss = 0.0
    metrics_logger = MetricsLogger()
    num_batches = 0
    
    for bev, trajectory, road_mask in dataloader:
        bev = bev.to(device, non_blocking=True)
        trajectory = trajectory.to(device, non_blocking=True)
        road_mask = road_mask.to(device, non_blocking=True)
        
        # Encoder
        conditioning, pred_road_mask = encoder(bev)
        
        # Diffusion
        batch_size = trajectory.shape[0]
        t = diffusion_model.scheduler.sample_timesteps(batch_size)
        noise = torch.randn_like(trajectory)
        x_t, _ = diffusion_model.forward_diffusion(trajectory, t, noise)
        t_emb = diffusion_model.timestep_embedding(t)
        predicted_noise = diffusion_model.denoising_network(x_t, conditioning, t_emb)
        
        # Loss
        loss, _ = criterion(predicted_noise, noise, pred_road_mask, road_mask)
        total_loss += loss.item()
        
        # Sample and compute trajectory metrics
        pred_trajectories = diffusion_model.sample(conditioning, num_samples=5)
        metrics = compute_trajectory_metrics(pred_trajectories, trajectory, threshold=2.0)
        metrics_logger.update(metrics, count=trajectory.shape[0])
        
        num_batches += 1
    
    return total_loss / num_batches, metrics_logger.get_averages()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha_road', type=float, default=0.1, 
                        help='Weight for road segmentation loss (Equation 5)')
    parser.add_argument('--sequences', type=str, nargs='+', default=['00'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--denoiser_arch', type=str, default='mlp')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    print("=" * 70)
    print("JOINT TRAINING: Encoder + Diffusion")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Loss: L_total = L_diffusion + {args.alpha_road} * L_road")
    print("=" * 70)
    
    # Models
    print("\nBuilding models...")
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    denoising_net = build_denoising_network(
        args.denoiser_arch, num_waypoints=8, coord_dim=2,
        conditioning_dim=512, timestep_dim=256
    ).to(device)
    diffusion_model = TrajectoryDiffusionModel(denoising_net, num_timesteps=10, device=device)
    
    enc_params = sum(p.numel() for p in encoder.parameters())
    diff_params = sum(p.numel() for p in denoising_net.parameters())
    print(f"  Encoder: {enc_params:,} params")
    print(f"  Denoiser: {diff_params:,} params")
    print(f"  Total: {enc_params + diff_params:,} params")
    
    # Loss
    criterion = TopoDiffuserLoss(alpha_road=args.alpha_road)
    
    # Optimizer (train both encoder and diffusion)
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(denoising_net.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Data
    print("\nLoading datasets...")
    train_dataset = KITTIJointDataset(sequences=args.sequences, split='train')
    val_dataset = KITTIJointDataset(sequences=args.sequences, split='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Training loop
    history = {
        'train_loss': [], 'train_diff': [], 'train_road': [],
        'val_loss': [], 'val_metrics': [], 'lr': []
    }
    best_minADE = float('inf')
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_losses = train_epoch(encoder, diffusion_model, train_loader, 
                                   criterion, optimizer, scaler, device, epoch)
        val_loss, val_metrics = validate(encoder, diffusion_model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_losses['total'])
        history['train_diff'].append(train_losses['diffusion'])
        history['train_road'].append(train_losses['road'])
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch [{epoch}/{args.epochs}] {epoch_time:.1f}s:")
        print(f"  Train: L_total={train_losses['total']:.4f} "
              f"(L_diff={train_losses['diffusion']:.4f}, L_road={train_losses['road']:.4f})")
        print(f"  Val: Loss={val_loss:.4f} | "
              f"minADE={val_metrics['minADE']:.3f}m | "
              f"minFDE={val_metrics['minFDE']:.3f}m | "
              f"HitRate={val_metrics['hit_rate']:.3f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best
        if val_metrics['minADE'] < best_minADE:
            best_minADE = val_metrics['minADE']
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'denoiser_state_dict': denoising_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, os.path.join(args.save_dir, 'joint_best.pth'))
            print(f"  ✓ Best saved (minADE: {best_minADE:.3f}m)")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'denoiser_state_dict': denoising_net.state_dict(),
            'history': history
        }, os.path.join(args.save_dir, 'joint_latest.pth'))
        
        with open(os.path.join(args.save_dir, 'joint_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"COMPLETE | Best minADE: {best_minADE:.3f}m")
    print("=" * 70)


if __name__ == "__main__":
    main()
