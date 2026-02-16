#!/usr/bin/env python3
"""
TopoDiffuser - Full Integrated Pipeline

Unified entry point for all operations:
- Training (joint or separate)
- Evaluation
- Inference

Usage:
    # Training with config
    python pipeline.py --mode train --config configs/lidar_only.yaml
    
    # Training with overrides
    python pipeline.py --mode train --config configs/default.yaml \
        --override training.batch_size=128 training.epochs=100
    
    # Evaluation
    python pipeline.py --mode eval --config configs/lidar_only.yaml \
        --checkpoint checkpoints/joint_best.pth
    
    # Inference
    python pipeline.py --mode infer --config configs/lidar_only.yaml \
        --checkpoint checkpoints/joint_best.pth \
        --sequence 00 --frame 1000
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))
sys.path.insert(0, str(project_root / "utils"))

from utils.config import load_config, override_config, validate_config, get_default_config
from models.bev_rasterization import BEVRasterizer, load_kitti_lidar
from models.encoder import build_encoder
from models.diffusion import TrajectoryDiffusionModel
from models.denoising_network import build_denoising_network
from models.losses import TopoDiffuserLoss, get_loss_function
from models.metrics import compute_trajectory_metrics, MetricsLogger


# =============================================================================
# Dataset
# =============================================================================

class KITTIDataset(torch.utils.data.Dataset):
    """
    Unified KITTI Dataset for TopoDiffuser.
    
    Supports both:
    - Encoder-only training (bev, road_mask)
    - Diffusion-only training (bev, trajectory)
    - Joint training (bev, trajectory, road_mask)
    """
    
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.data_root = config.paths.data_root
        self.rasterizer = BEVRasterizer()
        
        # Load sequences
        if split == 'train':
            sequences = config.data.train_sequences
        elif split == 'val':
            sequences = config.data.val_sequences
        else:  # test or all
            sequences = config.data.get(f"{split}_sequences", config.data.test_sequences)
        
        # Trajectory settings
        self.num_future = config.data.trajectory.num_future
        self.waypoint_spacing = config.data.trajectory.waypoint_spacing
        
        # Road mask settings
        self.mask_h = config.data.road_mask.height
        self.mask_w = config.data.road_mask.width
        self.mask_x_range = config.data.road_mask.x_range
        self.mask_y_range = config.data.road_mask.y_range
        self.mask_dilation = config.data.road_mask.dilation_iterations
        
        # Load all samples
        self.samples = []
        for seq in sequences:
            self._load_sequence(seq)
        
        print(f"[{split}] Loaded {len(self.samples)} samples from {sequences}")
    
    def _load_sequence(self, sequence):
        """Load samples from a sequence."""
        lidar_dir = Path(self.data_root) / 'sequences' / sequence / 'velodyne'
        pose_file = Path(self.data_root) / 'poses' / f'{sequence}.txt'
        
        if not pose_file.exists():
            return
        
        # Load poses
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(np.array(values).reshape(3, 4))
        
        # Create samples
        for frame_idx in range(len(poses) - self.num_future - 1):
            lidar_path = lidar_dir / f'{frame_idx:06d}.bin'
            if lidar_path.exists():
                self.samples.append({
                    'sequence': sequence,
                    'frame_idx': frame_idx,
                    'lidar_path': str(lidar_path),
                    'poses': poses
                })
    
    def _get_trajectory(self, poses, frame_idx):
        """Extract future trajectory at 2m intervals."""
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
        """Create binary road mask from trajectory."""
        mask = np.zeros((1, self.mask_h, self.mask_w), dtype=np.float32)
        
        if trajectory is None or len(trajectory) < 2:
            return mask
        
        res_x = (self.mask_x_range[1] - self.mask_x_range[0]) / self.mask_w
        res_y = (self.mask_y_range[1] - self.mask_y_range[0]) / self.mask_h
        
        def w2p(pt):
            px = int((pt[0] - self.mask_x_range[0]) / res_x)
            py = int((pt[1] - self.mask_y_range[0]) / res_y)
            return (max(0, min(self.mask_w-1, px)), max(0, min(self.mask_h-1, py)))
        
        # Draw trajectory on mask
        for i in range(len(trajectory) - 1):
            p1, p2 = w2p(trajectory[i]), w2p(trajectory[i + 1])
            
            # Bresenham's line algorithm
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
        mask[0] = binary_dilation(mask[0] > 0.5, iterations=self.mask_dilation).astype(np.float32)
        
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
        
        # Create road mask
        road_mask = self._create_road_mask(trajectory)
        
        return {
            'bev': torch.from_numpy(bev),
            'trajectory': torch.from_numpy(trajectory),
            'road_mask': torch.from_numpy(road_mask),
            'sequence': sample['sequence'],
            'frame_idx': sample['frame_idx']
        }


# =============================================================================
# Model Builder
# =============================================================================

class TopoDiffuserModel(nn.Module):
    """
    Complete TopoDiffuser model integrating all components.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = build_encoder(
            input_channels=config.model.encoder.input_channels,
            conditioning_dim=config.model.encoder.conditioning_dim
        )
        
        # Diffusion
        denoising_net = build_denoising_network(
            architecture=config.model.diffusion.denoising_network.architecture,
            num_waypoints=config.model.diffusion.denoising_network.num_waypoints,
            coord_dim=config.model.diffusion.denoising_network.coord_dim,
            conditioning_dim=config.model.diffusion.denoising_network.conditioning_dim,
            timestep_dim=config.model.diffusion.denoising_network.timestep_dim,
            hidden_dim=config.model.diffusion.denoising_network.hidden_dim,
            num_layers=config.model.diffusion.denoising_network.num_layers
        )
        
        self.diffusion = TrajectoryDiffusionModel(
            denoising_network=denoising_net,
            num_timesteps=config.model.diffusion.num_timesteps,
            beta_start=config.model.diffusion.beta_start,
            beta_end=config.model.diffusion.beta_end,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def forward(self, bev, trajectory=None, mode='full'):
        """
        Forward pass.
        
        Args:
            bev: [B, 3, 300, 400] - BEV input
            trajectory: [B, 8, 2] - Ground truth trajectory (for training)
            mode: 'full', 'encoder_only', 'diffusion_only'
            
        Returns:
            Dict with outputs depending on mode
        """
        # Encoder
        conditioning, road_mask_pred = self.encoder(bev)
        
        outputs = {
            'conditioning': conditioning,
            'road_mask_pred': road_mask_pred
        }
        
        if mode == 'encoder_only':
            return outputs
        
        # Diffusion training (if trajectory provided)
        if trajectory is not None:
            batch_size = trajectory.shape[0]
            t = self.diffusion.scheduler.sample_timesteps(batch_size)
            noise = torch.randn_like(trajectory)
            x_t, _ = self.diffusion.forward_diffusion(trajectory, t, noise)
            t_emb = self.diffusion.timestep_embedding(t)
            predicted_noise = self.diffusion.denoising_network(x_t, conditioning, t_emb)
            
            outputs.update({
                'predicted_noise': predicted_noise,
                'target_noise': noise,
                'timesteps': t
            })
        
        return outputs
    
    @torch.no_grad()
    def sample(self, bev, num_samples=5):
        """
        Generate trajectory samples.
        
        Args:
            bev: [B, 3, 300, 400]
            num_samples: K trajectory samples
            
        Returns:
            trajectories: [B, K, 8, 2]
        """
        self.eval()
        conditioning, _ = self.encoder(bev)
        return self.diffusion.sample(conditioning, num_samples=num_samples)


def build_model(config):
    """Build TopoDiffuser model from config."""
    return TopoDiffuserModel(config)


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Unified trainer for TopoDiffuser.
    
    Supports:
    - Joint training (encoder + diffusion)
    - Encoder-only training
    - Diffusion-only training (frozen encoder)
    """
    
    def __init__(self, config, model, device='cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = TopoDiffuserLoss(
            alpha_road=config.training.loss.alpha_road
        )
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler() if config.training.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_diff': [], 'train_road': [],
            'val_metrics': [], 'lr': []
        }
    
    def _build_optimizer(self):
        """Build optimizer based on training mode."""
        mode = self.config.training.mode
        lr = self.config.training.optimizer.lr
        weight_decay = self.config.training.optimizer.weight_decay
        
        if mode == 'joint':
            # Train everything
            params = list(self.model.encoder.parameters()) + \
                     list(self.model.diffusion.denoising_network.parameters())
        elif mode == 'encoder_only':
            # Train only encoder
            params = self.model.encoder.parameters()
        elif mode == 'diffusion_only':
            # Train only diffusion (encoder frozen)
            params = self.model.diffusion.denoising_network.parameters()
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_type = self.config.training.scheduler.type
        
        if scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.scheduler.T_max,
                eta_min=self.config.training.scheduler.eta_min
            )
        else:
            return None
    
    def train_epoch(self, dataloader):
        """Train one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_diff = 0.0
        total_road = 0.0
        num_batches = 0
        
        for batch in dataloader:
            bev = batch['bev'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            road_mask = batch['road_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward
            if self.config.training.mixed_precision and self.scaler:
                with autocast('cuda'):
                    outputs = self.model(bev, trajectory, mode='full')
                
                # Compute loss outside autocast (BCE + Sigmoid incompatible with autocast)
                loss, loss_dict = self.criterion(
                    outputs['predicted_noise'],
                    outputs['target_noise'],
                    outputs['road_mask_pred'],
                    road_mask
                )
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(bev, trajectory, mode='full')
                loss, loss_dict = self.criterion(
                    outputs['predicted_noise'],
                    outputs['target_noise'],
                    outputs['road_mask_pred'],
                    road_mask
                )
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss_dict['total']
            total_diff += loss_dict['diffusion']
            total_road += loss_dict['road']
            num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'diffusion': total_diff / num_batches,
            'road': total_road / num_batches
        }
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate and compute metrics."""
        self.model.eval()
        
        total_loss = 0.0
        metrics_logger = MetricsLogger()
        num_batches = 0
        
        for batch in dataloader:
            bev = batch['bev'].to(self.device)
            trajectory = batch['trajectory'].to(self.device)
            road_mask = batch['road_mask'].to(self.device)
            
            # Forward
            outputs = self.model(bev, trajectory, mode='full')
            loss, _ = self.criterion(
                outputs['predicted_noise'],
                outputs['target_noise'],
                outputs['road_mask_pred'],
                road_mask
            )
            total_loss += loss.item()
            
            # Sample and compute metrics
            pred_trajectories = self.model.sample(bev, num_samples=self.config.validation.num_samples)
            metrics = compute_trajectory_metrics(
                pred_trajectories, trajectory,
                threshold=self.config.validation.hitrate_threshold
            )
            metrics_logger.update(metrics, count=trajectory.shape[0])
            
            num_batches += 1
        
        return total_loss / num_batches, metrics_logger.get_averages()
    
    def train(self, train_loader, val_loader):
        """Full training loop."""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print(f"Mode: {self.config.training.mode}")
        print(f"Epochs: {self.config.training.epochs}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        for epoch in range(1, self.config.training.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            if epoch % self.config.validation.frequency == 0:
                val_loss, val_metrics = self.validate(val_loader)
            else:
                val_loss, val_metrics = 0.0, {}
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            self.history['train_loss'].append(train_losses['total'])
            self.history['train_diff'].append(train_losses['diffusion'])
            self.history['train_road'].append(train_losses['road'])
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Print
            print(f"\nEpoch [{epoch}/{self.config.training.epochs}] {epoch_time:.1f}s:")
            print(f"  Train: L={train_losses['total']:.4f} "
                  f"(L_diff={train_losses['diffusion']:.4f}, L_road={train_losses['road']:.4f})")
            if val_metrics:
                print(f"  Val: L={val_loss:.4f} | "
                      f"minADE={val_metrics.get('minADE', 0):.3f}m | "
                      f"HitRate={val_metrics.get('hit_rate', 0):.3f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Best minADE: {self.best_metric:.3f}m")
        print("=" * 70)
    
    def _save_checkpoint(self, epoch, val_metrics):
        """Save checkpoint."""
        save_dir = Path(self.config.paths.checkpoint_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history
        }
        
        # Save latest
        torch.save(checkpoint, save_dir / 'latest.pth')
        
        # Save best (based on minADE)
        if val_metrics and val_metrics.get('minADE', float('inf')) < self.best_metric:
            self.best_metric = val_metrics['minADE']
            torch.save(checkpoint, save_dir / 'best.pth')
            print(f"  ✓ Best saved (minADE: {self.best_metric:.3f}m)")
        
        # Save periodic
        if epoch % self.config.training.checkpoint.save_every == 0:
            torch.save(checkpoint, save_dir / f'epoch_{epoch:03d}.pth')
        
        # Save history
        with open(save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TopoDiffuser Pipeline')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval', 'infer', 'debug'],
                        help='Operation mode')
    
    # Config
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--override', type=str, nargs='*',
                        help='Override config values (e.g., training.batch_size=64)')
    
    # Checkpoint (for eval/infer)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for eval/infer')
    
    # Inference args
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--frame', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Apply overrides
    if args.override:
        overrides = {}
        for o in args.override:
            if '=' in o:
                k, v = o.split('=', 1)
                # Handle boolean strings
                if v.lower() == 'true':
                    v = True
                elif v.lower() == 'false':
                    v = False
                else:
                    try:
                        v = eval(v)
                    except:
                        pass
                overrides[k] = v
        config = override_config(config, overrides)
    
    # Validate
    validate_config(config)
    
    print(f"Config: {config.project.name}")
    print(f"Mode: {args.mode}")
    
    # Device
    device = torch.device(config.hardware.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Execute mode
    if args.mode == 'train':
        # Create datasets
        train_dataset = KITTIDataset(config, 'train')
        val_dataset = KITTIDataset(config, 'val')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=config.training.persistent_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.validation.batch_size,
            shuffle=False,
            num_workers=config.validation.num_workers,
            pin_memory=config.training.pin_memory
        )
        
        # Build model
        model = build_model(config)
        
        # Create trainer
        trainer = Trainer(config, model, device)
        
        # Train
        trainer.train(train_loader, val_loader)
        
    elif args.mode == 'eval':
        raise NotImplementedError("Evaluation mode coming soon")
        
    elif args.mode == 'infer':
        raise NotImplementedError("Inference mode coming soon")
        
    elif args.mode == 'debug':
        # Quick test with minimal data
        print("\nRunning debug mode...")
        config.training.epochs = 2
        config.data.train_sequences = ['00']
        config.data.val_sequences = ['00']
        
        train_dataset = KITTIDataset(config, 'train')
        val_dataset = KITTIDataset(config, 'val')
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        model = build_model(config)
        trainer = Trainer(config, model, device)
        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
