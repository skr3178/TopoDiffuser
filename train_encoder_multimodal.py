#!/usr/bin/env python3
"""
Train Full Multimodal Encoder (5-channel: LiDAR + History + OSM).

This is a separate training script that uses the new FullMultimodalEncoder
and KITTIFullMultimodalDataset, leaving the existing pipeline untouched.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'models')
sys.path.insert(0, 'utils')

from multimodal_encoder import build_full_multimodal_encoder
from dataset_multimodal import KITTIFullMultimodalDataset
from metrics import compute_trajectory_metrics, MetricsLogger


def combined_loss(conditioning_pred, conditioning_target, 
                  road_pred, road_target, 
                  lambda_road=0.5):
    """
    Combined loss for multimodal encoder.
    
    Args:
        conditioning_pred: Predicted conditioning [B, 512]
        conditioning_target: Target conditioning from trajectory [B, 512]
        road_pred: Predicted road segmentation [B, 1, H, W]
        road_target: Target road mask [B, 1, H, W]
        lambda_road: Weight for road segmentation loss
    """
    # Conditioning loss (MSE)
    loss_conditioning = nn.functional.mse_loss(conditioning_pred, conditioning_target)
    
    # Road segmentation loss (BCE with logits)
    loss_road = nn.functional.binary_cross_entropy_with_logits(
        road_pred, road_target
    )
    
    total_loss = loss_conditioning + lambda_road * loss_road
    
    return total_loss, {
        'conditioning': loss_conditioning.item(),
        'road': loss_road.item(),
        'total': total_loss.item()
    }


def train_epoch(encoder, dataloader, optimizer, scaler, device, epoch,
                lambda_road=0.5, max_grad_norm=1.0):
    """Train for one epoch."""
    encoder.train()
    total_loss = 0.0
    loss_breakdown = {'conditioning': 0.0, 'road': 0.0}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if len(batch) == 4:
            input_bev, target_traj, road_target, debug_info = batch
        else:
            input_bev, target_traj, road_target = batch
        
        input_bev = input_bev.to(device, non_blocking=True)
        target_traj = target_traj.to(device, non_blocking=True)
        road_target = road_target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            conditioning, road_pred = encoder(input_bev)
            
            # For conditioning target, we use a simple encoding of trajectory
            # In practice, this could be a pretrained trajectory encoder
            # For now, use MLP to encode target_traj to 512 dims
            target_traj_flat = target_traj.view(target_traj.size(0), -1)
            conditioning_target = torch.cat([
                target_traj_flat,
                torch.zeros(target_traj.size(0), 512 - target_traj_flat.size(1), 
                           device=device)
            ], dim=1)
            
            loss, loss_dict = combined_loss(
                conditioning, conditioning_target,
                road_pred, road_target,
                lambda_road
            )
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k in loss_breakdown:
            loss_breakdown[k] += loss_dict[k]
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(Cond: {loss_dict['conditioning']:.4f}, "
                  f"Road: {loss_dict['road']:.4f})")
    
    avg_loss = total_loss / num_batches
    avg_breakdown = {k: v / num_batches for k, v in loss_breakdown.items()}
    
    return avg_loss, avg_breakdown


@torch.no_grad()
def validate(encoder, dataloader, device, lambda_road=0.5):
    """Validate the encoder."""
    encoder.eval()
    total_loss = 0.0
    loss_breakdown = {'conditioning': 0.0, 'road': 0.0}
    metrics_logger = MetricsLogger()
    num_batches = 0
    
    for batch in dataloader:
        if len(batch) == 4:
            input_bev, target_traj, road_target, debug_info = batch
        else:
            input_bev, target_traj, road_target = batch
        
        input_bev = input_bev.to(device, non_blocking=True)
        target_traj = target_traj.to(device, non_blocking=True)
        road_target = road_target.to(device, non_blocking=True)
        
        conditioning, road_pred = encoder(input_bev)
        
        # Compute target conditioning
        target_traj_flat = target_traj.view(target_traj.size(0), -1)
        conditioning_target = torch.cat([
            target_traj_flat,
            torch.zeros(target_traj.size(0), 512 - target_traj_flat.size(1),
                       device=device)
        ], dim=1)
        
        loss, loss_dict = combined_loss(
            conditioning, conditioning_target,
            road_pred, road_target,
            lambda_road
        )
        
        total_loss += loss.item()
        for k in loss_breakdown:
            loss_breakdown[k] += loss_dict[k]
        
        # Compute trajectory metrics (dummy - actual diffusion model needed)
        # For now, just log that we processed the batch
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_breakdown = {k: v / num_batches for k, v in loss_breakdown.items()}
    
    return avg_loss, avg_breakdown, metrics_logger.get_averages()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Full Multimodal Encoder (5-channel)'
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data/kitti',
                       help='Path to KITTI odometry dataset')
    parser.add_argument('--raw_data_root', type=str, default='data/raw_data',
                       help='Path to KITTI raw data with OXTS')
    parser.add_argument('--train_sequences', type=str, nargs='+',
                       default=['00', '02', '05', '07'],
                       help='Training sequences')
    parser.add_argument('--val_sequences', type=str, nargs='+',
                       default=['08', '09', '10'],
                       help='Validation sequences')
    
    # Model arguments
    parser.add_argument('--input_channels', type=int, default=5,
                       help='Number of input channels (5 for multimodal)')
    parser.add_argument('--conditioning_dim', type=int, default=512,
                       help='Dimension of conditioning vector')
    parser.add_argument('--init_from_3ch', type=str, default=None,
                       help='Path to 3-channel encoder for weight initialization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_road', type=float, default=0.5,
                       help='Weight for road segmentation loss')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--viz_dir', type=str, default='viz/multimodal',
                       help='Directory to save visualizations')
    parser.add_argument('--enable_debug', action='store_true',
                       help='Enable debug mode with metrics')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    print("=" * 70)
    print("FULL MULTIMODAL ENCODER TRAINING")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Input channels: {args.input_channels}")
    print(f"Train sequences: {args.train_sequences}")
    print(f"Val sequences: {args.val_sequences}")
    print("=" * 70)
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_dataset = KITTIFullMultimodalDataset(
        data_root=args.data_root,
        sequences=args.train_sequences,
        raw_data_root=args.raw_data_root,
        enable_debug=args.enable_debug,
        viz_dir=args.viz_dir if args.enable_debug else None
    )
    
    val_dataset = KITTIFullMultimodalDataset(
        data_root=args.data_root,
        sequences=args.val_sequences,
        raw_data_root=args.raw_data_root,
        enable_debug=False  # No debug for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nBuilding model...")
    encoder = build_full_multimodal_encoder(
        input_channels=args.input_channels,
        conditioning_dim=args.conditioning_dim,
        init_from_3ch_path=args.init_from_3ch
    ).to(device)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_loss, train_breakdown = train_epoch(
            encoder, train_loader, optimizer, scaler, device, epoch,
            args.lambda_road
        )
        
        val_loss, val_breakdown, val_metrics = validate(
            encoder, val_loader, device, args.lambda_road
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} (Cond: {train_breakdown['conditioning']:.4f}, "
              f"Road: {train_breakdown['road']:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Cond: {val_breakdown['conditioning']:.4f}, "
              f"Road: {val_breakdown['road']:.4f})")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'encoder_full_multimodal_best.pth'))
            print(f"  Best saved (val_loss: {best_val_loss:.4f})")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': history,
            'args': vars(args)
        }, os.path.join(args.save_dir, 'encoder_full_multimodal_latest.pth'))
        
        # Save history
        with open(os.path.join(args.save_dir, 'encoder_multimodal_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"COMPLETE | Best val loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
