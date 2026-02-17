#!/usr/bin/env python3
"""
Resume diffusion training from checkpoint.
Continues training the denoiser with frozen encoder.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder import build_encoder
from models.diffusion import TrajectoryDiffusionModel
from models.metrics import compute_trajectory_metrics, aggregate_metrics, MetricsLogger
from pipeline import DiffusionConfig, LiDARDataset, TrajectoryScaler


def precompute_conditioning(encoder, dataset, device, batch_size=16):
    """Precompute conditioning vectors (same as original)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    
    all_conditioning = []
    all_trajectories = []
    
    print(f"  Precomputing conditioning vectors (one-time cost)...")
    total_samples = len(dataset)
    processed = 0
    start_time = datetime.now()
    
    encoder.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            lidar = batch['lidar'].to(device)
            history = batch['history'].to(device)
            
            # Forward through encoder
            conditioning = encoder(lidar, history)
            
            all_conditioning.append(conditioning.cpu())
            all_trajectories.append(batch['trajectory'])
            
            processed += lidar.size(0)
            if batch_idx % 10 == 0 or processed >= total_samples:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_samples - processed) / rate if rate > 0 else 0
                print(f"    [{processed}/{total_samples}] {rate:.0f} samples/s, ETA {eta:.0f}s")
    
    conditioning = torch.cat(all_conditioning, dim=0)
    trajectories = torch.cat(all_trajectories, dim=0)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  Precompute done: {len(conditioning)} samples in {elapsed:.1f}s ({len(conditioning)/elapsed:.0f} samples/s)")
    
    return conditioning, trajectories


class PrecomputedDataset(torch.utils.data.Dataset):
    """Dataset with precomputed conditioning."""
    def __init__(self, conditioning, trajectories):
        self.conditioning = conditioning
        self.trajectories = trajectories
        
    def __len__(self):
        return len(self.conditioning)
    
    def __getitem__(self, idx):
        return {
            'conditioning': self.conditioning[idx],
            'trajectory': self.trajectories[idx]
        }


def train_epoch(denoising_net, train_loader, optimizer, device, history=None):
    """Train for one epoch."""
    denoising_net.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        conditioning = batch['conditioning'].to(device)
        trajectory_gt = batch['trajectory'].to(device)
        
        optimizer.zero_grad()
        loss = DiffusionLoss(denoising_net, trajectory_gt, conditioning)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoising_net.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 49:
            avg_loss = total_loss / num_batches
            print(f"    [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}")
    
    return total_loss / num_batches


@torch.no_grad()
def validate(denoising_net, val_loader, scaler, config, device, num_samples=5):
    """Validate and compute metrics."""
    denoising_net.eval()
    total_loss = 0
    num_batches = 0
    metrics_logger = MetricsLogger()
    
    for batch in val_loader:
        conditioning = batch['conditioning'].to(device)
        trajectory_gt = batch['trajectory'].to(device)
        
        # Loss
        loss = DiffusionLoss(denoising_net, trajectory_gt, conditioning)
        total_loss += loss.item()
        num_batches += 1
        
        # Sample trajectories for metrics
        B = trajectory_gt.size(0)
        for i in range(B):
            cond_single = conditioning[i:i+1]
            gt_single = trajectory_gt[i:i+1]
            
            # Sample K trajectories
            pred_trajs = []
            for _ in range(num_samples):
                pred = denoising_net.sample(cond_single)
                pred_trajs.append(pred)
            pred_trajs = torch.cat(pred_trajs, dim=0).unsqueeze(0)  # [1, K, N, 2]
            
            # Compute metrics
            metrics = compute_trajectory_metrics(pred_trajs, gt_single, threshold=2.0)
            metrics_logger.update(metrics)
    
    avg_loss = total_loss / num_batches
    avg_metrics = metrics_logger.get_averages()
    
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Resume diffusion training')
    parser.add_argument('--encoder_ckpt', type=str, default='checkpoints/encoder_expanded_best.pth')
    parser.add_argument('--resume_ckpt', type=str, default='checkpoints/diffusion_unet_latest.pth',
                        help='Checkpoint to resume from')
    parser.add_argument('--additional_epochs', type=int, default=100,
                        help='Additional epochs to train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--train_sequences', type=str, nargs='+',
                        default=['00', '02', '05', '07'])
    parser.add_argument('--val_sequences', type=str, nargs='+', default=['08'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--cache_dir', type=str, default='checkpoints/cache_paper_split')
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--precompute_batch', type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    print("=" * 70)
    print("RESUMING DIFFUSION TRAINING (Frozen Encoder)")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Resume from: {args.resume_ckpt}")
    print("=" * 70)
    
    # Load checkpoint to resume
    start_epoch = 1
    best_minADE = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    if os.path.exists(args.resume_ckpt):
        print(f"\nLoading checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        start_epoch = ckpt.get('epoch', 0) + 1
        history = ckpt.get('history', history)
        
        # Find best minADE from history
        if history['val_metrics']:
            best_minADE = min([m.get('minADE', float('inf')) for m in history['val_metrics']])
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Previous best minADE: {best_minADE:.3f}m")
    else:
        print(f"\nWarning: Checkpoint not found at {args.resume_ckpt}")
        print("  Starting from scratch...")
    
    # Load frozen encoder
    print("\nLoading encoder (frozen)...")
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        checkpoint = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"  Loaded from {args.encoder_ckpt}")
    else:
        print(f"  Warning: Encoder checkpoint not found!")
    
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    print(f"  Encoder frozen ({sum(p.numel() for p in encoder.parameters()):,} params)")
    
    # Setup scaler and config
    scaler = TrajectoryScaler()
    config = DiffusionConfig()
    config.noise_schedule = args.noise_schedule
    
    # Load datasets
    print("\nLoading datasets...")
    train_datasets = []
    for seq in args.train_sequences:
        ds = LiDARDataset([seq], scaler=scaler)
        train_datasets.append(ds)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    
    val_datasets = []
    for seq in args.val_sequences:
        ds = LiDARDataset([seq], scaler=scaler)
        val_datasets.append(ds)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    # Precompute or load conditioning
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Train cache
    train_cache = os.path.join(args.cache_dir, f'cached_cond_train_{"_".join(args.train_sequences)}.pt')
    if os.path.exists(train_cache):
        print(f"  Loading cached train conditioning from {train_cache}")
        train_data = torch.load(train_cache, weights_only=False)
        train_cond, train_traj = train_data['conditioning'], train_data['trajectories']
    else:
        train_cond, train_traj = precompute_conditioning(encoder, train_dataset, device, args.precompute_batch)
        torch.save({'conditioning': train_cond, 'trajectories': train_traj}, train_cache)
    
    # Val cache
    val_cache = os.path.join(args.cache_dir, f'cached_cond_val_{"_".join(args.val_sequences)}.pt')
    if os.path.exists(val_cache):
        print(f"  Loading cached val conditioning from {val_cache}")
        val_data = torch.load(val_cache, weights_only=False)
        val_cond, val_traj = val_data['conditioning'], val_data['trajectories']
    else:
        val_cond, val_traj = precompute_conditioning(encoder, val_dataset, device, args.precompute_batch)
        torch.save({'conditioning': val_cond, 'trajectories': val_traj}, val_cache)
    
    del encoder  # Free memory
    torch.cuda.empty_cache()
    
    print(f"\nTrain: {len(train_cond)} samples")
    print(f"Val:   {len(val_cond)} samples")
    
    # Create precomputed datasets
    train_dataset_pre = PrecomputedDataset(train_cond, train_traj)
    val_dataset_pre = PrecomputedDataset(val_cond, val_traj)
    
    train_loader = DataLoader(train_dataset_pre, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset_pre, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    
    # Build denoising network
    print("\nBuilding diffusion model...")
    denoising_net = TrajectoryDiffusionModel(
        trajectory_dim=16,
        conditioning_dim=512,
        time_emb_dim=128,
        hidden_dim=256
    ).to(device)
    
    total_params = sum(p.numel() for p in denoising_net.parameters())
    print(f"  Denoiser: {total_params:,} params")
    
    # Load denoiser weights from checkpoint
    if os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        denoising_net.load_state_dict(ckpt['denoiser_state_dict'])
        print(f"  Loaded denoiser weights from checkpoint")
    
    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(denoising_net.parameters(), lr=args.lr)
    
    # Calculate total epochs
    total_epochs = start_epoch + args.additional_epochs - 1
    print(f"\nTraining from epoch {start_epoch} to {total_epochs}")
    print(f"  Additional epochs: {args.additional_epochs}")
    print(f"  Target metrics: minADE 0.56m, minFDE 0.26m, HitRate 0.93, HD 1.33m")
    
    # Training loop
    print("\n" + "=" * 70)
    print("RESUMING TRAINING")
    print("=" * 70)
    
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = datetime.now()
        
        # Train
        train_loss = train_epoch(denoising_net, train_loader, optimizer, device, history)
        
        # Validate
        val_loss, val_metrics = validate(denoising_net, val_loader, scaler, config, device)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch [{epoch}/{total_epochs}] ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  minADE: {val_metrics['minADE']:.3f}m | minFDE: {val_metrics['minFDE']:.3f}m")
        print(f"  HitRate: {val_metrics['hit_rate']:.3f} | HD: {val_metrics['hausdorff']:.3f}m")
        print(f"  LR: {current_lr:.6f}")
        
        # Check if we hit paper targets
        target_hit = (
            val_metrics['minADE'] <= 0.56 and
            val_metrics['minFDE'] <= 0.26 and
            val_metrics['hit_rate'] >= 0.93 and
            val_metrics['hausdorff'] <= 1.33
        )
        
        # Save best
        if val_metrics['minADE'] < best_minADE:
            best_minADE = val_metrics['minADE']
            torch.save({
                'epoch': epoch,
                'denoiser_state_dict': denoising_net.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, os.path.join(args.save_dir, 'diffusion_unet_best.pth'))
            print(f"  Best saved (minADE: {best_minADE:.3f}m)")
            
            if target_hit:
                print(f"  *** PAPER TARGETS ACHIEVED! ***")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'denoiser_state_dict': denoising_net.state_dict(),
            'history': history
        }, os.path.join(args.save_dir, 'diffusion_unet_latest.pth'))
        
        # Save history
        with open(os.path.join(args.save_dir, 'diffusion_history.json'), 'w') as f:
            def convert(obj):
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(item) for item in obj]
                elif hasattr(obj, 'item'):
                    return obj.item()
                return obj
            json.dump(convert(history), f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"COMPLETE | Best minADE: {best_minADE:.3f}m")
    print(f"Target: minADE 0.56m, minFDE 0.26m, HitRate 0.93, HD 1.33m")
    print("=" * 70)


if __name__ == "__main__":
    main()
