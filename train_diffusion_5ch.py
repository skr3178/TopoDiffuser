#!/usr/bin/env python3
"""
Train Diffusion with 5-Channel Encoder (Frozen) — Fast Version

Stage 2 training: Train denoising network with frozen 5-channel encoder.
Supports modality dropout during training.

Usage:
    python train_diffusion_5ch.py \
        --encoder_ckpt checkpoints/encoder_full_5ch_best.pth \
        --bev_cache_dir data/kitti/bev_cache_5ch \
        --epochs 120 \
        --batch_size 64 \
        --modality_dropout
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'models')
sys.path.insert(0, 'utils')

from multimodal_encoder import build_full_multimodal_encoder
from diffusion import TrajectoryDiffusionModel
from denoising_network import build_denoising_network
from metrics import compute_trajectory_metrics, MetricsLogger


class ModalityDropout:
    """
    Randomly zero out modality channels during training.
    
    Helps the model learn to work with incomplete inputs.
    """
    def __init__(self, history_prob=0.05, osm_prob=0.10):
        """
        Args:
            history_prob: Probability of dropping history channel (ch 3)
            osm_prob: Probability of dropping OSM channel (ch 4)
        """
        self.history_prob = history_prob
        self.osm_prob = osm_prob
    
    def apply(self, bev_batch):
        """
        Apply modality dropout to a batch of BEVs.
        
        Args:
            bev_batch: [B, 5, H, W] tensor
        
        Returns:
            bev_batch with some channels zeroed
        """
        if not self.training:
            return bev_batch
        
        B = bev_batch.size(0)
        
        # Drop history channel (ch 3)
        if np.random.random() < self.history_prob:
            mask = torch.rand(B, 1, 1, 1, device=bev_batch.device) > self.history_prob
            bev_batch[:, 3:4] = bev_batch[:, 3:4] * mask
        
        # Drop OSM channel (ch 4)
        if np.random.random() < self.osm_prob:
            mask = torch.rand(B, 1, 1, 1, device=bev_batch.device) > self.osm_prob
            bev_batch[:, 4:5] = bev_batch[:, 4:5] * mask
        
        return bev_batch


def _get_trajectory(poses, frame_idx, num_future, waypoint_spacing):
    """Compute future trajectory from poses."""
    current_pose = poses[frame_idx]
    cx, cy = current_pose[0, 3], current_pose[1, 3]
    trajectory = []
    for i in range(1, len(poses) - frame_idx):
        pose = poses[frame_idx + i]
        x, y = pose[0, 3], pose[1, 3]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist >= waypoint_spacing * (len(trajectory) + 1):
            trajectory.append([x - cx, y - cy])
            if len(trajectory) >= num_future:
                break
    while len(trajectory) < num_future:
        trajectory.append(trajectory[-1] if trajectory else [0.0, 0.0])
    return np.array(trajectory[:num_future], dtype=np.float32)


def _cache_key(encoder_ckpt, sequences, bev_cache_dir):
    """Deterministic cache filename based on encoder + sequences + cache."""
    h = hashlib.md5()
    h.update(encoder_ckpt.encode())
    h.update(','.join(sorted(sequences)).encode())
    h.update(bev_cache_dir.encode())
    if os.path.exists(encoder_ckpt):
        h.update(str(os.path.getmtime(encoder_ckpt)).encode())
    return h.hexdigest()[:12]


def precompute_conditioning(encoder, encoder_ckpt, sequences, data_root,
                            bev_cache_dir, device, batch_size=16,
                            modality_dropout=None):
    """
    Run frozen 5-channel encoder on all samples once and cache results.
    
    Args:
        encoder: 5-channel encoder model
        encoder_ckpt: Path to encoder checkpoint
        sequences: List of sequences to process
        data_root: Path to KITTI data
        bev_cache_dir: Path to 5ch BEV cache
        device: torch device
        batch_size: Batch size for encoding
        modality_dropout: Optional ModalityDropout for training data
    
    Returns:
        conditioning: [N, 512] tensor
        trajectories: [N, 8, 2] tensor
    """
    tag = _cache_key(encoder_ckpt, sequences, bev_cache_dir)
    seq_str = '_'.join(sequences)
    cache_path = Path('checkpoints/cache_5ch') / f'cached_cond_{seq_str}_{tag}.pt'
    
    if cache_path.exists():
        print(f"  Loading cached conditioning from {cache_path}")
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        cond = data['conditioning'].float()
        traj = data['trajectories'].float()
        print(f"  {cond.shape[0]} samples loaded from cache")
        return cond, traj
    
    print(f"  Precomputing conditioning vectors (one-time cost)...")
    
    # Collect samples
    samples = []
    for seq in sequences:
        cache_seq_dir = Path(bev_cache_dir) / seq
        pose_file = Path(data_root) / 'poses' / f'{seq}.txt'
        
        if not pose_file.exists() or not cache_seq_dir.exists():
            print(f"  Warning: Missing data for sequence {seq}")
            continue
        
        poses = np.loadtxt(pose_file)
        
        for frame_idx in range(len(poses) - 9):
            bev_path = cache_seq_dir / f'{frame_idx:06d}.npy'
            if bev_path.exists():
                traj = _get_trajectory(poses, frame_idx, 8, 2.0)
                samples.append((str(bev_path), traj))
    
    print(f"  {len(samples)} samples to process")
    
    encoder.eval()
    
    all_cond = []
    all_traj = []
    t0 = time.time()
    
    # Process in batches
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        batch_bevs = []
        batch_trajs = []
        
        for bev_path, traj in samples[start:end]:
            bev = np.load(bev_path)
            bev = torch.from_numpy(bev).float()
            batch_bevs.append(bev)
            batch_trajs.append(torch.from_numpy(traj))
        
        bev_batch = torch.stack(batch_bevs).to(device)
        
        # Apply modality dropout for training data
        if modality_dropout is not None:
            bev_batch = modality_dropout.apply(bev_batch)
        
        with torch.no_grad(), autocast('cuda'):
            cond, _ = encoder(bev_batch)
        
        all_cond.append(cond.float().cpu())
        all_traj.append(torch.stack(batch_trajs))
        
        done = end
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(samples) - done) / rate if rate > 0 else 0
        if (done // batch_size) % 50 == 0 or done == len(samples):
            print(f"    [{done}/{len(samples)}] {rate:.0f} samples/s, ETA {eta:.0f}s")
    
    conditioning = torch.cat(all_cond, dim=0)
    trajectories = torch.cat(all_traj, dim=0)
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'conditioning': conditioning, 'trajectories': trajectories}, cache_path)
    total = time.time() - t0
    print(f"  Precompute done: {len(samples)} samples in {total:.1f}s")
    print(f"  Cached to {cache_path}")
    
    return conditioning, trajectories


def train_epoch(diffusion_model, dataloader, optimizer, scaler, device, epoch,
                max_grad_norm=1.0):
    """Train for one epoch."""
    diffusion_model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (conditioning, trajectory) in enumerate(dataloader):
        conditioning = conditioning.to(device, non_blocking=True)
        trajectory = trajectory.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            batch_size = trajectory.shape[0]
            t = diffusion_model.scheduler.sample_timesteps(batch_size)
            noise = torch.randn_like(trajectory)
            x_t, _ = diffusion_model.forward_diffusion(trajectory, t, noise)
            t_emb = diffusion_model.timestep_embedding(t)
            predicted_noise = diffusion_model.denoising_network(x_t, conditioning, t_emb)
            loss = nn.functional.mse_loss(predicted_noise, noise)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(diffusion_model.denoising_network.parameters(),
                                       max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


@torch.no_grad()
def validate(diffusion_model, dataloader, device):
    """Validate the diffusion model."""
    diffusion_model.eval()
    total_loss = 0.0
    metrics_logger = MetricsLogger()
    num_batches = 0
    
    for conditioning, trajectory in dataloader:
        conditioning = conditioning.to(device, non_blocking=True)
        trajectory = trajectory.to(device, non_blocking=True)
        
        batch_size = trajectory.shape[0]
        t = diffusion_model.scheduler.sample_timesteps(batch_size)
        noise = torch.randn_like(trajectory)
        x_t, _ = diffusion_model.forward_diffusion(trajectory, t, noise)
        t_emb = diffusion_model.timestep_embedding(t)
        predicted_noise = diffusion_model.denoising_network(x_t, conditioning, t_emb)
        
        loss = nn.functional.mse_loss(predicted_noise, noise)
        total_loss += loss.item()
        
        pred_trajectories = diffusion_model.sample(conditioning, num_samples=5)
        metrics = compute_trajectory_metrics(pred_trajectories, trajectory, threshold=2.0)
        metrics_logger.update(metrics, count=batch_size)
        num_batches += 1
    
    return total_loss / num_batches, metrics_logger.get_averages()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_ckpt', type=str, 
                       default='checkpoints/encoder_full_5ch_best.pth')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_sequences', type=str, nargs='+',
                       default=['00', '02', '05', '07'])
    parser.add_argument('--val_sequences', type=str, nargs='+', 
                       default=['08', '09', '10'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--bev_cache_dir', type=str, 
                       default='data/kitti/bev_cache_5ch')
    parser.add_argument('--denoiser_arch', type=str, default='unet')
    parser.add_argument('--noise_schedule', type=str, default='cosine',
                       choices=['cosine', 'linear'])
    parser.add_argument('--precompute_batch', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--modality_dropout', action='store_true',
                       help='Enable modality dropout during training')
    parser.add_argument('--history_dropout_prob', type=float, default=0.05)
    parser.add_argument('--osm_dropout_prob', type=float, default=0.10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("DIFFUSION TRAINING (5-Channel Frozen Encoder)")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Modality dropout: {args.modality_dropout}")
    if args.modality_dropout:
        print(f"  History dropout: {args.history_dropout_prob}")
        print(f"  OSM dropout: {args.osm_dropout_prob}")
    print("=" * 70)

    # Load frozen 5-channel encoder
    print("\nLoading 5-channel encoder (frozen)...")
    encoder = build_full_multimodal_encoder(
        input_channels=5, conditioning_dim=512
    ).to(device)

    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        checkpoint = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['model_state_dict'])
        elif 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"  Loaded from {args.encoder_ckpt}")
    else:
        print(f"  WARNING: No checkpoint found at {args.encoder_ckpt}")

    for param in encoder.parameters():
        param.requires_grad = False
    print(f"  Encoder frozen ({sum(p.numel() for p in encoder.parameters()):,} params)")

    # Setup modality dropout for training
    train_modality_dropout = None
    if args.modality_dropout:
        train_modality_dropout = ModalityDropout(
            history_prob=args.history_dropout_prob,
            osm_prob=args.osm_dropout_prob
        )
        train_modality_dropout.training = True
        print(f"  Modality dropout enabled for training")

    # Precompute conditioning
    print("\nPrecomputing conditioning vectors...")
    data_root = 'data/kitti'

    train_cond, train_traj = precompute_conditioning(
        encoder, args.encoder_ckpt, args.train_sequences, data_root,
        args.bev_cache_dir, device, batch_size=args.precompute_batch,
        modality_dropout=train_modality_dropout)

    val_cond, val_traj = precompute_conditioning(
        encoder, args.encoder_ckpt, args.val_sequences, data_root,
        args.bev_cache_dir, device, batch_size=args.precompute_batch,
        modality_dropout=None)  # No dropout for validation

    # Free encoder
    del encoder
    torch.cuda.empty_cache()
    print("  Encoder freed from GPU memory")

    # Create dataloaders
    train_dataset = TensorDataset(train_cond, train_traj)
    val_dataset = TensorDataset(val_cond, val_traj)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True
    )

    print(f"\nTrain: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # Build diffusion model
    print("\nBuilding diffusion model...")
    denoising_net = build_denoising_network(
        args.denoiser_arch, num_waypoints=8, coord_dim=2,
        conditioning_dim=512, timestep_dim=256
    ).to(device)
    diffusion_model = TrajectoryDiffusionModel(
        denoising_net, num_timesteps=10, schedule=args.noise_schedule, device=device)
    print(f"  Denoiser: {sum(p.numel() for p in denoising_net.parameters()):,} params")
    print(f"  Noise schedule: {args.noise_schedule}")

    optimizer = optim.Adam(denoising_net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # Resume if requested
    start_epoch = 1
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': [], 'lr': []}
    best_minADE = float('inf')

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        denoising_net.load_state_dict(ckpt['denoiser_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        if 'history' in ckpt and ckpt['history']:
            history = ckpt['history']
        if 'val_metrics' in ckpt and ckpt['val_metrics']:
            best_minADE = ckpt['val_metrics'].get('minADE', float('inf'))
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, best minADE: {best_minADE:.3f}m")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Early stopping config
    PAPER_TARGETS = {
        'minADE': 0.26,
        'minFDE': 0.56,
        'hit_rate': 0.93,
        'hausdorff': 1.33,
    }
    TOLERANCE = 0.05
    patience = 10
    epochs_no_improve = 0

    def within_paper_targets(metrics):
        ade_ok = metrics['minADE'] <= PAPER_TARGETS['minADE'] * (1 + TOLERANCE)
        fde_ok = metrics['minFDE'] <= PAPER_TARGETS['minFDE'] * (1 + TOLERANCE)
        hr_ok = metrics['hit_rate'] >= PAPER_TARGETS['hit_rate'] * (1 - TOLERANCE)
        hd_ok = metrics['hausdorff'] <= PAPER_TARGETS['hausdorff'] * (1 + TOLERANCE)
        return ade_ok and fde_ok and hr_ok and hd_ok

    print(f"\nPaper targets (within {TOLERANCE:.0%}):")
    print(f"  minADE <= {PAPER_TARGETS['minADE'] * (1 + TOLERANCE):.3f}m")
    print(f"  minFDE <= {PAPER_TARGETS['minFDE'] * (1 + TOLERANCE):.3f}m")
    print(f"  HitRate >= {PAPER_TARGETS['hit_rate'] * (1 - TOLERANCE):.3f}")
    print(f"  HD <= {PAPER_TARGETS['hausdorff'] * (1 + TOLERANCE):.3f}m")

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        
        train_loss = train_epoch(diffusion_model, train_loader, optimizer,
                                 scaler, device, epoch)
        val_loss, val_metrics = validate(diffusion_model, val_loader, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['lr'].append(current_lr)
        
        print(f"\nEpoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  minADE: {val_metrics['minADE']:.3f}m | minFDE: {val_metrics['minFDE']:.3f}m")
        print(f"  HitRate: {val_metrics['hit_rate']:.3f} | HD: {val_metrics['hausdorff']:.3f}m")
        print(f"  LR: {current_lr:.6f}")

        # Save best
        improved = False
        if val_metrics['minADE'] < best_minADE:
            best_minADE = val_metrics['minADE']
            improved = True
            torch.save({
                'epoch': epoch,
                'denoiser_state_dict': denoising_net.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, os.path.join(args.save_dir, 'diffusion_5ch_best.pth'))
            print(f"  Best saved (minADE: {best_minADE:.3f}m)")
        
        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{patience} epoch(s)")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'denoiser_state_dict': denoising_net.state_dict(),
            'val_metrics': val_metrics,
            'history': history
        }, os.path.join(args.save_dir, 'diffusion_5ch_latest.pth'))
        
        # Save history
        with open(os.path.join(args.save_dir, 'diffusion_5ch_history.json'), 'w') as f:
            def convert(obj):
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(item) for item in obj]
                elif hasattr(obj, 'item'):
                    return obj.item()
                return obj
            json.dump(convert(history), f, indent=2)
        
        # Check early stopping
        if within_paper_targets(val_metrics):
            print(f"\n{'=' * 70}")
            print(f"PAPER TARGETS REACHED at epoch {epoch}!")
            print(f"{'=' * 70}")
            break
        
        if epochs_no_improve >= patience:
            print(f"\n{'=' * 70}")
            print(f"EARLY STOPPING at epoch {epoch}")
            print(f"{'=' * 70}")
            break

    print("\n" + "=" * 70)
    print(f"COMPLETE | Best minADE: {best_minADE:.3f}m")
    print("=" * 70)


if __name__ == "__main__":
    main()
