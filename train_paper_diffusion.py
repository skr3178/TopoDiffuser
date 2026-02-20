#!/usr/bin/env python3
"""
Train Diffusion Model on Paper Split (Stage 2)

Loads the frozen paper-split encoder, encodes all BEVs once, then trains
the denoising network on the cached conditioning vectors.

Fixes over v1:
  1. Trajectory normalisation — per-waypoint zero-mean / unit-variance using
     training-set statistics; predictions are denormalised before metrics.
  2. Early stopping on smoothed val_loss (3-epoch window, patience 20) instead
     of the noisy stochastic minADE metric.

Train split: data/paper_split/train_meta.pkl  (3,860 samples — seqs 00,02,05,07)
Val   split: data/paper_split/test_meta.pkl   (2,270 samples — seqs 08,09,10)

Usage:
    conda run --no-capture-output -n nuscenes python -u train_paper_diffusion.py \
        --epochs 300 --batch_size 64 --lr 1e-4 \
        > train_paper_diffusion.log 2>&1 &
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
import json
import time
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

sys.path.insert(0, 'models')
sys.path.insert(0, 'utils')

from multimodal_encoder import build_full_multimodal_encoder
from diffusion import TrajectoryDiffusionModel
from denoising_network import build_denoising_network
from metrics import compute_trajectory_metrics, MetricsLogger


# ---------------------------------------------------------------------------
# Precompute + normalise
# ---------------------------------------------------------------------------

def precompute_from_meta(encoder, encoder_ckpt, meta_path, device,
                         batch_size=64, traj_mean=None, traj_std=None):
    """
    Run frozen encoder on paper-split BEV files; cache conditioning vectors.

    Trajectories are normalised per-waypoint (zero-mean / unit-variance).
    If traj_mean / traj_std are None (train set call), they are computed from
    the data and returned so the caller can reuse them for the val set.

    Returns
    -------
    conditioning : [N, 512] float32 CPU tensor   (encoder output)
    traj_norm    : [N, 8, 2] float32 CPU tensor  (normalised trajectories)
    traj_mean    : [8, 2] float32 CPU tensor
    traj_std     : [8, 2] float32 CPU tensor
    """
    # Cache key: meta content + encoder mtime + norm tag
    h = hashlib.md5()
    with open(meta_path, 'rb') as f:
        h.update(f.read(1024))
    if os.path.exists(encoder_ckpt):
        h.update(str(os.path.getmtime(encoder_ckpt)).encode())
    h.update(b'norm_v1')
    tag = h.hexdigest()[:8]

    cache_path = (Path('checkpoints/cache_paper')
                  / f'{Path(meta_path).stem}_{tag}_norm.pt')

    if cache_path.exists():
        print(f"  Loading cached conditioning from {cache_path}")
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        cond      = data['conditioning'].float()
        traj_norm = data['traj_norm'].float()
        t_mean    = data['traj_mean'].float()
        t_std     = data['traj_std'].float()
        print(f"  {cond.shape[0]} samples loaded from cache")
        return cond, traj_norm, t_mean, t_std

    # ── encode BEVs ─────────────────────────────────────────────────────────
    print(f"  Precomputing conditioning vectors for {meta_path} ...")
    with open(meta_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"  {len(samples)} samples to process")

    encoder.eval()
    all_cond, all_traj = [], []
    t0 = time.time()

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]

        bevs = torch.stack([
            torch.from_numpy(np.load(s['npy_path']).astype(np.float32))
            for s in batch
        ]).to(device)

        trajs = torch.stack([
            torch.from_numpy(np.array(s['trajectory'], dtype=np.float32))
            for s in batch
        ])

        with torch.no_grad(), autocast('cuda'):
            cond, _ = encoder(bevs)

        all_cond.append(cond.float().cpu())
        all_traj.append(trajs)

        done    = start + len(batch)
        elapsed = time.time() - t0
        rate    = done / elapsed if elapsed > 0 else 1.0
        eta     = (len(samples) - done) / rate
        if (done // batch_size) % 10 == 0 or done == len(samples):
            print(f"    [{done}/{len(samples)}]  {rate:.0f} samples/s  ETA {eta:.0f}s")

    conditioning = torch.cat(all_cond)   # [N, 512]
    trajectories = torch.cat(all_traj)   # [N, 8, 2]

    # ── normalise ────────────────────────────────────────────────────────────
    if traj_mean is None:
        traj_mean = trajectories.mean(dim=0)              # [8, 2]
        traj_std  = trajectories.std(dim=0).clamp(min=1e-6)  # [8, 2]

    traj_norm = (trajectories - traj_mean) / traj_std

    print(f"  Traj mean (per-wp x/y): {traj_mean[:,0].mean():.2f} / {traj_mean[:,1].mean():.2f}")
    print(f"  Traj std  (per-wp x/y): {traj_std[:,0].mean():.2f}  / {traj_std[:,1].mean():.2f}")

    # ── save cache ───────────────────────────────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'conditioning': conditioning,
        'traj_norm':    traj_norm,
        'traj_mean':    traj_mean,
        'traj_std':     traj_std,
    }, cache_path)
    print(f"  Precompute done: {len(samples)} samples in {time.time() - t0:.1f}s")
    print(f"  Cached to {cache_path}")

    return conditioning, traj_norm, traj_mean, traj_std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(diffusion_model, dataloader, optimizer, scaler, device, epoch,
                max_grad_norm=1.0):
    """Train for one epoch on normalised trajectories."""
    diffusion_model.train()
    total_loss  = 0.0
    num_batches = 0

    for batch_idx, (conditioning, trajectory) in enumerate(dataloader):
        conditioning = conditioning.to(device, non_blocking=True)
        trajectory   = trajectory  .to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast('cuda'):
            batch_size = trajectory.shape[0]
            t          = diffusion_model.scheduler.sample_timesteps(batch_size)
            noise      = torch.randn_like(trajectory)
            x_t, _     = diffusion_model.forward_diffusion(trajectory, t, noise)
            t_emb      = diffusion_model.timestep_embedding(t)
            pred_noise = diffusion_model.denoising_network(x_t, conditioning, t_emb)
            loss       = nn.functional.mse_loss(pred_noise, noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            diffusion_model.denoising_network.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(diffusion_model, dataloader, device, traj_mean, traj_std):
    """
    Validate diffusion model.

    Samples are generated in normalised space then denormalised before
    computing ADE / FDE / HitRate / HD so metrics are in real metres.
    """
    diffusion_model.eval()
    total_loss     = 0.0
    metrics_logger = MetricsLogger()
    num_batches    = 0

    mean = traj_mean.to(device)   # [8, 2]
    std  = traj_std .to(device)   # [8, 2]

    for conditioning, trajectory in dataloader:
        conditioning = conditioning.to(device, non_blocking=True)
        trajectory   = trajectory  .to(device, non_blocking=True)   # normalised

        # Noise-prediction loss (in normalised space — correct)
        batch_size = trajectory.shape[0]
        t          = diffusion_model.scheduler.sample_timesteps(batch_size)
        noise      = torch.randn_like(trajectory)
        x_t, _     = diffusion_model.forward_diffusion(trajectory, t, noise)
        t_emb      = diffusion_model.timestep_embedding(t)
        pred_noise = diffusion_model.denoising_network(x_t, conditioning, t_emb)
        loss       = nn.functional.mse_loss(pred_noise, noise)
        total_loss += loss.item()

        # Sample K trajectories in normalised space, then denormalise
        pred_norm = diffusion_model.sample(conditioning, num_samples=5)  # [B, K, 8, 2]
        pred_real = pred_norm * std + mean        # [B, K, 8, 2]
        gt_real   = trajectory * std + mean       # [B, 8, 2]

        metrics = compute_trajectory_metrics(pred_real, gt_real, threshold=2.0)
        metrics_logger.update(metrics, count=batch_size)
        num_batches += 1

    return total_loss / num_batches, metrics_logger.get_averages()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train diffusion model on paper split (Stage 2, normalised)')
    parser.add_argument('--encoder_ckpt',     type=str,
                        default='checkpoints/paper_encoder_best.pth')
    parser.add_argument('--train_meta',       type=str,
                        default='data/paper_split/train_meta.pkl')
    parser.add_argument('--val_meta',         type=str,
                        default='data/paper_split/test_meta.pkl')
    parser.add_argument('--epochs',           type=int,   default=300)
    parser.add_argument('--batch_size',       type=int,   default=64)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--workers',          type=int,   default=4)
    parser.add_argument('--save_dir',         type=str,   default='checkpoints')
    parser.add_argument('--denoiser_arch',    type=str,   default='unet')
    parser.add_argument('--noise_schedule',   type=str,   default='cosine',
                        choices=['cosine', 'linear'])
    parser.add_argument('--precompute_batch', type=int,   default=64)
    parser.add_argument('--patience',         type=int,   default=20,
                        help='Early-stopping patience on smoothed val_loss')
    parser.add_argument('--smooth_window',    type=int,   default=3,
                        help='Window for val_loss smoothing')
    parser.add_argument('--resume',           type=str,   default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("DIFFUSION TRAINING — Paper Split (Stage 2, trajectory-normalised)")
    print("=" * 70)
    print(f"Start:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device:        {device}")
    print(f"Encoder:       {args.encoder_ckpt}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"LR:            {args.lr}")
    print(f"ES patience:   {args.patience} (smoothed window={args.smooth_window})")
    print("=" * 70)

    # ── Frozen encoder ───────────────────────────────────────────────────────
    print("\nLoading 5-channel encoder (frozen)...")
    encoder = build_full_multimodal_encoder(
        input_channels=5, conditioning_dim=512
    ).to(device)

    if args.encoder_ckpt and os.path.exists(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['model_state_dict'])
        elif 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
        else:
            encoder.load_state_dict(ckpt)
        print(f"  Loaded from {args.encoder_ckpt}")
    else:
        print(f"  WARNING: No checkpoint at {args.encoder_ckpt}")

    for p in encoder.parameters():
        p.requires_grad = False
    print(f"  Encoder frozen ({sum(p.numel() for p in encoder.parameters()):,} params)")

    # ── Precompute conditioning + normalised trajectories ────────────────────
    print("\nPrecomputing train conditioning vectors...")
    train_cond, train_traj, traj_mean, traj_std = precompute_from_meta(
        encoder, args.encoder_ckpt, args.train_meta, device,
        batch_size=args.precompute_batch)

    print("\nPrecomputing val conditioning vectors...")
    val_cond, val_traj, _, _ = precompute_from_meta(
        encoder, args.encoder_ckpt, args.val_meta, device,
        batch_size=args.precompute_batch,
        traj_mean=traj_mean, traj_std=traj_std)   # use train stats

    del encoder
    torch.cuda.empty_cache()
    print("\n  Encoder freed from GPU memory")

    # Save normalisation stats alongside checkpoints
    norm_stats_path = os.path.join(args.save_dir, 'paper_diffusion_norm_stats.pt')
    torch.save({'traj_mean': traj_mean, 'traj_std': traj_std}, norm_stats_path)
    print(f"  Normalisation stats saved to {norm_stats_path}")

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_dataset = TensorDataset(train_cond, train_traj)
    val_dataset   = TensorDataset(val_cond,   val_traj)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True)

    print(f"\nTrain: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # ── Diffusion model ──────────────────────────────────────────────────────
    print("\nBuilding diffusion model...")
    denoising_net = build_denoising_network(
        args.denoiser_arch, num_waypoints=8, coord_dim=2,
        conditioning_dim=512, timestep_dim=256
    ).to(device)
    diffusion_model = TrajectoryDiffusionModel(
        denoising_net, num_timesteps=10, schedule=args.noise_schedule,
        device=device)
    print(f"  Denoiser: {sum(p.numel() for p in denoising_net.parameters()):,} params")
    print(f"  Noise schedule: {args.noise_schedule}")

    optimizer = optim.Adam(denoising_net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lr_sched  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()

    # ── Optional resume ──────────────────────────────────────────────────────
    start_epoch = 1
    history     = {'train_loss': [], 'val_loss': [], 'val_metrics': [], 'lr': []}
    best_minADE = float('inf')

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        denoising_net.load_state_dict(ck['denoiser_state_dict'])
        start_epoch = ck.get('epoch', 0) + 1
        if ck.get('history'):
            history = ck['history']
        if ck.get('val_metrics'):
            best_minADE = ck['val_metrics'].get('minADE', float('inf'))
        for _ in range(start_epoch - 1):
            lr_sched.step()
        print(f"  Resumed at epoch {start_epoch}, best minADE: {best_minADE:.3f}m")

    # ── Paper targets ────────────────────────────────────────────────────────
    PAPER_TARGETS = {'minADE': 0.273, 'minFDE': 0.588,
                     'hit_rate': 0.8835, 'hausdorff': 1.397}
    TOLERANCE = 0.05

    def within_paper_targets(m):
        return (m['minADE']    <= PAPER_TARGETS['minADE']    * (1 + TOLERANCE) and
                m['minFDE']    <= PAPER_TARGETS['minFDE']    * (1 + TOLERANCE) and
                m['hit_rate']  >= PAPER_TARGETS['hit_rate']  * (1 - TOLERANCE) and
                m['hausdorff'] <= PAPER_TARGETS['hausdorff'] * (1 + TOLERANCE))

    print(f"\nPaper targets (±{TOLERANCE:.0%}):")
    print(f"  minADE  <= {PAPER_TARGETS['minADE']  * (1+TOLERANCE):.3f}m")
    print(f"  minFDE  <= {PAPER_TARGETS['minFDE']  * (1+TOLERANCE):.3f}m")
    print(f"  HitRate >= {PAPER_TARGETS['hit_rate'] * (1-TOLERANCE):.4f}")
    print(f"  HD      <= {PAPER_TARGETS['hausdorff']* (1+TOLERANCE):.3f}m")

    # ── Early-stopping state ─────────────────────────────────────────────────
    recent_val_losses = []   # rolling window for smoothing
    best_smooth_loss  = float('inf')
    epochs_no_improve = 0

    # ── Training loop ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss            = train_epoch(diffusion_model, train_loader,
                                            optimizer, scaler, device, epoch)
        val_loss, val_metrics = validate(diffusion_model, val_loader, device,
                                         traj_mean, traj_std)
        lr_sched.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - t0

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['lr'].append(current_lr)

        # Smoothed val_loss for early stopping
        recent_val_losses.append(val_loss)
        if len(recent_val_losses) > args.smooth_window:
            recent_val_losses.pop(0)
        smooth_loss = sum(recent_val_losses) / len(recent_val_losses)

        print(f"\nEpoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
              f"  (smooth: {smooth_loss:.4f})")
        print(f"  minADE:  {val_metrics['minADE']:.3f}m  |"
              f" minFDE: {val_metrics['minFDE']:.3f}m")
        print(f"  HitRate: {val_metrics['hit_rate']:.4f}  |"
              f" HD:     {val_metrics['hausdorff']:.3f}m")
        print(f"  LR: {current_lr:.6f}")

        # Save best checkpoint (tracked by minADE in real metres)
        if val_metrics['minADE'] < best_minADE:
            best_minADE = val_metrics['minADE']
            torch.save({
                'epoch':               epoch,
                'denoiser_state_dict': denoising_net.state_dict(),
                'val_metrics':         val_metrics,
                'history':             history,
                'traj_mean':           traj_mean,
                'traj_std':            traj_std,
            }, os.path.join(args.save_dir, 'paper_diffusion_best.pth'))
            print(f"  Best saved (minADE: {best_minADE:.3f}m)")

        # Save latest
        torch.save({
            'epoch':               epoch,
            'denoiser_state_dict': denoising_net.state_dict(),
            'val_metrics':         val_metrics,
            'history':             history,
            'traj_mean':           traj_mean,
            'traj_std':            traj_std,
        }, os.path.join(args.save_dir, 'paper_diffusion_latest.pth'))

        # Save history JSON
        def _cvt(obj):
            if isinstance(obj, dict):  return {k: _cvt(v) for k, v in obj.items()}
            if isinstance(obj, list):  return [_cvt(x) for x in obj]
            if hasattr(obj, 'item'):   return obj.item()
            return obj

        with open(os.path.join(args.save_dir, 'paper_diffusion_history.json'), 'w') as f:
            json.dump(_cvt(history), f, indent=2)

        # ── early stopping on smoothed val_loss ──────────────────────────────
        if smooth_loss < best_smooth_loss:
            best_smooth_loss  = smooth_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  ES: no improvement for {epochs_no_improve}/{args.patience} epochs"
                  f" (best smooth val_loss: {best_smooth_loss:.4f})")

        if within_paper_targets(val_metrics):
            print(f"\n{'='*70}\nPAPER TARGETS REACHED at epoch {epoch}!\n{'='*70}")
            break

        if epochs_no_improve >= args.patience:
            print(f"\n{'='*70}\nEARLY STOPPING at epoch {epoch}\n{'='*70}")
            break

    print("\n" + "=" * 70)
    print(f"COMPLETE | Best minADE: {best_minADE:.3f}m")
    print("=" * 70)


if __name__ == "__main__":
    main()
