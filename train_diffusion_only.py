#!/usr/bin/env python3
"""
Train Diffusion Only (Frozen Encoder) — Fast Version

Stage 2 training: Train denoising network with frozen pretrained encoder.
Precomputes conditioning vectors once, then trains on tiny cached tensors.

Usage:
    python train_diffusion_only.py \
        --encoder_ckpt checkpoints/encoder_full_best.pth \
        --epochs 120 \
        --batch_size 8
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

sys.path.insert(0, 'models')
from bev_rasterization import BEVRasterizer, load_kitti_lidar
from encoder import build_encoder
from diffusion import TrajectoryDiffusionModel
from denoising_network import build_denoising_network
from losses import DiffusionLoss
from metrics import compute_trajectory_metrics, MetricsLogger


# ---------------------------------------------------------------------------
# Precomputation: run frozen encoder once, cache (conditioning, trajectory)
# ---------------------------------------------------------------------------

def _collect_samples(sequences, data_root, num_future=8, waypoint_spacing=2.0):
    """Collect (lidar_path, trajectory) pairs from KITTI sequences."""
    samples = []
    for seq in sequences:
        lidar_dir = os.path.join(data_root, 'sequences', seq, 'velodyne')
        pose_file = os.path.join(data_root, 'poses', f'{seq}.txt')
        if not os.path.exists(pose_file):
            continue

        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(np.array(values).reshape(3, 4))

        print(f"  Sequence {seq}: {len(poses)} frames")

        for frame_idx in range(len(poses) - num_future - 1):
            lidar_path = os.path.join(lidar_dir, f'{frame_idx:06d}.bin')
            if not os.path.exists(lidar_path):
                continue
            # Compute trajectory
            traj = _get_trajectory(poses, frame_idx, num_future, waypoint_spacing)
            samples.append((lidar_path, traj))
    return samples


def _get_trajectory(poses, frame_idx, num_future, waypoint_spacing):
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


def _cache_key(encoder_ckpt, sequences):
    """Deterministic cache filename based on encoder checkpoint + sequences."""
    h = hashlib.md5()
    h.update(encoder_ckpt.encode())
    h.update(','.join(sorted(sequences)).encode())
    # Include encoder file mtime for invalidation on retrain
    if os.path.exists(encoder_ckpt):
        h.update(str(os.path.getmtime(encoder_ckpt)).encode())
    return h.hexdigest()[:12]


def precompute_conditioning(encoder, encoder_ckpt, sequences, data_root,
                            cache_dir, device, batch_size=16):
    """
    Run frozen encoder on all samples once and cache results.

    Returns:
        conditioning: [N, 512] tensor
        trajectories: [N, 8, 2] tensor
    """
    tag = _cache_key(encoder_ckpt, sequences)
    seq_str = '_'.join(sequences)
    cache_path = os.path.join(cache_dir, f'cached_cond_{seq_str}_{tag}.pt')

    if os.path.exists(cache_path):
        print(f"  Loading cached conditioning from {cache_path}")
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        cond = data['conditioning'].float()  # Ensure float32
        traj = data['trajectories'].float()
        print(f"  {cond.shape[0]} samples loaded from cache (dtype={cond.dtype})")
        return cond, traj

    print(f"  Precomputing conditioning vectors (one-time cost)...")
    samples = _collect_samples(sequences, data_root)
    print(f"  {len(samples)} samples to process")

    rasterizer = BEVRasterizer()
    encoder.eval()

    all_cond = []
    all_traj = []
    t0 = time.time()

    # Process in batches for GPU efficiency
    for start in range(0, len(samples), batch_size):
        end = min(start + batch_size, len(samples))
        batch_bevs = []
        batch_trajs = []

        for lidar_path, traj in samples[start:end]:
            points = load_kitti_lidar(lidar_path)
            bev = rasterizer.rasterize_lidar(points)
            batch_bevs.append(torch.from_numpy(bev))
            batch_trajs.append(torch.from_numpy(traj))

        bev_batch = torch.stack(batch_bevs).to(device)
        with torch.no_grad(), autocast('cuda'):
            cond, _ = encoder(bev_batch)

        all_cond.append(cond.float().cpu())  # Ensure float32 (autocast produces float16)
        all_traj.append(torch.stack(batch_trajs))

        done = end
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(samples) - done) / rate if rate > 0 else 0
        if (done // batch_size) % 50 == 0 or done == len(samples):
            print(f"    [{done}/{len(samples)}] {rate:.0f} samples/s, ETA {eta:.0f}s")

    conditioning = torch.cat(all_cond, dim=0)    # [N, 512]
    trajectories = torch.cat(all_traj, dim=0)     # [N, 8, 2]

    # Save cache
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'conditioning': conditioning, 'trajectories': trajectories}, cache_path)
    total = time.time() - t0
    print(f"  Precompute done: {len(samples)} samples in {total:.1f}s ({len(samples)/total:.0f} samples/s)")
    print(f"  Cached to {cache_path} ({os.path.getsize(cache_path)/1024/1024:.1f} MB)")

    return conditioning, trajectories


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------

def train_epoch(diffusion_model, dataloader, optimizer, scaler, device, epoch,
                max_grad_norm=1.0):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_ckpt', type=str, default='checkpoints/encoder_full_best.pth')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (1e-4 recommended for denoiser-only)')
    parser.add_argument('--train_sequences', type=str, nargs='+',
                        default=['00', '02', '05', '07'])
    parser.add_argument('--val_sequences', type=str, nargs='+', default=['08'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--cache_dir', type=str, default='checkpoints/cache')
    parser.add_argument('--denoiser_arch', type=str, default='unet')
    parser.add_argument('--noise_schedule', type=str, default='cosine',
                        choices=['cosine', 'linear'],
                        help='Noise schedule (cosine required for T=10)')
    parser.add_argument('--precompute_batch', type=int, default=16,
                        help='Batch size for precomputing conditioning (higher = faster)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to diffusion checkpoint to resume training from')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("DIFFUSION TRAINING (Frozen Encoder) — FAST MODE")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load frozen encoder (only needed for precomputation)
    # ------------------------------------------------------------------
    print("\nLoading encoder (frozen)...")
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)

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

    # ------------------------------------------------------------------
    # 2. Precompute conditioning vectors (one-time, cached to disk)
    # ------------------------------------------------------------------
    print("\nPrecomputing conditioning vectors...")
    data_root = 'data/kitti'

    train_cond, train_traj = precompute_conditioning(
        encoder, args.encoder_ckpt, args.train_sequences, data_root,
        args.cache_dir, device, batch_size=args.precompute_batch)

    val_cond, val_traj = precompute_conditioning(
        encoder, args.encoder_ckpt, args.val_sequences, data_root,
        args.cache_dir, device, batch_size=args.precompute_batch)

    # Free encoder from GPU — no longer needed
    del encoder
    torch.cuda.empty_cache()
    print("  Encoder freed from GPU memory")

    # ------------------------------------------------------------------
    # 3. Create lightweight dataloaders (pure tensor, no I/O)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4. Build diffusion model
    # ------------------------------------------------------------------
    print("\nBuilding diffusion model...")
    denoising_net = build_denoising_network(
        args.denoiser_arch, num_waypoints=8, coord_dim=2,
        conditioning_dim=512, timestep_dim=256
    ).to(device)
    diffusion_model = TrajectoryDiffusionModel(
        denoising_net, num_timesteps=10, schedule=args.noise_schedule, device=device)
    print(f"  Denoiser: {sum(p.numel() for p in denoising_net.parameters()):,} params")
    print(f"  Noise schedule: {args.noise_schedule}")
    print(f"  alphas_cumprod[T]: {diffusion_model.scheduler.alphas_cumprod[-1]:.4f} "
          f"(signal at final step: {diffusion_model.scheduler.alphas_cumprod[-1].sqrt():.2%})")

    optimizer = optim.Adam(denoising_net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # ------------------------------------------------------------------
    # 5. Resume from checkpoint if requested
    # ------------------------------------------------------------------
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
        # Advance scheduler to correct position
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, best minADE: {best_minADE:.3f}m")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ------------------------------------------------------------------
    # 6. Early stopping config
    # ------------------------------------------------------------------
    # Paper Table 1 targets (TopoDiffuser row)
    PAPER_TARGETS = {
        'minADE': 0.26,
        'minFDE': 0.56,
        'hit_rate': 0.93,
        'hausdorff': 1.33,
    }
    TOLERANCE = 0.05  # within 5% of paper target = close enough to stop
    patience = 10
    epochs_no_improve = 0

    def within_paper_targets(metrics):
        """Check if all metrics are within 5% of paper targets."""
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
    print(f"  Early stop patience: {patience} epochs")

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
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
            }, os.path.join(args.save_dir, 'diffusion_unet_best.pth'))
            print(f"  Best saved (minADE: {best_minADE:.3f}m)")

        # Early stopping: track no-improvement epochs
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

        # Check early stopping conditions
        if within_paper_targets(val_metrics):
            print(f"\n{'=' * 70}")
            print(f"PAPER TARGETS REACHED at epoch {epoch}!")
            print(f"  minADE: {val_metrics['minADE']:.3f}m (target <= {PAPER_TARGETS['minADE'] * (1 + TOLERANCE):.3f}m)")
            print(f"  minFDE: {val_metrics['minFDE']:.3f}m (target <= {PAPER_TARGETS['minFDE'] * (1 + TOLERANCE):.3f}m)")
            print(f"  HitRate: {val_metrics['hit_rate']:.3f} (target >= {PAPER_TARGETS['hit_rate'] * (1 - TOLERANCE):.3f})")
            print(f"  HD: {val_metrics['hausdorff']:.3f}m (target <= {PAPER_TARGETS['hausdorff'] * (1 + TOLERANCE):.3f}m)")
            print(f"{'=' * 70}")
            break

        if epochs_no_improve >= patience:
            print(f"\n{'=' * 70}")
            print(f"EARLY STOPPING at epoch {epoch} — no improvement for {patience} epochs")
            print(f"  Best minADE: {best_minADE:.3f}m")
            print(f"{'=' * 70}")
            break

    print("\n" + "=" * 70)
    print(f"COMPLETE | Best minADE: {best_minADE:.3f}m")
    print("=" * 70)


if __name__ == "__main__":
    main()
