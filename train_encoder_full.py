#!/usr/bin/env python3
"""
Train Encoder on Full Paper Dataset — Fast Version

Train sequences: 00, 02, 05, 07 (3,860 samples)
Val sequence: 08 (held-out)

Caches BEV rasterizations to disk on first run so subsequent epochs
skip LiDAR loading + rasterization entirely.

Usage:
    python train_encoder_full.py --epochs 50 --batch_size 128
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
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'models')
from bev_rasterization import BEVRasterizer, load_kitti_lidar
from encoder import build_encoder


class KITTIFullDataset(Dataset):
    """
    Dataset for encoder training with on-disk BEV caching.

    First access of each sample: loads .bin + rasterizes + saves .npy cache
    Subsequent accesses: loads cached .npy directly (~3x faster)
    """

    def __init__(self, sequences=['00', '02', '05', '07'], split='train',
                 data_root='data/kitti', cache_dir='data/kitti/bev_cache',
                 num_future=8, waypoint_spacing=2.0):
        self.data_root = data_root
        self.cache_dir = cache_dir
        self.rasterizer = BEVRasterizer()
        self.num_future = num_future
        self.waypoint_spacing = waypoint_spacing
        self.split = split

        self.samples = []
        for seq in sequences:
            self._load_sequence(seq)

        # 80/20 split
        if split != 'all':
            n = len(self.samples)
            if split == 'train':
                self.samples = self.samples[:int(n * 0.8)]
            else:
                self.samples = self.samples[int(n * 0.8):]

        print(f"[{split}] {len(self.samples)} samples from {sequences}")

        # Pre-warm cache if needed
        self._ensure_cache(sequences)

    def _load_sequence(self, sequence):
        lidar_dir = Path(self.data_root) / 'sequences' / sequence / 'velodyne'
        pose_file = Path(self.data_root) / 'poses' / f'{sequence}.txt'

        if not pose_file.exists():
            print(f"  Warning: {pose_file} not found")
            return

        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(np.array(values).reshape(3, 4))

        print(f"  Sequence {sequence}: {len(poses)} frames")

        for frame_idx in range(len(poses) - self.num_future - 1):
            lidar_path = lidar_dir / f'{frame_idx:06d}.bin'
            if not lidar_path.exists():
                continue

            trajectory = self._compute_trajectory(poses, frame_idx)
            road_mask = self._create_road_mask(trajectory)

            self.samples.append({
                'sequence': sequence,
                'frame_idx': frame_idx,
                'lidar_path': str(lidar_path),
                'road_mask': road_mask,
            })

    def _ensure_cache(self, sequences):
        """Pre-rasterize all BEVs to disk cache if not already cached."""
        # Check if cache is already complete
        uncached = []
        for s in self.samples:
            cache_path = self._bev_cache_path(s['sequence'], s['frame_idx'])
            if not cache_path.exists():
                uncached.append(s)

        if not uncached:
            print(f"  BEV cache complete ({len(self.samples)} samples)")
            return

        print(f"  Caching {len(uncached)} BEV rasterizations (one-time cost)...")
        t0 = time.time()
        for i, s in enumerate(uncached):
            cache_path = self._bev_cache_path(s['sequence'], s['frame_idx'])
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            points = load_kitti_lidar(s['lidar_path'])
            bev = self.rasterizer.rasterize_lidar(points)
            np.save(cache_path, bev)

            if (i + 1) % 1000 == 0 or (i + 1) == len(uncached):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(uncached) - i - 1) / rate
                print(f"    [{i+1}/{len(uncached)}] {rate:.0f} samples/s, ETA {eta:.0f}s")

        total = time.time() - t0
        print(f"  BEV cache done: {len(uncached)} samples in {total:.1f}s")

    def _bev_cache_path(self, sequence, frame_idx):
        return Path(self.cache_dir) / sequence / f'{frame_idx:06d}.npy'

    def _compute_trajectory(self, poses, frame_idx):
        current_pose = poses[frame_idx]
        cx, cy = current_pose[0, 3], current_pose[1, 3]
        trajectory = []
        for i in range(1, len(poses) - frame_idx):
            pose = poses[frame_idx + i]
            x, y = pose[0, 3], pose[1, 3]
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist >= self.waypoint_spacing * (len(trajectory) + 1):
                trajectory.append([x - cx, y - cy])
                if len(trajectory) >= self.num_future:
                    break
        while len(trajectory) < self.num_future:
            trajectory.append(trajectory[-1] if trajectory else [0.0, 0.0])
        return np.array(trajectory[:self.num_future], dtype=np.float32)

    def _create_road_mask(self, trajectory):
        H, W = 37, 50
        mask = np.zeros((1, H, W), dtype=np.float32)
        if len(trajectory) < 2:
            return mask

        x_range, y_range = (-20, 20), (-10, 30)
        res_x = (x_range[1] - x_range[0]) / W
        res_y = (y_range[1] - y_range[0]) / H

        def w2p(pt):
            px = int((pt[0] - x_range[0]) / res_x)
            py = int((pt[1] - y_range[0]) / res_y)
            return (max(0, min(W - 1, px)), max(0, min(H - 1, py)))

        for i in range(len(trajectory) - 1):
            p1, p2 = w2p(trajectory[i]), w2p(trajectory[i + 1])
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

        from scipy.ndimage import binary_dilation
        mask[0] = binary_dilation(mask[0] > 0.5, iterations=2).astype(np.float32)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load from cache (fast) or rasterize (fallback)
        cache_path = self._bev_cache_path(sample['sequence'], sample['frame_idx'])
        if cache_path.exists():
            bev = np.load(cache_path)
        else:
            points = load_kitti_lidar(sample['lidar_path'])
            bev = self.rasterizer.rasterize_lidar(points)

        road_mask = sample['road_mask']

        # Data augmentation (training only): random horizontal flip
        if self.split == 'train' and np.random.random() < 0.5:
            bev = bev[:, :, ::-1].copy()          # flip BEV laterally
            road_mask = road_mask[:, :, ::-1].copy()  # flip mask to match

        return torch.from_numpy(bev), torch.from_numpy(road_mask)


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (bev, road_mask) in enumerate(dataloader):
        bev = bev.to(device, non_blocking=True)
        road_mask = road_mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            _, pred_road_mask = model(bev)
            loss = nn.functional.binary_cross_entropy_with_logits(pred_road_mask, road_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for bev, road_mask in dataloader:
        bev = bev.to(device, non_blocking=True)
        road_mask = road_mask.to(device, non_blocking=True)

        _, pred_road_mask = model(bev)
        loss = nn.functional.binary_cross_entropy_with_logits(pred_road_mask, road_mask)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache_dir', type=str, default='data/kitti/bev_cache')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # Use ReduceLROnPlateau to lower LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    scaler = GradScaler('cuda')

    # Data
    print("\nLoading datasets...")
    # Fair comparison: same train as before, but validate on ALL held-out sequences
    train_sequences = ['00', '02', '05', '07']  # Keep as is for fair comparison
    val_sequences = ['08', '09', '10']          # Use ALL three for validation
    
    train_dataset = KITTIFullDataset(
        sequences=train_sequences, split='train',
        cache_dir=args.cache_dir
    )
    val_dataset = KITTIFullDataset(
        sequences=val_sequences, split='all',
        cache_dir=args.cache_dir
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True
    )

    print(f"\nTrain: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    print(f"Starting training...\n")

    best_val = float('inf')
    epochs_no_improve = 0
    patience = 10
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)
        epoch_time = time.time() - t0

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ({epoch_time:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/encoder_full_best.pth')
            print(f"  New best model saved (val_loss={val_loss:.4f})\n")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)\n")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                print(f"Best validation loss: {best_val:.4f}")
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, 'checkpoints/encoder_full_latest.pth')

    print(f"\n=== Training Complete ===")
    print(f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
