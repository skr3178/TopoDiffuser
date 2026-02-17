#!/usr/bin/env python3
"""
Train 5-Channel Encoder on Full Paper Dataset.

Channels:
  0-2: LiDAR (height, intensity, density)
  3: Trajectory history
  4: OSM roads (or trajectory proxy)

Usage:
    # Precompute 5ch BEV cache first:
    python precompute_bev_5ch.py --sequences 00 02 05 07 08 09 10
    
    # Then train with warm-start from 3ch encoder:
    python train_encoder_full_5ch.py \
        --epochs 50 \
        --batch_size 128 \
        --init_from_3ch checkpoints/encoder_expanded_best.pth
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
sys.path.insert(0, 'utils')

from encoder import build_encoder
from multimodal_encoder import build_full_multimodal_encoder


class KITTIFull5chDataset(Dataset):
    """
    Dataset for 5-channel encoder training.
    
    Loads precomputed 5ch BEV from cache.
    """

    def __init__(self, sequences=['00', '02', '05', '07'], split='train',
                 data_root='data/kitti', cache_dir='data/kitti/bev_cache_5ch',
                 num_future=8, waypoint_spacing=2.0):
        self.data_root = data_root
        self.cache_dir = cache_dir
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
        
        # Verify cache exists
        self._verify_cache()

    def _load_sequence(self, sequence):
        """Load samples from sequence using precomputed 5ch cache."""
        cache_seq_dir = Path(self.cache_dir) / sequence
        pose_file = Path(self.data_root) / 'poses' / f'{sequence}.txt'

        if not pose_file.exists():
            print(f"  Warning: {pose_file} not found")
            return
        
        if not cache_seq_dir.exists():
            print(f"  Warning: Cache directory {cache_seq_dir} not found")
            print(f"  Run: python precompute_bev_5ch.py --sequences {sequence}")
            return

        # Load poses to get trajectory for road mask
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(np.array(values).reshape(3, 4))

        print(f"  Sequence {sequence}: {len(poses)} frames")

        for frame_idx in range(len(poses) - self.num_future - 1):
            cache_path = cache_seq_dir / f'{frame_idx:06d}.npy'
            if not cache_path.exists():
                continue

            trajectory = self._compute_trajectory(poses, frame_idx)
            road_mask = self._create_road_mask(trajectory)

            self.samples.append({
                'sequence': sequence,
                'frame_idx': frame_idx,
                'cache_path': str(cache_path),
                'road_mask': road_mask,
            })

    def _verify_cache(self):
        """Verify that cache files exist."""
        missing = []
        for s in self.samples:
            if not Path(s['cache_path']).exists():
                missing.append(s)
        
        if missing:
            print(f"  Warning: {len(missing)} samples missing from cache")
            # Remove missing samples
            self.samples = [s for s in self.samples if Path(s['cache_path']).exists()]
            print(f"  Proceeding with {len(self.samples)} valid samples")

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

        # Load 5ch BEV from cache
        bev = np.load(sample['cache_path'])
        
        # Verify shape
        if bev.shape[0] != 5:
            raise ValueError(f"Expected 5-channel BEV, got {bev.shape[0]} channels. "
                           f"Run precompute_bev_5ch.py first.")

        road_mask = sample['road_mask']

        # Data augmentation (training only): random horizontal flip
        if self.split == 'train' and np.random.random() < 0.5:
            bev = bev[:, :, ::-1].copy()
            road_mask = road_mask[:, :, ::-1].copy()

        return torch.from_numpy(bev), torch.from_numpy(road_mask)


def init_encoder_from_3ch(encoder_5ch, checkpoint_path):
    """
    Initialize 5-channel encoder from 3-channel encoder checkpoint.
    
    Copies weights for shared layers, expands first conv layer.
    """
    print(f"Loading 3-channel encoder from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict_3ch = checkpoint['model_state_dict']
    elif 'encoder_state_dict' in checkpoint:
        state_dict_3ch = checkpoint['encoder_state_dict']
    else:
        state_dict_3ch = checkpoint
    
    # Get 5ch state dict
    state_dict_5ch = encoder_5ch.state_dict()
    
    # Track what we're loading
    loaded = []
    expanded = []
    skipped = []
    
    for name, param_5ch in state_dict_5ch.items():
        if name in state_dict_3ch:
            param_3ch = state_dict_3ch[name]
            
            if param_5ch.shape == param_3ch.shape:
                # Direct copy
                state_dict_5ch[name].copy_(param_3ch)
                loaded.append(name)
            elif 'c1.conv.weight' in name:
                # Expand first conv: 3 -> 5 channels
                print(f"  Expanding {name}: {param_3ch.shape} -> {param_5ch.shape}")
                with torch.no_grad():
                    # Copy existing 3 channels
                    state_dict_5ch[name][:, :3, :, :].copy_(param_3ch)
                    # Initialize channels 3-4 with mean of existing
                    mean_weight = param_3ch.mean(dim=1, keepdim=True)
                    state_dict_5ch[name][:, 3:4, :, :].copy_(mean_weight)
                    state_dict_5ch[name][:, 4:5, :, :].copy_(mean_weight)
                expanded.append(name)
            else:
                skipped.append((name, param_3ch.shape, param_5ch.shape))
        else:
            skipped.append((name, 'not in 3ch', param_5ch.shape))
    
    print(f"  Loaded {len(loaded)} layers directly")
    print(f"  Expanded {len(expanded)} layers")
    print(f"  Skipped {len(skipped)} layers (new or mismatched)")


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
    parser.add_argument('--cache_dir', type=str, default='data/kitti/bev_cache_5ch',
                       help='Path to 5-channel BEV cache')
    parser.add_argument('--init_from_3ch', type=str, default=None,
                       help='Path to 3-channel encoder checkpoint for warm-start')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("5-Channel Encoder Training")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cache: {args.cache_dir}")
    print("=" * 70)

    # Build 5-channel encoder
    model = build_full_multimodal_encoder(
        input_channels=5, 
        conditioning_dim=512
    ).to(device)
    
    # Warm-start from 3-channel encoder if specified
    if args.init_from_3ch:
        init_encoder_from_3ch(model, args.init_from_3ch)
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    scaler = GradScaler('cuda')

    # Data
    print("\nLoading datasets...")
    train_sequences = ['00', '02', '05', '07']
    val_sequences = ['08', '09', '10']
    
    train_dataset = KITTIFull5chDataset(
        sequences=train_sequences, split='train',
        cache_dir=args.cache_dir
    )
    val_dataset = KITTIFull5chDataset(
        sequences=val_sequences, split='all',
        cache_dir=args.cache_dir
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - t0
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'encoder_full_5ch_best.pth'))
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(args.save_dir, 'encoder_full_5ch_latest.pth'))

    print("\n" + "=" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
