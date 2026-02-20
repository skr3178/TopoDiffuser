#!/usr/bin/env python3
"""
Train Multimodal Encoder on Paper Split Dataset.

Uses the precomputed paper-split BEV data:
  Train: data/paper_split/train/  (3,860 samples — seqs 00,02,05,07)
  Val:   data/paper_split/test/   (2,270 samples — seqs 08,09,10)

Input:  [B, 4, 300, 400] float16 BEV  (height, intensity, density, OSM)
Output: [B, 512] conditioning vector  +  [B, 1, 37, 50] road segmentation

Training objective: predict future-trajectory road mask (auxiliary self-supervision).

Usage:
  conda run -n nuscenes python train_paper_encoder.py
  conda run -n nuscenes python train_paper_encoder.py --epochs 50 --batch_size 64
  conda run -n nuscenes python train_paper_encoder.py --init_from_3ch checkpoints/encoder_expanded_best.pth
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, 'models')
from multimodal_encoder import build_full_multimodal_encoder


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PaperSplitDataset(Dataset):
    """
    Loads precomputed 5-channel BEV maps from the paper split.

    Each sample:
      bev       — float32 [5, 300, 400]  (loaded from float16 .npy)
      road_mask — float32 [1,  37,  50]  (derived from future trajectory)
    """

    def __init__(self, meta_path: str, augment: bool = False):
        with open(meta_path, 'rb') as f:
            self.samples = pickle.load(f)
        self.augment = augment
        print(f"  Loaded {len(self.samples)} samples from {meta_path}")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s         = self.samples[idx]
        bev       = np.load(s['npy_path']).astype(np.float32)      # float16 → float32
        road_mask = s['road_mask'].astype(np.float32)               # float16 → float32

        # Augmentation: random horizontal flip (left/right symmetry)
        if self.augment and np.random.random() < 0.5:
            bev       = bev      [:, :, ::-1].copy()
            road_mask = road_mask[:, :, ::-1].copy()

        return torch.from_numpy(bev), torch.from_numpy(road_mask)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss, n = 0.0, 0
    t0 = time.time()

    for i, (bev, mask) in enumerate(loader):
        bev  = bev .to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type='cuda'):
            _, pred_mask = model(bev)
            loss = nn.functional.binary_cross_entropy_with_logits(pred_mask, mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n += 1

        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{epoch}][{i:4d}/{len(loader)}] "
                  f"loss={loss.item():.4f}  "
                  f"({elapsed:.0f}s elapsed)")

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for bev, mask in loader:
        bev  = bev .to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with autocast(device_type='cuda'):
            _, pred_mask = model(bev)
            loss = nn.functional.binary_cross_entropy_with_logits(pred_mask, mask)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train 5-ch multimodal encoder')
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--workers',       type=int,   default=4)
    parser.add_argument('--init_from_3ch', type=str,   default=None,
                        help='3-ch encoder .pth for warm-start')
    parser.add_argument('--resume',        type=str,   default=None,
                        help='Resume from checkpoint .pth')
    parser.add_argument('--save_dir',      type=str,   default='checkpoints')
    parser.add_argument('--train_meta',    type=str,
                        default='data/paper_split/train_meta.pkl')
    parser.add_argument('--val_meta',      type=str,
                        default='data/paper_split/test_meta.pkl')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('  TopoDiffuser — Train Multimodal Encoder (Paper Split)')
    print('=' * 65)
    print(f'  Device:     {device}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Epochs:     {args.epochs}')
    print(f'  LR:         {args.lr}')

    # ── Datasets ────────────────────────────────────────────────────────────
    print('\nLoading datasets...')
    train_ds = PaperSplitDataset(args.train_meta, augment=True)
    val_ds   = PaperSplitDataset(args.val_meta,   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0), drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0))

    print(f'  Train batches: {len(train_loader)}')
    print(f'  Val   batches: {len(val_loader)}')

    # ── Model ────────────────────────────────────────────────────────────────
    print('\nBuilding encoder...')
    model = build_full_multimodal_encoder(
        input_channels=4, conditioning_dim=512
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {total_params:,}  ({total_params*4/1e6:.1f} MB)')

    start_epoch = 1
    best_val    = float('inf')

    if args.resume:
        print(f'  Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val    = ckpt.get('best_val_loss', float('inf'))
        print(f'  Resuming at epoch {start_epoch}, best val={best_val:.4f}')
    elif args.init_from_3ch:
        print(f'  Warm-starting from 3-ch encoder: {args.init_from_3ch}')
        model.init_from_3channel_encoder(args.init_from_3ch)

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    # ── Training loop ────────────────────────────────────────────────────────
    print('\n' + '─' * 65)
    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t_ep
        lr_now  = optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch:3d}/{args.epochs}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={lr_now:.2e}  ({elapsed:.0f}s)')

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        ckpt = {
            'epoch':            epoch,
            'model_state_dict': model.state_dict(),
            'val_loss':         val_loss,
            'best_val_loss':    best_val,
        }
        torch.save(ckpt, save_dir / 'paper_encoder_latest.pth')
        if is_best:
            torch.save(ckpt, save_dir / 'paper_encoder_best.pth')
            print(f'  ✓ New best val loss: {best_val:.4f}')

    print('\n' + '=' * 65)
    print(f'  Training complete.  Best val loss: {best_val:.4f}')
    print(f'  Checkpoint: {save_dir}/paper_encoder_best.pth')
    print('=' * 65)


if __name__ == '__main__':
    main()
