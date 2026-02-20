#!/usr/bin/env python3
"""
Probe conditioning vector statistics from the trained encoder.
Runs encoder over all train+val BEVs and reports per-feature std.
"""
import pickle
import sys
import numpy as np
import torch
from pathlib import Path
from torch.amp import autocast

sys.path.insert(0, 'models')
from multimodal_encoder import build_full_multimodal_encoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT   = 'checkpoints/paper_encoder_best.pth'
TRAIN  = 'data/paper_split/train_meta.pkl'
VAL    = 'data/paper_split/test_meta.pkl'
BATCH  = 128

def encode_meta(encoder, meta_path):
    with open(meta_path, 'rb') as f:
        samples = pickle.load(f)
    all_cond = []
    for start in range(0, len(samples), BATCH):
        batch = samples[start:start+BATCH]
        bevs  = torch.stack([
            torch.from_numpy(np.load(s['npy_path']).astype(np.float32))
            for s in batch
        ]).to(DEVICE)
        with torch.no_grad(), autocast('cuda'):
            cond, _ = encoder(bevs)
        all_cond.append(cond.float().cpu())
        print(f'  {start+len(batch)}/{len(samples)}', end='\r')
    print()
    return torch.cat(all_cond)   # [N, 512]

def main():
    print(f'Loading encoder from {CKPT}')
    encoder = build_full_multimodal_encoder(input_channels=5, conditioning_dim=512).to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt['model_state_dict'])
    encoder.eval()
    epoch    = ckpt.get('epoch', '?')
    best_val = ckpt.get('best_val_loss', float('nan'))
    print(f'  Epoch {epoch},  best_val_loss={best_val:.4f}')

    print('\nEncoding TRAIN set...')
    cond_train = encode_meta(encoder, TRAIN)   # [3860, 512]

    print('Encoding VAL set...')
    cond_val   = encode_meta(encoder, VAL)     # [2270, 512]

    cond_all = torch.cat([cond_train, cond_val])   # [6130, 512]

    # ── Statistics ────────────────────────────────────────────────────────────
    per_feat_std  = cond_all.std(dim=0)        # [512]
    per_feat_mean = cond_all.mean(dim=0)       # [512]

    print('\n' + '='*55)
    print('  CONDITIONING VECTOR STATISTICS')
    print('='*55)
    print(f'  Samples        : {cond_all.shape[0]}')
    print(f'  Feature dim    : {cond_all.shape[1]}')
    print()
    print(f'  Mean of per-feature std  : {per_feat_std.mean().item():.4f}')
    print(f'  Median of per-feature std: {per_feat_std.median().item():.4f}')
    print(f'  Min per-feature std      : {per_feat_std.min().item():.4f}')
    print(f'  Max per-feature std      : {per_feat_std.max().item():.4f}')
    print()
    print(f'  Global mean              : {per_feat_mean.mean().item():.4f}')
    print(f'  Global std (all features): {cond_all.std().item():.4f}')
    print()

    # Distribution of per-feature stds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    print('  Per-feature std distribution:')
    for thr in thresholds:
        frac = (per_feat_std < thr).float().mean().item()
        print(f'    std < {thr:.2f}: {frac*100:.1f}% of features')

    print('='*55)

if __name__ == '__main__':
    main()
