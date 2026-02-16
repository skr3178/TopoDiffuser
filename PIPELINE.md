# TopoDiffuser - Full Integrated Pipeline

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: LiDAR Point Cloud                          │
│                              [N, 4] → [3, 300, 400]                        │
│                         (Height, Intensity, Density)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENCODER (Block 2 - Trainable)                       │
│                                                                             │
│   Input [3,300,400]                                                         │
│      ↓                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ c1 [32,150,200] → c2 [64,75,100] → c3 [128,38,50] → c4 [256,19,25] │  │
│   │                                           ↓                         │  │
│   │                                         c5 [512,10,13]              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│      ↓                              ↓                                       │
│   Decoder p5 [64,8,8]           Decoder p4 [64,19,25]                       │
│                                    ↓                                        │
│                           ┌─────────────────┐                               │
│                           │  SHARED BACKBONE │                              │
│                           │  CB → Conv+Sigmoid│                             │
│                           └────────┬────────┘                               │
│                                    ↓                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    ↓                               ↓                        │
│            Head A: Segmentation           Head B: Conditioning             │
│            [1,37,50] → BCE Loss           Conv → FC → [512]                │
│              (Auxiliary)                      ↓                            │
│                                           Diffusion                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DIFFUSION POLICY (Block 4 - Trainable)                 │
│                                                                             │
│   Forward Process (Training):                                               │
│   τ_0 [B,8,2] ──→ Add Noise ──→ τ_t [B,8,2]                                │
│   τ_t = √γ_t·τ_0 + √(1-γ_t)·ε                                               │
│                                                                             │
│   Reverse Process (g_φ):                                                    │
│   Input: τ_t [B,8,2], c [B,512], t_emb [B,256]                             │
│      ↓                                                                      │
│   Denoising Network (MLP):                                                  │
│      Flatten → Concat → FC[512] × 4 → Output ε̂_t [B,8,2]                   │
│                                                                             │
│   Denoising Step:                                                           │
│   τ_{t-1} = (τ_t - (1-α_t)/√(1-ᾱ_t)·ε̂_t) / √α_t                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LOSS FUNCTIONS                                 │
│                                                                             │
│   Equation 3:  L_diffusion = MSE(ε̂_t, ε)                                    │
│   Equation 4:  L_road = BCE(mask_pred, mask_gt)                             │
│   Equation 5:  L_total = L_diffusion + α_road · L_road                      │
│                                                                             │
│   α_road = 0.1 (default)                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT: Trajectories                             │
│                                                                             │
│   During Inference:                                                         │
│   1. Sample τ_N ~ N(0, I) for K=5 modes                                     │
│   2. Denoise for N=10 steps → τ_0 [K,8,2]                                   │
│   3. Select best via minADE                                                 │
│                                                                             │
│   Metrics: minADE, minFDE, maxADE, HitRate@2m, Hausdorff                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Components

### 1. Configuration (`configs/`)
- `default.yaml` - Full settings
- `lidar_only.yaml` - 3-channel ablation
- `multimodal.yaml` - 5-channel with OSM
- `debug.yaml` - Fast testing

### 2. Core Models (`models/`)
- `encoder.py` - Multimodal encoder with dual heads
- `diffusion.py` - DDPM diffusion process
- `denoising_network.py` - g_φ network (MLP/CNN)
- `losses.py` - Combined loss (Eq 3,4,5)
- `metrics.py` - Evaluation metrics
- `bev_rasterization.py` - LiDAR → BEV

### 3. Pipeline (`pipeline.py`)
Unified entry point for all operations:
```bash
# Training
python pipeline.py --mode train --config configs/lidar_only.yaml

# With overrides
python pipeline.py --mode train --config configs/default.yaml \
    --override training.batch_size=128 training.loss.alpha_road=0.5

# Debug (quick test)
python pipeline.py --mode debug --config configs/debug.yaml
```

### 4. Utilities (`utils/`)
- `config.py` - Config loading, validation, overrides

## Training Modes

### Mode 1: Joint Training (Recommended)
```bash
python pipeline.py --mode train --config configs/lidar_only.yaml
```
- Trains encoder + diffusion together
- End-to-end gradient flow
- Single training loop

### Mode 2: Separate Training
```bash
# Stage 1: Encoder only
python train_encoder_kitti_optimized.py

# Stage 2: Diffusion only (frozen encoder)
python run_diffusion.py --mode train --encoder_ckpt checkpoints/encoder_best.pth
```

### Mode 3: Debug Mode
```bash
python pipeline.py --mode debug --config configs/debug.yaml
```
- Minimal data (sequence 00 only)
- 2-3 epochs
- Small batch size
- No mixed precision (easier debugging)

## Loss Configuration

Control the auxiliary task weight:

```yaml
# configs/custom.yaml
training:
  loss:
    alpha_road: 0.1   # L_total = L_diff + 0.1 * L_road
```

| alpha_road | Effect |
|------------|--------|
| 0.0 | Diffusion only |
| 0.1 | Balanced (default) |
| 0.5 | Strong road supervision |
| 1.0 | Equal weighting |

## Metrics

All metrics computed during validation:
- **minADE**: Min ADE over K samples
- **minFDE**: Min FDE over K samples
- **maxADE**: Max displacement (best prediction)
- **HitRate@2m**: % within 2m threshold
- **Hausdorff**: Shape similarity

## File Structure

```
TopoDiffuser/
├── configs/
│   ├── default.yaml          # Full config
│   ├── lidar_only.yaml       # 3-channel ablation
│   ├── multimodal.yaml       # 5-channel full
│   ├── debug.yaml            # Quick testing
│   └── README.md
│
├── models/
│   ├── encoder.py            # Multimodal encoder
│   ├── diffusion.py          # DDPM
│   ├── denoising_network.py  # g_φ
│   ├── losses.py             # Eq 3,4,5
│   ├── metrics.py            # Evaluation
│   └── bev_rasterization.py
│
├── utils/
│   └── config.py             # Config system
│
├── pipeline.py               # Main entry point
├── train_joint.py            # Alternative joint training
├── run_diffusion.py          # Alternative separate training
│
├── LOSS_EQUATIONS.md         # Loss documentation
└── PIPELINE.md               # This file
```

## Quick Start

```bash
# 1. Test config system
python utils/config.py

# 2. Debug mode (2 epochs, small data)
python pipeline.py --mode debug --config configs/debug.yaml

# 3. Full training (LiDAR-only)
python pipeline.py --mode train --config configs/lidar_only.yaml

# 4. With custom settings
python pipeline.py --mode train --config configs/default.yaml \
    --override training.epochs=100 training.batch_size=64
```
