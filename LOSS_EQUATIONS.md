# TopoDiffuser Loss Functions

This document describes the loss functions from the paper and their implementation.

---

## Paper Equations

### Equation 3: Diffusion Loss (Main Task)

```
L_diffusion = ||ε - g_φ(τ_t, c, t)||²
```

**Where:**
- `ε` ~ N(0, I): Ground truth Gaussian noise
- `g_φ`: Denoising network (MLP/CNN)
- `τ_t`: Noised trajectory at timestep t
- `c`: Conditioning vector [B, 512] from encoder
- `t`: Timestep

**Purpose**: Train the diffusion model to predict the noise that was added to the trajectory during the forward process.

**Implementation**: `models/losses.py::DiffusionLoss`
```python
loss = MSELoss(predicted_noise, target_noise)
```

---

### Equation 4: Road Segmentation Loss (Auxiliary Task)

```
L_road = -Σ [y·log(x) + (1-y)·log(1-x)]
```

**Where:**
- `x`: Predicted road mask [B, 1, 37, 50] (after sigmoid)
- `y`: Ground truth road mask [B, 1, 37, 50] (binary)

**Purpose**: Auxiliary supervision to help the encoder learn spatial features of drivable areas. The road mask is created by rasterizing the future trajectory.

**Implementation**: `models/losses.py::RoadSegmentationLoss`
```python
loss = BCELoss(pred_mask, gt_mask)
```

---

### Equation 5: Total Loss

```
L_total = L_diffusion + α_road · L_road
```

**Where:**
- `α_road`: Weight for road loss (paper suggests 0.1-1.0, default: 0.1)

**Purpose**: Multi-task learning that combines:
1. **Main task**: Trajectory prediction via diffusion
2. **Auxiliary task**: Road segmentation

The auxiliary task provides additional gradient signal to the encoder, improving spatial representations.

**Implementation**: `models/losses.py::TopoDiffuserLoss`
```python
l_total = l_diffusion + alpha_road * l_road
```

---

## Training Modes

### Mode 1: Separate Training (Current)

```
Stage 1: Train Encoder (with L_road only)
   Encoder → Segmentation Head → BCE Loss

Stage 2: Train Diffusion (with L_diffusion only, encoder frozen)
   Encoder (frozen) → Conditioning → Diffusion → MSE Loss
```

**Scripts**: `train_encoder_kitti_optimized.py` → `run_diffusion.py --mode train`

---

### Mode 2: Joint Training (Recommended)

```
Train Everything Together:
   Encoder → Segmentation Head → L_road (BCE)
         ↓
   Conditioning → Diffusion → L_diffusion (MSE)
         ↓
   L_total = L_diffusion + α_road · L_road
```

**Script**: `train_joint.py`

**Advantages**:
- End-to-end gradient flow
- Better feature learning through multi-task supervision
- Single training loop

**Usage**:
```bash
python train_joint.py \
    --alpha_road 0.1 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4
```

---

## File Structure

```
models/
├── losses.py              # Loss function implementations
│   ├── DiffusionLoss      # Equation 3
│   ├── RoadSegmentationLoss  # Equation 4
│   ├── TopoDiffuserLoss   # Equation 5
│   └── UncertaintyWeightedLoss  # Optional: learned weights
│
train_joint.py             # Joint training script
run_diffusion.py           # Separate training/eval/inference
```

---

## Hyperparameter: α_road

| α_road | Effect | Use Case |
|--------|--------|----------|
| 0.0 | Diffusion only | Ablation study |
| 0.1 | Slight road supervision | **Default (recommended)** |
| 0.5 | Balanced multi-task | If road quality is important |
| 1.0 | Equal weighting | If segmentation is equally important |

**Paper Recommendation**: α_road ≈ 0.1-1.0 (Section III-C)

**Note**: The auxiliary loss should not dominate. If L_road >> L_diffusion, reduce α_road.

---

## Training Monitoring

During joint training, monitor both losses:

```
Epoch [10/50]:
  Train: L_total=1.234 (L_diff=1.123, L_road=1.111)
  Val: minADE=0.85m | HitRate=0.92
```

**Healthy Training Signs**:
- Both L_diff and L_road decreasing
- L_diff ≈ 0.5-2.0 (MSE on noise prediction)
- L_road ≈ 0.1-0.5 (BCE on segmentation)
- minADE < 1.0m
- HitRate@2m > 0.9

**Warning Signs**:
- L_road dominates → reduce α_road
- L_diff doesn't decrease → check learning rate
- minADE high > 2.0m → train longer or check data
