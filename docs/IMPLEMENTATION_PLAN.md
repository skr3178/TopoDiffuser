# TopoDiffuser Implementation Plan

## Paper Overview

**Title:** TopoDiffuser: A Diffusion-Based Multimodal Trajectory Prediction Model with Topometric Maps

**Core Idea:** A diffusion-based framework for multimodal trajectory prediction that incorporates topometric maps (OpenStreetMap/OSM) to generate accurate, diverse, and road-compliant future motion forecasts.

---

## 1. Architecture Overview

### Development Strategy: Incremental Modality Addition

Based on Table II ablation results, **LiDAR-only is already a strong baseline** (FDE 0.55, HitRate 0.93). We implement modalities incrementally:

1. **Phase 1:** LiDAR-only (3 channels) - Strong baseline
2. **Phase 2:** Add Trajectory History (+1 channel) - Major improvement (HD↓ 14.8%)
3. **Phase 3:** Add Topometric Map (+1 channel) - Road compliance boost

```
┌─────────────────────────────────────────────────────────────────┐
│                        TopoDiffuser                             │
├─────────────────────────────────────────────────────────────────┤
│  Input Modalities (Phased Implementation)                       │
│  ├── Phase 1: LiDAR BEV (Height, Intensity, Density)  [H×W×3]  │
│  ├── Phase 2: + Trajectory History (Binary mask)      [H×W×1]  │
│  └── Phase 3: + Topometric Map (OSM Route mask)       [H×W×1]  │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Multimodal Conditioning Encoder (CNN)           │   │
│  │  - CBR Blocks (Conv → BatchNorm → ReLU)                 │   │
│  │  - Deconvolution layers for upsampling                  │   │
│  │  - 1×1 Conv for dimension adjustment                    │   │
│  │  - Output: c ∈ R^(H2×W2) → flattened conditioning vector │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Conditional Diffusion Model                   │   │
│  │  - Lightweight U-Net denoising network                  │   │
│  │  - N denoising steps (default: 10)                      │   │
│  │  - Predicts noise ε̂ at each step                        │   │
│  │  - Output: τ₀ ∈ R^(Tf×2) future trajectory              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Key Components

### 2.1 Input Representation (Phased)

All inputs are rasterized into a unified BEV frame with resolution **H₀ × W₀ = 300 × 400**.

| Phase | Input | Channels | Shape | Description |
|-------|-------|----------|-------|-------------|
| 1 | LiDAR BEV | 3 | [300, 400, 3] | Height, Intensity, Density |
| 2 | LiDAR + History | 4 | [300, 400, 4] | + Past 5 keyframes (2m spacing) |
| 3 | LiDAR + History + Map | 5 | [300, 400, 5] | + OSM route binary mask |

**Tensor Shapes by Phase:**
```python
# Phase 1: LiDAR-only
x_input = [8, 3, 300, 400]       # Batch=8, 3 channels (LiDAR only)

# Phase 2: LiDAR + History
x_input = [8, 4, 300, 400]       # Batch=8, 4 channels (LiDAR×3 + history×1)

# Phase 3: Full model
x_input = [8, 5, 300, 400]       # Batch=8, 5 channels (LiDAR×3 + history×1 + map×1)

# Common outputs
obs_cond_before = [8, 64, 8]     # Intermediate features
obs_cond_after = [8, 512]        # Flattened conditioning vector (64×8)
diffusion_output = [40, 8, 2]    # 5 samples × 8 waypoints = 40, each (x,y)
```

### 2.2 Multimodal Conditioning Encoder

**Architecture:**
```
Input [B, C, 300, 400]  where C = 3, 4, or 5 (by phase)
    ↓
CBR Blocks (c1 → c2 → c3 → c4 → c5)
    ↓
Deconvolution layers (p5, p4) for top-down fusion
    ↓
1×1 Conv + Sigmoid → Road segmentation head (auxiliary)
    ↓
Flatten → obs_cond [B, 512]
```

**CBR Block:** Conv → BatchNorm → ReLU  
**CB Block:** Conv → BatchNorm  
**FC:** Fully Connected layer  
**Deconvolution layers** for upsampling  
**1×1 Convolution layers** for dimension adjustment  
**Sigmoid activations** for binary prediction tasks


**Dual Output:**
1. **Conditioning vector** `c` → fed into diffusion model
2. **Road segmentation mask** → auxiliary supervision (Binary Cross-Entropy Loss)

### 2.3 Conditional Diffusion Model

**Forward Process (Noise Schedule):**

The forward process progressively perturbs the ground-truth trajectory $\tau_0$ by adding Gaussian noise:

$$
q(\tau_t | \tau_0) = \mathcal{N}(\tau_t; \sqrt{\gamma_t}\tau_0, (1-\gamma_t)I)
$$

where $\gamma_t$ is a monotonically decreasing noise schedule.

**Reverse Process:**
- Denoising network $g_\phi$ predicts noise $\hat{\varepsilon}$ given $(\tau_t, t, c)$
- Timestep $t$ embedded using sinusoidal positional encodings
- Context $c$ from multimodal encoder

**Training Loss:**

The diffusion process is supervised using a mean squared error (MSE) loss between the predicted and actual noise:

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t,\tau_0,\varepsilon}\left[ \left\| \varepsilon - g_\phi(\tau_t, t, c) \right\|_2^2 \right]
$$

where $\tau_0$ is the ground-truth trajectory, $\tau_t$ is the noisy trajectory at timestep $t$, $\varepsilon$ is the sampled Gaussian noise, $c$ is the conditioning context, and $g_\phi$ is the denoising network parameterized by $\phi$.

To enhance road-awareness, an auxiliary road segmentation head is trained using binary cross-entropy loss:

$$
\mathcal{L}_{\text{road}} = -\sum_{i=1}^{H'} \sum_{j=1}^{W'} \left[ y_{i,j} \log(x_{i,j}) + (1-y_{i,j}) \log(1-x_{i,j}) \right]
$$

The final training objective combines both losses:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diffusion}} + \lambda_{\text{road}} \cdot \mathcal{L}_{\text{road}}
$$

---

## 3. Implementation Details

### 3.1 Dataset: KITTI

| Split | Sequences | Samples | Purpose |
|-------|-----------|---------|---------|
| Train | 00, 02, 05, 07 | 3,860 | Training all phases |
| Test-08 | 08 | 1,391 | Evaluation |
| Test-09 | 09 | 530 | Evaluation (main benchmark) |
| Test-10 | 10 | 349 | Evaluation |

**Trajectory Specifications:**
- History: 5 keyframes @ 2-meter intervals
- Future: 8 waypoints to predict
- Topometric map: OSM route centered at ego position (Phase 3)

### 3.2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| BEV Resolution | 300 × 400 |
| Learning Rate | 3×10⁻³ (with cosine decay) |
| Optimizer | Adam |
| Batch Size | 8 |
| Epochs | 120 |
| Denoising Steps (N) | 10 |
| Diffusion Steps (Training) | 10 |
| Number of Samples (Inference) | 5 |

### 3.3 Evaluation Metrics

| Metric | Description | Target (KITTI-09) |
|--------|-------------|-------------------|
| **FDE** ↓ | Final Displacement Error | 0.31m (full), 0.55m (LiDAR-only) |
| **minADE** ↓ | Min Average Displacement Error | 0.13m (full), 0.28m (LiDAR-only) |
| **HitRate** ↑ | % within threshold d | 0.99 (full), 0.93 (LiDAR-only) |
| **HD** ↓ | Hausdorff Distance | 1.21m (full), 1.55m (LiDAR-only) |

---

## 4. Project Structure

```
TopoDiffuser/
├── configs/
│   ├── lidar_only.yaml           # Phase 1: LiDAR-only config (3 ch)
│   ├── lidar_history.yaml        # Phase 2: LiDAR + History config (4 ch)
│   └── full_model.yaml           # Phase 3: Full model config (5 ch)
├── data/
│   ├── kitti/                    # KITTI raw dataset
│   ├── osm/                      # OpenStreetMap route data
│   └── preprocess.py             # Data preprocessing scripts
├── models/
│   ├── __init__.py
│   ├── encoder.py                # Multimodal conditioning encoder
│   ├── diffusion.py              # Conditional diffusion model
│   └── unet.py                   # Lightweight U-Net denoiser
├── utils/
│   ├── __init__.py
│   ├── bev_utils.py              # BEV rasterization utilities
│   ├── diffusion_schedule.py     # Noise schedules
│   └── metrics.py                # Evaluation metrics
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── inference.py                  # Inference demo
└── requirements.txt
```

---

## 5. Implementation Tasks

### Phase 1: LiDAR-Only Baseline ⭐ PRIORITY
**Goal:** Strong baseline using only LiDAR BEV (3 channels)
**Target Performance:** FDE 0.55, minADE 0.28, HitRate 0.93, HD 1.55

- [x] Download KITTI raw dataset (sequences 00, 02, 05, 07, 08, 09, 10)
- [x] Implement BEV rasterization for LiDAR (height, intensity, density)
- [x] Implement CBR and CB blocks
- [x] Implement Multimodal Conditioning Encoder (adapted for 3 channels)
- [ ] Implement sinusoidal timestep embeddings
- [ ] Implement lightweight U-Net denoiser
- [ ] Implement forward diffusion process
- [ ] Implement reverse diffusion sampling
- [ ] Implement training loop with MSE loss
- [ ] Implement evaluation metrics (FDE, minADE, HitRate, HD)
- [ ] Train LiDAR-only baseline
- [ ] Evaluate on KITTI-08, 09, 10
- [ ] Document baseline results

### Phase 2: Add Trajectory History
**Goal:** Add ego-motion history (+1 channel) for better road alignment
**Expected Improvement:** HD↓ from 1.55 to ~1.40 (14.8% of full improvement)

- [x] Implement trajectory history rasterization
- [ ] Update encoder to accept 4 channels
- [ ] Update dataset to return history masks
- [ ] Fine-tune from Phase 1 checkpoint
- [ ] Evaluate and compare with Phase 1
- [ ] Ablation: History vs no History

### Phase 3: Add Topometric Map (OSM)
**Goal:** Full model with OSM road topology (+1 channel)
**Expected Improvement:** Complete model (HD 1.32)

- [x] Extract OSM routes for KITTI sequences
- [ ] Implement OSM route mask generation
- [ ] Update encoder to accept 5 channels
- [ ] Update dataset to return OSM masks
- [ ] Fine-tune from Phase 2 checkpoint
- [ ] Evaluate full model
- [ ] Complete ablation studies (Table II reproduction)

### Phase 4: Analysis & Optimization
- [ ] Study effect of denoising steps (5, 10, 20)
- [ ] Study effect of number of samples (1, 4, 8, 16)
- [ ] Compare with baselines (CoverNet, MTP, TP)
- [ ] Inference time optimization
- [ ] Export trained models

---

## 6. Key Design Decisions

### 6.1 Why Start with LiDAR-Only?

**From Table II Ablation:**

| Model | FDE ↓ | minADE ↓ | HitRate ↑ | HD ↓ |
|-------|-------|----------|-----------|------|
| LiDAR only | 0.55 | 0.28 | 0.93 | 1.55 |
| LiDAR + Map | 0.55 | 0.25 | 0.93 | 1.50 |
| LiDAR + Map + History | 0.55 | 0.26 | 0.93 | 1.32 |

**Key Insights:**
1. **LiDAR carries the strongest signal** - Already achieves 0.55 FDE, 93% HitRate
2. **History provides major improvement** - HD↓ 0.23m (14.8% improvement)
3. **Map provides modest gains** - HD↓ 0.05m (structural guidance)
4. **Incremental development reduces risk** - Validate each component before adding complexity

### 6.2 Training Strategy

- **Phase 1:** Train LiDAR-only from scratch
- **Phase 2:** Fine-tune Phase 1 with history added
- **Phase 3:** Fine-tune Phase 2 with OSM added
- **Alternative:** Train full model from scratch if fine-tuning fails

### 6.3 Multimodal Fusion

- **LiDAR:** Captures scene geometry and obstacles (core signal)
- **History:** Provides motion dynamics context (major boost to road alignment)
- **OSM:** Provides road topology and compliance (structural refinement)

---

## 7. Ablation Results (from paper)

| Model | FDE ↓ | minADE ↓ | HitRate ↑ | HD ↓ |
|-------|-------|----------|-----------|------|
| LiDAR only | 0.55 | 0.28 | 0.93 | 1.55 |
| LiDAR + Map | 0.55 | 0.25 | 0.93 | 1.50 |
| LiDAR + Map + History | 0.55 | 0.26 | 0.93 | 1.32 |

**Key Insight:** History trajectory significantly improves HD by 14.8%, indicating better alignment with road structure.

---

## 8. Performance Benchmarks (KITTI-09)

| Method | FDE ↓ | minADE ↓ | HitRate ↑ | HD ↓ | Infer Time (s) |
|--------|-------|----------|-----------|------|----------------|
| CoverNet | 4.48 | 0.43 | 0.85 | 2.74 | 0.006 |
| MTP | 1.07 | 0.18 | 0.98 | 1.62 | 0.013 |
| TP | 0.55 | 0.23 | 0.98 | 2.49 | 0.018 |
| **TopoDiffuser (LiDAR only)** | **0.55** | **0.28** | **0.93** | **1.55** | ~0.04 |
| **TopoDiffuser (Full)** | **0.31** | **0.13** | **0.99** | **1.21** | 0.055 |

---

## 9. References & Resources

- **Paper:** TopoDiffuser (ICRA submission)
- **Code Repository:** https://github.com/EI-Nav/TopoDiffuser
- **Dataset:** KITTI Raw (http://www.cvlibs.net/datasets/kitti/)
- **Map Data:** OpenStreetMap (https://www.openstreetmap.org/)

### Related Works
- **Diffusion Policy:** Diffusion-based behavior cloning
- **CoverNet:** Multi-modal trajectory prediction using coverage
- **MTP:** Multiple trajectory prediction for autonomous driving
- **TP:** Trajectory prediction with transformers

---

## 10. Notes

### Inference Time vs Accuracy Trade-off
- LiDAR-only inference: ~0.04s (estimated)
- Full model inference: ~0.055s
- Baseline models: 0.005-0.018s
- However, achieves 28-44% improvement in accuracy metrics
- Acceptable trade-off for safety-critical applications

### Denoising Steps Sweet Spot
- Performance saturates after ~20 steps
- Recommended: 10 steps for practical deployment
- Paper shows 5→20 improves metrics, then plateaus

### Number of Samples
- Performance saturates at ~8 samples
- Recommended: 5-8 samples during inference
- Default: 5 samples

### Development Priority
1. **Get LiDAR-only working first** - This validates the core diffusion architecture
2. **Add history second** - This provides the biggest accuracy improvement
3. **Add OSM last** - This provides structural refinement

---

## 11. Current Implementation Status

### ✅ Completed Components

| Component | File | Status |
|-----------|------|--------|
| **Multimodal Encoder** | `models/encoder.py` | ✅ CBR blocks, conditioning output [B,512], configurable input channels |
| **LiDAR BEV Encoding** | `utils/bev_utils.py` | ✅ Height, Intensity, Density channels |
| **Trajectory BEV Encoding** | `utils/bev_utils.py` | ✅ Binary mask from pose history |
| **OSM BEV Encoding** | `utils/bev_utils.py` | ✅ Binary mask (placeholder works, real OSM available) |
| **KITTI Dataset** | `utils/dataset.py` | ✅ Loads LiDAR, poses, creates BEV inputs (configurable channels) |

### 📊 Tested Output Shapes

```python
# Phase 1: LiDAR-only (3 channels)
Input shape: torch.Size([8, 3, 300, 400])      # Batch=8, 3 channels
Conditioning shape: torch.Size([8, 512])       # Matches paper ✓

# Phase 2: LiDAR + History (4 channels)
Input shape: torch.Size([8, 4, 300, 400])      # Batch=8, 4 channels
Conditioning shape: torch.Size([8, 512])       # Matches paper ✓

# Phase 3: Full model (5 channels)
Input shape: torch.Size([8, 5, 300, 400])      # Batch=8, 5 channels
Conditioning shape: torch.Size([8, 512])       # Matches paper ✓
Road mask shape: torch.Size([8, 1, 300, 400])  # Aux segmentation output

# Dataset outputs
Target trajectory shape: torch.Size([8, 2])    # 8 future waypoints (x,y)
Road mask shape: torch.Size([1, 300, 400])     # Ground truth road mask
```

### ⬜ Missing Components (Priority Order)

| Component | Priority | Phase | Notes |
|-----------|----------|-------|-------|
| **Diffusion Model** | High | 1 | U-Net denoiser, forward/reverse diffusion process |
| **Timestep Embeddings** | High | 1 | Sinusoidal positional encodings for diffusion steps |
| **Training Script** | High | 1 | Loss functions, optimizer, training loop for LiDAR-only |
| **Evaluation Metrics** | High | 1 | FDE, minADE, HitRate, Hausdorff Distance |
| **Phase 2 Integration** | Medium | 2 | Add history channel, fine-tuning support |
| **Phase 3 Integration** | Medium | 3 | Add OSM channel, full model training |

### 🔗 Dataset Availability

| Source | Status | Path |
|--------|--------|------|
| KITTI Raw | ✅ Available | `data/kitti/` (symlink to CoPilot4D) |
| LiDAR (.bin) | ✅ Available | `data/kitti/sequences/XX/velodyne/` |
| Poses (.txt) | ✅ Available | `data/kitti/poses/XX.txt` |
| OSM Routes | ✅ Available | `data/osm/XX_osm.pkl` (370-717 edges per seq) |

### 📈 Model Parameters

```
MultimodalEncoder (3 channels):  ~10M parameters
MultimodalEncoder (4 channels):  ~12M parameters  
MultimodalEncoder (5 channels):  ~13M parameters
```

*Last Updated: 2026-02-14*
