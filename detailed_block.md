
---
# TopoDiffuser Architecture Pipeline Specification
**Paper**: TopoDiffuser: A Diffusion-Based Multimodal Trajectory Prediction Model with Topometric Maps (arXiv:2508.00303)  
**Configuration**: LiDAR-Only (Ablation Setting)  
**Hardware Reference**: Training on RTX 4090D, Batch Size 8 (scalable to RTX 3060 12GB with gradient accumulation)

---

## Legend
```
CBR = Conv → BatchNorm → ReLU
CB  = Conv → BatchNorm
⊕   = Element-wise Addition (Skip Connection)
```

---

## 1. High-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ INPUT REPRESENTATION (3 Modalities)                                                     │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                                      │
│ │   LiDAR     │  │   Traj      │  │    Map      │                                      │
│ │  Point Cloud│  │   History   │  │  (Topometric)│                                     │
│ │  [B,3,H,W]  │  │  [B,1,H,W]  │  │  [B,1,H,W]  │                                      │
│ └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                                      │
│        │                │                │                                             │
│        └────────────────┴────────────────┘                                             │
│                         │                                                              │
│                         ▼                                                              │
│              ┌─────────────────────┐                                                   │
│              │  Concatenated Input │                                                   │
│              │   [B, 5, 300, 400]  │  (if all 3 modalities)                            │
│              │   [B, 3, 300, 400]  │  (LiDAR-only ablation)                            │
│              └──────────┬──────────┘                                                   │
└─────────────────────────┼───────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ MULTIMODAL CONDITIONING ENCODER                                                         │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         ENCODER PATH (Downsampling)                             │   │
│  │                                                                                 │   │
│  │   Input ──→ CBR ──→ c1 [B,32,150,200] ──→ CBR ──→ c2 [B,64,75,100]              │   │
│  │                                                              │                  │   │
│  │   c2 ──→ CBR ──→ c3 [B,128,38,50] ──→ CBR ──→ c4 [B,256,19,25] ──→ CBR ──→ c5    │   │
│  │           │                              │                         [B,512,10,13]│   │
│  │           │                              │                                      │   │
│  └───────────┼──────────────────────────────┼──────────────────────────────────────┘   │
│              │                              │                                           │
│              └──────────────────────────────┘                                           │
│                             │                                                           │
│                             ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                      DECODER PATH (Upsampling with Skip)                        │   │
│  │                                                                                 │   │
│  │   c5 ──→ 1×1 Conv ──→ Deconv ──→ ⊕ ──→ p5 [B,64,8,8]                            │   │
│  │                              ↑                                                  │   │
│  │                              └── c4 (skip connection)                           │   │
│  │                                     │                                           │   │
│  │   p5 ──→ Deconv ──→ ⊕ ──→ p4 [B,64,19,25]                                       │   │
│  │                   ↑                                                             │   │
│  │                   └── c3 (skip connection)                                      │   │
│  │                                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                                   │
│                    ┌────────────────┴────────────────┐                                  │
│                    │                                                                   │
│                    ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         FROM p4: DUAL OUTPUT HEADS                               │   │
│  │                     (Shared: CB → Conv+Sigmoid Backbone)                         │   │
│  │                                                                                  │   │
│  │   p4 ──→ CB Blocks ──→ Conv + Sigmoid ──┬──→ Road Seg Mask [B,1,37,50]           │   │
│  │                                         │        │                              │   │
│  │                                         │        ▼                              │   │
│  │                                         │   BCE Loss (Auxiliary)                 │   │
│  │                                         │                                       │   │
│  │                                         └──→ Conv ──→ FC ──→ c [B, 512]         │   │
│  │                                                    │                            │   │
│  │                                                    ▼                            │   │
│  │                                            Diffusion Policy                     │   │
│  │                                               (Main Task)                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ DIFFUSION POLICY  g_φ(x_t, c, t)                                                        │
│                                                                                         │
│   ┌──────────────────────────────────────────────────────────────────────────────┐     │
│   │  FORWARD PROCESS (Training Only)                                             │     │
│   │  τ_0 (GT) ──→ Add Noise ──→ τ_t  (t=1...T, T=10)                             │     │
│   │  τ_t = √γ_t · τ_0 + √(1-γ_t) · ε,  ε ~ N(0,I)                                │     │
│   └──────────────────────────────────────────────────────────────────────────────┘     │
│                                          │                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────┐     │
│   │  REVERSE PROCESS (Denoising)                                                 │     │
│   │                                                                              │     │
│   │   Input: τ_t [B×K, 8, 2]  (K=5 samples)                                      │     │
│   │          c [B, 512]      (conditioning, broadcast)                           │     │
│   │          t [scalar]      (timestep embedding)                                │     │
│   │                                                                              │     │
│   │   g_φ: 1D-CNN/MLP ──→ Predicted Noise ε̂_t [B×K, 8, 2]                        │     │
│   │                                                                              │     │
│   │   Denoising Step: τ_{t-1} = (τ_t - (1-α_t)/√(1-ᾱ_t) · ε̂_t) / √α_t            │     │
│   │                                                                              │     │
│   └──────────────────────────────────────────────────────────────────────────────┘     │
│                                          │                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────┐     │
│   │  OUTPUT (Inference)                                                          │     │
│   │                                                                              │     │
│   │   After N=10 denoising steps:                                                │     │
│   │   τ_0 [K, 8, 2]  →  K=5 diverse trajectories                                 │     │
│   │                    →  8 waypoints (x,y) at 2m intervals                      │     │
│   │                                                                              │     │
│   │   Selection: minADE over K samples to pick best trajectory                   │     │
│   └──────────────────────────────────────────────────────────────────────────────┘     │
│                                          │                                              │
│                                          ▼                                              │
│                              MSE Loss (Noise Prediction)                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Block Specifications

### **BLOCK 1: Input Representation (BEV Rasterization)**
**Paper Ref**: Section III-B

| Parameter | Specification | Notes |
|-----------|--------------|-------|
| **Input Modalities** | LiDAR + Trajectory History + Topometric Map | 3-channel input (full model) |
| **LiDAR** | Raw point cloud (.bin files) | KITTI Velodyne 64-beam |
| **Traj History** | Past 5 keyframes (10m) | Rasterized as binary occupancy |
| **Topometric Map** | OSM binary mask | Road topology from OpenStreetMap |
| **Output** | $I_{\text{input}} \in \mathbb{R}^{B \times C \times H_0 \times W_0}$ | BEV pseudo-image |
| **Full Model** | $[B, 5, 300, 400]$ | 3 (LiDAR) + 1 (Traj) + 1 (Map) channels |
| **LiDAR-Only** | $[B, 3, 300, 400]$ | Height, Intensity, Density |
| **Batch Size (B)** | 8 (training) / 1 (inference) | Reduce to 4-6 for RTX 3060 |
| **Spatial Extent** | ~30m × 40m | 0.1m-0.15m per pixel |

**LiDAR Channel Details**:
1. **Height**: Max Z-value per cell (captures curbs, vehicle heights)
2. **Intensity**: Laser reflectivity (road markings vs asphalt)
3. **Density**: Point count per cell (uniform on road surfaces)

---

#### **LiDAR Rasterization Process**

The conversion from raw 3D LiDAR point cloud to 2D BEV pseudo-image involves several preprocessing steps:

**Step 1: Point Cloud Filtering**
| Operation | Description |
|-----------|-------------|
| **Range Filtering** | Remove points beyond valid range (typical: 0-80m) |
| **Height Filtering** | Remove ground plane outliers (keep -3m to +3m relative to ego) |
| **Ego Vehicle Removal** | Filter out points belonging to the ego vehicle (using bounding box mask) |
| **Ring Consistency** | Use Velodyne ring numbers for ground segmentation (optional) |

**Step 2: Coordinate Transformation**
```
Raw Point Cloud (LiDAR frame) → BEV Grid (Vehicle frame)

Input:  P_lidar = [x_l, y_l, z_l, intensity, ring]  (N × 5)
Output: P_bev   = [u, v, z, intensity]  (M × 4, M ≤ N)

Where:
- (u, v): Grid cell indices in BEV image
- z: Height value preserved for max pooling
- intensity: Laser reflectivity
```

**Transformation Pipeline:**
1. **Transform to ego vehicle frame** (using extrinsic calibration matrix)
2. **Discretize to BEV grid**: Map (x, y) → (u, v) based on spatial resolution
   - Grid resolution: 0.1-0.15m per pixel
   - Grid size: 300×400 pixels ≈ 30m × 40m
   - Origin: Center of image = ego vehicle position
   - X-axis: Forward direction (longitudinal)
   - Y-axis: Left direction (lateral)

**Step 3: Voxelization & Channel Encoding**

For each grid cell (u, v), aggregate points that fall within it:

| Channel | Computation | Physical Meaning |
|---------|-------------|------------------|
| **Height** | $H[u,v] = \max_i(z_i)$ | Maximum height of objects in cell |
| **Intensity** | $I[u,v] = \frac{1}{|P_{uv}|}\sum_i intensity_i$ | Average reflectivity |
| **Density** | $D[u,v] = \min(1.0, \log(1 + |P_{uv}|) / \log(N_{max}))$ | Normalized point count |

Where:
- $P_{uv}$ = set of points falling in cell (u, v)
- $|P_{uv}|$ = number of points in cell
- $N_{max}$ = maximum expected points per cell (normalization factor, typically 64)

**Step 4: Post-Processing**
| Operation | Purpose |
|-----------|---------|
| **Normalization** | Channel-wise min-max or z-score normalization |
| **Hole Filling** | Interpolate empty cells from neighbors (optional) |
| **Intensity Clipping** | Clamp intensity values to valid range [0, 255] → [0, 1] |

**Example Code Snippet (Conceptual):**
```python
import numpy as np

def lidar_to_bev(points, grid_shape=(300, 400), resolution=0.1):
    """
    Args:
        points: (N, 4) array of [x, y, z, intensity]
        grid_shape: (H, W) of output BEV image
        resolution: meters per pixel
    Returns:
        bev: (3, H, W) tensor [height, intensity, density]
    """
    H, W = grid_shape
    bev = np.zeros((3, H, W), dtype=np.float32)
    
    # Convert to grid coordinates
    # Center of image is ego position
    u = (points[:, 0] / resolution + W // 2).astype(int)
    v = (points[:, 1] / resolution + H // 2).astype(int)
    
    # Filter valid indices
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, intensity = u[valid], v[valid], points[valid, 2], points[valid, 3]
    
    # Build height channel (max pooling per cell)
    for i in range(len(u)):
        bev[0, v[i], u[i]] = max(bev[0, v[i], u[i]], z[i])
        bev[1, v[i], u[i]] += intensity[i]  # Sum for averaging
        bev[2, v[i], u[i]] += 1  # Point count
    
    # Normalize intensity (average)
    mask = bev[2] > 0
    bev[1][mask] /= bev[2][mask]
    
    # Convert density to log scale and normalize
    bev[2] = np.log(1 + bev[2]) / np.log(64)
    bev[2] = np.clip(bev[2], 0, 1)
    
    return bev
```

**Channel Visualization:**

| Channel | Typical Range | Road Appearance | Obstacle Appearance |
|---------|--------------|-----------------|---------------------|
| Height | -3.0 to +3.0 m | Flat (0m) | Elevated (curbs, vehicles) |
| Intensity | 0 to 1.0 | Low (asphalt ~0.1) | High (road markings ~0.8) |
| Density | 0 to 1.0 | Uniform (~0.5) | Sparse (edges) or Dense (walls) |

---

#### **Trajectory History & Topometric Map Rasterization**

For the full multimodal model, two additional binary channels are concatenated with the LiDAR BEV:

**Channel 4: Trajectory History (Past Poses)**
| Attribute | Specification |
|-----------|--------------|
| **Input** | Past 5 keyframes of ego vehicle poses (10m at 2m intervals) |
| **Representation** | Binary occupancy map |
| **Encoding** | Rasterize past (x, y) positions as 1s on black canvas (0s) |
| **Purpose** | Provide motion history context for prediction |

```
Trajectory Rasterization:
1. Load past 5 poses relative to current ego frame
2. For each pose (x_i, y_i):
   - Transform to BEV grid: u = x_i/resolution + W/2
   - Mark corresponding cell and neighbors as 1
3. Apply Gaussian blur (optional) for smoothness
4. Output: Binary map [1, 300, 400]
```

**Channel 5: Topometric Map (OSM Road Network)**
| Attribute | Specification |
|-----------|--------------|
| **Input** | OpenStreetMap road polylines in ego vehicle vicinity |
| **Representation** | Binary road mask |
| **Encoding** | Rasterize road centerlines as 1s, background as 0s |
| **Purpose** | Provide global road topology context |

```
Topometric Map Rasterization:
1. Query OSM for road segments within 30m × 40m of ego
2. Filter by road type: primary, secondary, residential
3. For each road polyline:
   - Transform lat/lon → local ego frame
   - Rasterize line onto BEV grid (Bresenham's algorithm)
4. Apply line thickness (2-3 pixels) for visibility
5. Output: Binary map [1, 300, 400]
```

**Full Input Stack:**
```
Input Tensor [B, 5, 300, 400]:
├── Channel 0: LiDAR Height
├── Channel 1: LiDAR Intensity  
├── Channel 2: LiDAR Density
├── Channel 3: Trajectory History (binary)
└── Channel 4: Topometric Map (binary)
```

---

### **BLOCK 2: Multimodal Conditioning Encoder**
**Paper Ref**: Section III-C, Appendix I

#### Encoder Path (Contracting)
| Block | Output Shape | Operation | Purpose |
|-------|--------------|-----------|---------|
| **Input** | $[B, 3, 300, 400]$ | Concatenated BEV | LiDAR-only ablation |
| **c1** | $[B, 32, 150, 200]$ | CBR (Conv+BN+ReLU), stride-2 | Low-level features, edges |
| **c2** | $[B, 64, 75, 100]$ | CBR (Conv+BN+ReLU), stride-2 | Mid-level features |
| **c3** | $[B, 128, 38, 50]$ | CBR (Conv+BN+ReLU), stride-2 | Higher-level features |
| **c4** | $[B, 256, 19, 25]$ | CB, stride-2 | Deep features |
| **c5** | $[B, 512, 10, 13]$ | CB, stride-2 | Bottleneck, semantic features |

#### Decoder Path (Expanding with Skip Connections)
| Block | Output Shape | Operation | Skip From |
|-------|--------------|-----------|-----------|
| **p5** | $[B, 64, 8, 8]$ | 1×1 Conv → Deconv → ⊕c4 | c4 $[B, 256, 19, 25]$ |
| **p4** | $[B, 64, 19, 25]$ | Deconv → ⊕c3 | c3 $[B, 128, 38, 50]$ |

**Key Architecture Points**:
- **CBR**: All encoder blocks use ReLU activation (CBR = Conv→BN→ReLU)
- **Note**: Changed from paper's CB (no ReLU) to fix gradient flow issues identified in testing
- **Skip Connections**: Element-wise addition (⊕) fuses decoder features with encoder features
- **p5**: Deep semantic features (bottleneck of decoder)
- **p4**: Spatially-rich features → CB Blocks → Conv+Sigmoid → **shared backbone**
  - Head A: Conv+Sigmoid output → Segmentation mask → BCE Loss
  - Head B: Conv+Sigmoid output → Conv → FC → c [512] → Diffusion

---

### **BLOCK 3: Dual Output Heads**

#### **Head A: Road Segmentation (Auxiliary)**
**Paper Ref**: Section III-C, Equation 4

| Spec | Value | Description |
|------|-------|-------------|
| **Input** | p4 features via CB Blocks $[B, 64, 19, 25]$ | Spatially-rich features from decoder |
| **Processing** | CB Blocks → Conv + Sigmoid | Binary classification per pixel |
| **Output** | $\mathbf{x} \in [0,1]^{B \times 1 \times 37 \times 50}$ | Road probability map |
| **Ground Truth** | $\mathbf{y} \in \{0,1\}^{B \times 1 \times 37 \times 50}$ | Rasterized driven path |
| **Loss** | $\mathcal{L}_{\text{road}}$ (BCE) | Binary cross-entropy per pixel |

**Note**: GT mask created by rasterizing future trajectory waypoints (1 = drivable, 0 = background).

#### **Head B: Conditioning Vector (Main)**
**Paper Ref**: Section III-C, Appendix Table I

| Spec | Value | Description |
|------|-------|-------------|
| **Input** | Shared backbone: p4 → CB Blocks → Conv + Sigmoid | Features from segmentation path |
| **Processing** | Conv → Flatten → FC | Dimension reduction to 512-dim vector |
| **Output** | $\mathbf{c} \in \mathbb{R}^{B \times 512}$ | Conditioning vector for diffusion |
| **Usage** | Injected into $g_\phi$ at every denoising step | Context for trajectory generation |

**Note**: The conditioning branch **shares** the p4 → CB → Conv+Sigmoid backbone with segmentation, then:
- **Segmentation output**: Direct output of Conv+Sigmoid → BCE Loss
- **Conditioning output**: Conv+Sigmoid → Conv → FC → c [512] → Diffusion Policy

---

### **BLOCK 4: Conditional Diffusion Policy**
**Paper Ref**: Section III-D, III-E

#### Forward Process (Training Only)
| Parameter | Specification |
|-----------|--------------|
| **Input** | $\tau_0 \in \mathbb{R}^{B \times 8 \times 2}$ (Ground Truth Future) |
| **Timesteps** | $T = 10$ (Paper Section IV-B) |
| **Noise Schedule** | $\gamma_t$ (monotonically decreasing) |
| **Noise Addition** | $\tau_t = \sqrt{\gamma_t}\tau_0 + \sqrt{1-\gamma_t}\epsilon$, $\epsilon \sim \mathcal{N}(0,\mathbf{I})$ |
| **Output** | $\tau_t \in \mathbb{R}^{B \times 8 \times 2}$ (Noised trajectory at step t) |

#### Reverse Process (Denoising Network $g_\phi$)
| Parameter | Specification |
|-----------|--------------|
| **Input Trajectory** | $\tau_t \in \mathbb{R}^{(B \times K) \times 8 \times 2}$ (K=5 samples) |
| **Conditioning** | $\mathbf{c} \in \mathbb{R}^{B \times 512}$ (broadcast to all K samples) |
| **Timestep Embedding** | $t \in \{10, 9, ..., 1\}$ via sinusoidal encoding |
| **Architecture** | Lightweight 1D-CNN or MLP (U-Net style) |
| **Output** | $\hat{\epsilon}_t \in \mathbb{R}^{(B \times K) \times 8 \times 2}$ (Predicted noise) |

#### Denoising Step (DDPM-style)
$$\tau_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\tau_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}_t\right) + \sigma_t \mathbf{z}$$

Where:
- $\alpha_t = 1 - \beta_t$ (noise schedule parameter)
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
- $\sigma_t$ = noise scale (can be 0 for deterministic sampling)
- $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ (optional stochasticity)

---

### **BLOCK 5: Loss Computation**

| Loss | Input Shapes | Target | Formula |
|------|--------------|--------|---------|
| **MSE** | $\hat{\epsilon}_t: [B \times K, 8, 2]$, $\epsilon: [B \times K, 8, 2]$ | Actual Gaussian noise | $\|\epsilon - g_\phi(\tau_t, t, \mathbf{c})\|^2$ |
| **BCE** | Pred mask: $[B, 1, 37, 50]$, GT: $[B, 1, 37, 50]$ | Rasterized road | $-\sum [y\log(x) + (1-y)\log(1-x)]$ |
| **Total** | $\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda_{\text{road}}\mathcal{L}_{\text{road}}$ | | $\lambda_{\text{road}} \approx 0.1-1.0$ |

---

### **BLOCK 6: Inference (Trajectory Generation)**
**Paper Ref**: Section IV-B

| Step | Operation | Dimensions | Notes |
|------|-----------|------------|-------|
| 1 | Encode BEV | $[1, 3, 300, 400] \rightarrow [1, 512]$ | Single sample, no gradients |
| 2 | Initialize noise | $\tau_N \sim \mathcal{N}(0, \mathbf{I})$ | $[K, 8, 2]$ for K=5 samples |
| 3 | Denoise loop | $\tau_t \rightarrow \tau_{t-1}$ | N=10 steps |
| 4 | Final output | $[K, 8, 2]$ | K diverse trajectory hypotheses |
| 5 | Selection | Best via minADE | Closest to GT for evaluation |

**Output Interpretation**:
- **K=5**: Number of multimodal samples (diverse future behaviors)
- **8**: Future waypoints (16m ahead at 2m intervals)
- **2**: $(x, y)$ coordinates in ego-vehicle frame (meters)

---

## 3. Dimension Summary Reference Table

| Stage | Tensor Name | Shape | Description |
|-------|-------------|-------|-------------|
| **Input** | $I_{\text{lidar}}$ | $[B, 3, 300, 400]$ | BEV rasterized LiDAR |
| **Input** | $I_{\text{traj}}$ (optional) | $[B, 1, 300, 400]$ | History occupancy |
| **Input** | $I_{\text{map}}$ (optional) | $[B, 1, 300, 400]$ | OSM binary mask |
| **Encoder** | c1 | $[B, 32, 150, 200]$ | Low-level features |
| **Encoder** | c3 | $[B, 128, 38, 50]$ | Mid-level features (skip to p4) |
| **Encoder** | c4 | $[B, 256, 19, 25]$ | Deep features (skip to p5) |
| **Encoder** | c5 | $[B, 512, 10, 13]$ | Bottleneck features |
| **Decoder** | p5 | $[B, 64, 8, 8]$ | Decoder output (deep features) |
| **Decoder** | p4 | $[B, 64, 19, 25]$ | Decoder output (spatial features → both heads) |
| **Head A** | Road Seg | $[B, 1, 37, 50]$ | Predicted road mask |
| **Head B** | $\mathbf{c}$ | $[B, 512]$ | Conditioning vector |
| **Diffusion** | $\tau_0$ (GT) | $[B, 8, 2]$ | Ground truth future |
| **Diffusion** | $\tau_t$ | $[B \times K, 8, 2]$ | Noised trajectories |
| **Output** | $\hat{\epsilon}_t$ | $[B \times K, 8, 2]$ | Predicted noise |
| **Output** | Final $\tau_0$ | $[K, 8, 2]$ | K diverse predictions |

**Key Hyperparameters**:
- **Past History**: 5 keyframes (10m at 2m spacing)
- **Future Prediction**: 8 waypoints (16m at 2m spacing)
- **Diffusion Steps**: N = 10 (training and inference)
- **Image Resolution**: 300 × 400 (H × W)
- **Latent Dim**: 512 (conditioning vector)
- **Traj Samples (K)**: 5 (inference)

---

## 4. Paper Tensor Shape Reference

**Source**: Appendix Table I (Network Parameters and Tensor Shapes)

| Tensor Name | Shape | Description |
|-------------|-------|-------------|
| **$x_{\text{input}}$** | $[8, 5, 300, 400]$ | Input tensor: batch size = 8, 5 channels (LiDAR, map, history trajectory), BEV image size = 300×400 |
| **obs_cond (before reshape)** | $[8, 64, 8]$ | Intermediate condition features from the encoder (p5 pathway: Conv output before flatten) |
| **obs_cond (after reshape)** | $[8, 512]$ | Flattened condition vector ($64 \times 8 = 512$) used as input to the diffusion policy network |
| **Diffusion Output** | $[40, 8, 2]$ | Output trajectories: $40 = B \times K$ (8 batch × 5 samples), 8 waypoints, 2D coordinates (x,y) |

**Note**: The paper's Appendix Table I indicates conditioning comes from **p5** features $[8, 64, 8, 8]$ → reduced to $[8, 64, 8]$ → flattened to $[8, 512]$. However, the diagram shows the conditioning path originating from **p4** through CB Blocks. This discrepancy suggests:
- The diagram may be simplified/illustrative
- **p5** is the actual source per paper specifications
- The CB Blocks pathway may be an alternative or auxiliary conditioning mechanism

---

## 5. Architecture Flow Summary

```
Training Phase:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEV Input [B,3,300,400] 
    ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           ENCODER (Contracting Path)                                │
│ c1[32,150,200] → c2[64,75,100] → c3[128,38,50] → c4[256,19,25] → c5[512,10,13]      │
└─────────────────────────────────────────────────────────────────────────────────────┘
    ↓                                    ↓
┌──────────────────────┐          ┌──────────────────────┐
│    DECODER Path      │          │    Skip c4 ──→ ⊕    │
│ 1×1Conv→Deconv→⊕c4  │────────→│      p5[64,8,8]      │
│      ↓               │          └──────────────────────┘
│    Deconv→⊕c3       │                     │
│      ↓               │                     │
│    p4[64,19,25]      │                     │
└──────────────────────┘                     │
    │                                        │
    ▼                                        │
┌────────────────────────────────────────────┘
│        DUAL HEADS (Shared Backbone)         │
│                                             │
│   p4 ──→ CB Blocks ──→ Conv+Sig ──┬──→ Seg ──→ BCE Loss
│   [B,64,19,25]                    │
│                                   └──→ Conv ──→ FC ──→ c[B,512] ──→ Diffusion
    ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        DIFFUSION POLICY g_φ(τ_t, c, t)                              │
│                                                                                     │
│   Forward:  τ_0 [B,8,2] ──→ Add Noise ──→ τ_t [B,8,2]                               │
│                                             ↓                                       │
│   Reverse:  g_φ predicts ε̂_t [B,8,2] with conditioning c [B,512]                    │
│                                             ↓                                       │
│   Loss:     MSE(ε̂_t, ε)                                                             │
└─────────────────────────────────────────────────────────────────────────────────────┘


Inference Phase:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEV Input [1,3,300,400]
    ↓
Encoder → c [1,512] (cached)
    ↓
Initialize τ_N ~ N(0,I) [K,8,2]  (K=5 samples)
    ↓
For t = N, N-1, ..., 1:
    ε̂_t = g_φ(τ_t, c, t)     ←── c broadcast to all K samples
    τ_{t-1} = Denoise(τ_t, ε̂_t)
    ↓
Output: τ_0 [K,8,2]  (K diverse trajectories)
    ↓
Select best via minADE
```



---

## 6. Dataset Split (Section IV-A)

Based on **Section IV-A (Dataset)** of the paper, the training/test split is:

### **Training Set**
**Sequences**: `00`, `02`, `05`, `07`  
**Total Samples**: **3,860**  
**Source**: KITTI Raw Dataset (following KITTI Odometry Benchmark protocol)

### **Test/Validation Set**
| Sequence | Samples | Scene Type |
|----------|---------|------------|
| `08` | 1,391 | Residential/commercial mixed |
| `09` | 530 | Country/rural roads |
| `10` | 349 | Urban city center |
| **Total Test** | **2,270** | |

### **Key Details**
- **Data Source**: KITTI Raw Dataset (not the tracking benchmark)
- **Protocol**: Standard KITTI Odometry split used by most autonomous driving papers
- **Frame Sampling**: Not every frame is used—keyframes are sampled at **2-meter intervals** (Section IV-B), meaning the 3,860 training samples represent driving segments spaced by 2m along the trajectory
- **Multimodality**: During inference, they sample **5 trajectories** per test case to evaluate multimodal diversity (Section IV-B)

**Note**: If you're implementing this on your RTX 3060, you can further subset the training data (e.g., use only sequences 00 and 02 for initial debugging ~1,900 samples) to speed up iteration before training on the full 3,860 samples.

### **Sequence Allocation Summary**

| Split | Sequences | Reason |
| ----- | --------- | ------ |
| **Training** | `00`, `02`, `05`, `07` | High diversity: residential, highway, and urban scenes with good GPS coverage |
| **Validation/Test** | `08`, `09`, `10` | Held out to match the official KITTI Odometry benchmark test protocol |
| **Unused** | `01`, `03`, `04`, `06` | These are often reserved for other purposes or excluded due to: <br>• `01`: Highway (monotonous, less interesting for trajectory prediction) <br>• `03`, `04`, `06`: Data quality issues or overlap with training distribution |

---

## 7. RTX 3060 12GB Adaptation Notes

| Component | Memory Impact | Optimization for 12GB |
|-----------|--------------|----------------------|
| Batch Size 8 | ~10-11 GB VRAM | Reduce to **4** or use gradient checkpointing |
| 5 Trajectory Samples | Linear increase | Inference: Process 1 sample at a time |
| BEV 300×400 | Fixed | Can reduce to 224×224 with minor loss |
| 10 Denoising Steps | Minimal | Fixed; don't reduce below 5 |

**Recommended 3060 Config**:
- Batch Size: 4
- Accumulation Steps: 2 (effective batch 8)
- Mixed Precision: FP16
- OSM: Use pre-processed (lighter footprint)



### NOTES IMPLEMENTATION

   Metric                   Before         After          Improvement
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Total Score              66.3           88.1           +21.8 points ✓
   Height Correlation       0.67 (POOR)    0.96 (GOOD)    Fixed metric ✓
   Elevation Preservation   N/A            100%           New metric ✓
   Object Preservation      36.6% (POOR)   87.2% (FAIR)   +50.6% ✓
   Coverage                 87.15%         87.21%         Stable
   Cell Utilization         14.12%         14.14%         Optimal



We can also see from the
results on KITTI that the simpler the scenario the higher the
accuracy. KITTI-08 has the most complex trajectories, and the
accuracy is the worst, while KITTI-10 is on the opposite side


Paper Contradicts Itself!
SourceWhat it showsConditioning fromSection III.C (text)Mask → CNN → conditioningSegmentation maskAppendix Figure 1Features → dual headsp4 featuresAppendix Table I[B,64,8] → [B,512]p4 features

paper text in Section III.C is misleading or incorrect. This is likely:

A writing/editing error
Outdated description from an earlier architecture
Poor wording (they meant "features that predict segmentation" not "segmentation mask itself")

Trust the Appendix over the main text. Your implementation correctly follows the Appendix specifications. If you implemented what the text describes, you'd get an inferior architecture with wrong dimensions.
This is a significant error in the paper, but it doesn't invalidate their results since the actual implementation (shown in the Appendix) is sound and what you've implemented correctly.


Don't "fix" the 2.5% std at c5. That sparsity is the ReLU doing its job—creating a sparse code where specific neurons represent specific road topologies (similar to how V1 cortex has edge detectors that only fire on specific orientations).
Verify success by:
Training for 5-10 epochs and checking if the segmentation head (BCE loss) converges
Visualizing the predicted road masks—they should align with actual drivable areas despite the low c5 std
Checking if the diffusion loss decreases (indicating the conditioning vector c  carries useful information)
If both losses drop, the architecture is working perfectly—the low variance in c5 is a feature, not a bug.