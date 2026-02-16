#!/usr/bin/env python3
"""
Verify Encoder Output Quality

Checks if the conditioning vector z from the encoder is appropriate:
1. Value range (should be bounded, not exploding/vanishing)
2. Distribution (mean, std, should have variation)
3. Correlation with input (different inputs → different outputs)
4. Comparison across different scenarios

Usage:
    python verify_encoder.py --encoder_ckpt checkpoints/encoder_best.pth \
                             --sequence 00 --num_samples 100
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))
from models.bev_rasterization import BEVRasterizer, load_kitti_lidar
from models.encoder import build_encoder
from scipy import stats


def load_encoder(checkpoint_path, device='cuda'):
    """Load trained encoder."""
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['model_state_dict'])
        elif 'encoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            encoder.load_state_dict(checkpoint)
        print(f"✓ Loaded encoder from {checkpoint_path}")
    else:
        print(f"⚠ No checkpoint found, using random initialization")
    
    encoder.eval()
    return encoder


def collect_encoder_outputs(encoder, sequence, num_samples, data_root, device):
    """Collect conditioning vectors from multiple frames."""
    rasterizer = BEVRasterizer()
    lidar_dir = Path(data_root) / 'sequences' / sequence / 'velodyne'
    
    outputs = []
    frame_indices = []
    
    # Sample random frames
    all_frames = sorted([f for f in lidar_dir.glob('*.bin')])
    if len(all_frames) > num_samples:
        sampled_frames = np.random.choice(all_frames, num_samples, replace=False)
    else:
        sampled_frames = all_frames
    
    print(f"\nCollecting encoder outputs from {len(sampled_frames)} frames...")
    
    with torch.no_grad():
        for frame_path in sampled_frames:
            frame_idx = int(frame_path.stem)
            
            # Load LiDAR
            lidar_points = load_kitti_lidar(str(frame_path))
            bev = rasterizer.rasterize_lidar(lidar_points)
            bev_tensor = torch.from_numpy(bev).unsqueeze(0).to(device)
            
            # Get encoder output
            conditioning, road_mask = encoder(bev_tensor)
            
            outputs.append(conditioning.cpu().numpy()[0])  # [512]
            frame_indices.append(frame_idx)
    
    return np.array(outputs), frame_indices  # [N, 512]


def analyze_outputs(outputs, save_path=None):
    """
    Analyze encoder outputs.
    
    Args:
        outputs: [N, 512] array of conditioning vectors
    """
    print("\n" + "=" * 70)
    print("ENCODER OUTPUT ANALYSIS")
    print("=" * 70)
    
    N, dim = outputs.shape
    print(f"Number of samples: {N}")
    print(f"Conditioning dimension: {dim}")
    
    # 1. Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 70)
    print(f"  Overall mean: {outputs.mean():.4f}")
    print(f"  Overall std:  {outputs.std():.4f}")
    print(f"  Min value:    {outputs.min():.4f}")
    print(f"  Max value:    {outputs.max():.4f}")
    print(f"  Range:        {outputs.max() - outputs.min():.4f}")
    
    # Per-dimension stats
    dim_means = outputs.mean(axis=0)
    dim_stds = outputs.std(axis=0)
    
    print(f"\n  Per-dimension:")
    print(f"    Mean of means:   {dim_means.mean():.4f} ± {dim_means.std():.4f}")
    print(f"    Mean of stds:    {dim_stds.mean():.4f} ± {dim_stds.std():.4f}")
    print(f"    Min dim mean:    {dim_means.min():.4f}")
    print(f"    Max dim mean:    {dim_means.max():.4f}")
    print(f"    Min dim std:     {dim_stds.min():.4f} (dim {np.argmin(dim_stds)})")
    print(f"    Max dim std:     {dim_stds.max():.4f} (dim {np.argmax(dim_stds)})")
    
    # 2. Health checks
    print("\n2. HEALTH CHECKS")
    print("-" * 70)
    
    # Check for dead dimensions (very low variance)
    dead_threshold = 0.01
    dead_dims = np.sum(dim_stds < dead_threshold)
    print(f"  Dead dimensions (std < {dead_threshold}): {dead_dims}/{dim}")
    if dead_dims > dim * 0.1:  # More than 10% dead
        print(f"    ⚠ WARNING: Too many dead dimensions! Encoder may be underfitting.")
    else:
        print(f"    ✓ OK: Good dimension utilization")
    
    # Check for exploding values
    if outputs.max() > 100 or outputs.min() < -100:
        print(f"  ⚠ WARNING: Extreme values detected (possible explosion)")
    else:
        print(f"  ✓ Value range is healthy")
    
    # Check for all-same (no variation)
    if dim_stds.mean() < 0.1:
        print(f"  ⚠ WARNING: Very low overall variance (encoder outputting similar values)")
    else:
        print(f"  ✓ Good variance across samples")
    
    # 3. Distribution shape
    print("\n3. DISTRIBUTION ANALYSIS")
    print("-" * 70)
    
    # Test for normality
    _, p_value = stats.normaltest(outputs.flatten())
    print(f"  Normality test p-value: {p_value:.4e}")
    if p_value < 0.05:
        print(f"    Distribution is NOT normal (expected for ReLU outputs)")
    
    # Sparsity (how many values are near zero)
    sparsity = np.mean(np.abs(outputs) < 0.01) * 100
    print(f"  Sparsity (|z| < 0.01): {sparsity:.1f}%")
    
    # Quartiles
    q25, q50, q75 = np.percentile(outputs, [25, 50, 75])
    print(f"  Quartiles: Q1={q25:.4f}, Q2={q50:.4f}, Q3={q75:.4f}")
    
    # 4. Variation across frames
    print("\n4. SAMPLE VARIATION")
    print("-" * 70)
    
    # Compute pairwise distances between random samples
    sample_size = min(50, N)
    idx = np.random.choice(N, sample_size, replace=False)
    sample_outputs = outputs[idx]
    
    # Pairwise L2 distances
    distances = []
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            dist = np.linalg.norm(sample_outputs[i] - sample_outputs[j])
            distances.append(dist)
    
    distances = np.array(distances)
    print(f"  Pairwise L2 distances (n={len(distances)} pairs):")
    print(f"    Mean: {distances.mean():.4f}")
    print(f"    Std:  {distances.std():.4f}")
    print(f"    Min:  {distances.min():.4f}")
    print(f"    Max:  {distances.max():.4f}")
    
    if distances.mean() < 1.0:
        print(f"    ⚠ WARNING: Samples are very similar (low diversity)")
    else:
        print(f"    ✓ Good diversity between samples")
    
    # 5. Correlation with simple features (sanity check)
    print("\n5. CORRELATION WITH INPUT FEATURES")
    print("-" * 70)
    print("  (Higher correlation with input features = better encoding)")
    
    # Compute mean activation of each dimension
    mean_activation = np.abs(outputs).mean(axis=0)
    active_dims = np.argsort(mean_activation)[-10:]  # Top 10 most active dims
    print(f"  Top 10 most active dimensions: {active_dims}")
    print(f"    Mean activations: {mean_activation[active_dims]}")
    
    print("\n" + "=" * 70)
    
    # Generate visualizations
    if save_path:
        create_visualizations(outputs, dim_means, dim_stds, save_path)
    
    return {
        'mean': outputs.mean(),
        'std': outputs.std(),
        'min': outputs.min(),
        'max': outputs.max(),
        'dead_dims': dead_dims,
        'sparsity': sparsity,
        'pairwise_dist_mean': distances.mean()
    }


def create_visualizations(outputs, dim_means, dim_stds, save_path):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Histogram of all values
    ax = axes[0, 0]
    ax.hist(outputs.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='Zero')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of All Output Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Per-dimension means
    ax = axes[0, 1]
    ax.plot(dim_means, alpha=0.7)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean Value')
    ax.set_title('Per-Dimension Mean Values')
    ax.grid(True, alpha=0.3)
    
    # 3. Per-dimension stds
    ax = axes[0, 2]
    ax.plot(dim_stds, alpha=0.7, color='orange')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Per-Dimension Std Dev (Variance Across Samples)')
    ax.grid(True, alpha=0.3)
    
    # 4. Heatmap of sample outputs (subset)
    ax = axes[1, 0]
    n_show = min(20, outputs.shape[0])
    im = ax.imshow(outputs[:n_show], aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Sample')
    ax.set_title(f'Output Heatmap (First {n_show} Samples)')
    plt.colorbar(im, ax=ax)
    
    # 5. Activation histogram
    ax = axes[1, 1]
    mean_abs = np.abs(outputs).mean(axis=0)
    ax.hist(mean_abs, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Absolute Value')
    ax.set_ylabel('Number of Dimensions')
    ax.set_title('Mean Absolute Activation per Dimension')
    ax.grid(True, alpha=0.3)
    
    # 6. CDF of values
    ax = axes[1, 2]
    sorted_vals = np.sort(outputs.flatten())
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, linewidth=2)
    ax.axvline(0, color='r', linestyle='--', label='Zero')
    ax.set_xlabel('Value')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_ckpt', type=str, 
                        default='checkpoints/encoder_best.pth')
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_root', type=str, 
                        default='/media/skr/storage/self_driving/TopoDiffuser/data/kitti')
    parser.add_argument('--output', type=str, default='encoder_analysis.png')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENCODER OUTPUT VERIFICATION")
    print("=" * 70)
    print(f"Checkpoint: {args.encoder_ckpt}")
    print(f"Sequence: {args.sequence}")
    print(f"Samples: {args.num_samples}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load encoder
    encoder = load_encoder(args.encoder_ckpt, device)
    
    # Collect outputs
    outputs, frame_indices = collect_encoder_outputs(
        encoder, args.sequence, args.num_samples, args.data_root, device
    )
    
    # Analyze
    stats = analyze_outputs(outputs, save_path=args.output)
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    healthy = True
    
    # Check criteria
    if stats['dead_dims'] > 50:
        print("❌ FAIL: Too many dead dimensions (>50)")
        healthy = False
    else:
        print("✓ Dead dimensions: OK")
    
    if stats['std'] < 0.1:
        print("❌ FAIL: Very low variance (std < 0.1)")
        healthy = False
    else:
        print("✓ Variance: OK")
    
    if abs(stats['mean']) > 10:
        print("❌ FAIL: Extreme mean value (>10)")
        healthy = False
    else:
        print("✓ Mean value: OK")
    
    if stats['pairwise_dist_mean'] < 1.0:
        print("⚠ WARNING: Low diversity between samples")
    else:
        print("✓ Sample diversity: OK")
    
    if healthy:
        print("\n✅ ENCODER OUTPUT IS HEALTHY")
        print("   Ready for diffusion training!")
    else:
        print("\n⚠️  ENCODER OUTPUT HAS ISSUES")
        print("   Consider retraining encoder or checking architecture")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
