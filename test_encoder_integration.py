"""
Integration test for Block 1 (BEV Rasterization) + Block 2 (Encoder).

Tests the full pipeline from raw LiDAR to conditioning vector.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from bev_rasterization import BEVRasterizer, load_kitti_lidar
from encoder import build_encoder


def load_kitti_poses(pose_file):
    """Load KITTI poses from file."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            poses.append(values)
    return np.array(poses)


def test_end_to_end(sequence='00', frame_idx=500):
    """
    Test complete pipeline: LiDAR → BEV → Encoder → Conditioning.
    
    Args:
        sequence: KITTI sequence number
        frame_idx: Frame index to test
    """
    print("="*70)
    print(f"End-to-End Test: Block 1 + Block 2")
    print(f"Sequence: {sequence}, Frame: {frame_idx}")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========== BLOCK 1: BEV RASTERIZATION ==========
    print("\n" + "-"*70)
    print("BLOCK 1: BEV Rasterization")
    print("-"*70)
    
    # Load real KITTI data
    lidar_path = f'/media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/{sequence}/velodyne/{frame_idx:06d}.bin'
    
    if not os.path.exists(lidar_path):
        print(f"Warning: {lidar_path} not found, using synthetic data")
        # Create synthetic data
        np.random.seed(42)
        lidar_points = np.zeros((50000, 4), dtype=np.float32)
        lidar_points[:, 0] = np.random.uniform(-15, 15, 50000)
        lidar_points[:, 1] = np.random.uniform(-5, 25, 50000)
        lidar_points[:, 2] = np.random.uniform(-2, 2, 50000)
        lidar_points[:, 3] = np.random.uniform(0, 255, 50000)
    else:
        lidar_points = load_kitti_lidar(lidar_path)
        print(f"Loaded {len(lidar_points):,} LiDAR points from KITTI")
    
    # Rasterize
    rasterizer = BEVRasterizer()
    bev = rasterizer.rasterize_lidar(lidar_points)
    print(f"BEV shape: {bev.shape}")
    print(f"  Height range: [{bev[0].min():.3f}, {bev[0].max():.3f}]")
    print(f"  Intensity range: [{bev[1].min():.3f}, {bev[1].max():.3f}]")
    print(f"  Density range: [{bev[2].min():.3f}, {bev[2].max():.3f}]")
    
    # Convert to torch tensor [1, 3, 300, 400]
    bev_tensor = torch.from_numpy(bev).unsqueeze(0).to(device)
    print(f"PyTorch tensor: {bev_tensor.shape}, dtype={bev_tensor.dtype}")
    
    # ========== BLOCK 2: ENCODER ==========
    print("\n" + "-"*70)
    print("BLOCK 2: Multimodal Encoder")
    print("-"*70)
    
    # Build encoder
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    encoder.eval()
    
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")
    
    # Forward pass
    with torch.no_grad():
        conditioning, road_seg = encoder(bev_tensor)
    
    print(f"\nOutput shapes:")
    print(f"  Conditioning: {conditioning.shape} → Expected: [1, 512]")
    print(f"  Road Segmentation: {road_seg.shape} → Expected: [1, 1, 37, 50]")
    
    # Verify shapes
    assert conditioning.shape == (1, 512), f"Conditioning shape mismatch!"
    assert road_seg.shape == (1, 1, 37, 50), f"Road seg shape mismatch!"
    print("  ✓ All shapes correct")
    
    # Statistics
    print(f"\nConditioning vector statistics:")
    print(f"  Mean: {conditioning[0].mean():.4f}")
    print(f"  Std: {conditioning[0].std():.4f}")
    print(f"  Min: {conditioning[0].min():.4f}")
    print(f"  Max: {conditioning[0].max():.4f}")
    print(f"  L2 Norm: {torch.norm(conditioning[0]):.4f}")
    
    print(f"\nRoad segmentation statistics:")
    print(f"  Mean: {road_seg[0, 0].mean():.4f}")
    print(f"  Min: {road_seg[0, 0].min():.4f}")
    print(f"  Max: {road_seg[0, 0].max():.4f}")
    print(f"  Predicted road pixels (p>0.5): {(road_seg[0, 0] > 0.5).sum().item()}")
    
    return bev, conditioning.cpu().numpy(), road_seg.cpu().numpy()


def visualize_encoder_outputs(bev, conditioning, road_seg, frame_idx=500):
    """
    Visualize encoder inputs and outputs.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Block 1+2 Pipeline Visualization\nFrame {frame_idx}', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: BEV Input Channels
    titles = ['Height', 'Intensity', 'Density']
    cmaps = ['viridis', 'hot', 'Blues']
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(bev[i], origin='lower', cmap=cmaps[i])
        ax.set_title(f'BEV Input: {titles[i]}')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 1, Col 4: RGB Composite
    ax = fig.add_subplot(gs[0, 3])
    rgb = np.stack([bev[0], bev[1], bev[2]], axis=-1)
    for i in range(3):
        ch = rgb[:, :, i]
        if ch.max() > ch.min():
            rgb[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min())
    ax.imshow(rgb, origin='lower')
    ax.set_title('BEV RGB Composite\n(R=Height, G=Intensity, B=Density)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Row 2: Encoder Output - Road Segmentation
    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(road_seg[0, 0], origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('Block 2 Output: Road Segmentation\n(Auxiliary Task)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Road Probability')
    
    # Row 2: Conditioning Vector Visualization
    ax = fig.add_subplot(gs[1, 2:])
    cond_2d = conditioning[0].reshape(64, 8)
    im = ax.imshow(cond_2d, cmap='coolwarm', aspect='auto')
    ax.set_title('Block 2 Output: Conditioning Vector c\n[512-dim] reshaped to [64×8]')
    ax.set_xlabel('Feature Dim (8)')
    ax.set_ylabel('Channel (64)')
    plt.colorbar(im, ax=ax, label='Value')
    
    # Row 3: Conditioning Vector Statistics
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(conditioning[0], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title('Conditioning Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.axvline(conditioning[0].mean(), color='red', linestyle='--', label=f'Mean={conditioning[0].mean():.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(conditioning[0], color='steelblue', linewidth=0.5)
    ax.set_title('Conditioning Vector (Sequential)')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 2])
    # Show first 64 values as bar chart
    x_pos = np.arange(64)
    ax.bar(x_pos, conditioning[0, :64], color='steelblue', alpha=0.7)
    ax.set_title('First 64 Dimensions')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 3])
    # Summary statistics
    stats_text = f"""
Conditioning Vector Stats:
  Shape: {conditioning.shape}
  Mean: {conditioning[0].mean():.4f}
  Std: {conditioning[0].std():.4f}
  Min: {conditioning[0].min():.4f}
  Max: {conditioning[0].max():.4f}
  L2 Norm: {np.linalg.norm(conditioning[0]):.4f}

Road Segmentation:
  Shape: {road_seg.shape}
  Mean: {road_seg[0, 0].mean():.4f}
  Road Pixels: {(road_seg[0, 0] > 0.5).sum()}

Next: Block 3 (Diffusion Policy)
  Input: c [1, 512]
  Output: 5 trajectory samples
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    return fig


def benchmark_pipeline(sequence='00', num_frames=10):
    """
    Benchmark the throughput of Blocks 1+2.
    """
    import time
    
    print("\n" + "="*70)
    print("Pipeline Benchmark")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize
    rasterizer = BEVRasterizer()
    encoder = build_encoder(input_channels=3).to(device)
    encoder.eval()
    
    # Load test frames
    frames = []
    base_path = f'/media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/{sequence}/velodyne'
    
    for i in range(num_frames):
        lidar_path = os.path.join(base_path, f'{i:06d}.bin')
        if os.path.exists(lidar_path):
            points = load_kitti_lidar(lidar_path)
            frames.append(points)
        else:
            # Synthetic fallback
            np.random.seed(i)
            points = np.random.randn(30000, 4).astype(np.float32)
            points[:, 0] = np.random.uniform(-15, 15, 30000)
            points[:, 1] = np.random.uniform(-5, 25, 30000)
            points[:, 2] = np.random.uniform(-2, 2, 30000)
            points[:, 3] = np.random.uniform(0, 255, 30000)
            frames.append(points)
    
    print(f"Testing with {len(frames)} frames on {device}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        bev = rasterizer.rasterize_lidar(frames[0])
        bev_tensor = torch.from_numpy(bev).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = encoder(bev_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark Block 1
    print("\nBenchmarking Block 1 (BEV Rasterization)...")
    times_bev = []
    for frame in frames:
        start = time.time()
        bev = rasterizer.rasterize_lidar(frame)
        times_bev.append(time.time() - start)
    
    avg_bev = np.mean(times_bev) * 1000
    print(f"  Average: {avg_bev:.2f} ms/frame")
    print(f"  Throughput: {1000/avg_bev:.1f} frames/sec")
    
    # Benchmark Block 2
    print("\nBenchmarking Block 2 (Encoder)...")
    bev_tensors = [torch.from_numpy(rasterizer.rasterize_lidar(f)).unsqueeze(0).to(device) 
                   for f in frames]
    
    times_enc = []
    for bev_tensor in bev_tensors:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = encoder(bev_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_enc.append(time.time() - start)
    
    avg_enc = np.mean(times_enc) * 1000
    print(f"  Average: {avg_enc:.2f} ms/frame")
    print(f"  Throughput: {1000/avg_enc:.1f} frames/sec")
    
    # End-to-end
    print("\nEnd-to-End (Block 1 + Block 2)...")
    times_total = []
    for frame in frames:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        bev = rasterizer.rasterize_lidar(frame)
        bev_tensor = torch.from_numpy(bev).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = encoder(bev_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_total.append(time.time() - start)
    
    avg_total = np.mean(times_total) * 1000
    print(f"  Average: {avg_total:.2f} ms/frame")
    print(f"  Throughput: {1000/avg_total:.1f} frames/sec")
    
    print(f"\nBreakdown:")
    print(f"  Block 1 (BEV): {avg_bev/avg_total*100:.1f}%")
    print(f"  Block 2 (Encoder): {avg_enc/avg_total*100:.1f}%")


if __name__ == "__main__":
    # Test end-to-end pipeline
    bev, conditioning, road_seg = test_end_to_end(sequence='00', frame_idx=500)
    
    # Visualize
    print("\n" + "="*70)
    print("Creating visualization...")
    print("="*70)
    
    fig = visualize_encoder_outputs(bev, conditioning, road_seg, frame_idx=500)
    output_path = '/media/skr/storage/self_driving/TopoDiffuser/encoder_integration_test.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)
    
    # Benchmark
    benchmark_pipeline(sequence='00', num_frames=10)
    
    print("\n" + "="*70)
    print("Block 2 (Encoder) Integration Test Complete!")
    print("="*70)
