"""
Integration of Block 1 (BEV Rasterization) with Block 2 (Encoder).

This demonstrates the full pipeline from raw sensor data to conditioning vector.
"""

import torch
import numpy as np
from bev_rasterization import BEVRasterizer, BEVRasterizationBlock
from encoder import MultimodalEncoder


def demo_block1_to_block2():
    """
    Demonstrate the full pipeline from raw data to conditioning vector.
    """
    print("=" * 70)
    print("TopoDiffuser Pipeline: Block 1 → Block 2 Integration Demo")
    print("=" * 70)
    
    # Configuration
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    
    # Initialize blocks
    print("\n" + "-" * 70)
    print("Initializing Blocks")
    print("-" * 70)
    
    bev_block = BEVRasterizationBlock(device=str(device))
    encoder = MultimodalEncoder(input_channels=5, conditioning_dim=512).to(device)
    
    print(f"✓ Block 1 (BEV Rasterization) initialized")
    print(f"  - Output shape: [B, 5, 300, 400]")
    print(f"✓ Block 2 (Encoder) initialized")
    print(f"  - Input channels: 5")
    print(f"  - Conditioning dim: 512")
    
    # Create synthetic raw data
    print("\n" + "-" * 70)
    print("Generating Synthetic Raw Data")
    print("-" * 70)
    
    np.random.seed(42)
    batch_data = []
    
    for b in range(batch_size):
        # LiDAR points (simulating KITTI-like data)
        num_points = 30000 + np.random.randint(10000)
        lidar_points = np.zeros((num_points, 4), dtype=np.float32)
        lidar_points[:, 0] = np.random.uniform(-15, 15, num_points)   # x
        lidar_points[:, 1] = np.random.uniform(-5, 25, num_points)    # y
        lidar_points[:, 2] = np.random.uniform(-2, 2, num_points)     # z
        lidar_points[:, 3] = np.random.uniform(0, 255, num_points)    # intensity
        
        # Trajectory history (5 waypoints, 10m at 2m spacing)
        traj_y = np.linspace(-5, 5, 5)
        traj_x = np.random.randn(5) * 0.5  # Some lateral deviation
        trajectory = np.column_stack([traj_x, traj_y]).astype(np.float32)
        
        # OSM road network (straight road ahead)
        osm_y = np.linspace(-10, 30, 20)
        osm_x = np.zeros_like(osm_y)
        osm_coords = np.column_stack([osm_x, osm_y]).astype(np.float32)
        
        # Ego pose
        ego_pose = (0.0, 0.0, 0.0)  # x, y, yaw
        
        batch_data.append({
            'lidar': lidar_points,
            'trajectory': trajectory,
            'osm_coords': osm_coords,
            'ego_pose': ego_pose,
            'mode': 'full'
        })
    
    print(f"✓ Generated {batch_size} synthetic samples")
    print(f"  - LiDAR points per sample: ~30-40k")
    print(f"  - Trajectory waypoints: 5")
    print(f"  - OSM road segments: 1 polyline")
    
    # Block 1: BEV Rasterization
    print("\n" + "-" * 70)
    print("Block 1: BEV Rasterization")
    print("-" * 70)
    
    bev_tensor = bev_block.forward(batch_data)
    print(f"✓ BEV tensor generated")
    print(f"  - Shape: {bev_tensor.shape}")
    print(f"  - Dtype: {bev_tensor.dtype}")
    print(f"  - Device: {bev_tensor.device}")
    
    # Show channel statistics
    for i, name in enumerate(['Height', 'Intensity', 'Density', 'Trajectory', 'OSM Map']):
        channel = bev_tensor[0, i]
        print(f"  - Channel {i} ({name}): mean={channel.mean():.3f}, max={channel.max():.3f}")
    
    # Block 2: Encoder
    print("\n" + "-" * 70)
    print("Block 2: Multimodal Encoder")
    print("-" * 70)
    
    encoder.eval()
    with torch.no_grad():
        conditioning, road_mask = encoder(bev_tensor)
    
    print(f"✓ Encoder forward pass complete")
    print(f"  - Conditioning vector: {conditioning.shape}")
    print(f"    → Expected: [4, 512] ✓" if conditioning.shape == (4, 512) else "    → Mismatch!")
    print(f"  - Road segmentation mask: {road_mask.shape}")
    print(f"    → Expected: [4, 1, 300, 400] ✓" if road_mask.shape == (4, 1, 300, 400) else "    → Mismatch!")
    
    # Show conditioning vector statistics
    print(f"\n  Conditioning vector statistics:")
    print(f"    - Mean: {conditioning.mean():.4f}")
    print(f"    - Std: {conditioning.std():.4f}")
    print(f"    - Min: {conditioning.min():.4f}")
    print(f"    - Max: {conditioning.max():.4f}")
    
    # Show road mask statistics
    print(f"\n  Road mask statistics:")
    print(f"    - Mean: {road_mask.mean():.4f}")
    print(f"    - Min: {road_mask.min():.4f}")
    print(f"    - Max: {road_mask.max():.4f}")
    print(f"    - Predicted road pixels: {(road_mask[0, 0] > 0.5).sum().item()}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"""
Input (Raw Data):
  - LiDAR point cloud: [N, 4] points per sample
  - Trajectory history: [5, 2] waypoints per sample
  - OSM road network: [M, 2] coordinates per sample

Block 1 (BEV Rasterization):
  ↓ Processed to BEV tensor: {bev_tensor.shape}
  - Channel 0: LiDAR Height (max Z per cell)
  - Channel 1: LiDAR Intensity (average reflectivity)
  - Channel 2: LiDAR Density (log-normalized point count)
  - Channel 3: Trajectory History (binary mask)
  - Channel 4: Topometric Map (OSM road binary mask)

Block 2 (Multimodal Encoder):
  ↓ Encoded to:
  - Conditioning vector: {conditioning.shape} for diffusion policy
  - Road mask: {road_mask.shape} (auxiliary task)

Next: Block 3 (Conditional Diffusion Policy)
  - Uses conditioning vector c to guide trajectory generation
  - Performs N=10 denoising steps
  - Outputs K=5 diverse trajectory samples
""")
    
    return bev_tensor, conditioning, road_mask


def benchmark_pipeline():
    """
    Benchmark the throughput of Blocks 1+2.
    """
    import time
    
    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    batch_size = 8
    num_iterations = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bev_block = BEVRasterizationBlock(device=str(device))
    encoder = MultimodalEncoder(input_channels=5, conditioning_dim=512).to(device)
    encoder.eval()
    
    # Generate test data
    np.random.seed(123)
    batch_data = []
    for b in range(batch_size):
        lidar_points = np.random.randn(35000, 4).astype(np.float32)
        lidar_points[:, 0] = np.random.uniform(-15, 15, 35000)
        lidar_points[:, 1] = np.random.uniform(-5, 25, 35000)
        lidar_points[:, 2] = np.random.uniform(-2, 2, 35000)
        lidar_points[:, 3] = np.random.uniform(0, 255, 35000)
        
        trajectory = np.array([[-2, -4], [-1, -2], [0, 0], [0.5, 2], [1, 4]], dtype=np.float32)
        osm_coords = np.array([[0, -10], [0, 30]], dtype=np.float32)
        
        batch_data.append({
            'lidar': lidar_points,
            'trajectory': trajectory,
            'osm_coords': osm_coords,
            'ego_pose': (0.0, 0.0, 0.0),
            'mode': 'full'
        })
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        bev = bev_block.forward(batch_data)
        with torch.no_grad():
            _ = encoder(bev)
    
    # Benchmark Block 1
    print("\nBenchmarking Block 1 (BEV Rasterization)...")
    times_block1 = []
    for _ in range(num_iterations):
        start = time.time()
        bev = bev_block.forward(batch_data)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_block1.append(time.time() - start)
    
    avg_block1 = np.mean(times_block1) * 1000  # ms
    std_block1 = np.std(times_block1) * 1000
    print(f"  Time: {avg_block1:.2f} ± {std_block1:.2f} ms")
    print(f"  Throughput: {batch_size / (avg_block1 / 1000):.1f} samples/sec")
    
    # Benchmark Block 2
    print("\nBenchmarking Block 2 (Encoder)...")
    bev = bev_block.forward(batch_data)
    times_block2 = []
    for _ in range(num_iterations):
        start = time.time()
        with torch.no_grad():
            _ = encoder(bev)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_block2.append(time.time() - start)
    
    avg_block2 = np.mean(times_block2) * 1000  # ms
    std_block2 = np.std(times_block2) * 1000
    print(f"  Time: {avg_block2:.2f} ± {std_block2:.2f} ms")
    print(f"  Throughput: {batch_size / (avg_block2 / 1000):.1f} samples/sec")
    
    # Benchmark end-to-end
    print("\nBenchmarking End-to-End (Block 1 + Block 2)...")
    times_total = []
    for _ in range(num_iterations):
        start = time.time()
        bev = bev_block.forward(batch_data)
        with torch.no_grad():
            _ = encoder(bev)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times_total.append(time.time() - start)
    
    avg_total = np.mean(times_total) * 1000  # ms
    std_total = np.std(times_total) * 1000
    print(f"  Time: {avg_total:.2f} ± {std_total:.2f} ms")
    print(f"  Throughput: {batch_size / (avg_total / 1000):.1f} samples/sec")
    
    # Parameter count
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nModel Size:")
    print(f"  Encoder parameters: {encoder_params:,} ({encoder_params * 4 / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    # Run integration demo
    bev_tensor, conditioning, road_mask = demo_block1_to_block2()
    
    # Run benchmark
    benchmark_pipeline()
    
    print("\n" + "=" * 70)
    print("Integration test complete!")
    print("=" * 70)
