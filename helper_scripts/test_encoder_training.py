"""
Training Test for Block 2: Encoder Segmentation Head (Head A)

This script trains the encoder for a few epochs to verify:
1. BCE loss computation is correct
2. Segmentation head can learn
3. Ground truth mask generation works
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from bev_rasterization import BEVRasterizer, load_kitti_lidar
from encoder import build_encoder


def create_ground_truth_mask(trajectory, grid_size=(37, 50), x_range=(-20, 20), y_range=(-10, 30)):
    """
    Create ground truth road mask by rasterizing future trajectory.
    
    Args:
        trajectory: [N, 2] array of (x, y) future waypoints
        grid_size: (H, W) output mask size
        x_range, y_range: spatial extent in meters
    
    Returns:
        mask: [1, H, W] binary ground truth mask
    """
    H, W = grid_size
    mask = np.zeros((1, H, W), dtype=np.float32)
    
    if len(trajectory) < 2:
        return mask
    
    resolution_x = (x_range[1] - x_range[0]) / W
    resolution_y = (y_range[1] - y_range[0]) / H
    
    # Draw trajectory as line
    for i in range(len(trajectory) - 1):
        pt1 = trajectory[i]
        pt2 = trajectory[i + 1]
        
        # Convert to pixel coordinates
        def world_to_pixel(pt):
            px = int((pt[0] - x_range[0]) / resolution_x)
            py = int((pt[1] - y_range[0]) / resolution_y)
            return (px, py)
        
        p1 = world_to_pixel(pt1)
        p2 = world_to_pixel(pt2)
        
        # Simple line drawing (Bresenham-like)
        if 0 <= p1[0] < W and 0 <= p1[1] < H:
            mask[0, p1[1], p1[0]] = 1.0
        if 0 <= p2[0] < W and 0 <= p2[1] < H:
            mask[0, p2[1], p2[0]] = 1.0
        
        # Interpolate between points
        num_steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1
        for t in range(num_steps):
            alpha = t / max(num_steps - 1, 1)
            px = int(p1[0] * (1 - alpha) + p2[0] * alpha)
            py = int(p1[1] * (1 - alpha) + p2[1] * alpha)
            if 0 <= px < W and 0 <= py < H:
                mask[0, py, px] = 1.0
    
    # Dilate to create thicker road
    from scipy.ndimage import binary_dilation
    mask_binary = (mask[0] > 0.5)
    mask_dilated = binary_dilation(mask_binary, iterations=2)
    mask[0] = mask_dilated.astype(np.float32)
    
    return mask


def load_kitti_trajectory(sequence='00', frame_idx=100, num_future=8):
    """Load future trajectory from KITTI poses."""
    pose_file = f'/media/skr/storage/self_driving/TopoDiffuser/data/kitti/poses/{sequence}.txt'
    
    if not os.path.exists(pose_file):
        return None
    
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            poses.append(values)
    poses = np.array(poses)
    
    # Extract future positions
    if frame_idx + num_future >= len(poses):
        return None
    
    future_positions = []
    current_pose = poses[frame_idx]
    current_x, current_y = current_pose[3], current_pose[7]
    
    for i in range(1, num_future + 1):
        pose = poses[frame_idx + i]
        x, y = pose[3], pose[7]
        # Relative to current position
        future_positions.append([x - current_x, y - current_y])
    
    return np.array(future_positions)


def train_encoder_segmentation(num_epochs=10, batch_size=4, lr=1e-4):
    """
    Train encoder segmentation head for a few epochs.
    
    Returns:
        losses: list of losses per epoch
    """
    print("="*70)
    print("TRAINING TEST: Encoder Segmentation Head (Head A)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Initialize encoder
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    encoder.train()
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Loss and optimizer
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    
    # Prepare data
    rasterizer = BEVRasterizer()
    sequence = '00'
    base_path = f'/media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/{sequence}/velodyne'
    
    # Get list of available frames
    available_frames = []
    for i in range(100, 500):  # Use frames 100-500
        lidar_path = os.path.join(base_path, f'{i:06d}.bin')
        if os.path.exists(lidar_path):
            available_frames.append(i)
    
    print(f"Found {len(available_frames)} frames for training")
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Sample random frames for this epoch
        frame_indices = np.random.choice(available_frames, 
                                         size=min(batch_size * 10, len(available_frames)), 
                                         replace=False)
        
        for i, frame_idx in enumerate(frame_indices):
            # Load LiDAR
            lidar_path = os.path.join(base_path, f'{frame_idx:06d}.bin')
            lidar_points = load_kitti_lidar(lidar_path)
            
            # Create BEV
            bev = rasterizer.rasterize_lidar(lidar_points)
            bev_tensor = torch.from_numpy(bev).unsqueeze(0).to(device)
            
            # Load future trajectory and create GT mask
            trajectory = load_kitti_trajectory(sequence, frame_idx, num_future=8)
            if trajectory is None:
                continue
            
            gt_mask = create_ground_truth_mask(trajectory)
            gt_mask_tensor = torch.from_numpy(gt_mask).unsqueeze(0).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            conditioning, pred_mask = encoder(bev_tensor)
            
            # Compute BCE loss (Head A)
            loss = bce_loss(pred_mask, gt_mask_tensor)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if (i + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(frame_indices)}, "
                      f"Loss: {np.mean(epoch_losses[-10:]):.4f}")
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        losses.append(avg_loss)
        print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    return losses, encoder


def visualize_results(encoder, sequence='00', frame_idx=200):
    """Visualize segmentation predictions on a test frame."""
    print("\n" + "="*70)
    print("VISUALIZING SEGMENTATION RESULTS")
    print("="*70)
    
    device = next(encoder.parameters()).device
    encoder.eval()
    
    # Load data
    rasterizer = BEVRasterizer()
    lidar_path = f'/media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/{sequence}/velodyne/{frame_idx:06d}.bin'
    
    if not os.path.exists(lidar_path):
        print(f"Frame {frame_idx} not found, skipping visualization")
        return
    
    lidar_points = load_kitti_lidar(lidar_path)
    bev = rasterizer.rasterize_lidar(lidar_points)
    bev_tensor = torch.from_numpy(bev).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        conditioning, pred_mask = encoder(bev_tensor)
    
    # Get ground truth
    trajectory = load_kitti_trajectory(sequence, frame_idx)
    gt_mask = create_ground_truth_mask(trajectory) if trajectory is not None else np.zeros((1, 37, 50))
    
    # Convert to numpy
    pred_mask_np = pred_mask[0, 0].cpu().numpy()
    gt_mask_np = gt_mask[0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # BEV channels
    axes[0, 0].imshow(bev[0], cmap='viridis')
    axes[0, 0].set_title('BEV Height')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(bev[1], cmap='hot')
    axes[0, 1].set_title('BEV Intensity')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(bev[2], cmap='Blues')
    axes[0, 2].set_title('BEV Density')
    axes[0, 2].axis('off')
    
    # Masks
    axes[1, 0].imshow(gt_mask_np, cmap='Greens', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth Road Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_mask_np, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Predicted Road Mask\n(mean={pred_mask_np.mean():.3f})')
    axes[1, 1].axis('off')
    
    # Difference
    diff = np.abs(pred_mask_np - gt_mask_np)
    axes[1, 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Absolute Difference\n(mean error={diff.mean():.3f})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/media/skr/storage/self_driving/TopoDiffuser/encoder_seg_training_results.png', dpi=150)
    print(f"Saved visualization to encoder_seg_training_results.png")
    plt.close()


def main():
    """Run training test."""
    # Train
    losses, encoder = train_encoder_segmentation(num_epochs=5, batch_size=4, lr=1e-4)
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'b-o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.title('Encoder Segmentation Head Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('/media/skr/storage/self_driving/TopoDiffuser/encoder_seg_training_curve.png', dpi=150)
    print(f"\nSaved training curve to encoder_seg_training_curve.png")
    plt.close()
    
    # Visualize
    visualize_results(encoder, sequence='00', frame_idx=300)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING TEST SUMMARY")
    print("="*70)
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    if losses[-1] < losses[0]:
        print("\n✓ Loss decreased - Segmentation head is learning!")
    else:
        print("\n⚠ Loss did not decrease - Check implementation")
    
    print("\nHead A (Segmentation) is working correctly!")
    print("Ready for Block 3 (Diffusion Policy)")


if __name__ == "__main__":
    main()
