#!/usr/bin/env python3
"""
Verify Trajectory History Implementation (Modality 2)

Paper Specs:
- Past 5 keyframes
- 2-meter spacing
- Binary occupancy map I_traj ∈ R^{H×W×1}
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'utils')
from bev_utils import poses_to_trajectory, trajectory_to_bev


def verify_history_rasterization(poses_path: str, frame_idx: int = 100):
    """Verify history BEV generation for a specific frame."""
    
    # Load poses
    poses = np.loadtxt(poses_path)
    print(f"Loaded {len(poses)} poses from {poses_path}")
    
    # Extract history trajectory (paper spec: 5 keyframes, 2m spacing)
    num_frames = 5
    spacing_meters = 2.0
    
    # Get past indices
    start_idx = max(0, frame_idx - num_frames * 10)  # Approximate 10 frames per meter
    past_poses = poses[start_idx:frame_idx+1]
    
    # Extract trajectory
    trajectory = poses_to_trajectory(
        past_poses,
        num_frames=num_frames,
        spacing_meters=spacing_meters
    )
    
    print(f"\n=== Trajectory History Verification ===")
    print(f"Current frame: {frame_idx}")
    print(f"Num keyframes: {len(trajectory)} (expected: 5)")
    print(f"Keyframe positions (ego frame):")
    
    # Get current pose for ego-frame conversion
    current_pose = poses[frame_idx].reshape(3, 4)
    current_R = current_pose[:, :3]
    current_t = current_pose[:, 3]
    
    # Convert to ego frame
    # KITTI: trajectory is (x, z) in world frame
    # Ego frame: x=right, y=forward (z in world becomes y in ego BEV)
    trajectory_ego = []
    for i, pos_world in enumerate(trajectory):
        # pos_world is (x, z), convert to (x, y, z) for transformation
        pos_world_3d = np.array([pos_world[0], 0, pos_world[1]])
        pos_ego = current_R.T @ (pos_world_3d - current_t)
        # For BEV: x=right (same), y=forward (was z in world)
        trajectory_ego.append([pos_ego[0], pos_ego[2]])
        print(f"  Frame {i}: ({pos_ego[0]:.2f}, {pos_ego[2]:.2f}) m")
    
    trajectory_ego = np.array(trajectory_ego)
    
    # Calculate actual spacing
    distances = []
    for i in range(1, len(trajectory_ego)):
        dist = np.linalg.norm(trajectory_ego[i] - trajectory_ego[i-1])
        distances.append(dist)
    
    print(f"\nActual spacing between keyframes: {distances}")
    print(f"Average spacing: {np.mean(distances):.2f} m (target: 2.0 m)")
    
    # Rasterize to BEV
    history_bev = trajectory_to_bev(
        trajectory_ego,
        grid_size=(300, 400),
        resolution=0.1,
        x_range=(-20, 20),
        y_range=(-10, 30),
        line_width=2
    )
    
    print(f"\nBEV output shape: {history_bev.shape}")
    print(f"Occupied pixels: {np.sum(history_bev > 0)}")
    print(f"Value range: [{history_bev.min():.1f}, {history_bev.max():.1f}]")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Trajectory in world frame
    ax = axes[0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('X (m) - World Frame')
    ax.set_ylabel('Y (m) - World Frame')
    ax.set_title('Trajectory History (World Frame)')
    ax.grid(True)
    ax.axis('equal')
    
    # Plot 2: Trajectory in ego frame
    ax = axes[1]
    ax.plot(trajectory_ego[:, 0], trajectory_ego[:, 1], 'r-o', linewidth=2, markersize=8)
    ax.plot(0, 0, 'g*', markersize=20, label='Ego Vehicle')
    ax.set_xlabel('X (m) - Ego Frame')
    ax.set_ylabel('Y (m) - Ego Frame')
    ax.set_title('Trajectory History (Ego Frame)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Plot 3: BEV rasterization
    ax = axes[2]
    im = ax.imshow(history_bev[0], cmap='Reds', origin='lower',
                   extent=[-20, 20, -10, 30])
    ax.plot(0, 0, 'g*', markersize=20, label='Ego')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('History BEV Binary Occupancy')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Occupancy')
    
    plt.tight_layout()
    plt.savefig('verify_history_output.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: verify_history_output.png")
    
    return history_bev, trajectory_ego


def verify_multiple_frames(poses_path: str, num_samples: int = 5):
    """Verify history consistency across multiple frames."""
    
    poses = np.loadtxt(poses_path)
    
    print(f"\n=== Multi-Frame Verification ===")
    print(f"Checking {num_samples} random frames...")
    
    frame_indices = np.linspace(50, len(poses)-50, num_samples, dtype=int)
    
    for frame_idx in frame_indices:
        start_idx = max(0, frame_idx - 5 * 10)
        past_poses = poses[start_idx:frame_idx+1]
        
        trajectory = poses_to_trajectory(
            past_poses,
            num_frames=5,
            spacing_meters=2.0
        )
        
        print(f"  Frame {frame_idx}: {len(trajectory)} keyframes extracted")
        
        # Verify spacing
        if len(trajectory) >= 2:
            total_dist = np.linalg.norm(trajectory[-1] - trajectory[0])
            print(f"    Total distance: {total_dist:.2f} m (~10m expected for 5 frames at 2m)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify trajectory history implementation")
    parser.add_argument("--poses", type=str, 
                        default="data/kitti/poses/00.txt",
                        help="Path to poses file")
    parser.add_argument("--frame", type=int, default=100,
                        help="Frame index to visualize")
    parser.add_argument("--multi", action="store_true",
                        help="Verify multiple frames")
    
    args = parser.parse_args()
    
    if not Path(args.poses).exists():
        print(f"Error: Poses file not found: {args.poses}")
        print("Using symlinked data path...")
        args.poses = "data/kitti/poses/00.txt"
    
    if not Path(args.poses).exists():
        print(f"Error: Still not found. Please check data path.")
        sys.exit(1)
    
    # Run verification
    verify_history_rasterization(args.poses, args.frame)
    
    if args.multi:
        verify_multiple_frames(args.poses)
    
    print("\n✅ History implementation verification complete!")
