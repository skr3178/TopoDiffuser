"""
Test BEV rasterization with REAL KITTI data.

Loads actual LiDAR point clouds and poses from KITTI dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from bev_rasterization import BEVRasterizer, load_kitti_lidar, extract_trajectory_from_poses


def load_kitti_poses(pose_file):
    """Load KITTI poses from file."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            poses.append(values)
    return np.array(poses)


def transform_lidar_to_ego(points, calib_file):
    """
    Transform LiDAR points from velodyne frame to camera/ego frame.
    
    KITTI calibration provides the transformation matrix.
    For simplicity, we assume velodyne is already close to ego frame
    with a small offset. In practice, you'd use the calibration file.
    """
    # Load calibration
    # For KITTI, velodyne to cam0 transform is in calib.txt
    # We'll use identity + height offset as approximation
    # (velodyne is at ~1.7m height in KITTI)
    
    # For now, just return points as-is (velodyne frame is close to ego)
    # The z-offset will be handled by z_range filtering
    return points


def visualize_real_kitti_rasterization(sequence='00', frame_idx=100):
    """
    Load and visualize real KITTI data rasterization.
    
    Args:
        sequence: KITTI sequence number (e.g., '00', '01', ...)
        frame_idx: Frame index to visualize
    """
    # Paths
    base_path = f'/media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/{sequence}'
    lidar_path = os.path.join(base_path, 'velodyne', f'{frame_idx:06d}.bin')
    pose_path = f'/media/skr/storage/self_driving/TopoDiffuser/data/kitti/poses/{sequence}.txt'
    calib_path = os.path.join(base_path, 'calib.txt')
    
    print(f"Loading KITTI Sequence {sequence}, Frame {frame_idx}")
    print(f"LiDAR: {lidar_path}")
    print(f"Poses: {pose_path}")
    
    # Check files exist
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    
    # Load LiDAR
    lidar_points = load_kitti_lidar(lidar_path)
    print(f"Loaded {len(lidar_points):,} LiDAR points")
    
    # Load poses and extract trajectory
    all_poses = load_kitti_poses(pose_path)
    
    # Get trajectory history (past 5 frames)
    trajectory = []
    for i in range(max(0, frame_idx - 4), frame_idx + 1):
        pose = all_poses[i]
        # Extract translation (tx, ty, tz) from flattened 3x4 matrix
        # Format: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
        tx, ty, tz = pose[3], pose[7], pose[11]
        trajectory.append([tx, ty])
    trajectory = np.array(trajectory, dtype=np.float32)
    
    # Transform trajectory to current frame (relative to current pose)
    current_pose = all_poses[frame_idx]
    current_tx, current_ty = current_pose[3], current_pose[7]
    trajectory = trajectory - np.array([current_tx, current_ty])
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Trajectory range X: [{trajectory[:,0].min():.2f}, {trajectory[:,0].max():.2f}]")
    print(f"Trajectory range Y: [{trajectory[:,1].min():.2f}, {trajectory[:,1].max():.2f}]")
    
    # Rasterize
    rasterizer = BEVRasterizer()
    bev = rasterizer.rasterize_lidar(lidar_points)
    
    print(f"BEV shape: {bev.shape}")
    print(f"  Height: min={bev[0].min():.3f}, max={bev[0].max():.3f}")
    print(f"  Intensity: min={bev[1].min():.3f}, max={bev[1].max():.3f}")
    print(f"  Density: min={bev[2].min():.3f}, max={bev[2].max():.3f}")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'KITTI Real Data BEV Rasterization\nSequence {sequence}, Frame {frame_idx} | '
                 f'{len(lidar_points):,} LiDAR Points', 
                 fontsize=14, fontweight='bold')
    
    x_range, y_range = rasterizer.x_range, rasterizer.y_range
    
    # ===== FRONT: Raw LiDAR =====
    
    # 1. Top-down view (X-Y)
    ax1 = fig.add_subplot(gs[0, 0])
    # Subsample for visualization (too many points otherwise)
    sample_idx = np.random.choice(len(lidar_points), min(10000, len(lidar_points)), replace=False)
    scatter = ax1.scatter(lidar_points[sample_idx, 0], lidar_points[sample_idx, 1], 
                          c=lidar_points[sample_idx, 2], cmap='viridis', 
                          s=1, alpha=0.5)
    if len(trajectory) > 0:
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Past trajectory')
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=4)
    ax1.set_xlabel('X (m) - Lateral')
    ax1.set_ylabel('Y (m) - Longitudinal')
    ax1.set_title('Front: Top-Down View (X-Y)\nColored by Height (subsampled)')
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Height (m)')
    
    # 2. Side view (Y-Z)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(lidar_points[sample_idx, 1], lidar_points[sample_idx, 2], 
                           c=lidar_points[sample_idx, 3], cmap='hot', 
                           s=1, alpha=0.5)
    ax2.set_xlabel('Y (m) - Longitudinal')
    ax2.set_ylabel('Z (m) - Height')
    ax2.set_title('Front: Side View (Y-Z)\nColored by Intensity')
    ax2.set_xlim(y_range)
    ax2.set_ylim(rasterizer.z_range)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Intensity')
    
    # 3. 3D view
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    sample_3d = np.random.choice(len(lidar_points), min(5000, len(lidar_points)), replace=False)
    ax3.scatter(lidar_points[sample_3d, 0], 
                lidar_points[sample_3d, 1], 
                lidar_points[sample_3d, 2],
                c=lidar_points[sample_3d, 2], cmap='viridis',
                s=0.5, alpha=0.5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('Front: 3D View\n(Subsampled)')
    ax3.set_xlim(x_range)
    ax3.set_ylim(y_range)
    ax3.set_zlim(rasterizer.z_range)
    
    # 4. Intensity histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(lidar_points[:, 3], bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Point Count')
    ax4.set_title('Front: Intensity Distribution')
    ax4.grid(True, alpha=0.3)
    
    # ===== BACK: BEV Rasterization =====
    
    # 5. Height channel
    ax5 = fig.add_subplot(gs[0, 2])
    im1 = ax5.imshow(bev[0], origin='lower', cmap='viridis', 
                     extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     aspect='auto')
    if len(trajectory) > 0:
        ax5.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2)
        ax5.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=4)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Back: BEV Height Channel\n(Max Z per cell)')
    plt.colorbar(im1, ax=ax5, label='Normalized Height')
    
    # 6. Intensity channel
    ax6 = fig.add_subplot(gs[0, 3])
    im2 = ax6.imshow(bev[1], origin='lower', cmap='hot',
                     extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     aspect='auto')
    if len(trajectory) > 0:
        ax6.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2)
        ax6.plot(trajectory[:, 0], trajectory[:, 1], 'go', markersize=4)
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('Back: BEV Intensity Channel\n(Avg Reflectivity)')
    plt.colorbar(im2, ax=ax6, label='Normalized Intensity')
    
    # 7. Density channel
    ax7 = fig.add_subplot(gs[1, 2])
    im3 = ax7.imshow(bev[2], origin='lower', cmap='Blues',
                     extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     aspect='auto')
    if len(trajectory) > 0:
        ax7.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2)
        ax7.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=4)
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_title('Back: BEV Density Channel\n(Log Point Count)')
    plt.colorbar(im3, ax=ax7, label='Normalized Density')
    
    # 8. RGB composite
    ax8 = fig.add_subplot(gs[1, 3])
    rgb = np.stack([bev[0], bev[1], bev[2]], axis=-1)
    # Normalize per channel for visualization
    for i in range(3):
        if rgb[:,:,i].max() > rgb[:,:,i].min():
            rgb[:,:,i] = (rgb[:,:,i] - rgb[:,:,i].min()) / (rgb[:,:,i].max() - rgb[:,:,i].min())
    ax8.imshow(rgb, origin='lower',
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
               aspect='auto')
    if len(trajectory) > 0:
        ax8.plot(trajectory[:, 0], trajectory[:, 1], 'cyan', linewidth=2)
        ax8.plot(trajectory[:, 0], trajectory[:, 1], 'co', markersize=4)
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.set_title('Back: RGB Composite\n(R=Height, G=Intensity, B=Density)')
    
    # Add statistics text
    occupied = (bev[2] > 0).sum()
    stats_text = f"""
KITTI Sequence {sequence}, Frame {frame_idx}
LiDAR Points: {len(lidar_points):,}
BEV Grid: {bev.shape[1]} × {bev.shape[2]} ({rasterizer.H} × {rasterizer.W})
Resolution: {rasterizer.resolution} m/pixel

BEV Channel Stats:
Height: min={bev[0].min():.3f}, max={bev[0].max():.3f}, mean={bev[0].mean():.3f}
Intensity: min={bev[1].min():.3f}, max={bev[1].max():.3f}, mean={bev[1].mean():.3f}
Density: min={bev[2].min():.3f}, max={bev[2].max():.3f}, mean={bev[2].mean():.3f}

Occupied Cells: {occupied:,} / {bev[2].size:,} ({100*occupied/bev[2].size:.1f}%)
    """
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    return fig, bev, lidar_points, trajectory


def test_multiple_frames():
    """Test rasterization on multiple KITTI frames."""
    sequence = '00'
    frame_indices = [100, 500, 1000, 1500]  # Different frames
    
    for frame_idx in frame_indices:
        print(f"\n{'='*70}")
        try:
            fig, bev, points, traj = visualize_real_kitti_rasterization(sequence, frame_idx)
            
            # Save
            output_path = f'/media/skr/storage/self_driving/TopoDiffuser/kitti_rasterization_seq{sequence}_frame{frame_idx:06d}.png'
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue


if __name__ == "__main__":
    print("="*70)
    print("Real KITTI Data BEV Rasterization Test")
    print("="*70)
    
    # Test single frame first
    print("\nTesting single frame (Sequence 00, Frame 100)...")
    try:
        fig, bev, points, traj = visualize_real_kitti_rasterization('00', 100)
        
        output_path = '/media/skr/storage/self_driving/TopoDiffuser/kitti_rasterization_test.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")
        plt.close(fig)
        
        # Now test multiple frames
        print("\n" + "="*70)
        print("Testing multiple frames...")
        print("="*70)
        test_multiple_frames()
        
        print("\n" + "="*70)
        print("All real KITTI visualizations complete!")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure KITTI data is available at:")
        print("  /media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/")
        print("  /media/skr/storage/self_driving/TopoDiffuser/data/kitti/poses/")
