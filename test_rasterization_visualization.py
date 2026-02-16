"""
Test and visualize BEV rasterization: Compare front (raw LiDAR) vs back (BEV output).

This script creates side-by-side visualizations of:
- Front: Raw LiDAR point cloud (3D view)
- Back: BEV rasterization (Height, Intensity, Density channels)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from bev_rasterization import BEVRasterizer


def create_synthetic_scene(scene_type='straight_road'):
    """
    Create synthetic LiDAR data for different scene types.
    
    Args:
        scene_type: 'straight_road', 'curve', 'intersection', 'with_obstacles'
    
    Returns:
        lidar_points: [N, 4] array (x, y, z, intensity)
        trajectory: [5, 2] waypoints
        description: Scene description string
    """
    np.random.seed(42)
    points = []
    
    if scene_type == 'straight_road':
        # Road surface: flat plane with markings
        n_road = 15000
        road_y = np.random.uniform(-5, 25, n_road)
        road_x = np.random.uniform(-3, 3, n_road)
        road_z = np.random.uniform(-0.1, 0.1, n_road)  # Flat ground
        # Road markings (center line)
        marking_mask = np.abs(road_x) < 0.2
        road_intensity = np.where(marking_mask, 200, 50)  # High intensity for markings
        
        points.append(np.column_stack([road_x, road_y, road_z, road_intensity]))
        
        # Curbs on both sides
        n_curb = 3000
        for side in [-1, 1]:
            curb_y = np.random.uniform(-5, 25, n_curb // 2)
            curb_x = np.full_like(curb_y, side * 3.5) + np.random.normal(0, 0.1, n_curb // 2)
            curb_z = np.random.uniform(0.1, 0.3, n_curb // 2)  # Elevated curbs
            curb_intensity = np.full_like(curb_y, 100)
            points.append(np.column_stack([curb_x, curb_y, curb_z, curb_intensity]))
        
        trajectory = np.array([[0, -5], [0, 0], [0, 5], [0, 10], [0, 15]], dtype=np.float32)
        description = "Straight road with center markings and curbs"
    
    elif scene_type == 'curve':
        # Curved road
        n_road = 15000
        t = np.random.uniform(0, 1, n_road)
        road_y = t * 20
        # Curve: x varies with y
        base_x = 3 * np.sin(road_y * 0.15)
        road_x = base_x + np.random.uniform(-2.5, 2.5, n_road)
        road_z = np.random.uniform(-0.1, 0.1, n_road)
        road_intensity = np.where(np.abs(road_x - base_x) < 0.2, 200, 50)
        
        points.append(np.column_stack([road_x, road_y, road_z, road_intensity]))
        
        # Curbs following the curve
        n_curb = 3000
        for offset in [-2.8, 2.8]:
            t = np.random.uniform(0, 1, n_curb // 2)
            curb_y = t * 20
            curb_x = 3 * np.sin(curb_y * 0.15) + offset + np.random.normal(0, 0.1, n_curb // 2)
            curb_z = np.random.uniform(0.1, 0.3, n_curb // 2)
            curb_intensity = np.full_like(curb_y, 100)
            points.append(np.column_stack([curb_x, curb_y, curb_z, curb_intensity]))
        
        trajectory = np.array([[0, -5], [0.5, 0], [1.5, 5], [2.5, 10], [2.5, 15]], dtype=np.float32)
        description = "Curved road with sinusoidal path"
    
    elif scene_type == 'with_obstacles':
        # Straight road with obstacles (vehicles/pedestrians)
        n_road = 12000
        road_y = np.random.uniform(-5, 25, n_road)
        road_x = np.random.uniform(-3, 3, n_road)
        road_z = np.random.uniform(-0.1, 0.1, n_road)
        road_intensity = np.where(np.abs(road_x) < 0.2, 200, 50)
        points.append(np.column_stack([road_x, road_y, road_z, road_intensity]))
        
        # Add a vehicle ahead (box shape)
        n_vehicle = 2000
        veh_x = np.random.uniform(-1.5, 1.5, n_vehicle) + 1.0  # Right lane
        veh_y = np.random.uniform(8, 14, n_vehicle)
        veh_z = np.random.uniform(0, 1.5, n_vehicle)  # Vehicle height
        veh_intensity = np.full_like(veh_x, 150)
        points.append(np.column_stack([veh_x, veh_y, veh_z, veh_intensity]))
        
        # Add a pedestrian on the left
        n_ped = 500
        ped_x = np.random.uniform(-0.3, 0.3, n_ped) - 2.0
        ped_y = np.random.uniform(5, 6, n_ped)
        ped_z = np.random.uniform(0, 1.7, n_ped)
        ped_intensity = np.full_like(ped_x, 120)
        points.append(np.column_stack([ped_x, ped_y, ped_z, ped_intensity]))
        
        trajectory = np.array([[-0.5, -5], [-0.5, 0], [-0.5, 5], [-0.5, 10], [-0.5, 15]], dtype=np.float32)
        description = "Road with vehicle ahead and pedestrian on side"
    
    else:  # intersection
        # Intersection: roads crossing
        n_road = 20000
        # Main road (vertical)
        road1_y = np.random.uniform(-5, 25, n_road // 2)
        road1_x = np.random.uniform(-3, 3, n_road // 2)
        road1_z = np.random.uniform(-0.1, 0.1, n_road // 2)
        road1_intensity = np.where(np.abs(road1_x) < 0.2, 200, 50)
        points.append(np.column_stack([road1_x, road1_y, road1_z, road1_intensity]))
        
        # Cross road (horizontal)
        road2_x = np.random.uniform(-15, 15, n_road // 2)
        road2_y = np.random.uniform(-2, 2, n_road // 2) + 10
        road2_z = np.random.uniform(-0.1, 0.1, n_road // 2)
        road2_intensity = np.where(np.abs(road2_y - 10) < 0.2, 200, 50)
        points.append(np.column_stack([road2_x, road2_y, road2_z, road2_intensity]))
        
        trajectory = np.array([[0, -5], [0, 0], [0, 5], [0, 10], [0, 15]], dtype=np.float32)
        description = "4-way intersection with crossing roads"
    
    lidar_points = np.vstack(points).astype(np.float32)
    return lidar_points, trajectory, description


def visualize_front_back(lidar_points, trajectory, description, scene_name):
    """
    Visualize front (raw LiDAR) vs back (BEV rasterization).
    
    Creates a figure with:
    - Left: 3D point cloud view (top-down, side, 3D)
    - Right: BEV channels (Height, Intensity, Density)
    """
    # Rasterize
    rasterizer = BEVRasterizer()
    bev = rasterizer.rasterize_lidar(lidar_points)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'BEV Rasterization: Front (Raw) vs Back (BEV)\n{scene_name}: {description}', 
                 fontsize=14, fontweight='bold')
    
    # ===== FRONT: Raw LiDAR Point Cloud =====
    
    # 1. Top-down view (X-Y plane)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(lidar_points[:, 0], lidar_points[:, 1], 
                          c=lidar_points[:, 2], cmap='viridis', 
                          s=0.5, alpha=0.6)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=5)
    ax1.set_xlabel('X (m) - Lateral')
    ax1.set_ylabel('Y (m) - Longitudinal')
    ax1.set_title('Front: Top-Down View (X-Y)\nColored by Height')
    ax1.set_xlim(rasterizer.x_range)
    ax1.set_ylim(rasterizer.y_range)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Height (m)')
    
    # 2. Side view (Y-Z plane)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(lidar_points[:, 1], lidar_points[:, 2], 
                           c=lidar_points[:, 3], cmap='hot', 
                           s=0.5, alpha=0.6)
    ax2.set_xlabel('Y (m) - Longitudinal')
    ax2.set_ylabel('Z (m) - Height')
    ax2.set_title('Front: Side View (Y-Z)\nColored by Intensity')
    ax2.set_xlim(rasterizer.y_range)
    ax2.set_ylim(rasterizer.z_range)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Intensity')
    
    # 3. 3D view
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    # Subsample for 3D plot (too slow with all points)
    sample_idx = np.random.choice(len(lidar_points), min(5000, len(lidar_points)), replace=False)
    ax3.scatter(lidar_points[sample_idx, 0], 
                lidar_points[sample_idx, 1], 
                lidar_points[sample_idx, 2],
                c=lidar_points[sample_idx, 2], cmap='viridis',
                s=1, alpha=0.5)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], np.zeros_like(trajectory[:, 0]), 
             'r-', linewidth=2, label='Trajectory')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('Front: 3D View\n(Subsampled)')
    ax3.set_xlim(rasterizer.x_range)
    ax3.set_ylim(rasterizer.y_range)
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
                     extent=[rasterizer.x_range[0], rasterizer.x_range[1],
                             rasterizer.y_range[0], rasterizer.y_range[1]],
                     aspect='auto')
    ax5.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2)
    ax5.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=5)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Back: BEV Height Channel\n(Max Z per cell)')
    plt.colorbar(im1, ax=ax5, label='Normalized Height')
    
    # 6. Intensity channel
    ax6 = fig.add_subplot(gs[0, 3])
    im2 = ax6.imshow(bev[1], origin='lower', cmap='hot',
                     extent=[rasterizer.x_range[0], rasterizer.x_range[1],
                             rasterizer.y_range[0], rasterizer.y_range[1]],
                     aspect='auto')
    ax6.plot(trajectory[:, 0], trajectory[:, 1], 'g-', linewidth=2)
    ax6.plot(trajectory[:, 0], trajectory[:, 1], 'go', markersize=5)
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('Back: BEV Intensity Channel\n(Avg Reflectivity)')
    plt.colorbar(im2, ax=ax6, label='Normalized Intensity')
    
    # 7. Density channel
    ax7 = fig.add_subplot(gs[1, 2])
    im3 = ax7.imshow(bev[2], origin='lower', cmap='Blues',
                     extent=[rasterizer.x_range[0], rasterizer.x_range[1],
                             rasterizer.y_range[0], rasterizer.y_range[1]],
                     aspect='auto')
    ax7.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2)
    ax7.plot(trajectory[:, 0], trajectory[:, 1], 'ro', markersize=5)
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    ax7.set_title('Back: BEV Density Channel\n(Log Point Count)')
    plt.colorbar(im3, ax=ax7, label='Normalized Density')
    
    # 8. RGB composite
    ax8 = fig.add_subplot(gs[1, 3])
    # Create RGB composite (normalize each channel to 0-1)
    rgb = np.stack([bev[0], bev[1], bev[2]], axis=-1)
    rgb = (rgb - rgb.min(axis=(0, 1))) / (rgb.max(axis=(0, 1)) - rgb.min(axis=(0, 1)) + 1e-8)
    ax8.imshow(rgb, origin='lower',
               extent=[rasterizer.x_range[0], rasterizer.x_range[1],
                       rasterizer.y_range[0], rasterizer.y_range[1]],
               aspect='auto')
    ax8.plot(trajectory[:, 0], trajectory[:, 1], 'cyan', linewidth=2)
    ax8.plot(trajectory[:, 0], trajectory[:, 1], 'co', markersize=5)
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.set_title('Back: RGB Composite\n(R=Height, G=Intensity, B=Density)')
    
    # Add statistics text
    stats_text = f"""
Statistics:
LiDAR Points: {len(lidar_points):,}
BEV Grid: {bev.shape[1]} × {bev.shape[2]} ({rasterizer.H} × {rasterizer.W})
Resolution: {rasterizer.resolution} m/pixel

BEV Channel Stats:
Height: min={bev[0].min():.3f}, max={bev[0].max():.3f}, mean={bev[0].mean():.3f}
Intensity: min={bev[1].min():.3f}, max={bev[1].max():.3f}, mean={bev[1].mean():.3f}
Density: min={bev[2].min():.3f}, max={bev[2].max():.3f}, mean={bev[2].mean():.3f}

Occupied Cells: {(bev[2] > 0).sum():,} / {bev[2].size:,} ({100*(bev[2] > 0).sum()/bev[2].size:.1f}%)
    """
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    return fig, bev


def main():
    """Run visualization tests for different scene types."""
    print("=" * 70)
    print("BEV Rasterization Visualization Test")
    print("Front (Raw LiDAR) vs Back (BEV Rasterization)")
    print("=" * 70)
    
    scene_types = ['straight_road', 'curve', 'with_obstacles', 'intersection']
    
    for scene_type in scene_types:
        print(f"\n{'='*70}")
        print(f"Scene: {scene_type.replace('_', ' ').title()}")
        print(f"{'='*70}")
        
        # Create synthetic data
        lidar_points, trajectory, description = create_synthetic_scene(scene_type)
        
        print(f"Description: {description}")
        print(f"LiDAR points: {len(lidar_points):,}")
        print(f"Trajectory shape: {trajectory.shape}")
        
        # Visualize
        fig, bev = visualize_front_back(lidar_points, trajectory, description, 
                                        scene_type.replace('_', ' ').title())
        
        # Save figure
        output_path = f'/media/skr/storage/self_driving/TopoDiffuser/test_rasterization_{scene_type}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close(fig)
    
    print(f"\n{'='*70}")
    print("All visualizations saved!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
