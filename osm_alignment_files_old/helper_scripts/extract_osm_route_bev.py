#!/usr/bin/env python3
"""
Extract OSM route around ego vehicle and rasterize to BEV.

Paper spec: "5 keyframes into the past and 15 into the future, sampled every 2 meters"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'utils')
from osm_alignment import latlon_to_utm


def load_aligned_osm(seq: str):
    """Load pre-aligned OSM data."""
    aligned_file = f'osm_aligned_seq{seq}.npy'
    if Path(aligned_file).exists():
        return np.load(aligned_file)
    else:
        raise FileNotFoundError(f"Aligned OSM not found. Run align_osm_properly.py first.")


def load_poses(pose_file: str) -> np.ndarray:
    """Load KITTI poses."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses: np.ndarray) -> np.ndarray:
    """Extract (x, z) trajectory."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def extract_osm_route_near_ego(osm_local, ego_pose, search_radius=50.0):
    """
    Extract OSM road points near ego vehicle.
    
    Args:
        osm_local: OSM points in local frame [N, 2]
        ego_pose: (x, y, yaw) ego position
        search_radius: Search radius in meters
    
    Returns:
        osm_nearby: OSM points within search radius
    """
    ego_x, ego_y = ego_pose[0], ego_pose[1]
    
    # Compute distances
    distances = np.sqrt((osm_local[:, 0] - ego_x)**2 + (osm_local[:, 1] - ego_y)**2)
    
    # Filter points within radius
    mask = distances <= search_radius
    osm_nearby = osm_local[mask]
    
    return osm_nearby


def rasterize_osm_to_bev(osm_points, ego_pose, 
                         grid_size=(300, 400), 
                         resolution=0.1,
                         x_range=(-20, 20), 
                         y_range=(-10, 30),
                         line_width=3):
    """
    Rasterize OSM points to BEV binary mask.
    
    Args:
        osm_points: OSM points in local frame
        ego_pose: (x, y, yaw) ego position
        grid_size: (H, W) BEV grid
        resolution: meters per pixel
        x_range: x coordinate range in ego frame
        y_range: y coordinate range in ego frame
    
    Returns:
        bev_mask: [H, W] binary mask
    """
    H, W = grid_size
    mask = np.zeros((H, W), dtype=np.float32)
    
    if len(osm_points) == 0:
        return mask
    
    ego_x, ego_y, ego_yaw = ego_pose
    
    # Transform OSM points to ego frame
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    
    for x_world, y_world in osm_points:
        # World to ego
        dx = x_world - ego_x
        dy = y_world - ego_y
        
        x_ego = dx * cos_yaw - dy * sin_yaw
        y_ego = dx * sin_yaw + dy * cos_yaw
        
        # Check if in BEV range
        if x_range[0] <= x_ego <= x_range[1] and y_range[0] <= y_ego <= y_range[1]:
            # Convert to pixel
            px = int((x_ego - x_range[0]) / resolution)
            py = int((y_ego - y_range[0]) / resolution)
            
            # Clamp to bounds
            px = np.clip(px, 0, W - 1)
            py = np.clip(py, 0, H - 1)
            
            # Draw with line width
            for dx in range(-line_width//2, line_width//2 + 1):
                for dy in range(-line_width//2, line_width//2 + 1):
                    pxx = np.clip(px + dx, 0, W - 1)
                    pyy = np.clip(py + dy, 0, H - 1)
                    mask[pyy, pxx] = 1.0
    
    return mask


def visualize_osm_bev(seq: str = "00", data_root: str = "data"):
    """Visualize OSM BEV extraction at multiple frames."""
    
    # Load aligned OSM
    osm_local = load_aligned_osm(seq)
    print(f"Loaded {len(osm_local)} aligned OSM points")
    
    # Load poses
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses)
    
    # Select frames to visualize
    frame_indices = [100, len(trajectory)//2, len(trajectory)-100]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, frame_idx in enumerate(frame_indices):
        # Get ego pose (x, y, yaw)
        pose = poses[frame_idx]
        ego_x = pose[0, 3]
        ego_y = pose[2, 3]
        # Extract yaw from rotation matrix
        ego_yaw = np.arctan2(pose[1, 0], pose[0, 0])
        ego_pose = (ego_x, ego_y, ego_yaw)
        
        # Extract OSM route near ego
        osm_nearby = extract_osm_route_near_ego(osm_local, ego_pose, search_radius=50.0)
        
        # Rasterize to BEV
        bev_mask = rasterize_osm_to_bev(osm_nearby, ego_pose)
        
        # Top row: Local frame view
        ax = axes[0, idx]
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1, alpha=0.3)
        
        # Plot OSM points
        if len(osm_nearby) > 0:
            ax.scatter(osm_nearby[:, 0], osm_nearby[:, 1], s=5, c='blue', alpha=0.6, label='OSM roads')
        
        # Plot ego
        ax.plot(ego_x, ego_y, 'g*', markersize=20, label='Ego')
        
        # Draw BEV range
        bev_x = [ego_x - 20, ego_x + 20, ego_x + 20, ego_x - 20, ego_x - 20]
        bev_y = [ego_y - 10, ego_y - 10, ego_y + 30, ego_y + 30, ego_y - 10]
        ax.plot(bev_x, bev_y, 'g--', linewidth=2, label='BEV range')
        
        ax.set_xlim(ego_x - 60, ego_x + 60)
        ax.set_ylim(ego_y - 60, ego_y + 60)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Frame {frame_idx}: OSM + Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Bottom row: BEV view
        ax = axes[1, idx]
        
        im = ax.imshow(bev_mask, cmap='Blues', origin='lower', 
                      extent=[-20, 20, -10, 30], vmin=0, vmax=1)
        ax.plot(0, 0, 'r*', markersize=20, label='Ego')
        
        # Add direction arrow
        arrow_len = 5
        ax.arrow(0, 0, 0, arrow_len, head_width=2, head_length=1, 
                fc='red', ec='red', linewidth=2, label='Forward')
        
        ax.set_xlabel('X (m) - Right')
        ax.set_ylabel('Y (m) - Forward')
        ax.set_title(f'BEV Mask: {int(bev_mask.sum())} pixels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label='Road occupancy')
    
    plt.suptitle(f'Sequence {seq}: OSM Route Extraction for BEV (5 past + 15 future @ 2m)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = f'osm_bev_extraction_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract OSM route and create BEV")
    parser.add_argument("--seq", type=str, default="00", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root")
    
    args = parser.parse_args()
    
    visualize_osm_bev(args.seq, args.data_root)
    print("\n✅ Done!")
