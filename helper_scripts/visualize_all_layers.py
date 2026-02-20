#!/usr/bin/env python3
"""
Visualize all layers superimposed: OSM + Trajectory + History

Shows how the OSM road network relates to the actual driven trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def load_poses(pose_file: str) -> np.ndarray:
    """Load KITTI poses from file."""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses: np.ndarray) -> np.ndarray:
    """Extract (x, z) trajectory from poses (KITTI local frame)."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def visualize_all_layers(seq: str = "00", data_root: str = "data"):
    """Visualize OSM roads + full trajectory + sample history snippets."""
    
    # Load OSM edges (GPS coordinates)
    osm_edges_file = Path(data_root) / "osm" / f"{seq}_edges.npy"
    osm_polylines_file = Path(data_root) / "osm_polylines" / f"{seq}_polylines.pkl"
    
    # Load trajectory (local frame)
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # === LEFT: OSM Map Only (GPS coords) ===
    ax = axes[0]
    
    if osm_edges_file.exists():
        edges = np.load(osm_edges_file)
        ax.scatter(edges[:, 1], edges[:, 0], s=0.1, c='blue', alpha=0.5, label='OSM Roads')
        ax.set_title(f'OSM Road Network (GPS)\n{len(edges)} points', fontsize=12, fontweight='bold')
    elif osm_polylines_file.exists():
        with open(osm_polylines_file, 'rb') as f:
            polylines = pickle.load(f)
        for polyline in polylines:
            if len(polyline) >= 2:
                lats = [p[0] for p in polyline]
                lons = [p[1] for p in polyline]
                ax.plot(lons, lats, 'b-', linewidth=0.3, alpha=0.6)
        ax.set_title(f'OSM Road Network (GPS)\n{len(polylines)} segments', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'OSM data not found', ha='center', va='center')
    
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # === CENTER: Trajectory Only (Local frame) ===
    ax = axes[1]
    
    if pose_file.exists():
        poses = load_poses(pose_file)
        trajectory = extract_trajectory(poses)
        
        # Full trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1.5, alpha=0.7, label='Full Trajectory')
        
        # Start and end
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=20, label='End')
        
        # Sample 3 history snippets
        snippet_indices = [100, len(trajectory)//2, len(trajectory)-100]
        colors = ['purple', 'orange', 'cyan']
        
        for idx, color in zip(snippet_indices, colors):
            # Extract 5 keyframes at 2m spacing
            past_indices = []
            current_idx = idx
            dist_accum = 0
            
            for i in range(idx, max(0, idx-100), -1):
                dist = np.linalg.norm(trajectory[i] - trajectory[current_idx])
                if dist >= 2.0 and len(past_indices) < 5:
                    past_indices.append(i)
                    current_idx = i
                    dist_accum = 0
            
            past_indices = sorted(past_indices + [idx])
            
            if len(past_indices) >= 2:
                snippet = trajectory[past_indices]
                ax.plot(snippet[:, 0], snippet[:, 1], 'o-', color=color, 
                       linewidth=2, markersize=6, alpha=0.8,
                       label=f'History @ frame {idx}')
        
        ax.set_title(f'Trajectory + History (Local Frame)\n{len(trajectory)} frames', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m) - Right', fontsize=10)
        ax.set_ylabel('Z (m) - Forward', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    else:
        ax.text(0.5, 0.5, 'Pose file not found', ha='center', va='center')
    
    # === RIGHT: Overlay Attempt (showing coordinate misalignment) ===
    ax = axes[2]
    
    if osm_edges_file.exists() and pose_file.exists():
        # Plot OSM
        edges = np.load(osm_edges_file)
        # Normalize OSM to roughly match trajectory scale for visualization
        osm_center = np.mean(edges[:, :2], axis=0)
        osm_std = np.std(edges[:, :2], axis=0)
        traj_center = np.mean(trajectory, axis=0)
        traj_std = np.std(trajectory, axis=0)
        
        # Simple normalization (not proper alignment!)
        osm_normalized = (edges[:, :2] - osm_center) / osm_std * traj_std + traj_center
        
        ax.scatter(osm_normalized[:, 1], osm_normalized[:, 0], s=0.1, c='blue', alpha=0.3, label='OSM Roads (normalized)')
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=20)
        
        ax.set_title('Overlay: OSM + Trajectory\n(Normalized - NOT Properly Aligned!)', 
                    fontsize=12, fontweight='bold', color='red')
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add warning text
        ax.text(0.5, 0.02, 'WARNING: Proper GPS→Local alignment needed!', 
               transform=ax.transAxes, ha='center', fontsize=10, 
               color='red', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Data not available for overlay', ha='center', va='center')
    
    plt.suptitle(f'Sequence {seq}: All Layers Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = f'all_layers_seq{seq}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize all layers (OSM + Trajectory + History)")
    parser.add_argument("--seq", type=str, default="00", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root directory")
    
    args = parser.parse_args()
    
    visualize_all_layers(args.seq, args.data_root)
    print("\n✅ Done!")
