#!/usr/bin/env python3
"""
Visualize OSM alignment with trajectory and history overlaid.

Shows multiple views to verify alignment:
1. Side-by-side comparison
2. Overlaid view with transparency
3. Zoomed views at multiple locations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, 'utils')


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
    """Extract (x, z) trajectory from poses."""
    return np.array([[p[0, 3], p[2, 3]] for p in poses])


def visualize_alignment(seq: str = "00", data_root: str = "data"):
    """Create comprehensive alignment visualization."""
    
    # Load aligned OSM
    osm_file = f'osm_aligned_seq{seq}.npy'
    if not Path(osm_file).exists():
        print(f"Aligned OSM not found: {osm_file}")
        print("Run: python align_osm_properly.py --seq", seq)
        return
    
    osm_local = np.load(osm_file)
    print(f"Loaded {len(osm_local)} OSM points")
    
    # Load trajectory
    pose_file = Path(data_root) / "kitti" / "poses" / f"{seq}.txt"
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses)
    print(f"Loaded {len(trajectory)} trajectory points")
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === ROW 1: Full views ===
    
    # Left: OSM only
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(osm_local[:, 0], osm_local[:, 1], s=3, c='blue', alpha=0.6, label='OSM Roads')
    ax1.set_title('OSM Roads Only', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Center: Trajectory only
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    ax2.set_title('Trajectory Only', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Right: Overlay (OSM + Trajectory)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(osm_local[:, 0], osm_local[:, 1], s=3, c='blue', alpha=0.4, label='OSM Roads')
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2.5, alpha=0.8, label='Trajectory')
    ax3.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150, marker='*', label='End', zorder=5)
    ax3.set_title('OVERLAY: OSM + Trajectory', fontsize=12, fontweight='bold', color='darkgreen')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # === ROW 2: Zoomed views at different locations ===
    
    # Select 3 interesting frames
    frame_indices = [
        len(trajectory) // 4,
        len(trajectory) // 2,
        3 * len(trajectory) // 4
    ]
    
    zoom_radius = 40  # meters
    
    for idx, (frame_idx, ax_pos) in enumerate(zip(frame_indices, [gs[1, 0], gs[1, 1], gs[1, 2]])):
        ax = fig.add_subplot(ax_pos)
        
        ego_x, ego_y = trajectory[frame_idx]
        
        # Filter OSM to zoom area
        dists = np.sqrt((osm_local[:, 0] - ego_x)**2 + (osm_local[:, 1] - ego_y)**2)
        nearby_mask = dists <= zoom_radius
        osm_nearby = osm_local[nearby_mask]
        
        # Filter trajectory to zoom area
        traj_dists = np.sqrt((trajectory[:, 0] - ego_x)**2 + (trajectory[:, 1] - ego_y)**2)
        traj_nearby_mask = traj_dists <= zoom_radius
        traj_nearby = trajectory[traj_nearby_mask]
        
        # Plot OSM
        if len(osm_nearby) > 0:
            ax.scatter(osm_nearby[:, 0], osm_nearby[:, 1], s=20, c='blue', alpha=0.7, 
                      label=f'OSM ({len(osm_nearby)} pts)', zorder=2)
        
        # Plot trajectory
        if len(traj_nearby) > 0:
            ax.plot(traj_nearby[:, 0], traj_nearby[:, 1], 'r-', linewidth=3, 
                   alpha=0.9, label='Trajectory', zorder=3)
        
        # Plot full trajectory faintly
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1, alpha=0.2, zorder=1)
        
        # Plot ego
        ax.plot(ego_x, ego_y, 'g*', markersize=30, markeredgecolor='black', 
               markeredgewidth=2, label='Ego', zorder=5)
        
        # Plot history points (past 5 keyframes at 2m)
        history_points = []
        dist_accum = 0
        for i in range(frame_idx, max(0, frame_idx-200), -1):
            if i < frame_idx:
                dist = np.linalg.norm(trajectory[i] - trajectory[i+1])
                dist_accum += dist
                if dist_accum >= 2.0 and len(history_points) < 5:
                    history_points.append(trajectory[i])
                    dist_accum = 0
        
        if history_points:
            history_points = np.array(history_points)
            ax.scatter(history_points[:, 0], history_points[:, 1], s=80, c='orange', 
                      marker='o', edgecolors='black', linewidths=1.5,
                      label='History (5 keyframes)', zorder=4)
            # Connect history to ego
            for hp in history_points:
                ax.plot([hp[0], ego_x], [hp[1], ego_y], 'orange', linewidth=1.5, alpha=0.5)
        
        ax.set_xlim(ego_x - zoom_radius - 5, ego_x + zoom_radius + 5)
        ax.set_ylim(ego_y - zoom_radius - 5, ego_y + zoom_radius + 5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'ZOOM Frame {frame_idx}: OSM + Trajectory + History', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    # === ROW 3: Detailed comparison views ===
    
    # Left: OSM density heatmap
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Select middle frame for detailed view
    mid_frame = len(trajectory) // 2
    ego_x, ego_y = trajectory[mid_frame]
    
    # Create 2D histogram of OSM points
    hist_range = [[ego_x - 50, ego_x + 50], [ego_y - 50, ego_y + 50]]
    h, xedges, yedges = np.histogram2d(osm_local[:, 0], osm_local[:, 1], 
                                       bins=50, range=hist_range)
    
    im = ax7.imshow(h.T, origin='lower', extent=[ego_x - 50, ego_x + 50, ego_y - 50, ego_y + 50],
                   cmap='Blues', aspect='equal')
    ax7.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.7)
    ax7.plot(ego_x, ego_y, 'g*', markersize=25, markeredgecolor='black', markeredgewidth=2)
    ax7.set_xlim(ego_x - 50, ego_x + 50)
    ax7.set_ylim(ego_y - 50, ego_y + 50)
    ax7.set_title('OSM Density Heatmap', fontsize=12, fontweight='bold')
    ax7.set_xlabel('X (m)')
    ax7.set_ylabel('Y (m)')
    plt.colorbar(im, ax=ax7, label='OSM Point Count')
    
    # Center: Side-by-side comparison
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Split view: left half OSM, right half trajectory
    x_range = [ego_x - 40, ego_x + 40]
    y_range = [ego_y - 40, ego_y + 40]
    
    # Plot OSM in left half (clipped)
    osm_in_range = osm_local[
        (osm_local[:, 0] >= x_range[0]) & (osm_local[:, 0] <= ego_x) &
        (osm_local[:, 1] >= y_range[0]) & (osm_local[:, 1] <= y_range[1])
    ]
    ax8.scatter(osm_in_range[:, 0], osm_in_range[:, 1], s=15, c='blue', alpha=0.6, label='OSM (left half)')
    
    # Plot trajectory in right half (clipped)
    traj_in_range = trajectory[
        (trajectory[:, 0] >= ego_x) & (trajectory[:, 0] <= x_range[1]) &
        (trajectory[:, 1] >= y_range[0]) & (trajectory[:, 1] <= y_range[1])
    ]
    if len(traj_in_range) > 0:
        ax8.plot(traj_in_range[:, 0], traj_in_range[:, 1], 'r-', linewidth=3, label='Trajectory (right half)')
    
    # Plot center line
    ax8.axvline(x=ego_x, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Split line')
    ax8.plot(ego_x, ego_y, 'g*', markersize=25, markeredgecolor='black', markeredgewidth=2)
    
    ax8.set_xlim(x_range)
    ax8.set_ylim(y_range)
    ax8.set_title('Split View: OSM (left) | Trajectory (right)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Right: Difference/Error visualization
    ax9 = fig.add_subplot(gs[2, 2])
    
    # For each trajectory point, find nearest OSM point
    sample_indices = np.linspace(100, len(trajectory)-100, 50, dtype=int)
    errors = []
    
    for ti in sample_indices:
        tx, ty = trajectory[ti]
        dists = np.sqrt((osm_local[:, 0] - tx)**2 + (osm_local[:, 1] - ty)**2)
        min_dist = np.min(dists)
        errors.append(min_dist)
    
    ax9.hist(errors, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax9.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(errors):.1f}m')
    ax9.set_xlabel('Distance to Nearest OSM Road (m)', fontsize=11)
    ax9.set_ylabel('Count', fontsize=11)
    ax9.set_title(f'Alignment Error Distribution\n({len(errors)} sample points)', 
                 fontsize=12, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Sequence {seq}: OSM Alignment Verification\n' + 
                f'OSM Points: {len(osm_local)} | Trajectory Points: {len(trajectory)} | ' +
                f'Mean Alignment Error: {np.mean(errors):.2f}m',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = f'alignment_verification_seq{seq}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved comprehensive alignment verification to: {output_file}")
    print(f"   Mean alignment error: {np.mean(errors):.2f}m")
    print(f"   Max alignment error: {np.max(errors):.2f}m")
    print(f"   Min alignment error: {np.min(errors):.2f}m")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize OSM alignment with trajectory")
    parser.add_argument("--seq", type=str, default="00", help="Sequence number")
    parser.add_argument("--data_root", type=str, default="data", help="Data root")
    
    args = parser.parse_args()
    
    visualize_alignment(args.seq, args.data_root)
    print("\n🎉 Done!")
