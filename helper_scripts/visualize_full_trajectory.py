#!/usr/bin/env python3
"""
Visualize the full trajectory of a KITTI sequence.

Creates a bird's-eye view plot of the entire sequence path.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_poses(pose_file: str) -> np.ndarray:
    """Load KITTI poses from file.
    
    Each pose is a 3x4 transformation matrix flattened row-wise.
    Returns array of shape [N, 3, 4].
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def extract_trajectory(poses: np.ndarray) -> np.ndarray:
    """Extract (x, z) trajectory from poses.
    
    In KITTI: x=right, y=down, z=forward
    For ground plane: use (x, z)
    """
    trajectory = []
    for pose in poses:
        x = pose[0, 3]  # Translation in x
        z = pose[2, 3]  # Translation in z (forward)
        trajectory.append([x, z])
    return np.array(trajectory)


def plot_full_trajectory(pose_file: str, seq: str = "00", sample_every: int = 10):
    """Plot the full trajectory with style similar to verify_history_output.png."""
    
    print(f"Loading poses from: {pose_file}")
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses)
    
    print(f"Total frames: {len(trajectory)}")
    print(f"Trajectory length: {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.1f} meters")
    print(f"X range: [{trajectory[:, 0].min():.1f}, {trajectory[:, 0].max():.1f}] m")
    print(f"Z range: [{trajectory[:, 1].min():.1f}, {trajectory[:, 1].max():.1f}] m")
    
    # Create figure with similar style
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot full trajectory as a line
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            'b-', linewidth=2, alpha=0.6, label='Full Trajectory')
    
    # Sample points to show direction
    sampled_indices = np.arange(0, len(trajectory), sample_every)
    
    # Plot sampled points
    ax.scatter(trajectory[sampled_indices, 0], trajectory[sampled_indices, 1],
               c=sampled_indices, cmap='viridis', s=30, alpha=0.7, zorder=5)
    
    # Mark start and end
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=15, 
            label='Start', zorder=10)
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=20, 
            label='End', zorder=10)
    
    # Add direction arrows at regular intervals
    arrow_indices = np.arange(0, len(trajectory) - sample_every, sample_every * 5)
    for idx in arrow_indices:
        dx = trajectory[idx + sample_every, 0] - trajectory[idx, 0]
        dz = trajectory[idx + sample_every, 1] - trajectory[idx, 1]
        ax.arrow(trajectory[idx, 0], trajectory[idx, 1], dx * 3, dz * 3,
                head_width=2, head_length=3, fc='green', ec='green', alpha=0.5)
    
    # Labels and styling
    ax.set_xlabel('X (m) - Right', fontsize=12)
    ax.set_ylabel('Z (m) - Forward', fontsize=12)
    ax.set_title(f'Sequence {seq}: Full Vehicle Trajectory', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.axis('equal')
    
    # Add text box with stats
    stats_text = f"""Frames: {len(trajectory)}
Distance: {np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)):.0f} m
Start: ({trajectory[0, 0]:.1f}, {trajectory[0, 1]:.1f})
End: ({trajectory[-1, 0]:.1f}, {trajectory[-1, 1]:.1f})"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_file = f'seq_{seq}_full_trajectory.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")
    
    return fig, ax


def plot_trajectory_with_history_snippets(pose_file: str, seq: str = "00", num_snippets: int = 4):
    """Plot full trajectory with zoomed history snippets at different points."""
    
    print(f"Loading poses from: {pose_file}")
    poses = load_poses(pose_file)
    trajectory = extract_trajectory(poses)
    
    # Create figure with subplots: 2 rows, 3 cols
    # Top: full trajectory (spans all 3 cols)
    # Bottom: 4 snippets
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Main plot - full trajectory (spanning top row)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1.5, alpha=0.6)
    ax_main.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax_main.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')
    
    # Mark snippet locations
    snippet_indices = np.linspace(100, len(trajectory) - 100, num_snippets, dtype=int)
    colors = ['red', 'orange', 'green', 'purple']
    
    for i, idx in enumerate(snippet_indices):
        ax_main.plot(trajectory[idx, 0], trajectory[idx, 1], 'o', 
                    color=colors[i], markersize=10)
    
    ax_main.set_xlabel('X (m) - Right')
    ax_main.set_ylabel('Z (m) - Forward')
    ax_main.set_title(f'Sequence {seq}: Full Trajectory with History Snippet Locations')
    ax_main.legend(loc='best', fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.axis('equal')
    
    # Snippet plots (bottom row: 4 subplots)
    past_frames = 5
    spacing_meters = 2.0
    
    for i, idx in enumerate(snippet_indices):
        ax = fig.add_subplot(gs[1, i])
        
        # Extract history (similar to verify_history_implementation.py)
        start_idx = max(0, idx - past_frames * 10)
        past_trajectory = trajectory[start_idx:idx+1]
        
        # Sample at spacing
        sampled = [past_trajectory[-1]]  # Current position
        dist_accum = 0
        for j in range(len(past_trajectory) - 2, -1, -1):
            dist = np.linalg.norm(past_trajectory[j] - past_trajectory[j + 1])
            dist_accum += dist
            if dist_accum >= spacing_meters and len(sampled) < past_frames:
                sampled.insert(0, past_trajectory[j])
                dist_accum = 0
        
        sampled = np.array(sampled)
        
        # Transform to ego frame
        current_pos = past_trajectory[-1]
        ego_trajectory = sampled - current_pos
        
        # Plot
        ax.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], 
                'r-o', linewidth=2, markersize=8)
        ax.plot(0, 0, 'g*', markersize=20, label='Ego')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_title(f'Snippet {i+1} (Frame {idx})', color=colors[i], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(fontsize=8)
    
    plt.suptitle(f'Sequence {seq}: Full Trajectory + History Snippets (5 keyframes @ 2m)', 
                 fontsize=14, fontweight='bold')
    
    output_file = f'seq_{seq}_trajectory_with_snippets.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_file}")
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize full KITTI sequence trajectory")
    parser.add_argument("--pose_file", type=str, default="data/kitti/poses/00.txt",
                        help="Path to pose file")
    parser.add_argument("--seq", type=str, default="00",
                        help="Sequence number (for title)")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Sample every N frames for points")
    parser.add_argument("--with_snippets", action="store_true",
                        help="Also plot history snippets at different locations")
    
    args = parser.parse_args()
    
    if not Path(args.pose_file).exists():
        print(f"Error: Pose file not found: {args.pose_file}")
        exit(1)
    
    # Main plot
    plot_full_trajectory(args.pose_file, args.seq, args.sample_every)
    
    # Optional: With snippets
    if args.with_snippets:
        plt.close('all')
        plot_trajectory_with_history_snippets(args.pose_file, args.seq)
    
    print("\n✅ Done!")
