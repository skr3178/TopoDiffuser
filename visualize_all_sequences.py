#!/usr/bin/env python3
"""
Visualize full trajectories for all KITTI sequences.

Creates a grid plot showing all sequence paths.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


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
    trajectory = []
    for pose in poses:
        x = pose[0, 3]  # Translation in x (right)
        z = pose[2, 3]  # Translation in z (forward)
        trajectory.append([x, z])
    return np.array(trajectory)


def plot_all_sequences(data_root: str = "data/kitti", sequences=None):
    """Plot all KITTI sequences in a grid."""
    
    if sequences is None:
        sequences = [f"{i:02d}" for i in range(11)]  # 00-10
    
    # Calculate grid size
    n_seqs = len(sequences)
    n_cols = 3
    n_rows = (n_seqs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_seqs > 1 else [axes]
    
    stats = []
    
    for idx, seq in enumerate(sequences):
        ax = axes[idx]
        pose_file = Path(data_root) / "poses" / f"{seq}.txt"
        
        if not pose_file.exists():
            ax.text(0.5, 0.5, f"Sequence {seq}\nNot Found", 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            continue
        
        # Load and extract trajectory
        poses = load_poses(pose_file)
        trajectory = extract_trajectory(poses)
        
        # Calculate stats
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        duration = len(poses) / 10.0  # Assuming 10 Hz
        
        stats.append({
            'seq': seq,
            'frames': len(poses),
            'distance': total_distance,
            'duration': duration
        })
        
        # Plot trajectory with color gradient
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], 
                           c=np.arange(len(trajectory)), cmap='viridis',
                           s=5, alpha=0.7)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
               'b-', linewidth=1, alpha=0.3)
        
        # Mark start and end
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=12, label='End')
        
        # Title and labels
        ax.set_title(f'Sequence {seq}: {len(poses)} frames, {total_distance:.0f}m', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m) - Right', fontsize=9)
        ax.set_ylabel('Z (m) - Forward', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add stats text
        stats_text = f"Dist: {total_distance:.0f}m\nDur: {duration:.0f}s"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide extra subplots
    for idx in range(n_seqs, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('KITTI Dataset: All Sequence Trajectories', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_file = 'all_kitti_sequences.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    
    # Print summary table
    print("\n" + "="*60)
    print("KITTI Sequence Summary")
    print("="*60)
    print(f"{'Seq':<6}{'Frames':<10}{'Distance (m)':<15}{'Duration (s)':<15}")
    print("-"*60)
    for s in stats:
        print(f"{s['seq']:<6}{s['frames']:<10}{s['distance']:<15.1f}{s['duration']:<15.1f}")
    
    total_dist = sum(s['distance'] for s in stats)
    total_frames = sum(s['frames'] for s in stats)
    print("-"*60)
    print(f"{'TOTAL':<6}{total_frames:<10}{total_dist:<15.1f}")
    print("="*60)
    
    return fig, stats


def plot_sequences_comparison(data_root: str = "data/kitti"):
    """Plot all sequences on a single plot for comparison."""
    
    sequences = [f"{i:02d}" for i in range(11)]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sequences)))
    
    for idx, seq in enumerate(sequences):
        pose_file = Path(data_root) / "poses" / f"{seq}.txt"
        
        if not pose_file.exists():
            continue
        
        poses = load_poses(pose_file)
        trajectory = extract_trajectory(poses)
        
        # Plot with offset for visibility
        offset_x = idx * 50  # Small offset to separate overlapping paths
        offset_y = 0
        
        ax.plot(trajectory[:, 0] + offset_x, trajectory[:, 1] + offset_y, 
               '-', linewidth=1.5, color=colors[idx], alpha=0.8, label=f'Seq {seq}')
        ax.plot(trajectory[0, 0] + offset_x, trajectory[0, 1] + offset_y, 
               'o', color=colors[idx], markersize=6)
    
    ax.set_xlabel('X (m) - Right', fontsize=12)
    ax.set_ylabel('Z (m) - Forward', fontsize=12)
    ax.set_title('All KITTI Sequences Trajectories (Overlaid)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    output_file = 'all_sequences_overlaid.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nOverlaid plot saved to: {output_file}")
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize all KITTI sequence trajectories")
    parser.add_argument("--data_root", type=str, default="data/kitti",
                       help="Path to KITTI dataset")
    parser.add_argument("--sequences", nargs="+", default=None,
                       help="Specific sequences to plot (default: 00-10)")
    parser.add_argument("--overlaid", action="store_true",
                       help="Also create overlaid comparison plot")
    
    args = parser.parse_args()
    
    # Main grid plot
    plot_all_sequences(args.data_root, args.sequences)
    
    # Optional overlaid plot
    if args.overlaid:
        plot_sequences_comparison(args.data_root)
    
    print("\n✅ Done!")
