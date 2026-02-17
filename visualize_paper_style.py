#!/usr/bin/env python3
"""
Paper-style Trajectory Visualization

Generates figures matching the paper style (arXiv:2508.00303):
- Height-based LiDAR colormap on dark background
- Red dashed = ground truth, green = predictions
- Trajectories originate from ego vehicle center
- Forward direction is upward

Usage:
    python visualize_paper_style.py \
        --encoder_ckpt checkpoints/encoder_full_best.pth \
        --diffusion_ckpt checkpoints/diffusion_unet_best.pth \
        --num_scenes 5 \
        --num_samples_per_scene 5 \
        --save_dir results/paper_style_viz
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import argparse
import sys
import os

sys.path.insert(0, 'models')
from encoder import build_encoder
from diffusion import TrajectoryDiffusionModel
from denoising_network import build_denoising_network
from bev_rasterization import BEVRasterizer, load_kitti_lidar


# ---------------------------------------------------------------------------
# BEV coordinate constants (must match BEVRasterizer.DEFAULT_CONFIG)
# ---------------------------------------------------------------------------
X_RANGE = (-20, 20)
Y_RANGE = (-10, 30)
RESOLUTION = 0.1
GRID_H, GRID_W = 300, 400  # rows, cols


def load_models(encoder_ckpt, diffusion_ckpt, device):
    """Load trained encoder and diffusion models."""
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    encoder.load_state_dict(
        torch.load(encoder_ckpt, map_location=device)['model_state_dict']
    )
    encoder.eval()

    denoising_net = build_denoising_network().to(device)
    ckpt = torch.load(diffusion_ckpt, map_location=device)
    denoising_net.load_state_dict(ckpt['denoiser_state_dict'])
    denoising_net.eval()

    diffusion_model = TrajectoryDiffusionModel(
        denoising_network=denoising_net,
        num_timesteps=10,
        schedule='cosine',
        device=device,
    )

    return encoder, diffusion_model


# ---------------------------------------------------------------------------
# Dataset: load BEV + trajectory for visualization
# ---------------------------------------------------------------------------

def collect_viz_samples(sequences, data_root='data/kitti',
                        num_future=8, waypoint_spacing=2.0):
    """Return list of (lidar_path, trajectory[8,2]) dicts."""
    samples = []
    for seq in sequences:
        lidar_dir = Path(data_root) / 'sequences' / seq / 'velodyne'
        pose_file = Path(data_root) / 'poses' / f'{seq}.txt'
        if not pose_file.exists():
            print(f"  Warning: {pose_file} not found")
            continue

        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                poses.append(np.array(values).reshape(3, 4))

        for frame_idx in range(len(poses) - num_future - 1):
            lp = lidar_dir / f'{frame_idx:06d}.bin'
            if not lp.exists():
                continue
            traj = _compute_trajectory(poses, frame_idx, num_future, waypoint_spacing)
            samples.append({'lidar_path': str(lp), 'trajectory': traj})

    print(f"  {len(samples)} visualization samples from {sequences}")
    return samples


def _compute_trajectory(poses, frame_idx, num_future, waypoint_spacing):
    current_pose = poses[frame_idx]
    cx, cy = current_pose[0, 3], current_pose[1, 3]
    trajectory = []
    for i in range(1, len(poses) - frame_idx):
        pose = poses[frame_idx + i]
        x, y = pose[0, 3], pose[1, 3]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist >= waypoint_spacing * (len(trajectory) + 1):
            trajectory.append([x - cx, y - cy])
            if len(trajectory) >= num_future:
                break
    while len(trajectory) < num_future:
        trajectory.append(trajectory[-1] if trajectory else [0.0, 0.0])
    return np.array(trajectory[:num_future], dtype=np.float32)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def world_to_pixel(points):
    """
    Convert world coordinates to pixel coordinates for display.

    Uses the same mapping as BEVRasterizer (resolution-based), then flips Y
    so that the forward direction (positive Y) is at the TOP of the image.

    Args:
        points: [N, 2] array in ego frame (x=lateral, y=longitudinal)
    Returns:
        px, py: pixel coordinates (col, row) for matplotlib plotting
    """
    px = ((points[:, 0] - X_RANGE[0]) / RESOLUTION).astype(int)
    py = ((points[:, 1] - Y_RANGE[0]) / RESOLUTION).astype(int)
    px = np.clip(px, 0, GRID_W - 1)
    py = np.clip(py, 0, GRID_H - 1)
    # Flip Y so forward (high y) is at the top of the displayed image
    py = GRID_H - 1 - py
    return px, py


def prepend_origin(trajectory):
    """Prepend (0, 0) to trajectory so it starts from ego vehicle."""
    origin = np.array([[0.0, 0.0]])
    return np.vstack([origin, trajectory])


# ---------------------------------------------------------------------------
# BEV rendering
# ---------------------------------------------------------------------------

def create_bev_visualization(bev):
    """
    Create a paper-style BEV visualization using a height-based colormap.

    Maps LiDAR height to a perceptual colormap (turbo/inferno) on a dark
    background, modulated by density so empty cells stay black.
    """
    height = bev[0]      # [H, W] normalised to [0, 1]
    intensity = bev[1]   # [H, W]
    density = bev[2]     # [H, W]

    # Occupied mask — cells with any LiDAR points
    occupied = density > 0

    # Height-based colormap (turbo gives the vivid rainbow look from the paper)
    cmap = plt.cm.turbo
    height_colored = cmap(height)[:, :, :3]  # [H, W, 3] RGB in [0,1]

    # Blend with intensity for richer appearance
    blend = 0.7 * height_colored + 0.3 * intensity[:, :, None]
    blend = np.clip(blend, 0, 1)

    # Black background — only show occupied cells
    bev_rgb = np.zeros((GRID_H, GRID_W, 3), dtype=np.float32)
    bev_rgb[occupied] = blend[occupied]

    # Flip vertically so forward direction is UP
    bev_rgb = bev_rgb[::-1].copy()

    return bev_rgb


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_multiple_trajectories(encoder, diffusion_model, bev_tensor,
                                   device, num_samples=5):
    """Generate K trajectory samples from a single BEV."""
    bev_tensor = bev_tensor.unsqueeze(0).to(device)
    conditioning, _ = encoder(bev_tensor)
    trajectories = diffusion_model.sample(
        conditioning, num_samples=num_samples, return_all_steps=False
    )
    return trajectories[0].cpu().numpy()  # [K, 8, 2]


@torch.no_grad()
def generate_trajectory_with_steps(encoder, diffusion_model, bev_tensor,
                                   device, num_samples=5):
    """Generate trajectories and return intermediate denoising steps."""
    bev_tensor = bev_tensor.unsqueeze(0).to(device)
    conditioning, _ = encoder(bev_tensor)
    trajectories, all_steps = diffusion_model.sample(
        conditioning, num_samples=num_samples, return_all_steps=True
    )
    # all_steps[i]: [B, K, 8, 2] — index 0 is pure noise (t=T), last is final
    step_indices = [0, 2, 4, 6, 8, 10]  # T=10,8,6,4,2,0(final)
    selected = []
    for idx in step_indices:
        if idx < len(all_steps):
            selected.append(all_steps[idx][0].cpu().numpy())  # [K, 8, 2]
    return selected, trajectories[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_trajectories_on_ax(ax, bev_rgb, trajectories, gt_trajectory,
                             show_gt=True, pred_alpha=0.7):
    """Core plotting: BEV background + predicted + GT trajectories."""
    ax.imshow(bev_rgb, aspect='equal')

    # Predicted trajectories (green)
    K = trajectories.shape[0]
    for k in range(K):
        traj = prepend_origin(trajectories[k])
        px, py = world_to_pixel(traj)
        ax.plot(px, py, color='lime', linewidth=2.0, alpha=pred_alpha,
                solid_capstyle='round')
        ax.scatter(px[1:], py[1:], c='lime', s=20, alpha=pred_alpha * 0.8,
                   edgecolors='darkgreen', linewidths=0.5, zorder=5)

    # Ground truth (red dashed)
    if show_gt:
        gt = prepend_origin(gt_trajectory)
        px_gt, py_gt = world_to_pixel(gt)
        ax.plot(px_gt, py_gt, color='red', linewidth=2.5, linestyle='--',
                solid_capstyle='round', zorder=6)
        ax.scatter(px_gt[1:], py_gt[1:], c='red', s=30, marker='s',
                   edgecolors='darkred', linewidths=0.8, zorder=7)

    # Ego vehicle marker (white star at origin)
    ego_px, ego_py = world_to_pixel(np.array([[0.0, 0.0]]))
    ax.plot(ego_px[0], ego_py[0], 'w*', markersize=18,
            markeredgecolor='black', markeredgewidth=0.8, zorder=10)

    ax.axis('off')


# ---------------------------------------------------------------------------
# Visualisation functions
# ---------------------------------------------------------------------------

def visualize_multi_trajectory_scene(bev_np, trajectories, gt_trajectory,
                                     save_path, scene_idx):
    """Multi-trajectory overlay (paper Fig 1 style)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='black')
    bev_rgb = create_bev_visualization(bev_np)
    _plot_trajectories_on_ax(ax, bev_rgb, trajectories, gt_trajectory)
    ax.set_title(f'Scene {scene_idx}: Multi-modal Predictions',
                 fontsize=12, color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_denoising_refinement(bev_np, step_trajectories, gt_trajectory,
                                   save_path, scene_idx):
    """Denoising refinement grid (paper Appendix Fig 2 style)."""
    num_steps = len(step_trajectories)
    cols = min(num_steps, 6)
    rows = (num_steps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                             facecolor='black')
    if num_steps == 1:
        axes = [axes]
    else:
        axes = list(axes.flat) if rows > 1 else list(axes)

    step_labels = ['T=10 (Noise)', 'T=8', 'T=6', 'T=4', 'T=2', 'T=0 (Final)']
    bev_rgb = create_bev_visualization(bev_np)

    # Colour progression: yellow (noisy) → cyan → lime (clean)
    step_colors = ['#ffff00', '#ffcc00', '#00dddd', '#00cccc', '#66ff66', '#00ff00']

    for idx in range(len(axes)):
        ax = axes[idx]
        if idx >= num_steps:
            ax.axis('off')
            continue

        ax.imshow(bev_rgb, aspect='equal')
        step_trajs = step_trajectories[idx]
        color = step_colors[min(idx, len(step_colors) - 1)]
        alpha = 0.5 + 0.1 * idx  # increasing opacity

        K = step_trajs.shape[0]
        for k in range(K):
            traj = prepend_origin(step_trajs[k])
            px, py = world_to_pixel(traj)
            ax.plot(px, py, color=color, linewidth=2, alpha=min(alpha, 0.9))

        # Show GT only on final panel
        is_final = (idx == num_steps - 1)
        if is_final:
            gt = prepend_origin(gt_trajectory)
            px_gt, py_gt = world_to_pixel(gt)
            ax.plot(px_gt, py_gt, 'red', linewidth=2.5, linestyle='--', zorder=6)

        # Ego marker
        ego_px, ego_py = world_to_pixel(np.array([[0.0, 0.0]]))
        ax.plot(ego_px[0], ego_py[0], 'w*', markersize=12,
                markeredgecolor='black', markeredgewidth=0.5, zorder=10)

        label = step_labels[idx] if idx < len(step_labels) else f'Step {idx}'
        ax.set_title(label, fontsize=11, color='white')
        ax.axis('off')

    fig.suptitle(f'Scene {scene_idx}: Denoising Refinement',
                 fontsize=14, color='white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved: {save_path}")


def create_summary_figure(all_bevs, all_trajectories, all_gt, save_path):
    """Summary grid with multiple scenes."""
    num_scenes = len(all_bevs)
    cols = min(num_scenes, 5)
    rows = (num_scenes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             facecolor='black')
    if num_scenes == 1:
        axes = np.array([axes])
    axes = axes.flat if hasattr(axes, 'flat') else [axes]

    fig.suptitle('Predicted Trajectories on Representative Scenes',
                 fontsize=14, color='white')

    for idx, ax in enumerate(axes):
        if idx >= num_scenes:
            ax.axis('off')
            continue
        bev_rgb = create_bev_visualization(all_bevs[idx])
        _plot_trajectories_on_ax(ax, bev_rgb, all_trajectories[idx],
                                 all_gt[idx])
        ax.set_title(f'Scene {idx + 1}', fontsize=10, color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Summary saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_ckpt', type=str,
                        default='checkpoints/encoder_full_best.pth')
    parser.add_argument('--diffusion_ckpt', type=str,
                        default='checkpoints/diffusion_unet_best.pth')
    parser.add_argument('--num_scenes', type=int, default=5)
    parser.add_argument('--num_samples_per_scene', type=int, default=5)
    parser.add_argument('--save_dir', type=str,
                        default='results/paper_style_viz')
    parser.add_argument('--val_sequences', nargs='+', default=['08'])
    parser.add_argument('--scene_stride', type=int, default=200,
                        help='Stride between selected scenes for diversity')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    print(f"\nLoading models...")
    print(f"  Encoder:   {args.encoder_ckpt}")
    print(f"  Diffusion: {args.diffusion_ckpt}")
    encoder, diffusion_model = load_models(
        args.encoder_ckpt, args.diffusion_ckpt, device
    )

    # Collect samples (lightweight — just paths + trajectories)
    print(f"\nCollecting samples...")
    samples = collect_viz_samples(args.val_sequences)
    rasterizer = BEVRasterizer()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating paper-style visualizations...")
    print(f"  Scenes: {args.num_scenes}, Samples/scene: {args.num_samples_per_scene}")
    print(f"  Scene stride: {args.scene_stride}")

    all_bevs = []
    all_trajectories = []
    all_gt = []

    for i in range(args.num_scenes):
        sample_idx = i * args.scene_stride
        if sample_idx >= len(samples):
            sample_idx = i % len(samples)

        print(f"\nScene {i + 1}/{args.num_scenes} (sample {sample_idx})")

        sample = samples[sample_idx]
        gt_trajectory = sample['trajectory']

        # Rasterize BEV from LiDAR
        points = load_kitti_lidar(sample['lidar_path'])
        bev_np = rasterizer.rasterize_lidar(points)         # [3, H, W]
        bev_tensor = torch.from_numpy(bev_np)                # for encoder

        # Generate predictions
        trajectories = generate_multiple_trajectories(
            encoder, diffusion_model, bev_tensor, device,
            num_samples=args.num_samples_per_scene,
        )

        # Multi-trajectory figure
        save_path = save_dir / f'scene_{i:03d}_multi.png'
        visualize_multi_trajectory_scene(
            bev_np, trajectories, gt_trajectory, save_path, i
        )

        # Denoising refinement figure
        step_trajectories, final_trajectories = generate_trajectory_with_steps(
            encoder, diffusion_model, bev_tensor, device,
            num_samples=args.num_samples_per_scene,
        )
        save_path = save_dir / f'scene_{i:03d}_refinement.png'
        visualize_denoising_refinement(
            bev_np, step_trajectories, gt_trajectory, save_path, i
        )

        all_bevs.append(bev_np)
        all_trajectories.append(final_trajectories)
        all_gt.append(gt_trajectory)

    # Summary grid
    print(f"\nCreating summary figure...")
    summary_path = save_dir / 'summary_multi_trajectory.png'
    create_summary_figure(all_bevs, all_trajectories, all_gt, summary_path)

    print(f"\n{'=' * 60}")
    print(f"Visualization complete!")
    print(f"Results: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
