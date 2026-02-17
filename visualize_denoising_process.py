#!/usr/bin/env python3
"""
Visualize Trajectory Refinement Through Denoising Steps

Generates figures showing how trajectories refine from noise to final prediction,
similar to paper appendix figures.

Usage:
    python visualize_denoising_process.py \
        --encoder_ckpt checkpoints/encoder_best.pth \
        --diffusion_ckpt checkpoints/diffusion_unet_best.pth \
        --num_samples 5 \
        --save_dir results/denoising_visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.insert(0, 'models')
from encoder import build_encoder
from diffusion import TrajectoryDiffusionModel, DiffusionScheduler
from denoising_network import build_denoising_network
from train_diffusion_only import KITTIDiffusionDataset


def load_models(encoder_ckpt, diffusion_ckpt, device):
    """Load trained encoder and diffusion models."""
    # Build encoder
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device)['model_state_dict'])
    encoder.eval()
    
    # Build diffusion model
    denoising_net = build_denoising_network().to(device)
    
    # Load denoiser weights
    ckpt = torch.load(diffusion_ckpt, map_location=device)
    denoising_net.load_state_dict(ckpt['denoiser_state_dict'])
    denoising_net.eval()
    
    # Build diffusion wrapper
    diffusion_model = TrajectoryDiffusionModel(
        denoising_network=denoising_net,
        num_timesteps=10,
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )
    
    return encoder, diffusion_model


@torch.no_grad()
def generate_trajectory_with_intermediates(encoder, diffusion_model, bev, device):
    """
    Generate trajectory and save intermediate denoising steps.
    
    Returns:
        intermediates: List of trajectories at different timesteps
        conditioning: Conditioning vector from encoder
    """
    bev = bev.unsqueeze(0).to(device)
    
    # Get conditioning from encoder
    conditioning, _ = encoder(bev)
    
    # Use the sample method with return_all_steps=True
    trajectories, all_steps = diffusion_model.sample(
        conditioning, 
        num_samples=1, 
        return_all_steps=True
    )
    
    # Extract intermediates at specific timesteps
    # all_steps is [num_timesteps, B, K, num_waypoints, 2]
    save_steps = [10, 8, 6, 4, 2, 1]  # From noise to clean (1-indexed in diffusion model)
    intermediates = {}
    
    for step in save_steps:
        if step <= diffusion_model.num_timesteps:
            # Convert from 1-indexed to 0-indexed for the list
            idx = step - 1
            if idx < len(all_steps):
                # all_steps[idx] is [B, K, num_waypoints, 2]
                traj = all_steps[idx][0, 0].cpu().numpy()  # First batch, first sample
                intermediates[step] = traj
    
    return intermediates, conditioning.squeeze(0).cpu().numpy()


def visualize_denoising_process(intermediates, gt_trajectory, bev, save_path, sample_idx):
    """Create visualization figure showing denoising refinement."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Trajectory Refinement Through Denoising (Sample {sample_idx})', fontsize=14)
    
    # BEV visualization (top-left)
    ax_bev = axes[0, 0]
    bev_vis = np.zeros((300, 400, 3))
    bev_vis[:, :, 0] = bev[0]  # Height -> Red
    bev_vis[:, :, 1] = bev[1]  # Intensity -> Green
    bev_vis[:, :, 2] = bev[2]  # Density -> Blue
    ax_bev.imshow(np.clip(bev_vis, 0, 1))
    ax_bev.set_title('BEV Input (LiDAR)')
    ax_bev.axis('off')
    
    # Denoising steps (remaining subplots)
    steps = [10, 8, 6, 4, 2, 1]
    titles = ['T=10 (Noise)', 'T=8', 'T=6', 'T=4', 'T=2', 'T=1 (Final)']
    
    for idx, (step, title) in enumerate(zip(steps, titles)):
        if idx == 0:
            ax = axes[0, 0]
        else:
            ax = axes[idx // 3, idx % 3]
        
        if step not in intermediates:
            continue
            
        traj = intermediates[step]
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=2, markersize=6, label='Predicted')
        
        # Plot GT if final step
        if step == 1:
            ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'g--s', 
                   linewidth=2, markersize=6, label='Ground Truth')
            ax.legend()
        
        # Mark start point
        ax.plot(0, 0, 'r*', markersize=15, label='Current Position')
        
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(-20, 20)
        ax.set_ylim(-10, 30)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add arrow for direction
        if len(traj) > 1:
            dx = traj[-1, 0] - traj[-2, 0]
            dy = traj[-1, 1] - traj[-2, 1]
            ax.arrow(traj[-2, 0], traj[-2, 1], dx*0.5, dy*0.5,
                    head_width=0.8, head_length=0.5, fc='blue', ec='blue', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def create_summary_figure(all_intermediates, all_gt_trajectories, save_path):
    """Create a summary figure showing multiple samples side by side."""
    num_samples = len(all_intermediates)
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3*num_samples))
    fig.suptitle('Trajectory Refinement: Multiple Samples', fontsize=16)
    
    steps = [10, 8, 6, 4, 2, 1]
    titles = ['T=10', 'T=8', 'T=6', 'T=4', 'T=2', 'T=1']
    
    for sample_idx, (intermediates, gt_traj) in enumerate(zip(all_intermediates, all_gt_trajectories)):
        for step_idx, (step, title) in enumerate(zip(steps, titles)):
            ax = axes[sample_idx, step_idx] if num_samples > 1 else axes[step_idx]
            
            if step not in intermediates:
                continue
                
            traj = intermediates[step]
            
            # Plot trajectory
            ax.plot(traj[:, 0], traj[:, 1], 'b-o', linewidth=1.5, markersize=4)
            
            # Plot GT on final step
            if step == 1:
                ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g--s', linewidth=1.5, markersize=4, alpha=0.7)
            
            # Mark start
            ax.plot(0, 0, 'r*', markersize=10)
            
            ax.set_xlim(-20, 20)
            ax.set_ylim(-10, 30)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            if sample_idx == 0:
                ax.set_title(title, fontsize=10)
            
            if step_idx == 0:
                ax.set_ylabel(f'Sample {sample_idx+1}', fontsize=10)
            
            ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Summary saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_ckpt', type=str, default='checkpoints/encoder_best.pth')
    parser.add_argument('--diffusion_ckpt', type=str, default='checkpoints/diffusion_unet_best.pth')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='results/denoising_visualization')
    parser.add_argument('--val_sequences', nargs='+', default=['08'])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models
    print(f"\nLoading models...")
    print(f"  Encoder: {args.encoder_ckpt}")
    print(f"  Diffusion: {args.diffusion_ckpt}")
    encoder, diffusion_model = load_models(args.encoder_ckpt, args.diffusion_ckpt, device)
    
    # Load dataset
    print(f"\nLoading dataset (sequence {args.val_sequences})...")
    dataset = KITTIDiffusionDataset(sequences=args.val_sequences, split='all')
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print(f"\nGenerating visualizations for {args.num_samples} samples...")
    all_intermediates = []
    all_gt_trajectories = []
    
    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}")
        
        # Get sample
        bev, gt_trajectory = dataset[i]
        
        # Generate with intermediates
        intermediates, _ = generate_trajectory_with_intermediates(
            encoder, diffusion_model, bev, device
        )
        
        # Save individual figure
        save_path = save_dir / f'denoising_sample_{i:03d}.png'
        visualize_denoising_process(
            intermediates, 
            gt_trajectory.numpy(), 
            bev.numpy(), 
            save_path,
            i
        )
        
        all_intermediates.append(intermediates)
        all_gt_trajectories.append(gt_trajectory.numpy())
    
    # Create summary figure
    print(f"\nCreating summary figure...")
    summary_path = save_dir / 'denoising_summary.png'
    create_summary_figure(all_intermediates, all_gt_trajectories, summary_path)
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
