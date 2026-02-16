#!/usr/bin/env python3
"""
Generate Trajectories from Trained Checkpoints

Generates new trajectory samples using the trained diffusion model.
Can generate:
1. Single-frame visualization
2. Batch generation for evaluation
3. Diverse samples from same input

Usage:
    # Generate from single frame
    python generate_trajectories.py \
        --encoder_ckpt checkpoints/encoder_best.pth \
        --diffusion_ckpt checkpoints/diffusion_best.pth \
        --sequence 00 --frame 1000
    
    # Batch generation for evaluation
    python generate_trajectories.py \
        --checkpoint checkpoints/joint_best.pth \
        --mode batch \
        --sequences 08 09 10 \
        --output results/generated_trajectories.npz
"""

import torch
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "models"))
from bev_rasterization import BEVRasterizer, load_kitti_lidar
from encoder import build_encoder
from diffusion import TrajectoryDiffusionModel
from denoising_network import build_denoising_network
from metrics import compute_trajectory_metrics


class TrajectoryGenerator:
    """
    Trajectory generator using trained TopoDiffuser model.
    """
    
    def __init__(self, encoder_ckpt, diffusion_ckpt=None, joint_ckpt=None,
                 config=None, device='cuda'):
        self.device = device
        self.config = config
        
        # Load model
        if joint_ckpt and os.path.exists(joint_ckpt):
            self.model = self._load_joint_checkpoint(joint_ckpt)
        else:
            self.model = self._load_separate_checkpoints(encoder_ckpt, diffusion_ckpt)
        
        self.model.eval()
        self.rasterizer = BEVRasterizer()
        
        print(f"✓ Generator ready on {device}")
    
    def _load_joint_checkpoint(self, checkpoint_path):
        """Load joint-trained model."""
        from pipeline import build_model
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Build model from config
        if 'config' in checkpoint:
            from config import ConfigDict
            config = ConfigDict.from_dict(checkpoint['config'])
        else:
            # Default config
            config = self._default_config()
        
        model = build_model(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Loaded joint checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        
        return model
    
    def _load_separate_checkpoints(self, encoder_ckpt, diffusion_ckpt):
        """Load separately trained encoder and diffusion."""
        from pipeline import TopoDiffuserModel
        
        # Build default model
        model = TopoDiffuserModel(self._default_config()).to(self.device)
        
        # Load encoder
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            enc_checkpoint = torch.load(encoder_ckpt, map_location=self.device, weights_only=False)
            if 'model_state_dict' in enc_checkpoint:
                model.encoder.load_state_dict(enc_checkpoint['model_state_dict'])
            elif 'encoder_state_dict' in enc_checkpoint:
                model.encoder.load_state_dict(enc_checkpoint['encoder_state_dict'])
            else:
                model.encoder.load_state_dict(enc_checkpoint)
            print(f"✓ Loaded encoder from {encoder_ckpt}")
        
        # Load diffusion
        if diffusion_ckpt and os.path.exists(diffusion_ckpt):
            diff_checkpoint = torch.load(diffusion_ckpt, map_location=self.device, weights_only=False)
            model.diffusion.denoising_network.load_state_dict(diff_checkpoint['denoiser_state_dict'])
            print(f"✓ Loaded diffusion from {diffusion_ckpt}")
        
        return model
    
    def _default_config(self):
        """Create default config."""
        from config import ConfigDict
        return ConfigDict.from_dict({
            'model': {
                'encoder': {
                    'input_channels': 3,
                    'conditioning_dim': 512
                },
                'diffusion': {
                    'num_timesteps': 10,
                    'denoising_network': {
                        'architecture': 'mlp',
                        'num_waypoints': 8,
                        'coord_dim': 2,
                        'conditioning_dim': 512,
                        'timestep_dim': 256,
                        'hidden_dim': 512,
                        'num_layers': 4
                    }
                }
            }
        })
    
    @torch.no_grad()
    def generate(self, bev, num_samples=5, return_all_steps=False):
        """
        Generate trajectory samples from BEV input.
        
        Args:
            bev: [3, 300, 400] or [B, 3, 300, 400] - BEV input
            num_samples: K - number of trajectory samples
            return_all_steps: If True, return all denoising steps
            
        Returns:
            trajectories: [K, 8, 2] or [B, K, 8, 2] - Generated trajectories
            (optional) all_steps: List of intermediate trajectories
        """
        # Handle single input
        single_input = False
        if bev.ndim == 3:
            bev = bev.unsqueeze(0)
            single_input = True
        
        bev = bev.to(self.device)
        
        # Generate
        if return_all_steps:
            trajectories, all_steps = self.model.sample(bev, num_samples, return_all_steps=True)
        else:
            trajectories = self.model.sample(bev, num_samples, return_all_steps=False)
            all_steps = None
        
        if single_input:
            trajectories = trajectories[0]  # [K, 8, 2]
            if all_steps:
                all_steps = [s[0] for s in all_steps]
        
        if return_all_steps:
            return trajectories, all_steps
        return trajectories
    
    def generate_from_file(self, lidar_path, num_samples=5):
        """Generate from LiDAR file."""
        lidar_points = load_kitti_lidar(lidar_path)
        bev = self.rasterizer.rasterize_lidar(lidar_points)
        bev_tensor = torch.from_numpy(bev)
        
        return self.generate(bev_tensor, num_samples)
    
    def generate_from_sequence(self, sequence, frame_indices, data_root, num_samples=5):
        """Generate for multiple frames in a sequence."""
        lidar_dir = Path(data_root) / 'sequences' / sequence / 'velodyne'
        
        results = []
        for frame_idx in tqdm(frame_indices, desc=f"Generating for seq {sequence}"):
            lidar_path = lidar_dir / f'{frame_idx:06d}.bin'
            if lidar_path.exists():
                trajectories = self.generate_from_file(str(lidar_path), num_samples)
                results.append({
                    'sequence': sequence,
                    'frame': frame_idx,
                    'trajectories': trajectories.cpu().numpy()
                })
        
        return results


def visualize_generation(bev, trajectories, gt_trajectory=None, save_path=None):
    """Visualize generated trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # BEV
    ax = axes[0]
    bev_vis = bev.transpose(1, 2, 0)
    bev_vis = (bev_vis - bev_vis.min()) / (bev_vis.max() - bev_vis.min() + 1e-8)
    ax.imshow(bev_vis)
    ax.set_title('LiDAR BEV Input')
    ax.axis('off')
    
    # Trajectories
    ax = axes[1]
    
    # GT if available
    if gt_trajectory is not None:
        gt_x, gt_y = gt_trajectory[:, 0], gt_trajectory[:, 1]
        ax.plot(gt_x, gt_y, 'g-o', linewidth=3, markersize=8, 
                label='Ground Truth', zorder=10)
    
    # Generated samples
    K = len(trajectories)
    colors = plt.cm.rainbow(np.linspace(0, 1, K))
    
    for i, (traj, color) in enumerate(zip(trajectories, colors)):
        x, y = traj[:, 0], traj[:, 1]
        ax.plot(x, y, '--', color=color, alpha=0.7, linewidth=2,
                label=f'Sample {i+1}' if i < 3 else None)
        ax.scatter(x, y, color=color, s=30, alpha=0.6, zorder=5)
    
    # Ego position
    ax.scatter([0], [0], c='black', s=200, marker='*', zorder=20, label='Ego')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Generated Trajectories (K={K})', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-5, 20)
    ax.set_ylim(-10, 10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    
    # Checkpoint options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Joint checkpoint (encoder+diffusion)')
    parser.add_argument('--encoder_ckpt', type=str, default='checkpoints/encoder_best.pth')
    parser.add_argument('--diffusion_ckpt', type=str, default='checkpoints/diffusion_best.pth')
    
    # Mode
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'])
    
    # Single frame
    parser.add_argument('--sequence', type=str, default='00')
    parser.add_argument('--frame', type=int, default=1000)
    
    # Batch generation
    parser.add_argument('--sequences', type=str, nargs='+', default=['08', '09', '10'])
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames per sequence (None=all)')
    
    # Generation params
    parser.add_argument('--num_samples', type=int, default=5,
                        help='K trajectory samples')
    parser.add_argument('--data_root', type=str,
                        default='/media/skr/storage/self_driving/TopoDiffuser/data/kitti')
    
    # Output
    parser.add_argument('--output', type=str, default='generated_trajectories.npz')
    parser.add_argument('--viz', action='store_true', help='Visualize single frame')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Create generator
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    generator = TrajectoryGenerator(
        encoder_ckpt=args.encoder_ckpt,
        diffusion_ckpt=args.diffusion_ckpt,
        joint_ckpt=args.checkpoint,
        device=device
    )
    
    if args.mode == 'single':
        # Generate for single frame
        print(f"\nGenerating trajectories for {args.sequence}/{args.frame:06d}...")
        
        lidar_path = Path(args.data_root) / 'sequences' / args.sequence / 'velodyne' / f'{args.frame:06d}.bin'
        
        if not lidar_path.exists():
            print(f"✗ Frame not found: {lidar_path}")
            return
        
        # Generate
        trajectories = generator.generate_from_file(str(lidar_path), args.num_samples)
        trajectories_np = trajectories.cpu().numpy()
        
        print(f"\nGenerated {args.num_samples} trajectories:")
        for i, traj in enumerate(trajectories_np):
            print(f"  Sample {i+1}: {traj[0]} → {traj[-1]} (length={len(traj)})")
        
        # Diversity metrics
        pairwise_dists = []
        for i in range(args.num_samples):
            for j in range(i+1, args.num_samples):
                dist = np.linalg.norm(trajectories_np[i] - trajectories_np[j], axis=1).mean()
                pairwise_dists.append(dist)
        
        print(f"\nDiversity:")
        print(f"  Mean pairwise distance: {np.mean(pairwise_dists):.3f}m")
        print(f"  Min: {np.min(pairwise_dists):.3f}m, Max: {np.max(pairwise_dists):.3f}m")
        
        # Visualize
        if args.viz:
            lidar_points = load_kitti_lidar(str(lidar_path))
            bev = generator.rasterizer.rasterize_lidar(lidar_points)
            
            visualize_generation(bev, trajectories_np, 
                                 save_path=f"generation_{args.sequence}_{args.frame:06d}.png")
        
        # Save
        output_path = args.output if args.output.endswith('.npz') else args.output + '.npz'
        np.savez(output_path,
                 trajectories=trajectories_np,
                 sequence=args.sequence,
                 frame=args.frame)
        print(f"\nSaved to {output_path}")
        
    elif args.mode == 'batch':
        # Batch generation
        print(f"\nBatch generation for sequences: {args.sequences}")
        
        all_results = []
        
        for seq in args.sequences:
            lidar_dir = Path(args.data_root) / 'sequences' / seq / 'velodyne'
            all_frames = sorted([int(f.stem) for f in lidar_dir.glob('*.bin')])
            
            if args.max_frames:
                all_frames = all_frames[:args.max_frames]
            
            print(f"\nSequence {seq}: {len(all_frames)} frames")
            
            results = generator.generate_from_sequence(
                seq, all_frames, args.data_root, args.num_samples
            )
            all_results.extend(results)
        
        # Save batch results
        output_path = args.output if args.output.endswith('.npz') else args.output + '.npz'
        
        # Flatten for saving
        sequences = [r['sequence'] for r in all_results]
        frames = [r['frame'] for r in all_results]
        trajectories = np.array([r['trajectories'] for r in all_results])  # [N, K, 8, 2]
        
        np.savez(output_path,
                 sequences=sequences,
                 frames=frames,
                 trajectories=trajectories,
                 num_samples=args.num_samples,
                 generated_at=datetime.now().isoformat())
        
        print(f"\n{'='*70}")
        print(f"Batch generation complete!")
        print(f"  Total frames: {len(all_results)}")
        print(f"  Samples per frame: {args.num_samples}")
        print(f"  Saved to: {output_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
