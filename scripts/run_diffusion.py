#!/usr/bin/env python3
"""
Unified Diffusion Policy Training & Evaluation Script

Modes:
  --mode train    : Train diffusion policy with frozen encoder
  --mode eval     : Evaluate trained model on test set
  --mode infer    : Single-frame inference with visualization

Usage:
  # Training
  python run_diffusion.py --mode train --encoder_ckpt checkpoints/encoder_best.pth

  # Evaluation
  python run_diffusion.py --mode eval --encoder_ckpt checkpoints/encoder_best.pth \
                          --diffusion_ckpt checkpoints/diffusion_best.pth

  # Single inference
  python run_diffusion.py --mode infer --encoder_ckpt checkpoints/encoder_best.pth \
                          --diffusion_ckpt checkpoints/diffusion_best.pth \
                          --sequence 00 --frame 1000
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import sys
import time
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from models.bev_rasterization import BEVRasterizer, load_kitti_lidar
from models.encoder import build_encoder
from models.diffusion import TrajectoryDiffusionModel
from models.denoising_network import build_denoising_network
from models.metrics import compute_trajectory_metrics, MetricsLogger, aggregate_metrics


# =============================================================================
# Dataset
# =============================================================================

class KITTIDiffusionDataset(Dataset):
    """KITTI Dataset for Diffusion Training/Evaluation."""
    
    def __init__(self, sequences=['00'], split='train', data_root='/media/skr/storage/self_driving/TopoDiffuser/data/kitti', 
                 num_future=8, waypoint_spacing=2.0):
        self.data_root = data_root
        self.rasterizer = BEVRasterizer()
        self.num_future = num_future
        self.waypoint_spacing = waypoint_spacing
        self.split = split
        
        # Load all sequences
        self.samples = []
        for seq in sequences:
            self._load_sequence(seq)
        
        # Split: 80/20 if single sequence, else use sequences as split
        if len(sequences) == 1 and split != 'all':
            n = len(self.samples)
            if split == 'train':
                self.samples = self.samples[:int(n * 0.8)]
            else:
                self.samples = self.samples[int(n * 0.8):]
        
        print(f"[{split}] Loaded {len(self.samples)} samples from sequences {sequences}")
    
    def _load_sequence(self, sequence):
        """Load samples from a sequence."""
        lidar_dir = os.path.join(self.data_root, 'sequences', sequence, 'velodyne')
        pose_file = os.path.join(self.data_root, 'poses', f'{sequence}.txt')
        
        if not os.path.exists(pose_file):
            print(f"Warning: Pose file not found: {pose_file}")
            return
        
        # Load poses
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                pose = np.array(values).reshape(3, 4)
                poses.append(pose)
        
        # Create samples
        for frame_idx in range(len(poses) - self.num_future - 1):
            lidar_path = os.path.join(lidar_dir, f'{frame_idx:06d}.bin')
            if os.path.exists(lidar_path):
                self.samples.append({
                    'sequence': sequence,
                    'frame_idx': frame_idx,
                    'lidar_path': lidar_path,
                    'poses': poses
                })
    
    def _get_trajectory(self, poses, frame_idx):
        """Get future waypoints at 2m intervals."""
        current_pose = poses[frame_idx]
        current_x, current_y = current_pose[0, 3], current_pose[1, 3]
        
        trajectory = []
        for i in range(1, len(poses) - frame_idx):
            pose = poses[frame_idx + i]
            x, y = pose[0, 3], pose[1, 3]
            dist = np.sqrt((x - current_x)**2 + (y - current_y)**2)
            
            if dist >= self.waypoint_spacing * (len(trajectory) + 1):
                trajectory.append([x - current_x, y - current_y])
                if len(trajectory) >= self.num_future:
                    break
        
        while len(trajectory) < self.num_future:
            trajectory.append(trajectory[-1] if trajectory else [0.0, 0.0])
        
        return np.array(trajectory[:self.num_future], dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load LiDAR
        lidar_points = load_kitti_lidar(sample['lidar_path'])
        bev = self.rasterizer.rasterize_lidar(lidar_points)
        
        # Get trajectory
        trajectory = self._get_trajectory(sample['poses'], sample['frame_idx'])
        
        return torch.from_numpy(bev), torch.from_numpy(trajectory)


# =============================================================================
# Model Loading
# =============================================================================

def load_models(encoder_ckpt, diffusion_ckpt=None, denoiser_arch='mlp', device='cuda'):
    """Load encoder and optionally diffusion model."""
    # Encoder
    encoder = build_encoder(input_channels=3, conditioning_dim=512).to(device)
    enc_checkpoint = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in enc_checkpoint:
        encoder.load_state_dict(enc_checkpoint['model_state_dict'])
    else:
        encoder.load_state_dict(enc_checkpoint)
    
    diffusion_model = None
    if diffusion_ckpt and os.path.exists(diffusion_ckpt):
        denoising_net = build_denoising_network(
            denoiser_arch, num_waypoints=8, coord_dim=2,
            conditioning_dim=512, timestep_dim=256
        ).to(device)
        
        diff_checkpoint = torch.load(diffusion_ckpt, map_location=device, weights_only=False)
        denoising_net.load_state_dict(diff_checkpoint['denoiser_state_dict'])
        
        diffusion_model = TrajectoryDiffusionModel(denoising_net, num_timesteps=10, device=device)
    
    return encoder, diffusion_model


# =============================================================================
# Training
# =============================================================================

def train_epoch(encoder, diffusion_model, dataloader, optimizer, scaler, device, epoch):
    """Train one epoch."""
    diffusion_model.train()
    encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (bev, trajectory) in enumerate(dataloader):
        bev = bev.to(device, non_blocking=True)
        trajectory = trajectory.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            with torch.no_grad():
                conditioning, _ = encoder(bev)
            loss, _, _ = diffusion_model.training_step(trajectory, conditioning)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(encoder, diffusion_model, dataloader, device, num_samples=5):
    """Evaluate with comprehensive metrics."""
    encoder.eval()
    diffusion_model.eval()
    
    total_loss = 0.0
    metrics_logger = MetricsLogger()
    num_batches = 0
    
    for bev, trajectory in dataloader:
        bev = bev.to(device, non_blocking=True)
        trajectory = trajectory.to(device, non_blocking=True)
        
        conditioning, _ = encoder(bev)
        
        # Loss
        loss, _, _ = diffusion_model.training_step(trajectory, conditioning)
        total_loss += loss.item()
        
        # Sample and compute metrics
        pred_trajectories = diffusion_model.sample(conditioning, num_samples=num_samples)
        metrics = compute_trajectory_metrics(pred_trajectories, trajectory, threshold=2.0)
        metrics_logger.update(metrics, count=trajectory.shape[0])
        
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_metrics = metrics_logger.get_averages()
    
    return avg_loss, avg_metrics


def train(args, device):
    """Training mode."""
    print("=" * 70)
    print("DIFFUSION POLICY TRAINING")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    
    # Load encoder (frozen)
    print("\nLoading encoder...")
    encoder, _ = load_models(args.encoder_ckpt, None, args.denoiser_arch, device)
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params (frozen)")
    
    # Build diffusion model
    print("\nBuilding diffusion model...")
    denoising_net = build_denoising_network(
        args.denoiser_arch, num_waypoints=8, coord_dim=2,
        conditioning_dim=512, timestep_dim=256
    ).to(device)
    diffusion_model = TrajectoryDiffusionModel(denoising_net, num_timesteps=10, device=device)
    print(f"  Denoiser: {sum(p.numel() for p in denoising_net.parameters()):,} params")
    
    # Optimizer
    optimizer = optim.AdamW(denoising_net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Data
    print("\nLoading datasets...")
    train_dataset = KITTIDiffusionDataset(sequences=args.train_sequences, split='train')
    val_dataset = KITTIDiffusionDataset(sequences=args.train_sequences, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': [], 'lr': []}
    best_minADE = float('inf')
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_epoch(encoder, diffusion_model, train_loader, optimizer, scaler, device, epoch)
        val_loss, val_metrics = evaluate(encoder, diffusion_model, val_loader, device, num_samples=5)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch [{epoch}/{args.epochs}] {epoch_time:.1f}s:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  minADE: {val_metrics['minADE']:.3f}m | minFDE: {val_metrics['minFDE']:.3f}m | HitRate: {val_metrics['hit_rate']:.3f}")
        
        # Save best
        if val_metrics['minADE'] < best_minADE:
            best_minADE = val_metrics['minADE']
            torch.save({
                'epoch': epoch,
                'denoiser_state_dict': denoising_net.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'history': history
            }, os.path.join(args.save_dir, 'diffusion_best.pth'))
            print(f"  ✓ Best saved (minADE: {best_minADE:.3f}m)")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'denoiser_state_dict': denoising_net.state_dict(),
            'history': history
        }, os.path.join(args.save_dir, 'diffusion_latest.pth'))
        
        with open(os.path.join(args.save_dir, 'diffusion_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"COMPLETE | Best minADE: {best_minADE:.3f}m")
    print("=" * 70)


# =============================================================================
# Evaluation
# =============================================================================

def eval_mode(args, device):
    """Evaluation mode."""
    print("=" * 70)
    print("DIFFUSION POLICY EVALUATION")
    print("=" * 70)
    
    if not os.path.exists(args.diffusion_ckpt):
        print(f"✗ Diffusion checkpoint not found: {args.diffusion_ckpt}")
        return
    
    # Load models
    print("\nLoading models...")
    encoder, diffusion_model = load_models(args.encoder_ckpt, args.diffusion_ckpt, args.denoiser_arch, device)
    print("✓ Models loaded")
    
    # Load test data
    print(f"\nLoading test data: {args.test_sequences}...")
    test_dataset = KITTIDiffusionDataset(sequences=args.test_sequences, split='all')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    
    # Evaluate
    print("\nEvaluating...")
    test_loss, test_metrics = evaluate(encoder, diffusion_model, test_loader, device, num_samples=5)
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  Loss: {test_loss:.4f}")
    print(f"  minADE: {test_metrics['minADE']:.3f}m ± {test_metrics['minADE_std']:.3f}m")
    print(f"  minFDE: {test_metrics['minFDE']:.3f}m ± {test_metrics['minFDE_std']:.3f}m")
    print(f"  maxADE: {test_metrics['maxADE']:.3f}m ± {test_metrics['maxADE_std']:.3f}m")
    print(f"  HitRate@2m: {test_metrics['hit_rate']:.3f} ± {test_metrics['hit_rate_std']:.3f}")
    print(f"  Hausdorff: {test_metrics['hausdorff']:.3f}m ± {test_metrics['hausdorff_std']:.3f}m")
    print("=" * 70)


# =============================================================================
# Inference
# =============================================================================

def infer_mode(args, device):
    """Single-frame inference with visualization."""
    print("=" * 70)
    print("SINGLE-FRAME INFERENCE")
    print("=" * 70)
    print(f"Frame: {args.sequence}/{args.frame:06d}")
    
    if not os.path.exists(args.diffusion_ckpt):
        print(f"✗ Diffusion checkpoint not found: {args.diffusion_ckpt}")
        return
    
    # Load models
    print("\nLoading models...")
    encoder, diffusion_model = load_models(args.encoder_ckpt, args.diffusion_ckpt, args.denoiser_arch, device)
    print("✓ Models loaded")
    
    # Load single frame
    print("\nLoading frame...")
    dataset = KITTIDiffusionDataset(sequences=[args.sequence], split='all')
    
    # Find frame
    sample_idx = None
    for i, sample in enumerate(dataset.samples):
        if sample['frame_idx'] == args.frame:
            sample_idx = i
            break
    
    if sample_idx is None:
        print(f"✗ Frame {args.frame} not found in sequence {args.sequence}")
        return
    
    bev, gt_trajectory = dataset[sample_idx]
    
    # Inference
    print("\nRunning inference...")
    bev_tensor = bev.unsqueeze(0).to(device)
    
    with torch.no_grad():
        conditioning, _ = encoder(bev_tensor)
        pred_trajectories = diffusion_model.sample(conditioning, num_samples=args.num_samples)
    
    pred_trajectories = pred_trajectories[0].cpu().numpy()  # [K, 8, 2]
    gt = gt_trajectory.numpy()  # [8, 2]
    
    # Metrics
    pred_tensor = torch.from_numpy(pred_trajectories).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).to(device)
    metrics = compute_trajectory_metrics(pred_tensor, gt_tensor, threshold=2.0)
    
    print(f"\nResults:")
    print(f"  minADE: {metrics['minADE'].item():.3f}m")
    print(f"  minFDE: {metrics['minFDE'].item():.3f}m")
    print(f"  maxADE: {metrics['maxADE'].item():.3f}m")
    print(f"  HitRate@2m: {metrics['hit_rate'].item():.3f}")
    print(f"  Hausdorff: {metrics['hausdorff'].mean():.3f}m")
    
    # Visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # BEV
    ax = axes[0]
    bev_vis = bev.numpy().transpose(1, 2, 0)
    bev_vis = (bev_vis - bev_vis.min()) / (bev_vis.max() - bev_vis.min() + 1e-8)
    ax.imshow(bev_vis)
    ax.set_title('LiDAR BEV Input')
    ax.axis('off')
    
    # Trajectories
    ax = axes[1]
    gt_x, gt_y = gt[:, 0], gt[:, 1]
    ax.plot(gt_x, gt_y, 'g-o', linewidth=3, markersize=8, label='Ground Truth', zorder=10)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(pred_trajectories)))
    for i, (traj, color) in enumerate(zip(pred_trajectories, colors)):
        ax.plot(traj[:, 0], traj[:, 1], '--', color=color, alpha=0.7, linewidth=2)
        ax.scatter(traj[:, 0], traj[:, 1], color=color, s=20, alpha=0.5)
    
    ax.scatter([0], [0], c='black', s=200, marker='*', zorder=20, label='Ego')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Predictions (K={len(pred_trajectories)}) | minADE: {metrics["minADE"].item():.2f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-5, 20)
    ax.set_ylim(-10, 10)
    
    plt.tight_layout()
    
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.output}")
    else:
        plt.savefig(f'inference_{args.sequence}_{args.frame:06d}.png', dpi=150, bbox_inches='tight')
        print(f"Saved to inference_{args.sequence}_{args.frame:06d}.png")
    
    plt.close()
    print("\n✓ Complete!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Diffusion Policy Training & Evaluation')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'infer'],
                        help='Mode: train, eval, or infer')
    
    # Model paths
    parser.add_argument('--encoder_ckpt', type=str, default='checkpoints/encoder_best.pth')
    parser.add_argument('--diffusion_ckpt', type=str, default='checkpoints/diffusion_best.pth')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    
    # Data
    parser.add_argument('--train_sequences', type=str, nargs='+', default=['00'])
    parser.add_argument('--test_sequences', type=str, nargs='+', default=['08', '09', '10'])
    parser.add_argument('--sequence', type=str, default='00', help='For infer mode')
    parser.add_argument('--frame', type=int, default=1000, help='For infer mode')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--denoiser_arch', type=str, default='mlp', choices=['mlp', 'cnn1d'])
    
    # Inference
    parser.add_argument('--num_samples', type=int, default=5, help='K trajectory samples')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    if args.mode == 'train':
        train(args, device)
    elif args.mode == 'eval':
        eval_mode(args, device)
    elif args.mode == 'infer':
        infer_mode(args, device)


if __name__ == "__main__":
    main()
