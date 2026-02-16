#!/usr/bin/env python3
"""
Conditional Diffusion Policy for Trajectory Prediction

Implements DDPM-style diffusion with:
- Forward process: Add noise to GT trajectory
- Reverse process: Denoising network g_φ predicts noise
- Conditioning: Encoder output c [B, 512] injected at each step

Paper: TopoDiffuser (arXiv:2508.00303)
Section III-D, III-E
"""

import torch
import torch.nn as nn
import numpy as np


class DiffusionScheduler:
    """
    Noise schedule for diffusion process.
    
    Uses linear beta schedule (standard DDPM).
    β_t increases linearly from beta_start to beta_end.
    """
    
    def __init__(self, num_timesteps=10, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Linear schedule: beta_t increases linearly
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        
        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # ᾱ_t = cumulative product of α_i from i=1 to t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alphas_cumprod[:-1]
        ])
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_0: [B, num_waypoints, 2] - Ground truth trajectory
            t: [B] - Timestep indices (1-indexed: 1 to T)
            noise: [B, num_waypoints, 2] - Optional pre-generated noise
            
        Returns:
            x_t: [B, num_waypoints, 2] - Noised trajectory
            noise: [B, num_waypoints, 2] - The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Convert t to 0-indexed for indexing
        t_idx = t - 1  # [B]
        
        # Get sqrt(γ_t) and sqrt(1-γ_t) for each sample in batch
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t_idx].view(-1, 1, 1)  # [B, 1, 1]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_idx].view(-1, 1, 1)  # [B, 1, 1]
        
        # τ_t = √γ_t · τ_0 + √(1-γ_t) · ε
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        
        return x_t, noise
    
    def sample_timesteps(self, batch_size):
        """Sample random timesteps uniformly from [1, T]."""
        return torch.randint(1, self.num_timesteps + 1, (batch_size,), device=self.device)
    
    def denoise_step(self, x_t, predicted_noise, t):
        """
        Single denoising step: compute x_{t-1} from x_t
        
        Args:
            x_t: [B, num_waypoints, 2] - Current noised trajectory
            predicted_noise: [B, num_waypoints, 2] - Predicted noise ε̂_t
            t: int - Current timestep (1-indexed)
            
        Returns:
            x_{t-1}: [B, num_waypoints, 2] - Denoised trajectory
        """
        t_idx = t - 1  # 0-indexed
        
        alpha_t = self.alphas[t_idx]
        alpha_cumprod_t = self.alphas_cumprod[t_idx]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_idx]
        
        # DDPM denoising formula:
        # τ_{t-1} = (τ_t - (1-α_t)/√(1-ᾱ_t) · ε̂_t) / √α_t
        
        # Predicted x_0 (optional, used in some variants)
        # pred_x_0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        # Coefficients for noise prediction
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t
        
        # Mean of posterior q(x_{t-1} | x_t, x_0)
        mean = coef1 * (x_t - coef2 * predicted_noise)
        
        if t == 1:
            # Final step: no noise added
            return mean
        else:
            # Add variance noise
            variance = torch.sqrt(self.posterior_variance[t_idx])
            noise = torch.randn_like(x_t)
            return mean + variance * noise


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding as in Transformer/DDPM.
    
    Converts scalar timestep t into a vector embedding using
    sinusoidal functions at different frequencies.
    """
    
    def __init__(self, embedding_dim=256, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
    def forward(self, t):
        """
        Args:
            t: [B] - Timestep indices (can be int or float)
            
        Returns:
            emb: [B, embedding_dim] - Sinusoidal embeddings
        """
        half_dim = self.embedding_dim // 2
        
        # Create frequency bands: 1 / 10000^(2k/d)
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        
        # t * freqs: [B] * [half_dim] -> [B, half_dim] via broadcasting
        args = t[:, None].float() * freqs[None, :]  # [B, half_dim]
        
        # sin and cos embeddings
        emb_sin = torch.sin(args)
        emb_cos = torch.cos(args)
        
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # [B, embedding_dim]
        
        return emb


class TrajectoryDiffusionModel(nn.Module):
    """
    Complete diffusion model for trajectory prediction.
    
    Combines:
    - Noise scheduler
    - Timestep embedding
    - Denoising network g_φ
    """
    
    def __init__(self, denoising_network, num_timesteps=10, 
                 beta_start=0.0001, beta_end=0.02, device='cuda'):
        super().__init__()
        
        self.denoising_network = denoising_network
        self.scheduler = DiffusionScheduler(num_timesteps, beta_start, beta_end, device)
        self.timestep_embedding = SinusoidalTimestepEmbedding(embedding_dim=256)
        self.num_timesteps = num_timesteps
        self.device = device
        
    def forward_diffusion(self, x_0, t, noise=None):
        """
        Forward process: add noise to ground truth trajectory.
        
        Args:
            x_0: [B, num_waypoints, 2] - Ground truth trajectory
            t: [B] - Timesteps (1-indexed)
            noise: Optional pre-generated noise
            
        Returns:
            x_t, noise
        """
        return self.scheduler.add_noise(x_0, t, noise)
    
    def reverse_diffusion(self, x_t, conditioning, t):
        """
        Reverse process: predict noise and denoise.
        
        Args:
            x_t: [B, num_waypoints, 2] - Noised trajectory
            conditioning: [B, 512] - Conditioning vector from encoder
            t: [B] - Timesteps (1-indexed)
            
        Returns:
            predicted_noise: [B, num_waypoints, 2]
        """
        # Get timestep embeddings
        t_emb = self.timestep_embedding(t)  # [B, 256]
        
        # Predict noise
        predicted_noise = self.denoising_network(x_t, conditioning, t_emb)
        
        return predicted_noise
    
    def training_step(self, x_0, conditioning):
        """
        Single training step: sample t, add noise, predict, compute loss.
        
        Args:
            x_0: [B, num_waypoints, 2] - Ground truth trajectories
            conditioning: [B, 512] - Conditioning vectors
            
        Returns:
            loss: scalar MSE loss
            predicted_noise: [B, num_waypoints, 2]
            actual_noise: [B, num_waypoints, 2]
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = self.scheduler.sample_timesteps(batch_size)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion: add noise
        x_t, _ = self.forward_diffusion(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.reverse_diffusion(x_t, conditioning, t)
        
        # MSE loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss, predicted_noise, noise
    
    @torch.no_grad()
    def sample(self, conditioning, num_samples=5, return_all_steps=False):
        """
        Generate trajectory samples using reverse diffusion.
        
        Args:
            conditioning: [B, 512] - Conditioning vectors from encoder
            num_samples: K - Number of trajectory samples per conditioning
            return_all_steps: If True, return all denoising steps
            
        Returns:
            trajectories: [B, K, num_waypoints, 2] - Generated trajectories
            (optional) all_steps: list of intermediate trajectories
        """
        batch_size = conditioning.shape[0]
        num_waypoints = 8  # As per paper: 8 waypoints
        
        # Broadcast conditioning for K samples: [B, 512] -> [B*K, 512]
        conditioning_expanded = conditioning.unsqueeze(1).repeat(1, num_samples, 1)  # [B, K, 512]
        conditioning_expanded = conditioning_expanded.view(batch_size * num_samples, -1)  # [B*K, 512]
        
        # Initialize from noise: τ_N ~ N(0, I)
        x_t = torch.randn(batch_size * num_samples, num_waypoints, 2, device=self.device)
        
        all_steps = [x_t.view(batch_size, num_samples, num_waypoints, 2)] if return_all_steps else None
        
        # Reverse diffusion loop: t = T, T-1, ..., 1
        for t in range(self.num_timesteps, 0, -1):
            # Create timestep tensor
            t_tensor = torch.full((batch_size * num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.reverse_diffusion(x_t, conditioning_expanded, t_tensor)
            
            # Denoise step
            x_t = self.scheduler.denoise_step(x_t, predicted_noise, t)
            
            if return_all_steps:
                all_steps.append(x_t.view(batch_size, num_samples, num_waypoints, 2))
        
        # Reshape: [B*K, 8, 2] -> [B, K, 8, 2]
        trajectories = x_t.view(batch_size, num_samples, num_waypoints, 2)
        
        if return_all_steps:
            return trajectories, all_steps
        return trajectories


# Import metrics from metrics module
from metrics import compute_minADE


if __name__ == "__main__":
    # Test the diffusion model
    print("Testing Diffusion Model Components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test scheduler
    print("\n1. Testing DiffusionScheduler...")
    scheduler = DiffusionScheduler(num_timesteps=10, device=device)
    print(f"   Betas: {scheduler.betas[:5].cpu().numpy()}...")
    print(f"   Alphas cumprod: {scheduler.alphas_cumprod[:5].cpu().numpy()}...")
    
    # Test forward diffusion
    print("\n2. Testing Forward Diffusion...")
    x_0 = torch.randn(4, 8, 2, device=device)  # [B=4, 8 waypoints, 2D]
    t = torch.tensor([1, 3, 5, 10], device=device)
    x_t, noise = scheduler.add_noise(x_0, t)
    print(f"   x_0 shape: {x_0.shape}")
    print(f"   x_t shape: {x_t.shape}")
    print(f"   noise shape: {noise.shape}")
    print(f"   x_0 mean/std: {x_0.mean().item():.3f}/{x_0.std().item():.3f}")
    print(f"   x_t mean/std (t=10): {x_t[3].mean().item():.3f}/{x_t[3].std().item():.3f}")
    
    # Test timestep embedding
    print("\n3. Testing Timestep Embedding...")
    t_emb = SinusoidalTimestepEmbedding(embedding_dim=256)
    t_test = torch.tensor([1, 5, 10], device=device)
    emb = t_emb(t_test)
    print(f"   Input t: {t_test.cpu().numpy()}")
    print(f"   Output shape: {emb.shape}")
    print(f"   Embedding range: [{emb.min().item():.3f}, {emb.max().item():.3f}]")
    
    print("\n✓ All diffusion components working!")
