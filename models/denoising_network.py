#!/usr/bin/env python3
"""
Denoising Network g_φ for Diffusion Policy

Implements:
1. DenoisingMLP - Simple MLP baseline
2. DenoisingCNN1D - ResNet with FiLM conditioning
3. LightweightUNet1D - Paper-matching U-Net (default)

Paper Reference: Section III-D, IV-B ("lightweight U-Net")
"""

import torch
import torch.nn as nn
import math


class DenoisingMLP(nn.Module):
    """MLP-based denoising network (baseline)."""
    
    def __init__(self, num_waypoints=8, coord_dim=2, conditioning_dim=512, 
                 timestep_dim=256, hidden_dim=512, num_layers=4):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.coord_dim = coord_dim
        
        traj_flat_dim = num_waypoints * coord_dim
        input_dim = traj_flat_dim + conditioning_dim + timestep_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, traj_flat_dim)
        self.activation = nn.SiLU()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, trajectory, conditioning, timestep_emb):
        batch_size = trajectory.shape[0]
        
        traj_flat = trajectory.view(batch_size, -1)
        x = torch.cat([traj_flat, conditioning, timestep_emb], dim=-1)
        
        x = self.input_proj(x)
        x = self.activation(x)
        
        for layer, norm in zip(self.hidden_layers, self.norms):
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = norm(x)
            x = x + residual
        
        x = self.output_proj(x)
        predicted_noise = x.view(batch_size, self.num_waypoints, self.coord_dim)
        
        return predicted_noise


class DenoisingCNN1D(nn.Module):
    """1D-CNN ResNet with FiLM conditioning."""
    
    def __init__(self, num_waypoints=8, coord_dim=2, conditioning_dim=512,
                 timestep_dim=256, hidden_channels=128, num_layers=4):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.coord_dim = coord_dim
        self.hidden_channels = hidden_channels
        
        self.input_proj = nn.Conv1d(coord_dim, hidden_channels, kernel_size=3, padding=1)
        
        film_dim = conditioning_dim + timestep_dim
        self.film_generator = nn.Sequential(
            nn.Linear(film_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2)
        )
        
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
            )
            self.norm_layers.append(
                nn.GroupNorm(8, hidden_channels)
            )
        
        self.output_proj = nn.Conv1d(hidden_channels, coord_dim, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, trajectory, conditioning, timestep_emb):
        batch_size = trajectory.shape[0]
        
        x = trajectory.transpose(1, 2)
        x = self.input_proj(x)
        x = self.activation(x)
        
        film_input = torch.cat([conditioning, timestep_emb], dim=-1)
        film_params = self.film_generator(film_input)
        scale, shift = film_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            residual = x
            x = conv(x)
            x = norm(x)
            x = x * (1 + scale) + shift
            x = self.activation(x)
            x = x + residual
        
        x = self.output_proj(x)
        predicted_noise = x.transpose(1, 2)
        
        return predicted_noise


class LightweightUNet1D(nn.Module):
    """
    Lightweight U-Net denoiser as described in TopoDiffuser paper.
    
    Paper: Section III-D, IV-B ("lightweight U-Net")
    
    Architecture:
    - Encoder: 1D conv downsampling (8 → 4 → 2 waypoints)
    - Bottleneck: Processing at coarse resolution
    - Decoder: 1D conv upsampling with skip connections (2 → 4 → 8)
    - Conditioning injected at multiple scales via FiLM
    """
    
    def __init__(self, num_waypoints=8, coord_dim=2, conditioning_dim=512, 
                 timestep_dim=256, base_channels=64):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.coord_dim = coord_dim
        self.condition_dim = conditioning_dim + timestep_dim
        
        # FiLM generator factory
        def film_gen(in_ch):
            return nn.Sequential(
                nn.Linear(self.condition_dim, in_ch * 2),
                nn.SiLU(),
                nn.Linear(in_ch * 2, in_ch * 2)
            )
        
        # ENCODER (Downsampling)
        # Level 0: 8 waypoints → 8 waypoints (same res) → 4
        self.enc0_conv = nn.Conv1d(coord_dim, base_channels, 3, padding=1)
        self.enc0_norm = nn.BatchNorm1d(base_channels)
        self.enc0_film = film_gen(base_channels)
        self.enc0_pool = nn.MaxPool1d(2)  # 8 → 4
        
        # Level 1: 4 waypoints → 4 → 2
        self.enc1_conv = nn.Conv1d(base_channels, base_channels*2, 3, padding=1)
        self.enc1_norm = nn.BatchNorm1d(base_channels*2)
        self.enc1_film = film_gen(base_channels*2)
        self.enc1_pool = nn.MaxPool1d(2)  # 4 → 2
        
        # BOTTLENECK: 2 waypoints
        self.bottleneck_conv = nn.Conv1d(base_channels*2, base_channels*2, 3, padding=1)
        self.bottleneck_norm = nn.BatchNorm1d(base_channels*2)
        self.bottleneck_film = film_gen(base_channels*2)
        
        # DECODER (Upsampling with Skip Connections)
        # Level 1: 2 → 4 waypoints
        self.dec1_upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1_conv = nn.Conv1d(base_channels*4, base_channels, 3, padding=1)  # *4 due to skip
        self.dec1_norm = nn.BatchNorm1d(base_channels)
        self.dec1_film = film_gen(base_channels)
        
        # Level 0: 4 → 8 waypoints  
        self.dec0_upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec0_conv = nn.Conv1d(base_channels*2, base_channels, 3, padding=1)  # *2 due to skip
        self.dec0_norm = nn.BatchNorm1d(base_channels)
        self.dec0_film = film_gen(base_channels)
        
        # Output: back to coordinate dimension
        self.output_conv = nn.Conv1d(base_channels, coord_dim, 3, padding=1)
        
        self.activation = nn.SiLU()
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def apply_film(self, x, film_params):
        """Apply Feature-wise Linear Modulation."""
        scale, shift = film_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)  # [B, C, 1]
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift
        
    def forward(self, trajectory, conditioning, timestep_emb):
        """
        Args:
            trajectory: [B, num_waypoints, coord_dim]
            conditioning: [B, conditioning_dim]
            timestep_emb: [B, timestep_dim]
            
        Returns:
            predicted_noise: [B, num_waypoints, coord_dim]
        """
        B = trajectory.shape[0]
        cond = torch.cat([conditioning, timestep_emb], dim=-1)
        
        # Transpose to [B, channels, length]
        x = trajectory.transpose(1, 2)  # [B, 2, 8]
        
        # ENCODER
        # Level 0: 8 → 8 → 4
        e0 = self.enc0_conv(x)
        e0 = self.enc0_norm(e0)
        e0 = self.apply_film(e0, self.enc0_film(cond))
        e0 = self.activation(e0)
        e0_pooled = self.enc0_pool(e0)  # [B, 64, 4]
        
        # Level 1: 4 → 4 → 2
        e1 = self.enc1_conv(e0_pooled)
        e1 = self.enc1_norm(e1)
        e1 = self.apply_film(e1, self.enc1_film(cond))
        e1 = self.activation(e1)
        e1_pooled = self.enc1_pool(e1)  # [B, 128, 2]
        
        # BOTTLENECK: 2
        b = self.bottleneck_conv(e1_pooled)
        b = self.bottleneck_norm(b)
        b = self.apply_film(b, self.bottleneck_film(cond))
        b = self.activation(b)
        
        # DECODER with Skip Connections (U-Net style)
        # Level 1: 2 → 4, concat with e1
        d1 = self.dec1_upsample(b)  # [B, 128, 4]
        d1 = torch.cat([d1, e1], dim=1)  # [B, 256, 4] - Skip connection!
        d1 = self.dec1_conv(d1)  # [B, 64, 4]
        d1 = self.dec1_norm(d1)
        d1 = self.apply_film(d1, self.dec1_film(cond))
        d1 = self.activation(d1)
        
        # Level 0: 4 → 8, concat with e0
        d0 = self.dec0_upsample(d1)  # [B, 64, 8]
        d0 = torch.cat([d0, e0], dim=1)  # [B, 128, 8] - Skip connection!
        d0 = self.dec0_conv(d0)  # [B, 64, 8]
        d0 = self.dec0_norm(d0)
        d0 = self.apply_film(d0, self.dec0_film(cond))
        d0 = self.activation(d0)
        
        # Output
        out = self.output_conv(d0)  # [B, 2, 8]
        
        return out.transpose(1, 2)  # [B, 8, 2]


def build_denoising_network(architecture='unet', **kwargs):
    """
    Factory function to build denoising network.
    
    Args:
        architecture: 'mlp', 'cnn1d', or 'unet' (default, paper-matching)
        **kwargs: Passed to network constructor
        
    Returns:
        Denoising network instance
    """
    if architecture == 'mlp':
        return DenoisingMLP(**kwargs)
    elif architecture == 'cnn1d':
        return DenoisingCNN1D(**kwargs)
    elif architecture == 'unet':
        return LightweightUNet1D(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'mlp', 'cnn1d', or 'unet'")


if __name__ == "__main__":
    # Test all architectures
    print("Testing Denoising Networks...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_waypoints = 8
    
    trajectory = torch.randn(batch_size, num_waypoints, 2, device=device)
    conditioning = torch.randn(batch_size, 512, device=device)
    timestep_emb = torch.randn(batch_size, 256, device=device)
    
    for name, net_class in [('MLP', DenoisingMLP), ('CNN1D', DenoisingCNN1D), ('U-Net', LightweightUNet1D)]:
        print(f"\n{name}:")
        net = net_class().to(device)
        params = sum(p.numel() for p in net.parameters())
        output = net(trajectory, conditioning, timestep_emb)
        print(f"  Params: {params:,}")
        print(f"  Output: {output.shape}, range=[{output.min():.3f}, {output.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("✓ All networks working!")
