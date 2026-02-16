#!/usr/bin/env python3
"""
Denoising Network g_φ for Diffusion Policy

Architecture: MLP-based (simpler, faster, sufficient for 8 waypoints)
Alternative: 1D-CNN (more expressive, could use for larger trajectories)

Input:
  - τ_t: [B, num_waypoints, 2] - Noised trajectory
  - c: [B, 512] - Conditioning vector from encoder
  - t_emb: [B, 256] - Timestep embedding

Output:
  - ε̂_t: [B, num_waypoints, 2] - Predicted noise
"""

import torch
import torch.nn as nn
import math


class DenoisingMLP(nn.Module):
    """
    MLP-based denoising network g_φ.
    
    Architecture:
    1. Flatten trajectory [8,2] -> [16]
    2. Concat with conditioning [512] and timestep emb [256] -> [784]
    3. 4 FC layers with residual connections
    4. Reshape back to [8, 2]
    
    Paper: Section III-E
    """
    
    def __init__(self, num_waypoints=8, coord_dim=2, conditioning_dim=512, 
                 timestep_dim=256, hidden_dim=512, num_layers=4):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.coord_dim = coord_dim
        self.conditioning_dim = conditioning_dim
        self.timestep_dim = timestep_dim
        self.hidden_dim = hidden_dim
        
        # Trajectory flatten: [num_waypoints, coord_dim] -> [num_waypoints * coord_dim]
        traj_flat_dim = num_waypoints * coord_dim
        
        # Input projection
        input_dim = traj_flat_dim + conditioning_dim + timestep_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, traj_flat_dim)
        
        self.activation = nn.SiLU()  # Smooth activation, works well with diffusion
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, trajectory, conditioning, timestep_emb):
        """
        Predict noise given noised trajectory, conditioning, and timestep.
        
        Args:
            trajectory: [B, num_waypoints, coord_dim] - Noised trajectory τ_t
            conditioning: [B, conditioning_dim] - Conditioning vector c
            timestep_emb: [B, timestep_dim] - Timestep embedding
            
        Returns:
            predicted_noise: [B, num_waypoints, coord_dim] - Predicted noise ε̂_t
        """
        batch_size = trajectory.shape[0]
        
        # Flatten trajectory
        traj_flat = trajectory.view(batch_size, -1)  # [B, num_waypoints * coord_dim]
        
        # Concatenate all inputs
        x = torch.cat([traj_flat, conditioning, timestep_emb], dim=-1)  # [B, input_dim]
        
        # Input projection
        x = self.input_proj(x)
        x = self.activation(x)
        
        # Hidden layers with residual connections
        for layer, norm in zip(self.hidden_layers, self.norms):
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = norm(x)
            x = x + residual  # Residual connection
        
        # Output projection
        x = self.output_proj(x)  # [B, num_waypoints * coord_dim]
        
        # Reshape back to trajectory format
        predicted_noise = x.view(batch_size, self.num_waypoints, self.coord_dim)
        
        return predicted_noise


class DenoisingCNN1D(nn.Module):
    """
    1D-CNN based denoising network (alternative to MLP).
    
    Uses temporal convolutions over waypoints with conditioning
    injected via FiLM (Feature-wise Linear Modulation).
    
    Better for longer trajectories, but overkill for just 8 waypoints.
    """
    
    def __init__(self, num_waypoints=8, coord_dim=2, conditioning_dim=512,
                 timestep_dim=256, hidden_channels=128, num_layers=4):
        super().__init__()
        
        self.num_waypoints = num_waypoints
        self.coord_dim = coord_dim
        self.hidden_channels = hidden_channels
        
        # Project trajectory to hidden channels
        # Input: [B, coord_dim, num_waypoints]
        self.input_proj = nn.Conv1d(coord_dim, hidden_channels, kernel_size=3, padding=1)
        
        # FiLM generator: conditioning + timestep -> scale, shift
        film_dim = conditioning_dim + timestep_dim
        self.film_generator = nn.Sequential(
            nn.Linear(film_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels * 2)
        )
        
        # 1D ResNet blocks
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
            )
            self.norm_layers.append(
                nn.GroupNorm(8, hidden_channels)  # Group norm works well with conv
            )
        
        # Output projection
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
        """
        Args:
            trajectory: [B, num_waypoints, coord_dim]
            conditioning: [B, conditioning_dim]
            timestep_emb: [B, timestep_dim]
            
        Returns:
            predicted_noise: [B, num_waypoints, coord_dim]
        """
        batch_size = trajectory.shape[0]
        
        # Transpose for Conv1d: [B, num_waypoints, coord_dim] -> [B, coord_dim, num_waypoints]
        x = trajectory.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)  # [B, hidden_channels, num_waypoints]
        x = self.activation(x)
        
        # Generate FiLM parameters
        film_input = torch.cat([conditioning, timestep_emb], dim=-1)
        film_params = self.film_generator(film_input)  # [B, hidden_channels * 2]
        scale, shift = film_params.chunk(2, dim=-1)  # Each [B, hidden_channels]
        
        # Apply scale and shift to x (broadcast over waypoints)
        # scale/shift: [B, hidden_channels] -> [B, hidden_channels, 1]
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        # ResNet blocks with FiLM conditioning
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            residual = x
            x = conv(x)
            x = norm(x)
            # FiLM: x = x * (1 + scale) + shift
            x = x * (1 + scale) + shift
            x = self.activation(x)
            x = x + residual
        
        # Output projection
        x = self.output_proj(x)  # [B, coord_dim, num_waypoints]
        
        # Transpose back: [B, coord_dim, num_waypoints] -> [B, num_waypoints, coord_dim]
        predicted_noise = x.transpose(1, 2)
        
        return predicted_noise


def build_denoising_network(architecture='mlp', **kwargs):
    """
    Factory function to build denoising network.
    
    Args:
        architecture: 'mlp' or 'cnn1d'
        **kwargs: Passed to network constructor
        
    Returns:
        Denoising network instance
    """
    if architecture == 'mlp':
        return DenoisingMLP(**kwargs)
    elif architecture == 'cnn1d':
        return DenoisingCNN1D(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose 'mlp' or 'cnn1d'")


if __name__ == "__main__":
    # Test both architectures
    print("Testing Denoising Networks...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_waypoints = 8
    
    # Test inputs
    trajectory = torch.randn(batch_size, num_waypoints, 2, device=device)
    conditioning = torch.randn(batch_size, 512, device=device)
    timestep_emb = torch.randn(batch_size, 256, device=device)
    
    # Test MLP
    print("\n1. Testing DenoisingMLP...")
    mlp_net = DenoisingMLP().to(device)
    print(f"   Parameters: {sum(p.numel() for p in mlp_net.parameters()):,}")
    
    output_mlp = mlp_net(trajectory, conditioning, timestep_emb)
    print(f"   Input shape: {trajectory.shape}")
    print(f"   Output shape: {output_mlp.shape}")
    print(f"   Output stats: mean={output_mlp.mean().item():.4f}, std={output_mlp.std().item():.4f}")
    
    # Test CNN1D
    print("\n2. Testing DenoisingCNN1D...")
    cnn_net = DenoisingCNN1D().to(device)
    print(f"   Parameters: {sum(p.numel() for p in cnn_net.parameters()):,}")
    
    output_cnn = cnn_net(trajectory, conditioning, timestep_emb)
    print(f"   Input shape: {trajectory.shape}")
    print(f"   Output shape: {output_cnn.shape}")
    print(f"   Output stats: mean={output_cnn.mean().item():.4f}, std={output_cnn.std().item():.4f}")
    
    # Test factory function
    print("\n3. Testing build_denoising_network factory...")
    net1 = build_denoising_network('mlp').to(device)
    net2 = build_denoising_network('cnn1d').to(device)
    print(f"   MLP type: {type(net1).__name__}")
    print(f"   CNN type: {type(net2).__name__}")
    
    print("\n" + "=" * 60)
    print("✓ All denoising networks working!")
    
    # Parameter comparison
    print(f"\nParameter comparison:")
    print(f"  MLP:    {sum(p.numel() for p in mlp_net.parameters()):,}")
    print(f"  CNN1D:  {sum(p.numel() for p in cnn_net.parameters()):,}")
