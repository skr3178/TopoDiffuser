#!/usr/bin/env python3
"""
Loss Functions for TopoDiffuser

Implements Equations 3, 4, 5 from the paper:
- Equation 3: L_diffusion (MSE on noise prediction)
- Equation 4: L_road (BCE on road segmentation)
- Equation 5: L_total = L_diffusion + α_road * L_road

Paper Reference: TopoDiffuser (arXiv:2508.00303)
Section III-C, III-D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """
    Equation 3: Diffusion Loss (MSE)
    
    L_diff = ||ε - g_φ(τ_t, c, t)||²
    
    where:
    - ε: Ground truth noise sampled from N(0, I)
    - g_φ: Denoising network
    - τ_t: Noised trajectory at timestep t
    - c: Conditioning vector from encoder
    - t: Timestep
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, predicted_noise, target_noise):
        """
        Args:
            predicted_noise: [B, num_waypoints, 2] - ε̂_t from denoising network
            target_noise: [B, num_waypoints, 2] - ε (ground truth noise)
            
        Returns:
            loss: scalar MSE loss
        """
        return self.mse(predicted_noise, target_noise)


class RoadSegmentationLoss(nn.Module):
    """
    Equation 4: Road Segmentation Loss (BCE)
    
    L_road = -Σ[y·log(x) + (1-y)·log(1-x)]
    
    where:
    - x: Predicted road mask [B, 1, H, W]
    - y: Ground truth road mask [B, 1, H, W]
    
    Note: This is pixel-wise binary cross-entropy averaged over all pixels.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.bce = nn.BCELoss(reduction=reduction)
    
    def forward(self, pred_mask, gt_mask):
        """
        Args:
            pred_mask: [B, 1, H, W] - Predicted road probabilities (after sigmoid)
            gt_mask: [B, 1, H, W] - Ground truth binary mask
            
        Returns:
            loss: scalar BCE loss
        """
        return self.bce(pred_mask, gt_mask)


class TopoDiffuserLoss(nn.Module):
    """
    Equation 5: Total Loss
    
    L_total = L_diffusion + α_road * L_road
    
    This combines:
    1. Main task: Trajectory prediction via diffusion (L_diffusion)
    2. Auxiliary task: Road segmentation (L_road)
    
    The auxiliary task provides additional supervision to the encoder,
    helping it learn better spatial representations of drivable areas.
    """
    
    def __init__(self, alpha_road=0.1):
        """
        Args:
            alpha_road: Weight for road segmentation loss (default: 0.1)
                       Paper suggests 0.1-1.0 range.
        """
        super().__init__()
        self.alpha_road = alpha_road
        self.diffusion_loss = DiffusionLoss()
        self.road_loss = RoadSegmentationLoss()
    
    def forward(self, predicted_noise, target_noise, pred_mask, gt_mask):
        """
        Compute total loss.
        
        Args:
            predicted_noise: [B, num_waypoints, 2] - Predicted noise from diffusion
            target_noise: [B, num_waypoints, 2] - Ground truth noise
            pred_mask: [B, 1, H, W] - Predicted road segmentation
            gt_mask: [B, 1, H, W] - Ground truth road mask
            
        Returns:
            total_loss: scalar total loss
            loss_dict: dict with individual loss components
        """
        # Equation 3: Diffusion loss
        l_diff = self.diffusion_loss(predicted_noise, target_noise)
        
        # Equation 4: Road segmentation loss
        l_road = self.road_loss(pred_mask, gt_mask)
        
        # Equation 5: Total loss
        l_total = l_diff + self.alpha_road * l_road
        
        loss_dict = {
            'total': l_total.item(),
            'diffusion': l_diff.item(),
            'road': l_road.item(),
            'alpha_road': self.alpha_road
        }
        
        return l_total, loss_dict
    
    def forward_diffusion_only(self, predicted_noise, target_noise):
        """
        Compute only diffusion loss (for validation or ablation).
        
        Args:
            predicted_noise: [B, num_waypoints, 2]
            target_noise: [B, num_waypoints, 2]
            
        Returns:
            loss: scalar diffusion loss
        """
        return self.diffusion_loss(predicted_noise, target_noise)
    
    def forward_road_only(self, pred_mask, gt_mask):
        """
        Compute only road segmentation loss (for validation or ablation).
        
        Args:
            pred_mask: [B, 1, H, W]
            gt_mask: [B, 1, H, W]
            
        Returns:
            loss: scalar road loss
        """
        return self.road_loss(pred_mask, gt_mask)


class UncertaintyWeightedLoss(nn.Module):
    """
    Optional: Learnable uncertainty weighting (Kendall et al.).
    
    Instead of fixed α_road, learns the optimal weighting automatically:
    L_total = (1/(2*σ_diff²)) * L_diff + (1/(2*σ_road²)) * L_road + log(σ_diff * σ_road)
    
    This can be used as an alternative to fixed α_road.
    """
    
    def __init__(self):
        super().__init__()
        # Learnable log variances
        self.log_sigma_diff = nn.Parameter(torch.zeros(1))
        self.log_sigma_road = nn.Parameter(torch.zeros(1))
        
        self.diffusion_loss = DiffusionLoss()
        self.road_loss = RoadSegmentationLoss()
    
    def forward(self, predicted_noise, target_noise, pred_mask, gt_mask):
        """
        Compute uncertainty-weighted total loss.
        """
        l_diff = self.diffusion_loss(predicted_noise, target_noise)
        l_road = self.road_loss(pred_mask, gt_mask)
        
        # Precision weighted loss
        loss = (l_diff / (2 * torch.exp(2 * self.log_sigma_diff)) + 
                l_road / (2 * torch.exp(2 * self.log_sigma_road)) +
                self.log_sigma_diff + self.log_sigma_road)
        
        loss_dict = {
            'total': loss.item(),
            'diffusion': l_diff.item(),
            'road': l_road.item(),
            'sigma_diff': torch.exp(self.log_sigma_diff).item(),
            'sigma_road': torch.exp(self.log_sigma_road).item()
        }
        
        return loss, loss_dict


def get_loss_function(loss_type='combined', alpha_road=0.1):
    """
    Factory function to get loss function.
    
    Args:
        loss_type: 'combined', 'diffusion_only', 'road_only', 'uncertainty'
        alpha_road: Weight for road loss (for combined loss)
        
    Returns:
        Loss module
    """
    if loss_type == 'combined':
        return TopoDiffuserLoss(alpha_road=alpha_road)
    elif loss_type == 'diffusion_only':
        return DiffusionLoss()
    elif loss_type == 'road_only':
        return RoadSegmentationLoss()
    elif loss_type == 'uncertainty':
        return UncertaintyWeightedLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    
    # Test data
    pred_noise = torch.randn(batch_size, 8, 2, device=device)
    target_noise = torch.randn(batch_size, 8, 2, device=device)
    pred_mask = torch.sigmoid(torch.randn(batch_size, 1, 37, 50, device=device))
    gt_mask = torch.randint(0, 2, (batch_size, 1, 37, 50), dtype=torch.float32, device=device)
    
    print(f"\nTest shapes:")
    print(f"  Noise: {pred_noise.shape}")
    print(f"  Mask: {pred_mask.shape}")
    
    # Test 1: Diffusion loss only
    print("\n1. DiffusionLoss (Equation 3):")
    diff_loss = DiffusionLoss()
    l_diff = diff_loss(pred_noise, target_noise)
    print(f"   L_diffusion = {l_diff.item():.4f}")
    
    # Test 2: Road loss only
    print("\n2. RoadSegmentationLoss (Equation 4):")
    road_loss = RoadSegmentationLoss()
    l_road = road_loss(pred_mask, gt_mask)
    print(f"   L_road = {l_road.item():.4f}")
    
    # Test 3: Combined loss
    print("\n3. TopoDiffuserLoss (Equation 5):")
    for alpha in [0.1, 0.5, 1.0]:
        combined_loss = TopoDiffuserLoss(alpha_road=alpha)
        l_total, loss_dict = combined_loss(pred_noise, target_noise, pred_mask, gt_mask)
        print(f"   α={alpha}: L_total = {loss_dict['total']:.4f} "
              f"(L_diff={loss_dict['diffusion']:.4f}, L_road={loss_dict['road']:.4f})")
    
    # Test 4: Uncertainty weighted
    print("\n4. UncertaintyWeightedLoss:")
    unc_loss = UncertaintyWeightedLoss().to(device)
    l_total, loss_dict = unc_loss(pred_noise, target_noise, pred_mask, gt_mask)
    print(f"   L_total = {loss_dict['total']:.4f}")
    print(f"   σ_diff = {loss_dict['sigma_diff']:.4f}, σ_road = {loss_dict['sigma_road']:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All loss functions working!")
