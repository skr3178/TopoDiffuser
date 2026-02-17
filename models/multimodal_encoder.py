"""
Full Multimodal Encoder for TopoDiffuser.

5-channel input: LiDAR [3] + History [1] + OSM [1]
Output: 512-dim conditioning vector + road segmentation

This is a completely separate encoder from the existing 3-channel version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBRBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CBBlock(nn.Module):
    """Convolution + BatchNorm block (no activation)."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))


class FullMultimodalEncoder(nn.Module):
    """
    Full Multimodal Encoder - 5-channel input.
    
    Input: [B, 5, 300, 400] - LiDAR [3] + History [1] + OSM [1]
    Output: 
        - conditioning: [B, 512] conditioning vector for diffusion
        - road_seg: [B, 1, 37, 50] road segmentation prediction
    
    Architecture mirrors the existing encoder but handles 5 input channels.
    """
    
    def __init__(self, input_channels=5, conditioning_dim=512, dropout=0.3):
        super().__init__()

        self.input_channels = input_channels
        self.conditioning_dim = conditioning_dim

        # ========== ENCODER PATH (Contracting) ==========
        # c1: [B, 5, 300, 400] → [B, 32, 150, 200]
        self.c1 = CBRBlock(input_channels, 32, kernel_size=3, stride=2, padding=1)

        # c2: [B, 32, 150, 200] → [B, 64, 75, 100]
        self.c2 = CBRBlock(32, 64, kernel_size=3, stride=2, padding=1)

        # c3: [B, 64, 75, 100] → [B, 128, 38, 50]
        self.c3 = CBRBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.drop3 = nn.Dropout2d(dropout)

        # c4: [B, 128, 38, 50] → [B, 256, 19, 25]
        self.c4 = CBRBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.drop4 = nn.Dropout2d(dropout)

        # c5: [B, 256, 19, 25] → [B, 512, 10, 13]
        self.c5 = CBRBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.drop5 = nn.Dropout2d(dropout * 2)
        
        # ========== DECODER PATH (Expanding) ==========
        
        # p5: [B, 512, 10, 13] → 1×1 Conv + Deconv + ⊕c4 → [B, 64, 8, 8]
        self.p5_conv1x1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.p5_deconv = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.c4_skip_conv = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        
        # p4: [B, 64, 8, 8] → Deconv + ⊕c3 → [B, 64, 19, 25]
        self.p4_deconv = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.c3_skip_conv = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        
        # ========== DUAL HEADS FROM P4 (SHARED BACKBONE) ==========
        
        # Shared backbone: CB Blocks → Conv+BN
        self.shared_backbone = nn.Sequential(
            CBBlock(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            CBBlock(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),
        )
        
        # Head A: Road Segmentation (Auxiliary Task)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        
        # Head B: Conditioning Vector (Main Task)
        self.cond_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.cond_bn = nn.BatchNorm2d(64)
        self.cond_relu = nn.ReLU(inplace=True)
        self.cond_pool = nn.AdaptiveAvgPool2d((1, 8))  # [B, 64, 1, 8]
        self.cond_drop = nn.Dropout(dropout * 2)
        self.cond_fc = nn.Linear(64 * 8, conditioning_dim)  # [B, 512] → [B, 512]
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [B, 5, H, W] multimodal input tensor
               Channel 0-2: LiDAR (height, intensity, density)
               Channel 3: History trajectory
               Channel 4: OSM roads
        
        Returns:
            conditioning: [B, 512] conditioning vector
            road_seg: [B, 1, 37, 50] road segmentation prediction
        """
        B = x.shape[0]
        
        # ========== ENCODER ==========
        c1_out = self.c1(x)      # [B, 32, 150, 200]
        c2_out = self.c2(c1_out)  # [B, 64, 75, 100]
        c3_out = self.drop3(self.c3(c2_out))  # [B, 128, 38, 50]
        c4_out = self.drop4(self.c4(c3_out))  # [B, 256, 19, 25]
        c5_out = self.drop5(self.c5(c4_out))  # [B, 512, 10, 13]
        
        # ========== DECODER ==========
        
        # p5: [B, 512, 10, 13] → [B, 64, 8, 8]
        p5 = self.p5_conv1x1(c5_out)  # [B, 256, 10, 13]
        p5 = self.p5_deconv(p5)       # [B, 64, 20, 26]
        if p5.shape[2:] != (8, 8):
            p5 = F.interpolate(p5, size=(8, 8), mode='bilinear', align_corners=False)
        c4_skip = F.interpolate(self.c4_skip_conv(c4_out), size=(8, 8), 
                                mode='bilinear', align_corners=False)
        p5 = p5 + c4_skip  # [B, 64, 8, 8]
        
        # p4: [B, 64, 8, 8] → [B, 64, 19, 25]
        p4 = self.p4_deconv(p5)  # [B, 64, 16, 18]
        if p4.shape[2:] != (19, 25):
            p4 = F.interpolate(p4, size=(19, 25), mode='bilinear', align_corners=False)
        c3_skip = F.interpolate(self.c3_skip_conv(c3_out), size=(19, 25),
                                mode='bilinear', align_corners=False)
        p4 = p4 + c3_skip  # [B, 64, 19, 25]
        
        # ========== DUAL HEADS FROM P4 ==========
        shared_features = self.shared_backbone(p4)  # [B, 64, 19, 25]
        
        # Head A: Road Segmentation
        road_seg = self.seg_head(shared_features)  # [B, 1, 37, 50]
        if road_seg.shape[2:] != (37, 50):
            road_seg = F.interpolate(road_seg, size=(37, 50), mode='bilinear', align_corners=False)
        
        # Head B: Conditioning Vector
        cond_features = self.cond_conv(shared_features)  # [B, 64, 19, 25]
        cond_features = self.cond_bn(cond_features)
        cond_features = self.cond_relu(cond_features)
        cond_features = self.cond_pool(cond_features)  # [B, 64, 1, 8]
        cond_flat = cond_features.view(B, -1)  # [B, 512]
        cond_flat = self.cond_drop(cond_flat)
        conditioning = self.cond_fc(cond_flat)  # [B, 512]
        
        return conditioning, road_seg
    
    def init_from_3channel_encoder(self, encoder_3ch_path: str):
        """
        Initialize this 5-channel encoder from a 3-channel encoder.
        
        Loads weights for shared layers, handles first conv layer specially.
        
        Args:
            encoder_3ch_path: Path to 3-channel encoder checkpoint
        """
        checkpoint = torch.load(encoder_3ch_path, map_location='cpu', weights_only=False)
        
        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        else:
            state_dict = checkpoint
        
        # Load all layers except c1 (first conv)
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if name == 'c1.conv.weight':
                    # Special handling: expand 3-channel weights to 5 channels
                    # Copy LiDAR weights to first 3 channels
                    # Initialize history and OSM channels with mean of LiDAR weights
                    print(f"Expanding {name}: {param.shape} → {own_state[name].shape}")
                    with torch.no_grad():
                        # Copy existing 3 channels
                        own_state[name][:, :3, :, :] = param
                        # Initialize channel 3 (history) and 4 (OSM) as average of existing
                        mean_weight = param.mean(dim=1, keepdim=True)
                        own_state[name][:, 3:4, :, :] = mean_weight
                        own_state[name][:, 4:5, :, :] = mean_weight
                elif own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                else:
                    print(f"Shape mismatch for {name}: {param.shape} vs {own_state[name].shape}")
        
        print(f"Initialized from {encoder_3ch_path}")
        print("First conv layer expanded: 3 channels → 5 channels")


def build_full_multimodal_encoder(input_channels=5, conditioning_dim=512, 
                                  init_from_3ch_path=None):
    """
    Build full multimodal encoder.
    
    Args:
        input_channels: Number of input channels (5 for full multimodal)
        conditioning_dim: Dimension of conditioning vector (default: 512)
        init_from_3ch_path: Optional path to 3-channel encoder for weight initialization
    
    Returns:
        encoder: FullMultimodalEncoder instance
    """
    encoder = FullMultimodalEncoder(input_channels, conditioning_dim)
    
    if init_from_3ch_path is not None:
        encoder.init_from_3channel_encoder(init_from_3ch_path)
    
    return encoder


if __name__ == "__main__":
    print("=" * 70)
    print("Full Multimodal Encoder Test Suite")
    print("=" * 70)
    
    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    batch_size = 4
    input_channels = 5
    height, width = 300, 400
    
    x = torch.randn(batch_size, input_channels, height, width)
    print(f"   Input shape: {x.shape}")
    
    encoder = build_full_multimodal_encoder(input_channels, 512)
    conditioning, road_seg = encoder(x)
    
    print(f"   Conditioning shape: {conditioning.shape}")
    print(f"   Road segmentation shape: {road_seg.shape}")
    
    # Verify shapes
    assert conditioning.shape == (batch_size, 512)
    assert road_seg.shape == (batch_size, 1, 37, 50)
    print("   ✓ Output shapes correct")
    
    # Test 2: Parameter count
    print("\n2. Model statistics...")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test 3: Initialization from 3-channel encoder
    print("\n3. Testing initialization from 3-channel encoder...")
    # This would require a real checkpoint file
    print("   (Skipping - requires checkpoint file)")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
