"""
Test to verify whether c2-c5 need ReLU activation.

Paper says: c2-c5 use CB (Conv+BN) only, no ReLU
This test checks if that causes:
1. Vanishing gradients
2. Feature collapse (low diversity)
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from models.encoder import MultimodalEncoder, CBRBlock, CBBlock


def test_gradient_flow():
    """
    Test 1: Check if gradients flow properly through CB blocks (without ReLU).
    
    Theory: Without ReLU, gradients should actually flow BETTER (no cutoff).
    But if features become too linear/degenerate, gradients might be weak.
    """
    print("="*70)
    print("TEST 1: Gradient Flow Through Encoder")
    print("="*70)
    
    encoder = MultimodalEncoder(input_channels=5)
    encoder.train()
    
    # Forward pass
    x = torch.randn(2, 5, 300, 400, requires_grad=True)
    conditioning, road_seg = encoder(x)
    
    # Compute loss on both outputs
    loss = conditioning.sum() + road_seg.sum()
    loss.backward()
    
    print("\nGradient magnitudes at each encoder layer:")
    print(f"  Input grad:        {x.grad.abs().mean():.6f}")
    
    # Check gradients for each encoder block
    def get_grad_stats(module, name):
        if hasattr(module, 'conv') and module.conv.weight.grad is not None:
            grad_mag = module.conv.weight.grad.abs().mean()
            print(f"  {name} grad:       {grad_mag:.6f}")
            return grad_mag
        return None
    
    c1_grad = get_grad_stats(encoder.c1[0], "c1 (CBR)")
    c2_grad = get_grad_stats(encoder.c2[0], "c2 (CB) ")
    c3_grad = get_grad_stats(encoder.c3[0], "c3 (CB) ")
    c4_grad = get_grad_stats(encoder.c4[0], "c4 (CB) ")
    c5_grad = get_grad_stats(encoder.c5[0], "c5 (CB) ")
    
    # Analysis
    print("\nAnalysis:")
    if c1_grad and c2_grad and c5_grad:
        c2_vs_c1 = c2_grad / c1_grad if c1_grad > 0 else 0
        c5_vs_c1 = c5_grad / c1_grad if c1_grad > 0 else 0
        
        print(f"  c2/c1 ratio: {c2_vs_c1:.3f}")
        print(f"  c5/c1 ratio: {c5_vs_c1:.3f}")
        
        if c5_vs_c1 < 0.001:
            print("  ⚠️  WARNING: c5 gradients nearly vanished!")
            print("      → c2-c5 MAY need ReLU to maintain gradient magnitude")
        elif c5_vs_c1 > 0.1:
            print("  ✓ Gradients flow well through CB blocks")
            print("      → Paper's CB (no ReLU) design is OK")
        else:
            print(f"  ⚠️  Gradients weakened (ratio: {c5_vs_c1:.4f})")


def test_feature_diversity():
    """
    Test 2: Check feature diversity at each layer.
    
    Without ReLU, features might collapse (low std) because:
    - Linear(Linear(x)) is still linear
    - No non-linearity to create diverse features
    """
    print("\n" + "="*70)
    print("TEST 2: Feature Diversity at Each Layer")
    print("="*70)
    
    encoder = MultimodalEncoder(input_channels=5)
    encoder.eval()
    
    # Create input with known statistics
    x = torch.randn(8, 5, 300, 400) * 0.5 + 0.1  # mean=0.1, std=0.5
    
    print(f"\nInput statistics: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # Extract features at each layer
    with torch.no_grad():
        c1_out = encoder.c1(x)
        print(f"\nc1 (CBR - HAS ReLU):")
        print(f"  Shape: {c1_out.shape}")
        print(f"  Mean: {c1_out.mean():.4f}, Std: {c1_out.std():.4f}")
        print(f"  Min: {c1_out.min():.4f}, Max: {c1_out.max():.4f}")
        print(f"  Active units (% > 0): {(c1_out > 0).float().mean()*100:.1f}%")
        
        c2_out = encoder.c2(c1_out)
        print(f"\nc2 (CB - NO ReLU):")
        print(f"  Shape: {c2_out.shape}")
        print(f"  Mean: {c2_out.mean():.4f}, Std: {c2_out.std():.4f}")
        print(f"  Min: {c2_out.min():.4f}, Max: {c2_out.max():.4f}")
        print(f"  Active units (% > 0): {(c2_out > 0).float().mean()*100:.1f}%")
        
        c3_out = encoder.c3(c2_out)
        print(f"\nc3 (CB - NO ReLU):")
        print(f"  Shape: {c3_out.shape}")
        print(f"  Mean: {c3_out.mean():.4f}, Std: {c3_out.std():.4f}")
        
        c4_out = encoder.c4(c3_out)
        print(f"\nc4 (CB - NO ReLU):")
        print(f"  Shape: {c4_out.shape}")
        print(f"  Mean: {c4_out.mean():.4f}, Std: {c4_out.std():.4f}")
        
        c5_out = encoder.c5(c4_out)
        print(f"\nc5 (CB - NO ReLU):")
        print(f"  Shape: {c5_out.shape}")
        print(f"  Mean: {c5_out.mean():.4f}, Std: {c5_out.std():.4f}")
        print(f"  Min: {c5_out.min():.4f}, Max: {c5_out.max():.4f}")
    
    # Analysis
    print("\n" + "-"*70)
    print("Analysis:")
    
    # Check if std collapses
    stds = [c1_out.std().item(), c2_out.std().item(), c3_out.std().item(), 
            c4_out.std().item(), c5_out.std().item()]
    
    print(f"  Std progression: c1={stds[0]:.4f} → c5={stds[-1]:.4f}")
    
    if stds[-1] < stds[0] * 0.1:
        print("  ⚠️  WARNING: Feature std collapsed!")
        print("      → c2-c5 MAY need ReLU to maintain feature diversity")
    elif stds[-1] < stds[0] * 0.5:
        print(f"  ⚠️  Feature std reduced ({stds[-1]/stds[0]:.1%})")
        print("      → Consider adding ReLU for better feature diversity")
    else:
        print(f"  ✓ Feature diversity maintained ({stds[-1]/stds[0]:.1%})")
        print("      → Paper's CB (no ReLU) design preserves features")
    
    # Check for dead features
    dead_c5 = (c5_out.std(dim=[0,2,3]) < 0.01).sum().item()
    total_c5 = c5_out.shape[1]
    print(f"\n  Dead channels in c5: {dead_c5}/{total_c5} ({dead_c5/total_c5*100:.1f}%)")
    if dead_c5 / total_c5 > 0.3:
        print("  ⚠️  Too many dead channels - ReLU might help")


def test_with_vs_without_relu():
    """
    Test 3: Direct comparison of CB vs CBR for c2-c5.
    
    Train a tiny model with both configurations and compare.
    """
    print("\n" + "="*70)
    print("TEST 3: Direct Comparison - CB vs CBR for c2-c5")
    print("="*70)
    
    # Create two mini encoders
    class MiniEncoderCB(nn.Module):
        """c2-c5 with CB (no ReLU) - paper's design"""
        def __init__(self):
            super().__init__()
            self.c1 = CBRBlock(5, 32, stride=2)  # CBR
            self.c2 = CBBlock(32, 64, stride=2)   # CB (no ReLU)
            self.c3 = CBBlock(64, 128, stride=2)  # CB (no ReLU)
            
        def forward(self, x):
            x = self.c1(x)
            x = self.c2(x)
            x = self.c3(x)
            return x.mean()
    
    class MiniEncoderCBR(nn.Module):
        """c2-c5 with CBR (with ReLU) - alternative design"""
        def __init__(self):
            super().__init__()
            self.c1 = CBRBlock(5, 32, stride=2)   # CBR
            self.c2 = CBRBlock(32, 64, stride=2)  # CBR (with ReLU)
            self.c3 = CBRBlock(64, 128, stride=2) # CBR (with ReLU)
            
        def forward(self, x):
            x = self.c1(x)
            x = self.c2(x)
            x = self.c3(x)
            return x.mean()
    
    # Quick training test
    print("\nTraining both models for 100 iterations...")
    
    models = {
        'CB (Paper)': MiniEncoderCB(),
        'CBR (Alt)': MiniEncoderCBR()
    }
    
    results = {}
    
    for name, model in models.items():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        
        for i in range(100):
            x = torch.randn(4, 5, 300, 400)
            target = torch.randn(4, 128, 38, 50)
            
            optimizer.zero_grad()
            # Simulate a task: predict something
            out = model.c3(model.c2(model.c1(x)))  # Get c3 output
            loss = nn.MSELoss()(out, target)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check final gradient health
        x = torch.randn(2, 5, 300, 400, requires_grad=True)
        out = model(x)
        out.backward()
        
        results[name] = {
            'final_loss': losses[-1],
            'grad_norm': x.grad.norm().item(),
            'loss_trend': losses[-1] / losses[0]  # How much loss decreased
        }
    
    print("\nResults:")
    for name, res in results.items():
        print(f"\n  {name}:")
        print(f"    Final loss: {res['final_loss']:.4f}")
        print(f"    Loss reduction: {res['loss_trend']:.2%} of initial")
        print(f"    Input grad norm: {res['grad_norm']:.4f}")
    
    # Compare
    cb_better = results['CB (Paper)']['final_loss'] < results['CBR (Alt)']['final_loss']
    print("\n" + "-"*70)
    if cb_better:
        print("Result: CB (paper design) performs better for this task")
    else:
        print("Result: CBR (with ReLU) performs better for this task")
    print("Note: This is a simplified test - real performance may differ")


def main():
    print("\n" + "="*70)
    print("TESTING ENCODER ACTIVATION HYPOTHESIS")
    print("Paper says: c2-c5 use CB (Conv+BN), NO ReLU")
    print("Question: Does this cause problems?")
    print("="*70)
    
    # Run all tests
    test_gradient_flow()
    test_feature_diversity()
    test_with_vs_without_relu()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
Based on the tests above:

If gradients vanish OR features collapse:
  → Paper may have a typo, c2-c5 should use CBR (with ReLU)
  → Or BatchNorm alone provides enough non-linearity

If gradients flow well AND features are diverse:
  → Paper is correct, CB (no ReLU) is intentional
  → This is a valid design choice (similar to ResNet v2)

Common architectures WITHOUT ReLU in every block:
  - ResNet v2: Conv→BN→ReLU order
  - Pre-activation ResNet
  - Some U-Net variants

The skip connections (c3→p4, c4→p5) add back gradients,
so vanishing gradients may not be a problem even without ReLU.
""")


if __name__ == "__main__":
    main()
