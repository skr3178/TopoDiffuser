"""
Comprehensive Test Suite for Block 2: Multimodal Encoder

Tests to verify encoder correctness before proceeding to Block 3.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import time

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from models.encoder import build_encoder


def test_shape_consistency():
    """Test 1: Output shapes are correct for various inputs."""
    print("="*70)
    print("TEST 1: Shape Consistency")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.eval()
    
    test_cases = [
        (1, 5, 300, 400),   # Single sample
        (8, 5, 300, 400),   # Batch of 8 (paper)
        (16, 5, 300, 400),  # Larger batch
        (4, 3, 300, 400),   # LiDAR-only (will skip - different channels)
    ]
    
    all_passed = True
    for batch_size, channels, h, w in test_cases:
        # Skip LiDAR-only case if encoder expects different channels
        if channels != encoder.input_channels:
            print(f"⊘ Input: [{batch_size}, {channels}, {h}, {w}] → Skipped (encoder expects {encoder.input_channels} channels)")
            continue
        
        x = torch.randn(batch_size, channels, h, w)
        
        try:
            conditioning, road_seg = encoder(x)
            
            expected_cond = (batch_size, 512)
            expected_seg = (batch_size, 1, 37, 50)
            
            cond_ok = conditioning.shape == expected_cond
            seg_ok = road_seg.shape == expected_seg
            
            status = "✓" if cond_ok and seg_ok else "✗"
            print(f"{status} Input: {x.shape} → Cond: {conditioning.shape}, Seg: {road_seg.shape}")
            
            if not (cond_ok and seg_ok):
                all_passed = False
        except Exception as e:
            print(f"✗ Input: {x.shape} → Error: {e}")
            all_passed = False
    
    return all_passed


def test_gradient_flow():
    """Test 2: Gradients flow properly through all layers."""
    print("\n" + "="*70)
    print("TEST 2: Gradient Flow")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.train()
    
    x = torch.randn(4, 5, 300, 400, requires_grad=True)
    conditioning, road_seg = encoder(x)
    
    # Loss on both outputs
    loss = conditioning.sum() + road_seg.sum()
    loss.backward()
    
    # Check all layers have gradients
    layers_to_check = [
        ('c1', encoder.c1),
        ('c2', encoder.c2),
        ('c3', encoder.c3),
        ('c4', encoder.c4),
        ('c5', encoder.c5),
        ('p5_conv1x1', encoder.p5_conv1x1),
        ('shared_backbone', encoder.shared_backbone),
        ('seg_head', encoder.seg_head),
        ('cond_fc', encoder.cond_fc),
    ]
    
    all_ok = True
    for name, module in layers_to_check:
        has_grad = False
        grad_mag = 0.0
        
        if hasattr(module, 'conv') and module.conv.weight.grad is not None:
            has_grad = True
            grad_mag = module.conv.weight.grad.abs().mean().item()
        elif hasattr(module, 'weight') and module.weight.grad is not None:
            has_grad = True
            grad_mag = module.weight.grad.abs().mean().item()
        elif isinstance(module, nn.Sequential):
            # Check first layer with parameters
            for layer in module:
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    has_grad = True
                    grad_mag = layer.weight.grad.abs().mean().item()
                    break
        
        status = "✓" if has_grad and grad_mag > 1e-10 else "✗"
        print(f"{status} {name:20s}: grad_mag={grad_mag:.6f}")
        
        if not (has_grad and grad_mag > 1e-10):
            all_ok = False
    
    return all_ok


def test_shared_backbone():
    """Test 3: Both heads use the same shared features."""
    print("\n" + "="*70)
    print("TEST 3: Shared Backbone Verification")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.eval()
    
    # Hook to capture intermediate features
    shared_features = None
    def hook_fn(module, input, output):
        nonlocal shared_features
        shared_features = output
    
    encoder.shared_backbone.register_forward_hook(hook_fn)
    
    x = torch.randn(2, 5, 300, 400)
    conditioning, road_seg = encoder(x)
    
    print(f"Shared features shape: {shared_features.shape}")
    print(f"Shared features mean: {shared_features.mean():.4f}")
    print(f"Shared features std: {shared_features.std():.4f}")
    print(f"Value range: [{shared_features.min():.4f}, {shared_features.max():.4f}]")
    
    # Check sigmoid was applied (values in [0, 1])
    in_range = (shared_features >= 0).all() and (shared_features <= 1).all()
    print(f"Sigmoid applied (values in [0,1]): {'✓ Yes' if in_range else '✗ No'}")
    
    return in_range


def test_output_validity():
    """Test 4: Output values are valid (no NaN, Inf, etc.)."""
    print("\n" + "="*70)
    print("TEST 4: Output Validity")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.eval()
    
    x = torch.randn(8, 5, 300, 400)
    conditioning, road_seg = encoder(x)
    
    # Check for NaN/Inf
    cond_valid = torch.isfinite(conditioning).all()
    seg_valid = torch.isfinite(road_seg).all()
    
    print(f"Conditioning has NaN: {torch.isnan(conditioning).any().item()}")
    print(f"Conditioning has Inf: {torch.isinf(conditioning).any().item()}")
    print(f"Road seg has NaN: {torch.isnan(road_seg).any().item()}")
    print(f"Road seg has Inf: {torch.isinf(road_seg).any().item()}")
    
    # Check ranges
    print(f"\nConditioning range: [{conditioning.min():.4f}, {conditioning.max():.4f}]")
    print(f"Road seg range: [{road_seg.min():.4f}, {road_seg.max():.4f}]")
    print(f"Road seg is probability [0,1]: {(road_seg >= 0).all() and (road_seg <= 1).all()}")
    
    return cond_valid and seg_valid


def test_determinism():
    """Test 5: Same input produces same output (deterministic)."""
    print("\n" + "="*70)
    print("TEST 5: Determinism")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.eval()
    
    x = torch.randn(4, 5, 300, 400)
    
    # Run twice
    conditioning1, road_seg1 = encoder(x)
    conditioning2, road_seg2 = encoder(x)
    
    cond_same = torch.allclose(conditioning1, conditioning2)
    seg_same = torch.allclose(road_seg1, road_seg2)
    
    print(f"Conditioning identical: {'✓ Yes' if cond_same else '✗ No'}")
    print(f"Road seg identical: {'✓ Yes' if seg_same else '✗ No'}")
    
    if cond_same and seg_same:
        print(f"Max difference: {abs(conditioning1 - conditioning2).max():.10f}")
    
    return cond_same and seg_same


def test_batch_independence():
    """Test 6: Samples in batch are processed independently."""
    print("\n" + "="*70)
    print("TEST 6: Batch Independence")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.eval()
    
    # Create two identical samples
    x1 = torch.randn(1, 5, 300, 400)
    x2 = torch.randn(1, 5, 300, 400)
    
    # Process individually
    cond1, seg1 = encoder(x1)
    cond2, seg2 = encoder(x2)
    
    # Process together
    x_batch = torch.cat([x1, x2], dim=0)
    cond_batch, seg_batch = encoder(x_batch)
    
    # Check individual matches batch (with tolerance for numerical precision)
    cond_match = torch.allclose(cond1, cond_batch[0:1], atol=1e-6) and torch.allclose(cond2, cond_batch[1:2], atol=1e-6)
    seg_match = torch.allclose(seg1, seg_batch[0:1], atol=1e-6) and torch.allclose(seg2, seg_batch[1:2], atol=1e-6)
    
    print(f"Individual == Batch: {'✓ Yes' if cond_match and seg_match else '✗ No'}")
    
    return cond_match and seg_match


def test_memory_usage():
    """Test 7: Memory usage is reasonable."""
    print("\n" + "="*70)
    print("TEST 7: Memory Usage")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.eval()
    
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        x = torch.randn(8, 5, 300, 400).cuda()
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = encoder(x)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
        
        # Check if reasonable (should be < 1GB for batch=8)
        reasonable = peak_memory < 1024
        print(f"Memory usage reasonable (< 1GB): {'✓ Yes' if reasonable else '✗ No'}")
        return reasonable
    else:
        print("CUDA not available, skipping GPU memory test")
        return True


def test_with_real_data():
    """Test 8: Works with real KITTI data."""
    print("\n" + "="*70)
    print("TEST 8: Real Data Integration")
    print("="*70)
    
    try:
        from bev_rasterization import BEVRasterizer, load_kitti_lidar
        import os
        
        # Try to load real data
        lidar_path = '/media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/00/velodyne/000100.bin'
        
        if os.path.exists(lidar_path):
            points = load_kitti_lidar(lidar_path)
            print(f"Loaded {len(points):,} LiDAR points")
            
            # Rasterize
            rasterizer = BEVRasterizer()
            bev = rasterizer.rasterize_lidar(points)
            print(f"BEV shape: {bev.shape}")
            
            # Encode
            encoder = build_encoder(input_channels=3, conditioning_dim=512)
            encoder.eval()
            
            bev_tensor = torch.from_numpy(bev).unsqueeze(0).float()
            conditioning, road_seg = encoder(bev_tensor)
            
            print(f"Output shapes: cond={conditioning.shape}, seg={road_seg.shape}")
            print(f"Output valid: {torch.isfinite(conditioning).all() and torch.isfinite(road_seg).all()}")
            
            return True
        else:
            print(f"Real data not found at {lidar_path}")
            print("Skipping real data test")
            return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_parameter_count():
    """Test 9: Parameter count matches expectation."""
    print("\n" + "="*70)
    print("TEST 9: Parameter Count")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Should be around 2.5M
    expected_range = (2_000_000, 3_000_000)
    in_range = expected_range[0] <= total_params <= expected_range[1]
    
    print(f"Parameter count in expected range: {'✓ Yes' if in_range else '✗ No'}")
    
    return in_range


def test_inference_speed():
    """Test 10: Inference speed is acceptable."""
    print("\n" + "="*70)
    print("TEST 10: Inference Speed")
    print("="*70)
    
    encoder = build_encoder(input_channels=5, conditioning_dim=512)
    encoder.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    
    # Warmup
    x = torch.randn(8, 5, 300, 400).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = encoder(x)
    
    # Benchmark
    times = []
    for _ in range(20):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = encoder(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000
    
    print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {1000/avg_time:.1f} samples/sec (batch=8)")
    
    # Should be faster than 100ms
    fast_enough = avg_time < 100
    print(f"Speed acceptable (< 100ms): {'✓ Yes' if fast_enough else '✗ No'}")
    
    return fast_enough


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*70)
    print("COMPREHENSIVE ENCODER TEST SUITE")
    print("="*70)
    
    tests = [
        ("Shape Consistency", test_shape_consistency),
        ("Gradient Flow", test_gradient_flow),
        ("Shared Backbone", test_shared_backbone),
        ("Output Validity", test_output_validity),
        ("Determinism", test_determinism),
        ("Batch Independence", test_batch_independence),
        ("Memory Usage", test_memory_usage),
        ("Real Data Integration", test_with_real_data),
        ("Parameter Count", test_parameter_count),
        ("Inference Speed", test_inference_speed),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Encoder is ready for Block 3.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
