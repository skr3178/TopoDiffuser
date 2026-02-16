"""
Metrics to evaluate BEV rasterization effectiveness.

Tests information preservation, coverage, and quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
import os

sys.path.insert(0, '/media/skr/storage/self_driving/TopoDiffuser/models')
from models.bev_rasterization import BEVRasterizer, load_kitti_lidar


def point_coverage_metric(points, bev_density, rasterizer):
    """
    Measure percentage of points preserved in BEV.
    
    Returns: coverage_ratio (0-1), points_lost
    """
    H, W = bev_density.shape
    
    # Map points to grid (only XY, no Z filtering)
    px = ((points[:, 0] - rasterizer.x_range[0]) / rasterizer.resolution).astype(int)
    py = ((points[:, 1] - rasterizer.y_range[0]) / rasterizer.resolution).astype(int)
    
    # Filter valid XY only
    valid_mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    points_valid = points[valid_mask]
    px_valid = px[valid_mask]
    py_valid = py[valid_mask]
    
    if len(points_valid) == 0:
        return 0.0, len(points)
    
    # Check which map to occupied cells
    density_at_points = bev_density[py_valid, px_valid]
    occupied_mask = density_at_points > 0
    
    # Points covered / Total points (including those outside XY bounds)
    coverage_ratio = occupied_mask.sum() / len(points)
    points_lost = len(points) - occupied_mask.sum()
    
    return coverage_ratio, points_lost


def height_accuracy_metric(points, bev_height, rasterizer):
    """
    Measure if elevated regions in raw data appear as elevated in BEV.
    
    This checks structural preservation, not point-wise correlation.
    (Point-wise correlation is flawed because BEV uses max pooling)
    
    Returns: region_correlation (0-1), elevation_preservation_ratio
    """
    H, W = bev_height.shape
    
    # Map points to grid (XY only)
    px = ((points[:, 0] - rasterizer.x_range[0]) / rasterizer.resolution).astype(int)
    py = ((points[:, 1] - rasterizer.y_range[0]) / rasterizer.resolution).astype(int)
    
    # Filter valid XY
    valid_mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    points_valid = points[valid_mask]
    px_valid = px[valid_mask]
    py_valid = py[valid_mask]
    
    if len(points_valid) < 100:
        return 0.0, 0.0
    
    # Create a height map from points (max Z per cell, like BEV)
    point_height_map = np.zeros((H, W), dtype=np.float32)
    z_vals = points_valid[:, 2]
    
    for i in range(len(points_valid)):
        u, v = px_valid[i], py_valid[i]
        if z_vals[i] > point_height_map[v, u]:
            point_height_map[v, u] = z_vals[i]
    
    # Normalize both maps
    if point_height_map.max() > point_height_map.min():
        point_height_map = (point_height_map - point_height_map.min()) / (point_height_map.max() - point_height_map.min())
    
    # Compare only occupied cells
    occupied_mask = (bev_height > 0) | (point_height_map > 0)
    
    if occupied_mask.sum() < 100:
        return 0.0, 0.0
    
    # Calculate correlation between the two height maps
    bev_flat = bev_height[occupied_mask]
    point_flat = point_height_map[occupied_mask]
    
    corr, _ = pearsonr(bev_flat, point_flat)
    
    # Elevation preservation: % of elevated regions preserved
    elevated_threshold = 0.3  # Normalized height
    point_elevated = point_height_map > elevated_threshold
    bev_elevated = bev_height > elevated_threshold
    
    if point_elevated.sum() == 0:
        elevation_preservation = 1.0  # Nothing to preserve
    else:
        # How many of the point-elevated cells are also elevated in BEV?
        both_elevated = point_elevated & bev_elevated
        elevation_preservation = both_elevated.sum() / point_elevated.sum()
    
    return max(0, corr), elevation_preservation


def cell_utilization_metric(bev_density):
    """
    Measure what % of cells are actually used.
    
    Too low = inefficient (mostly empty)
    Too high = crowded (loss of information)
    """
    occupied = (bev_density > 0).sum()
    total = bev_density.size
    utilization = occupied / total
    
    return utilization, occupied, total


def object_preservation_metric(points, bev_height, rasterizer, object_z_threshold=0.5):
    """
    Check if elevated objects (obstacles) are preserved.
    
    FIX: Use relative threshold based on observed height distribution
    Counts points with Z > threshold that map to non-zero BEV cells.
    """
    H, W = bev_height.shape
    
    # Calculate relative threshold based on observed ground height
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    
    # Dynamic threshold: ground level + offset
    # Ground is typically at minimum Z in ego frame
    ground_level = z_min + 0.2  # Slightly above min (noise floor)
    relative_threshold = ground_level + object_z_threshold
    
    # Find elevated points (above ground + threshold)
    elevated_mask = points[:, 2] > relative_threshold
    elevated_points = points[elevated_mask]
    
    if len(elevated_points) == 0:
        return 1.0, 0, 0  # No objects to preserve
    
    # Map to grid
    px = ((elevated_points[:, 0] - rasterizer.x_range[0]) / rasterizer.resolution).astype(int)
    py = ((elevated_points[:, 1] - rasterizer.y_range[0]) / rasterizer.resolution).astype(int)
    
    valid_mask = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    px_valid = px[valid_mask]
    py_valid = py[valid_mask]
    
    if len(px_valid) == 0:
        return 0.0, len(elevated_points), 0
    
    # Check preservation (point maps to cell with any height > 0)
    height_at_points = bev_height[py_valid, px_valid]
    preserved = (height_at_points > 0).sum()
    
    preservation_ratio = preserved / len(elevated_points)
    
    return preservation_ratio, len(elevated_points), preserved


def evaluate_rasterization_quality(lidar_path, frame_idx=0):
    """
    Comprehensive evaluation of rasterization quality.
    """
    print(f"\n{'='*70}")
    print(f"Rasterization Quality Metrics - Frame {frame_idx}")
    print(f"{'='*70}")
    
    # Load data
    points = load_kitti_lidar(lidar_path)
    print(f"\nInput: {len(points):,} LiDAR points")
    
    # Rasterize
    rasterizer = BEVRasterizer()
    bev = rasterizer.rasterize_lidar(points)
    
    print(f"Output: {bev.shape} BEV tensor")
    
    # 1. Point Coverage
    print(f"\n{'-'*70}")
    print("1. POINT COVERAGE")
    print(f"{'-'*70}")
    coverage, lost = point_coverage_metric(points, bev[2], rasterizer)
    print(f"  Coverage Ratio: {coverage*100:.2f}%")
    print(f"  Points Preserved: {len(points) - lost:,} / {len(points):,}")
    print(f"  Points Lost: {lost:,}")
    print(f"  Status: {'✓ GOOD' if coverage > 0.95 else '✗ POOR'} (target: >95%)")
    
    # 2. Height Accuracy
    print(f"\n{'-'*70}")
    print("2. HEIGHT STRUCTURE PRESERVATION")
    print(f"{'-'*70}")
    corr, elev_preservation = height_accuracy_metric(points, bev[0], rasterizer)
    print(f"  Height Map Correlation: {corr:.4f}")
    print(f"  Elevation Preservation: {elev_preservation*100:.1f}%")
    print(f"  Status: {'✓ GOOD' if corr > 0.8 and elev_preservation > 0.8 else '⚠ FAIR' if corr > 0.6 else '✗ POOR'} (target: >0.8)")
    
    # 3. Cell Utilization
    print(f"\n{'-'*70}")
    print("3. CELL UTILIZATION")
    print(f"{'-'*70}")
    util, occupied, total = cell_utilization_metric(bev[2])
    print(f"  Utilization: {util*100:.2f}% ({occupied:,} / {total:,} cells)")
    print(f"  Status: {'✓ OPTIMAL' if 0.05 < util < 0.20 else '⚠ SUBOPTIMAL'} (target: 5-20%)")
    
    # 4. Object Preservation
    print(f"\n{'-'*70}")
    print("4. OBJECT PRESERVATION (Z > 0.5m)")
    print(f"{'-'*70}")
    obj_pres, total_obj, preserved_obj = object_preservation_metric(
        points, bev[0], rasterizer, object_z_threshold=0.5
    )
    print(f"  Preservation Ratio: {obj_pres*100:.2f}%")
    print(f"  Elevated Points: {total_obj:,}")
    print(f"  Preserved: {preserved_obj:,}")
    print(f"  Status: {'✓ GOOD' if obj_pres > 0.9 else '⚠ FAIR' if obj_pres > 0.8 else '✗ POOR'} (target: >90%)")
    
    # 5. Channel Statistics
    print(f"\n{'-'*70}")
    print("5. CHANNEL STATISTICS")
    print(f"{'-'*70}")
    for i, name in enumerate(['Height', 'Intensity', 'Density']):
        ch = bev[i]
        print(f"  {name}:")
        print(f"    Range: [{ch.min():.4f}, {ch.max():.4f}]")
        print(f"    Mean: {ch.mean():.4f}, Std: {ch.std():.4f}")
        print(f"    Non-zero: {(ch > 0).sum():,} pixels")
    
    # 6. Information Loss Analysis
    print(f"\n{'-'*70}")
    print("6. INFORMATION LOSS ANALYSIS")
    print(f"{'-'*70}")
    
    # Before: point-wise information
    unique_z_values = len(np.unique(points[:, 2]))
    unique_intensities = len(np.unique(points[:, 3]))
    
    # After: cell-wise information  
    unique_bev_heights = len(np.unique(bev[0][bev[0] > 0]))
    unique_bev_intensities = len(np.unique(bev[1][bev[1] > 0]))
    
    print(f"  Height Values: {unique_z_values:,} (raw) → {unique_bev_heights:,} (BEV)")
    print(f"    Compression Ratio: {unique_z_values / max(unique_bev_heights, 1):.1f}x")
    
    print(f"  Intensity Values: {unique_intensities:,} (raw) → {unique_bev_intensities:,} (BEV)")
    print(f"    Compression Ratio: {unique_intensities / max(unique_bev_intensities, 1):.1f}x")
    
    # Overall score
    print(f"\n{'='*70}")
    print("OVERALL QUALITY SCORE")
    print(f"{'='*70}")
    
    # Weighted average of metrics
    score_coverage = min(coverage / 0.95, 1.0) * 30  # 30% weight
    score_height = max(0, corr) * 15  # 15% weight (structural correlation)
    score_elevation = max(0, elev_preservation) * 15  # 15% weight (elevated regions)
    score_util = (1 - abs(util - 0.10) / 0.10) * 20 if 0 < util < 0.20 else 0  # 20% weight
    score_object = min(obj_pres / 0.90, 1.0) * 20  # 20% weight
    
    total_score = score_coverage + score_height + score_elevation + score_util + score_object
    
    print(f"  Coverage Score: {score_coverage:.1f}/30")
    print(f"  Height Structure Score: {score_height:.1f}/15")
    print(f"  Elevation Preservation Score: {score_elevation:.1f}/15")
    print(f"  Utilization Score: {score_util:.1f}/20")
    print(f"  Object Preservation Score: {score_object:.1f}/20")
    print(f"\n  TOTAL SCORE: {total_score:.1f}/100")
    
    if total_score >= 90:
        quality = "EXCELLENT"
    elif total_score >= 80:
        quality = "GOOD"
    elif total_score >= 60:
        quality = "FAIR"
    else:
        quality = "POOR - Needs Improvement"
    
    print(f"  QUALITY RATING: {quality}")
    
    return {
        'coverage': coverage,
        'height_correlation': corr,
        'utilization': util,
        'object_preservation': obj_pres,
        'total_score': total_score,
        'quality': quality
    }


def compare_resolutions(lidar_path):
    """
    Compare rasterization at different resolutions.
    """
    print(f"\n{'='*70}")
    print("Resolution Comparison")
    print(f"{'='*70}")
    
    points = load_kitti_lidar(lidar_path)
    resolutions = [0.05, 0.10, 0.15, 0.20]  # meters per pixel
    
    results = []
    for res in resolutions:
        config = {'resolution': res, 'grid_size': (int(40/res), int(40/res))}
        rasterizer = BEVRasterizer(config)
        bev = rasterizer.rasterize_lidar(points)
        
        coverage, _ = point_coverage_metric(points, bev[2], rasterizer)
        corr, _ = height_accuracy_metric(points, bev[0], rasterizer)
        util, occ, tot = cell_utilization_metric(bev[2])
        
        results.append({
            'res': res,
            'grid': bev.shape,
            'coverage': coverage,
            'height_corr': corr,
            'util': util
        })
        
        print(f"\nResolution: {res:.2f}m/px | Grid: {bev.shape}")
        print(f"  Coverage: {coverage*100:.1f}% | Height Corr: {corr:.3f} | Util: {util*100:.1f}%")
    
    # Recommendation
    print(f"\n{'-'*70}")
    print("RECOMMENDATION:")
    # Balance coverage vs computational cost
    best = max(results, key=lambda x: x['coverage'] * x['height_corr'])
    print(f"  Optimal Resolution: {best['res']:.2f}m/px")
    print(f"  Grid Size: {best['grid']}")
    
    return results


if __name__ == "__main__":
    # Test on real KITTI data
    kitti_path = '/media/skr/storage/self_driving/TopoDiffuser/data/kitti/sequences/00/velodyne/000500.bin'
    
    if not os.path.exists(kitti_path):
        print(f"KITTI data not found at {kitti_path}")
        print("Using synthetic data for testing...")
        
        # Create synthetic data
        np.random.seed(42)
        n_points = 50000
        points = np.zeros((n_points, 4), dtype=np.float32)
        points[:, 0] = np.random.uniform(-15, 15, n_points)
        points[:, 1] = np.random.uniform(-5, 25, n_points)
        points[:, 2] = np.random.uniform(-2, 2, n_points)
        points[:, 3] = np.random.uniform(0, 255, n_points)
        
        # Save temporarily
        import tempfile
        kitti_path = tempfile.mktemp(suffix='.bin')
        points.tofile(kitti_path)
    
    # Run evaluation
    metrics = evaluate_rasterization_quality(kitti_path, frame_idx=500)
    
    # Compare resolutions
    print("\n" + "="*70)
    print("RESOLUTION SWEEP (for optimization)")
    print("="*70)
    res_results = compare_resolutions(kitti_path)
    
    print("\n" + "="*70)
    print("Rasterization Quality Evaluation Complete!")
    print("="*70)
