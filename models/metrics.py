#!/usr/bin/env python3
"""
Trajectory Prediction Metrics

Implements standard metrics for trajectory prediction evaluation:
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- minADE (minimum ADE over K samples)
- minFDE (minimum FDE over K samples)
- HitRate@k (threshold-based success metric)
- Hausdorff Distance (shape similarity)
- maxADE (maximum displacement at any point)

Reference: Based on trajectory-prediction reference implementation
"""

import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def l2_distance(traj_pred, traj_gt, return_all=False):
    """
    Compute L2 distance between predicted and GT trajectories.
    
    Args:
        traj_pred: [N, 2] or [B, N, 2] - Predicted trajectory(s)
        traj_gt: [N, 2] or [B, N, 2] - Ground truth trajectory(s)
        return_all: If True, return all point-wise distances
        
    Returns:
        dict with:
            - ADE: Average Displacement Error (mean of L2 distances)
            - FDE: Final Displacement Error (L2 distance at last waypoint)
            - MAX: Maximum displacement at any point
            - distances: All L2 distances [N] or [B, N] (if return_all=True)
    """
    # Ensure tensors
    if isinstance(traj_pred, np.ndarray):
        traj_pred = torch.from_numpy(traj_pred)
    if isinstance(traj_gt, np.ndarray):
        traj_gt = torch.from_numpy(traj_gt)
    
    # Compute L2 distance at each waypoint
    distances = torch.norm(traj_pred - traj_gt, dim=-1)  # [N] or [B, N]
    
    result = {
        'ADE': distances.mean(dim=-1),  # Mean over waypoints
        'FDE': distances[..., -1],      # Last waypoint
        'MAX': distances.max(dim=-1)[0] if distances.dim() > 0 else distances.max()
    }
    
    if return_all:
        result['distances'] = distances
    
    return result


def compute_ade(traj_pred, traj_gt):
    """
    ADE: Average Displacement Error
    Mean L2 distance across all waypoints.
    
    Args:
        traj_pred: [B, N, 2] or [N, 2]
        traj_gt: [B, N, 2] or [N, 2]
        
    Returns:
        ADE: [B] or scalar
    """
    result = l2_distance(traj_pred, traj_gt)
    return result['ADE']


def compute_fde(traj_pred, traj_gt):
    """
    FDE: Final Displacement Error
    L2 distance at the final waypoint.
    
    Args:
        traj_pred: [B, N, 2] or [N, 2]
        traj_gt: [B, N, 2] or [N, 2]
        
    Returns:
        FDE: [B] or scalar
    """
    result = l2_distance(traj_pred, traj_gt)
    return result['FDE']


def compute_minADE(pred_trajectories, gt_trajectory, return_best_idx=False):
    """
    minADE: Minimum ADE over K trajectory samples.
    
    For multi-modal predictions, computes the minimum ADE across K samples.
    
    Args:
        pred_trajectories: [B, K, N, 2] - K predicted trajectories
        gt_trajectory: [B, N, 2] - Ground truth trajectory
        return_best_idx: If True, also return index of best prediction
        
    Returns:
        minADE: [B] - Minimum ADE for each batch sample
        best_idx: [B] - Index of best prediction (if return_best_idx=True)
    """
    if isinstance(pred_trajectories, np.ndarray):
        pred_trajectories = torch.from_numpy(pred_trajectories)
    if isinstance(gt_trajectory, np.ndarray):
        gt_trajectory = torch.from_numpy(gt_trajectory)
    
    B, K, N, _ = pred_trajectories.shape
    
    # Expand GT for broadcasting: [B, N, 2] -> [B, 1, N, 2]
    gt_expanded = gt_trajectory.unsqueeze(1)
    
    # Compute L2 distance: [B, K, N]
    distances = torch.norm(pred_trajectories - gt_expanded, dim=-1)
    
    # ADE: mean over waypoints -> [B, K]
    ade = distances.mean(dim=-1)
    
    # min over K samples -> [B]
    minADE, best_idx = ade.min(dim=-1)
    
    if return_best_idx:
        return minADE, best_idx
    return minADE


def compute_minFDE(pred_trajectories, gt_trajectory, return_best_idx=False):
    """
    minFDE: Minimum FDE over K trajectory samples.
    
    Args:
        pred_trajectories: [B, K, N, 2]
        gt_trajectory: [B, N, 2]
        return_best_idx: If True, also return index
        
    Returns:
        minFDE: [B]
        best_idx: [B] (optional)
    """
    if isinstance(pred_trajectories, np.ndarray):
        pred_trajectories = torch.from_numpy(pred_trajectories)
    if isinstance(gt_trajectory, np.ndarray):
        gt_trajectory = torch.from_numpy(gt_trajectory)
    
    # Expand GT
    gt_expanded = gt_trajectory.unsqueeze(1)
    
    # Compute L2 distance: [B, K, N]
    distances = torch.norm(pred_trajectories - gt_expanded, dim=-1)
    
    # FDE: last waypoint -> [B, K]
    fde = distances[..., -1]
    
    # min over K
    minFDE, best_idx = fde.min(dim=-1)
    
    if return_best_idx:
        return minFDE, best_idx
    return minFDE


def compute_hit_rate(pred_trajectories, gt_trajectory, threshold=2.0, mode='max'):
    """
    HitRate@K: Success rate based on displacement threshold.
    
    From reference: HitRate = 1 if MAX displacement < threshold, else 0
    Can also use ADE or FDE as criteria.
    
    Args:
        pred_trajectories: [B, K, N, 2] - K predicted trajectories
        gt_trajectory: [B, N, 2] - Ground truth
        threshold: Distance threshold in meters (default 2.0m)
        mode: 'max' (MAX displacement), 'ade', or 'fde'
        
    Returns:
        hit_rate: [B] - 1 if within threshold, 0 otherwise
        best_metric: [B] - The actual metric value for best prediction
    """
    if isinstance(pred_trajectories, np.ndarray):
        pred_trajectories = torch.from_numpy(pred_trajectories)
    if isinstance(gt_trajectory, np.ndarray):
        gt_trajectory = torch.from_numpy(gt_trajectory)
    
    # Expand GT
    gt_expanded = gt_trajectory.unsqueeze(1)
    
    # Compute distances
    distances = torch.norm(pred_trajectories - gt_expanded, dim=-1)  # [B, K, N]
    
    if mode == 'max':
        # MAX displacement for each prediction
        metric_values = distances.max(dim=-1)[0]  # [B, K]
    elif mode == 'ade':
        metric_values = distances.mean(dim=-1)  # [B, K]
    elif mode == 'fde':
        metric_values = distances[..., -1]  # [B, K]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Best metric (minimum) across K samples
    best_metric, _ = metric_values.min(dim=-1)  # [B]
    
    # Hit rate: 1 if within threshold
    hit_rate = (best_metric < threshold).float()  # [B]
    
    return hit_rate, best_metric


def compute_maxADE(pred_trajectories, gt_trajectory):
    """
    maxADE: Maximum displacement at any point (minimum across K samples).
    
    Args:
        pred_trajectories: [B, K, N, 2]
        gt_trajectory: [B, N, 2]
        
    Returns:
        maxADE: [B] - Minimum max displacement across K samples
    """
    if isinstance(pred_trajectories, np.ndarray):
        pred_trajectories = torch.from_numpy(pred_trajectories)
    if isinstance(gt_trajectory, np.ndarray):
        gt_trajectory = torch.from_numpy(gt_trajectory)
    
    gt_expanded = gt_trajectory.unsqueeze(1)
    distances = torch.norm(pred_trajectories - gt_expanded, dim=-1)  # [B, K, N]
    
    # Max displacement for each prediction
    max_disp = distances.max(dim=-1)[0]  # [B, K]
    
    # Min across K samples
    min_maxADE, _ = max_disp.min(dim=-1)
    
    return min_maxADE


def hausdorff_distance(traj1, traj2):
    """
    Hausdorff Distance between two trajectories.
    
    Computes the double-sided Hausdorff distance:
    H(A, B) = max(h(A, B), h(B, A))
    where h(A, B) = max_{a in A} min_{b in B} ||a - b||
    
    Args:
        traj1: [N, 2] or [B, N, 2] - First trajectory
        traj2: [N, 2] or [B, N, 2] - Second trajectory
        
    Returns:
        hd: scalar or [B] - Hausdorff distance
    """
    if isinstance(traj1, torch.Tensor):
        traj1 = traj1.cpu().numpy()
    if isinstance(traj2, torch.Tensor):
        traj2 = traj2.cpu().numpy()
    
    # Handle batch dimension
    if traj1.ndim == 3:
        # Batch of trajectories
        B = traj1.shape[0]
        hd_values = []
        for i in range(B):
            d1 = directed_hausdorff(traj1[i], traj2[i])[0]
            d2 = directed_hausdorff(traj2[i], traj1[i])[0]
            hd_values.append(max(d1, d2))
        return np.array(hd_values)
    else:
        # Single trajectory
        d1 = directed_hausdorff(traj1, traj2)[0]
        d2 = directed_hausdorff(traj2, traj1)[0]
        return max(d1, d2)


def compute_trajectory_metrics(pred_trajectories, gt_trajectory, threshold=2.0):
    """
    Compute all standard trajectory prediction metrics at once.
    
    Args:
        pred_trajectories: [B, K, N, 2] - K predicted trajectories
        gt_trajectory: [B, N, 2] - Ground truth
        threshold: HitRate threshold in meters
        
    Returns:
        dict with all metrics:
            - minADE: [B]
            - minFDE: [B]
            - hit_rate: [B]
            - maxADE: [B]
            - hausdorff: [B] (if B=1, scalar)
    """
    metrics = {}
    
    # minADE and minFDE
    metrics['minADE'], best_idx = compute_minADE(pred_trajectories, gt_trajectory, return_best_idx=True)
    metrics['minFDE'], _ = compute_minFDE(pred_trajectories, gt_trajectory, return_best_idx=True)
    
    # HitRate
    hit_rate, best_max = compute_hit_rate(pred_trajectories, gt_trajectory, threshold=threshold, mode='max')
    metrics['hit_rate'] = hit_rate
    metrics['maxADE'] = best_max  # The max displacement for best prediction
    
    # Hausdorff Distance (on best predictions)
    B = pred_trajectories.shape[0]
    if isinstance(pred_trajectories, torch.Tensor):
        pred_np = pred_trajectories.cpu().numpy()
    else:
        pred_np = pred_trajectories
    if isinstance(gt_trajectory, torch.Tensor):
        gt_np = gt_trajectory.cpu().numpy()
    else:
        gt_np = gt_trajectory
    
    # Extract best predictions
    best_preds = np.array([pred_np[i, best_idx[i].item()] for i in range(B)])
    
    hd_values = []
    for i in range(B):
        d1 = directed_hausdorff(best_preds[i], gt_np[i])[0]
        d2 = directed_hausdorff(gt_np[i], best_preds[i])[0]
        hd_values.append(max(d1, d2))
    
    metrics['hausdorff'] = np.array(hd_values)
    
    return metrics


def aggregate_metrics(metrics_dict):
    """
    Aggregate metrics across batch/samples.
    
    Args:
        metrics_dict: Dict with metric arrays
        
    Returns:
        dict with mean values
    """
    aggregated = {}
    for key, value in metrics_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        aggregated[key] = float(np.mean(value))
        aggregated[key + '_std'] = float(np.std(value))
    return aggregated


class MetricsLogger:
    """
    Logger for accumulating metrics over multiple batches.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.metrics_sum = {}
        self.metrics_count = {}
    
    def update(self, metrics_dict, count=1):
        """Update with new metrics."""
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            
            if key not in self.metrics_sum:
                self.metrics_sum[key] = 0.0
                self.metrics_count[key] = 0
            
            self.metrics_sum[key] += np.sum(value)
            self.metrics_count[key] += count if np.ndim(value) > 0 else count
    
    def get_averages(self):
        """Get average metrics."""
        averages = {}
        for key in self.metrics_sum:
            averages[key] = self.metrics_sum[key] / max(self.metrics_count[key], 1)
        return averages
    
    def __str__(self):
        """String representation of averages."""
        avgs = self.get_averages()
        parts = []
        for key in ['minADE', 'minFDE', 'hit_rate', 'maxADE', 'hausdorff']:
            if key in avgs:
                parts.append(f"{key}: {avgs[key]:.3f}")
        return " | ".join(parts)


if __name__ == "__main__":
    # Test all metrics
    print("Testing Trajectory Metrics...")
    print("=" * 60)
    
    # Create synthetic data
    B, K, N = 4, 5, 8
    
    # GT: straight line forward
    gt = torch.zeros(B, N, 2)
    gt[:, :, 0] = torch.linspace(0, 16, N)  # X: 0 to 16m
    
    # Predictions: GT + noise
    pred = gt.unsqueeze(1) + torch.randn(B, K, N, 2) * 2.0  # Add noise
    
    print(f"\nTest data:")
    print(f"  Predictions: {pred.shape} (Batch={B}, K={K}, Waypoints={N})")
    print(f"  Ground truth: {gt.shape}")
    
    # Test individual metrics
    print("\n1. Testing ADE/FDE...")
    single_pred = pred[:, 0, :, :]  # Take first prediction
    result = l2_distance(single_pred, gt, return_all=True)
    print(f"   ADE: {result['ADE'].mean().item():.3f}m")
    print(f"   FDE: {result['FDE'].mean().item():.3f}m")
    print(f"   MAX: {result['MAX'].mean().item():.3f}m")
    
    print("\n2. Testing minADE/minFDE...")
    minADE, best_idx = compute_minADE(pred, gt, return_best_idx=True)
    minFDE, _ = compute_minFDE(pred, gt, return_best_idx=True)
    print(f"   minADE: {minADE.mean().item():.3f}m ± {minADE.std().item():.3f}m")
    print(f"   minFDE: {minFDE.mean().item():.3f}m ± {minFDE.std().item():.3f}m")
    print(f"   Best indices: {best_idx.tolist()}")
    
    print("\n3. Testing HitRate@2m...")
    hit_rate, max_disp = compute_hit_rate(pred, gt, threshold=2.0)
    print(f"   HitRate@2m: {hit_rate.mean().item():.3f}")
    print(f"   Max displacement (best): {max_disp.mean().item():.3f}m")
    
    print("\n4. Testing Hausdorff Distance...")
    hd = hausdorff_distance(pred[:, 0, :, :], gt)  # Compare first prediction
    print(f"   Hausdorff: {hd.mean():.3f}m")
    
    print("\n5. Testing compute_trajectory_metrics (all at once)...")
    all_metrics = compute_trajectory_metrics(pred, gt, threshold=2.0)
    for key, value in all_metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        print(f"   {key}: {np.mean(value):.3f}")
    
    print("\n6. Testing MetricsLogger...")
    logger = MetricsLogger()
    for i in range(3):
        batch_pred = pred[:2, :, :, :]
        batch_gt = gt[:2, :, :]
        batch_metrics = compute_trajectory_metrics(batch_pred, batch_gt)
        logger.update(batch_metrics, count=2)
    
    print(f"   Logger averages: {logger}")
    
    print("\n" + "=" * 60)
    print("✓ All metrics working!")
