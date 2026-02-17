"""
Multimodal BEV Encoding Utilities with Debugging and Evaluation.

Handles:
- LiDAR point cloud to BEV rasterization (existing)
- Trajectory history to BEV binary mask with evaluation
- OSM route to BEV binary mask with GPS alignment and evaluation
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


# =============================================================================
# Coordinate Transformations
# =============================================================================

def latlon_to_utm(lat: float, lon: float, zone: int = 32) -> Tuple[float, float]:
    """
    Convert latitude/longitude to UTM coordinates.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zone: UTM zone (Karlsruhe is zone 32)
    
    Returns:
        (east, north) in meters
    """
    # WGS84 parameters
    a = 6378137.0  # Semi-major axis
    e = 0.0818191908426  # Eccentricity
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)  # Central meridian
    
    k0 = 0.9996  # Scale factor
    N = a / np.sqrt(1 - e**2 * np.sin(lat_rad)**2)
    T = np.tan(lat_rad)**2
    C = e**2 * np.cos(lat_rad)**2 / (1 - e**2)
    A = np.cos(lat_rad) * (lon_rad - lon0)
    
    # Meridional arc
    M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad -
             (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*lat_rad) +
             (15*e**4/256 + 45*e**6/1024) * np.sin(4*lat_rad) -
             (35*e**6/3072) * np.sin(6*lat_rad))
    
    east = k0 * N * (A + (1 - T + C) * A**3 / 6 + 
                     (5 - 18*T + T**2 + 72*C - 58*0.006739497) * A**5 / 120)
    east += 500000  # UTM easting offset
    
    north = k0 * (M + N * np.tan(lat_rad) * (A**2 / 2 + 
                  (5 - T + 9*C + 4*C**2) * A**4 / 24 +
                  (61 - 58*T + T**2 + 600*C - 330*0.006739497) * A**6 / 720))
    
    return east, north


def compute_gps_to_local_transform(oxts_data: np.ndarray, 
                                   poses: np.ndarray) -> Dict:
    """
    Compute transformation from GPS (UTM) to KITTI local frame.
    
    Args:
        oxts_data: [N, 30] OXTS readings (lat, lon in first 2 cols)
        poses: [N, 12] KITTI pose matrices (3x4 flattened)
    
    Returns:
        Dictionary with offset and rotation for alignment
    """
    # Extract GPS positions and convert to UTM
    lats = oxts_data[:, 0]
    lons = oxts_data[:, 1]
    
    utm_east = []
    utm_north = []
    for lat, lon in zip(lats, lons):
        e, n = latlon_to_utm(lat, lon)
        utm_east.append(e)
        utm_north.append(n)
    
    utm_east = np.array(utm_east)
    utm_north = np.array(utm_north)
    
    # Extract local positions from poses (translation part)
    local_x = poses[:, 3]   # First row, last column
    local_y = poses[:, 7]   # Second row, last column
    
    # Compute offset (difference between UTM and local at start)
    offset_east = utm_east[0] - local_x[0]
    offset_north = utm_north[0] - local_y[0]
    
    # Compute alignment error
    aligned_x = utm_east - offset_east
    aligned_y = utm_north - offset_north
    
    errors = np.sqrt((aligned_x - local_x)**2 + (aligned_y - local_y)**2)
    
    return {
        'offset_east': offset_east,
        'offset_north': offset_north,
        'utm_east': utm_east,
        'utm_north': utm_north,
        'aligned_x': aligned_x,
        'aligned_y': aligned_y,
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors)
    }


def world_to_ego(point_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Transform point from world frame to ego frame.
    
    Args:
        point_world: [2] or [3] (x, y) or (x, y, z) in world frame
        pose: [3, 4] transformation matrix [R|t]
    
    Returns:
        point_ego: [2] or [3] in ego frame
    """
    R = pose[:, :3]
    t = pose[:, 3]
    
    if len(point_world) == 2:
        point_world = np.array([point_world[0], point_world[1], 0])
    
    point_ego = R.T @ (point_world - t)
    return point_ego[:2]  # Return (x, y)


# =============================================================================
# BEV Rasterization
# =============================================================================

def lidar_to_bev(points: np.ndarray, 
                 grid_size: Tuple[int, int] = (300, 400),
                 resolution: float = 0.1,
                 x_range: Tuple[float, float] = (-20, 20),
                 y_range: Tuple[float, float] = (-10, 30),
                 z_range: Tuple[float, float] = (-2, 2)) -> np.ndarray:
    """
    Convert LiDAR point cloud to BEV representation.
    
    Args:
        points: [N, 4] array of (x, y, z, intensity)
        grid_size: (H, W) output BEV size
        resolution: meters per pixel
        x_range: (min, max) x coordinates in ego frame
        y_range: (min, max) y coordinates in ego frame
        z_range: (min, max) z coordinates (height)
    
    Returns:
        bev: [3, H, W] BEV image (height, intensity, density)
    """
    H, W = grid_size
    bev = np.zeros((3, H, W), dtype=np.float32)
    
    # Filter points by range
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1])
    )
    
    if points.shape[1] >= 4:
        mask = mask & (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1])
    
    points = points[mask]
    
    if len(points) == 0:
        return bev
    
    # Convert to pixel coordinates
    px = ((points[:, 0] - x_range[0]) / resolution).astype(np.int32)
    py = ((points[:, 1] - y_range[0]) / resolution).astype(np.int32)
    
    # Clip to grid bounds
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)
    
    # Fill BEV channels
    for i in range(len(points)):
        x, y = px[i], py[i]
        z = points[i, 2]
        
        # Height channel (max height)
        if z > bev[0, y, x]:
            bev[0, y, x] = z
        
        # Intensity channel (max intensity)
        if points.shape[1] >= 4:
            intensity = points[i, 3]
            if intensity > bev[1, y, x]:
                bev[1, y, x] = intensity
        
        # Density channel (count)
        bev[2, y, x] += 1
    
    # Normalize
    if bev[0].max() > 0:
        bev[0] = bev[0] / z_range[1]
    if bev[1].max() > 0:
        bev[1] = bev[1] / 255.0
    if bev[2].max() > 0:
        bev[2] = bev[2] / bev[2].max()
    
    return bev


def draw_line(grid: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
              width: int = 1) -> np.ndarray:
    """Draw a line on the grid using Bresenham's algorithm."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        # Draw pixel with width
        for w in range(-width//2, width//2 + 1):
            for h in range(-width//2, width//2 + 1):
                px, py = x1 + w, y1 + h
                if 0 <= px < grid.shape[1] and 0 <= py < grid.shape[0]:
                    grid[py, px] = 1.0
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return grid


def trajectory_to_bev(trajectory: np.ndarray,
                      current_pose: np.ndarray,
                      past_poses: np.ndarray,
                      grid_size: Tuple[int, int] = (300, 400),
                      resolution: float = 0.1,
                      x_range: Tuple[float, float] = (-20, 20),
                      y_range: Tuple[float, float] = (-10, 30),
                      line_width: int = 2,
                      debug: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Convert trajectory waypoints to BEV binary mask.
    
    Args:
        trajectory: [N, 2] array of (x, y) waypoints in WORLD frame
        current_pose: [3, 4] current ego pose in world frame
        past_poses: [M, 12] past poses for full trajectory
        grid_size: (H, W) output BEV size
        resolution: meters per pixel
        x_range: (min, max) x coordinates in ego frame
        y_range: (min, max) y coordinates in ego frame
        line_width: width of trajectory line in pixels
        debug: If True, return debug info
    
    Returns:
        bev: [1, H, W] binary mask
        debug_info: Dictionary with alignment metrics
    """
    H, W = grid_size
    bev = np.zeros((1, H, W), dtype=np.float32)
    
    debug_info = {
        'num_waypoints': len(trajectory),
        'waypoints_in_bounds': 0,
        'ego_frame_waypoints': [],
        'pixel_coords': []
    }
    
    if len(trajectory) < 2:
        if debug:
            return bev, debug_info
        return bev
    
    # Transform waypoints to ego frame
    ego_waypoints = []
    for wp in trajectory:
        ego_wp = world_to_ego(wp, current_pose)
        ego_waypoints.append(ego_wp)
        if debug:
            debug_info['ego_frame_waypoints'].append(ego_wp)
    
    ego_waypoints = np.array(ego_waypoints)
    
    # Convert to pixel coordinates
    def world_to_pixel(pt):
        px = int((pt[0] - x_range[0]) / resolution)
        py = int((pt[1] - y_range[0]) / resolution)
        return (px, py)
    
    pixel_coords = []
    for ego_wp in ego_waypoints:
        px, py = world_to_pixel(ego_wp)
        pixel_coords.append((px, py))
        
        # Check if within bounds
        if 0 <= px < W and 0 <= py < H:
            debug_info['waypoints_in_bounds'] += 1
    
    if debug:
        debug_info['pixel_coords'] = pixel_coords
    
    # Draw lines between consecutive waypoints
    for i in range(len(pixel_coords) - 1):
        pt1 = pixel_coords[i]
        pt2 = pixel_coords[i + 1]
        
        # Check if either point is within bounds
        if (0 <= pt1[0] < W and 0 <= pt1[1] < H) or \
           (0 <= pt2[0] < W and 0 <= pt2[1] < H):
            bev[0] = draw_line(bev[0], pt1, pt2, line_width)
    
    # Compute coverage metric
    total_pixels = H * W
    road_pixels = np.sum(bev[0] > 0)
    debug_info['coverage_percent'] = (road_pixels / total_pixels) * 100
    
    if debug:
        return bev, debug_info
    return bev


def osm_to_bev(osm_edges: List[List[Tuple[float, float]]],
               current_pose: np.ndarray,
               gps_transform: Dict,
               grid_size: Tuple[int, int] = (300, 400),
               resolution: float = 0.1,
               x_range: Tuple[float, float] = (-20, 20),
               y_range: Tuple[float, float] = (-10, 30),
               line_width: int = 3,
               max_distance: float = 50.0,
               debug: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Convert OSM road network to BEV binary mask with GPS alignment.
    
    Args:
        osm_edges: List of road segments, each [(lat, lon), ...]
        current_pose: [3, 4] current ego pose in KITTI world frame
        gps_transform: Dict with 'offset_east', 'offset_north' from compute_gps_to_local_transform
        grid_size: (H, W) output BEV size
        resolution: meters per pixel
        x_range: (min, max) x coordinates in ego frame
        y_range: (min, max) y coordinates in ego frame
        line_width: width of road line in pixels
        max_distance: Maximum distance from ego to include roads (meters)
        debug: If True, return debug info
    
    Returns:
        bev: [1, H, W] binary mask
        debug_info: Dictionary with alignment metrics
    """
    H, W = grid_size
    bev = np.zeros((1, H, W), dtype=np.float32)
    
    debug_info = {
        'num_edges': len(osm_edges),
        'edges_rendered': 0,
        'points_transformed': 0,
        'points_in_bounds': 0,
        'mean_edge_distance': 0.0,
        'transform_offset': (gps_transform.get('offset_east', 0), 
                            gps_transform.get('offset_north', 0))
    }
    
    if len(osm_edges) == 0:
        if debug:
            return bev, debug_info
        return bev
    
    # Extract current position in world frame
    current_x = current_pose[0, 3]
    current_y = current_pose[1, 3]
    
    # Get GPS to local transform
    offset_east = gps_transform.get('offset_east', 0)
    offset_north = gps_transform.get('offset_north', 0)
    
    edge_distances = []
    
    for edge in osm_edges:
        # Transform edge points: lat/lon → UTM → KITTI world → ego
        edge_points_world = []
        
        for lat, lon in edge:
            # GPS → UTM
            utm_e, utm_n = latlon_to_utm(lat, lon)
            
            # UTM → KITTI world
            world_x = utm_e - offset_east
            world_y = utm_n - offset_north
            
            edge_points_world.append((world_x, world_y))
            debug_info['points_transformed'] += 1
        
        # Transform to ego frame
        edge_points_ego = []
        for wp in edge_points_world:
            ego_pt = world_to_ego(np.array([wp[0], wp[1], 0]), current_pose)
            edge_points_ego.append(ego_pt)
        
        # Check if edge is within range of ego vehicle
        edge_center = np.mean(edge_points_ego, axis=0)
        distance = np.linalg.norm(edge_center)
        edge_distances.append(distance)
        
        if distance > max_distance:
            continue
        
        # Convert to pixel coordinates and draw
        def ego_to_pixel(pt):
            px = int((pt[0] - x_range[0]) / resolution)
            py = int((pt[1] - y_range[0]) / resolution)
            return (px, py)
        
        pixel_coords = []
        for ego_pt in edge_points_ego:
            px, py = ego_to_pixel(ego_pt)
            pixel_coords.append((px, py))
            if 0 <= px < W and 0 <= py < H:
                debug_info['points_in_bounds'] += 1
        
        # Draw lines
        for i in range(len(pixel_coords) - 1):
            pt1 = pixel_coords[i]
            pt2 = pixel_coords[i + 1]
            
            if (0 <= pt1[0] < W and 0 <= pt1[1] < H) or \
               (0 <= pt2[0] < W and 0 <= pt2[1] < H):
                bev[0] = draw_line(bev[0], pt1, pt2, line_width)
        
        debug_info['edges_rendered'] += 1
    
    if edge_distances:
        debug_info['mean_edge_distance'] = np.mean(edge_distances)
        debug_info['min_edge_distance'] = np.min(edge_distances)
        debug_info['max_edge_distance'] = np.max(edge_distances)
    
    # Coverage metric
    total_pixels = H * W
    road_pixels = np.sum(bev[0] > 0)
    debug_info['coverage_percent'] = (road_pixels / total_pixels) * 100
    
    if debug:
        return bev, debug_info
    return bev


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_alignment_metrics(pred_mask: np.ndarray, 
                              gt_mask: np.ndarray,
                              threshold: float = 0.5) -> Dict:
    """
    Compute alignment metrics between predicted road mask and ground truth.
    
    Args:
        pred_mask: [H, W] predicted binary mask (OSM or History)
        gt_mask: [H, W] ground truth mask (e.g., from LiDAR road detection)
        threshold: Threshold for binary conversion
    
    Returns:
        Dictionary with metrics
    """
    # Binarize
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > threshold).astype(np.float32)
    
    # Compute IoU
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(np.clip(pred_binary + gt_binary, 0, 1))
    iou = intersection / (union + 1e-8)
    
    # Compute precision/recall if we treat LiDAR drivable area as GT
    true_positives = intersection
    false_positives = np.sum(pred_binary * (1 - gt_binary))
    false_negatives = np.sum((1 - pred_binary) * gt_binary)
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Coverage comparison
    pred_coverage = np.sum(pred_binary) / pred_binary.size
    gt_coverage = np.sum(gt_binary) / gt_binary.size
    coverage_ratio = pred_coverage / (gt_coverage + 1e-8)
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pred_coverage': pred_coverage,
        'gt_coverage': gt_coverage,
        'coverage_ratio': coverage_ratio
    }


def evaluate_osm_alignment(osm_bev: np.ndarray,
                           lidar_bev: np.ndarray,
                           use_intensity_channel: bool = True) -> Dict:
    """
    Evaluate OSM alignment using LiDAR as reference.
    
    Args:
        osm_bev: [1, H, W] OSM road mask
        lidar_bev: [3, H, W] LiDAR BEV (uses intensity or density as proxy for drivable area)
        use_intensity_channel: If True, use intensity channel; else use density
    
    Returns:
        Alignment metrics
    """
    osm_mask = osm_bev[0]
    
    # Use LiDAR intensity as proxy for drivable area (roads have higher intensity)
    if use_intensity_channel:
        lidar_ref = lidar_bev[1]  # Intensity channel
    else:
        lidar_ref = lidar_bev[2]  # Density channel
    
    # Threshold LiDAR to get approximate drivable area
    threshold = np.percentile(lidar_ref[lidar_ref > 0], 50) if np.any(lidar_ref > 0) else 0.1
    lidar_mask = (lidar_ref > threshold).astype(np.float32)
    
    metrics = compute_alignment_metrics(osm_mask, lidar_mask)
    metrics['lidar_threshold'] = threshold
    
    return metrics


def evaluate_trajectory_alignment(history_bev: np.ndarray,
                                  future_trajectory: np.ndarray,
                                  current_pose: np.ndarray,
                                  grid_size: Tuple[int, int] = (300, 400),
                                  resolution: float = 0.1,
                                  x_range: Tuple[float, float] = (-20, 20),
                                  y_range: Tuple[float, float] = (-10, 30)) -> Dict:
    """
    Evaluate if history trajectory leads naturally to future trajectory.
    
    Args:
        history_bev: [1, H, W] history trajectory mask
        future_trajectory: [T, 2] future waypoints in ego frame
        current_pose: [3, 4] current pose
        grid_size: (H, W) BEV size
        resolution: meters per pixel
        x_range, y_range: coordinate ranges
    
    Returns:
        Continuity metrics
    """
    H, W = grid_size
    
    # Check if future trajectory starts near where history ends
    # History should end at origin (0, 0) in ego frame
    if len(future_trajectory) == 0:
        return {'continuity_error': float('inf'), 'is_continuous': False}
    
    # First future waypoint should be near ego position (origin)
    first_future = future_trajectory[0]
    distance_from_origin = np.linalg.norm(first_future)
    
    # Check trajectory smoothness (curvature)
    if len(future_trajectory) >= 3:
        # Compute turning angles
        angles = []
        for i in range(1, len(future_trajectory) - 1):
            v1 = future_trajectory[i] - future_trajectory[i-1]
            v2 = future_trajectory[i+1] - future_trajectory[i]
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)
                angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))
                angles.append(angle)
        
        mean_curvature = np.mean(angles) if angles else 0
        max_curvature = np.max(angles) if angles else 0
    else:
        mean_curvature = 0
        max_curvature = 0
    
    # Check overlap between history and future (they should connect)
    history_mask = history_bev[0] > 0
    
    # Rasterize first few future waypoints
    future_mask = np.zeros((H, W), dtype=np.float32)
    for i in range(min(3, len(future_trajectory))):
        pt = future_trajectory[i]
        px = int((pt[0] - x_range[0]) / resolution)
        py = int((pt[1] - y_range[0]) / resolution)
        if 0 <= px < W and 0 <= py < H:
            future_mask[py, px] = 1
    
    # Dilate both masks slightly
    from scipy.ndimage import binary_dilation
    history_dilated = binary_dilation(history_mask, iterations=2)
    future_dilated = binary_dilation(future_mask, iterations=2)
    
    overlap = np.sum(history_dilated & future_dilated)
    
    return {
        'continuity_error': distance_from_origin,
        'is_continuous': distance_from_origin < 2.0,  # Within 2 meters
        'mean_curvature': mean_curvature,
        'max_curvature': max_curvature,
        'history_future_overlap': int(overlap),
        'is_smooth': max_curvature < np.pi / 4  # Less than 45 degree turns
    }


# =============================================================================
# Full Multimodal Tensor Creation
# =============================================================================

def create_multimodal_input(lidar_bev: np.ndarray,
                           history_bev: np.ndarray,
                           osm_bev: np.ndarray,
                           debug: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Concatenate all BEV features into multimodal input tensor.
    
    Args:
        lidar_bev: [3, H, W] LiDAR BEV
        history_bev: [1, H, W] trajectory history BEV
        osm_bev: [1, H, W] OSM map BEV
        debug: If True, return debug info
    
    Returns:
        input_tensor: [5, H, W] concatenated input
        debug_info: Dictionary with combined metrics
    """
    input_tensor = np.concatenate([lidar_bev, history_bev, osm_bev], axis=0)
    
    debug_info = {
        'input_shape': input_tensor.shape,
        'lidar_coverage': np.sum(lidar_bev[2] > 0) / lidar_bev[2].size,  # Density channel
        'history_coverage': np.sum(history_bev[0] > 0) / history_bev[0].size,
        'osm_coverage': np.sum(osm_bev[0] > 0) / osm_bev[0].size,
    }
    
    # Check alignment: OSM and history should overlap
    history_mask = history_bev[0] > 0
    osm_mask = osm_bev[0] > 0
    
    if np.sum(history_mask) > 0 and np.sum(osm_mask) > 0:
        overlap = np.sum(history_mask & osm_mask)
        union = np.sum(history_mask | osm_mask)
        debug_info['history_osm_iou'] = overlap / (union + 1e-8)
    else:
        debug_info['history_osm_iou'] = 0.0
    
    if debug:
        return input_tensor, debug_info
    return input_tensor


# =============================================================================
# Visualization Utilities
# =============================================================================

def visualize_multimodal_bev(lidar_bev: np.ndarray,
                             history_bev: np.ndarray,
                             osm_bev: np.ndarray,
                             save_path: Optional[str] = None,
                             metrics: Optional[Dict] = None):
    """
    Visualize multimodal BEV input for debugging.
    
    Args:
        lidar_bev: [3, H, W] LiDAR BEV
        history_bev: [1, H, W] history BEV
        osm_bev: [1, H, W] OSM BEV
        save_path: Optional path to save figure
        metrics: Optional metrics dict to display
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # LiDAR channels
        axes[0, 0].imshow(lidar_bev[0], cmap='viridis')
        axes[0, 0].set_title('LiDAR Height')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(lidar_bev[1], cmap='hot')
        axes[0, 1].set_title('LiDAR Intensity')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(lidar_bev[2], cmap='plasma')
        axes[0, 2].set_title('LiDAR Density')
        axes[0, 2].axis('off')
        
        # History and OSM
        axes[1, 0].imshow(history_bev[0], cmap='Reds')
        axes[1, 0].set_title('History Trajectory')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(osm_bev[0], cmap='Blues')
        axes[1, 1].set_title('OSM Roads')
        axes[1, 1].axis('off')
        
        # Overlay
        overlay = np.zeros((*lidar_bev.shape[1:], 3))
        overlay[..., 0] = history_bev[0]  # Red: History
        overlay[..., 2] = osm_bev[0]  # Blue: OSM
        overlay[..., 1] = lidar_bev[2] / (lidar_bev[2].max() + 1e-8)  # Green: LiDAR density
        axes[1, 2].imshow(np.clip(overlay, 0, 1))
        axes[1, 2].set_title('Overlay (R:History, B:OSM, G:LiDAR)')
        axes[1, 2].axis('off')
        
        # Add metrics as text
        if metrics:
            metrics_text = '\n'.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                     for k, v in metrics.items()])
            fig.text(0.02, 0.02, metrics_text, fontsize=8, family='monospace',
                    verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available for visualization")


if __name__ == "__main__":
    print("=" * 70)
    print("Multimodal BEV Utilities - Test Suite")
    print("=" * 70)
    
    # Test LiDAR to BEV
    print("\n1. Testing LiDAR to BEV...")
    num_points = 10000
    lidar_points = np.random.randn(num_points, 4).astype(np.float32)
    lidar_points[:, 0] = np.random.uniform(-20, 20, num_points)
    lidar_points[:, 1] = np.random.uniform(-10, 30, num_points)
    lidar_points[:, 2] = np.random.uniform(-2, 2, num_points)
    lidar_points[:, 3] = np.random.uniform(0, 255, num_points)
    
    lidar_bev = lidar_to_bev(lidar_points)
    print(f"   LiDAR BEV shape: {lidar_bev.shape}")
    
    # Test trajectory to BEV
    print("\n2. Testing Trajectory to BEV...")
    trajectory = np.array([
        [-5, -5],
        [-3, 0],
        [0, 5],
        [2, 10],
        [1, 15]
    ])
    
    # Mock pose (identity at origin)
    pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    past_poses = np.tile(pose.flatten(), (10, 1))
    
    history_bev, history_debug = trajectory_to_bev(
        trajectory, pose, past_poses, debug=True
    )
    print(f"   History BEV shape: {history_bev.shape}")
    print(f"   Waypoints in bounds: {history_debug['waypoints_in_bounds']}/{history_debug['num_waypoints']}")
    print(f"   Coverage: {history_debug['coverage_percent']:.2f}%")
    
    # Test OSM to BEV (with mock data)
    print("\n3. Testing OSM to BEV...")
    osm_edges = [
        [(49.011, 8.418), (49.012, 8.419), (49.013, 8.420)],
        [(49.011, 8.418), (49.010, 8.417)],
    ]
    
    gps_transform = {
        'offset_east': 500000,
        'offset_north': 0
    }
    
    osm_bev, osm_debug = osm_to_bev(
        osm_edges, pose, gps_transform, debug=True
    )
    print(f"   OSM BEV shape: {osm_bev.shape}")
    print(f"   Edges rendered: {osm_debug['edges_rendered']}/{osm_debug['num_edges']}")
    print(f"   Coverage: {osm_debug['coverage_percent']:.2f}%")
    
    # Test full multimodal input
    print("\n4. Testing multimodal input creation...")
    multimodal_input, multimodal_debug = create_multimodal_input(
        lidar_bev, history_bev, osm_bev, debug=True
    )
    print(f"   Multimodal input shape: {multimodal_input.shape}")
    print(f"   History-OSM IoU: {multimodal_debug['history_osm_iou']:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
