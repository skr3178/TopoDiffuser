"""
BEV (Bird's Eye View) Encoding Utilities for TopoDiffuser.

Handles:
- LiDAR point cloud to BEV rasterization
- Trajectory history to BEV binary mask
- OSM route to BEV binary mask
"""

import numpy as np
import torch
import torch.nn.functional as F


def lidar_to_bev(points, grid_size=(300, 400), resolution=0.1, 
                 x_range=(-20, 20), y_range=(-10, 30), z_range=(-2, 2)):
    """
    Convert LiDAR point cloud to BEV representation.
    
    Following the paper, creates a 3-channel BEV:
    - Channel 0: Maximum height
    - Channel 1: Maximum intensity  
    - Channel 2: Point density
    
    Args:
        points: [N, 4] array of (x, y, z, intensity) or [N, 3] of (x, y, z)
        grid_size: (H, W) output BEV size (default: 300x400)
        resolution: meters per pixel (default: 0.1m)
        x_range: (min, max) x coordinates in ego frame (left-right)
        y_range: (min, max) y coordinates in ego frame (back-forward)
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
    # x (left-right) -> width (column)
    # y (back-forward) -> height (row)
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
        bev[0] = bev[0] / z_range[1]  # Normalize by max height
    if bev[1].max() > 0:
        bev[1] = bev[1] / 255.0  # Normalize by max intensity
    if bev[2].max() > 0:
        bev[2] = bev[2] / bev[2].max()  # Normalize density
    
    return bev


def trajectory_to_bev(trajectory, grid_size=(300, 400), resolution=0.1,
                      x_range=(-20, 20), y_range=(-10, 30), line_width=2):
    """
    Convert trajectory waypoints to BEV binary mask.
    
    Args:
        trajectory: [N, 2] array of (x, y) waypoints in ego frame
        grid_size: (H, W) output BEV size
        resolution: meters per pixel
        x_range: (min, max) x coordinates
        y_range: (min, max) y coordinates
        line_width: width of trajectory line in pixels
    
    Returns:
        bev: [1, H, W] binary mask
    """
    H, W = grid_size
    bev = np.zeros((1, H, W), dtype=np.float32)
    
    if len(trajectory) < 2:
        return bev
    
    # Convert waypoints to pixel coordinates
    def world_to_pixel(pt):
        px = int((pt[0] - x_range[0]) / resolution)
        py = int((pt[1] - y_range[0]) / resolution)
        return (px, py)
    
    # Draw lines between consecutive waypoints
    for i in range(len(trajectory) - 1):
        pt1 = world_to_pixel(trajectory[i])
        pt2 = world_to_pixel(trajectory[i + 1])
        
        # Check if points are within bounds
        if not (0 <= pt1[0] < W and 0 <= pt1[1] < H):
            continue
        if not (0 <= pt2[0] < W and 0 <= pt2[1] < H):
            continue
        
        # Draw line using Bresenham's line algorithm
        bev[0] = draw_line(bev[0], pt1, pt2, line_width)
    
    return bev


def draw_line(grid, pt1, pt2, width=1):
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


def osm_to_bev(osm_coords, ego_pose, grid_size=(300, 400), resolution=0.1,
               x_range=(-20, 20), y_range=(-10, 30), line_width=3):
    """
    Convert OSM road coordinates to BEV binary mask.
    
    Args:
        osm_coords: [N, 2] array of (lat, lon) or [N, 2] array of (x, y) in world frame
        ego_pose: (x, y, yaw) ego vehicle pose in world frame
        grid_size: (H, W) output BEV size
        resolution: meters per pixel
        x_range: (min, max) x coordinates in ego frame
        y_range: (min, max) y coordinates in ego frame
        line_width: width of road line in pixels
    
    Returns:
        bev: [1, H, W] binary mask
    """
    H, W = grid_size
    bev = np.zeros((1, H, W), dtype=np.float32)
    
    if len(osm_coords) < 2:
        return bev
    
    # Transform from world to ego frame
    ego_x, ego_y, ego_yaw = ego_pose
    
    def world_to_ego(pt):
        """Transform world coordinate to ego frame."""
        dx = pt[0] - ego_x
        dy = pt[1] - ego_y
        
        # Rotate by -ego_yaw
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        
        x_ego = dx * cos_yaw - dy * sin_yaw
        y_ego = dx * sin_yaw + dy * cos_yaw
        
        return (x_ego, y_ego)
    
    def ego_to_pixel(pt):
        """Transform ego coordinate to pixel."""
        px = int((pt[0] - x_range[0]) / resolution)
        py = int((pt[1] - y_range[0]) / resolution)
        return (px, py)
    
    # Transform and draw OSM coordinates
    pixel_coords = []
    for coord in osm_coords:
        ego_pt = world_to_ego(coord)
        pixel_pt = ego_to_pixel(ego_pt)
        pixel_coords.append(pixel_pt)
    
    # Draw lines
    for i in range(len(pixel_coords) - 1):
        pt1 = pixel_coords[i]
        pt2 = pixel_coords[i + 1]
        
        # Check if either point is within bounds
        if (0 <= pt1[0] < W and 0 <= pt1[1] < H) or \
           (0 <= pt2[0] < W and 0 <= pt2[1] < H):
            bev[0] = draw_line(bev[0], pt1, pt2, line_width)
    
    return bev


def load_lidar_bin(file_path):
    """
    Load KITTI LiDAR bin file.
    
    Args:
        file_path: path to .bin file
    
    Returns:
        points: [N, 4] array of (x, y, z, intensity)
    """
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points


def poses_to_trajectory(poses, num_frames=5, spacing_meters=2.0):
    """
    Extract trajectory history from KITTI pose file.
    
    KITTI poses are 3x4 transformation matrices [R|t] flattened row-wise.
    
    Args:
        poses: [N, 12] array of flattened 3x4 pose matrices
        num_frames: number of history frames to extract
        spacing_meters: spacing between keyframes in meters
    
    Returns:
        trajectory: [num_frames, 2] array of (x, y) positions
    """
    # Extract translation (last column of each 3x4 matrix)
    # Format: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    positions = poses[:, [3, 7, 11]]  # (x, y, z)
    
    # Select keyframes based on spacing
    trajectory = []
    last_pos = positions[-1] if len(positions) > 0 else np.array([0, 0, 0])
    trajectory.append(last_pos[:2])  # Current position
    
    dist_accum = 0
    for i in range(len(positions) - 2, -1, -1):
        dist = np.linalg.norm(positions[i] - positions[i + 1])
        dist_accum += dist
        
        if dist_accum >= spacing_meters and len(trajectory) < num_frames:
            trajectory.insert(0, positions[i][:2])
            dist_accum = 0
    
    # Pad if not enough frames
    while len(trajectory) < num_frames:
        if len(trajectory) > 0:
            trajectory.insert(0, trajectory[0])
        else:
            trajectory.insert(0, np.array([0, 0]))
    
    return np.array(trajectory[-num_frames:])


def create_input_tensor(lidar_bev, history_bev, map_bev):
    """
    Concatenate BEV features into input tensor.
    
    Args:
        lidar_bev: [3, H, W] LiDAR BEV
        history_bev: [1, H, W] trajectory history BEV
        map_bev: [1, H, W] OSM map BEV
    
    Returns:
        input_tensor: [5, H, W] concatenated input
    """
    return np.concatenate([lidar_bev, history_bev, map_bev], axis=0)


if __name__ == "__main__":
    # Test BEV encoding
    print("Testing BEV encoding utilities...")
    
    # Test LiDAR to BEV
    print("\n1. Testing LiDAR to BEV...")
    num_points = 10000
    lidar_points = np.random.randn(num_points, 4).astype(np.float32)
    lidar_points[:, 0] = np.random.uniform(-20, 20, num_points)  # x
    lidar_points[:, 1] = np.random.uniform(-10, 30, num_points)  # y
    lidar_points[:, 2] = np.random.uniform(-2, 2, num_points)    # z
    lidar_points[:, 3] = np.random.uniform(0, 255, num_points)   # intensity
    
    lidar_bev = lidar_to_bev(lidar_points)
    print(f"LiDAR BEV shape: {lidar_bev.shape}")  # Expected: (3, 300, 400)
    print(f"  Height range: [{lidar_bev[0].min():.2f}, {lidar_bev[0].max():.2f}]")
    print(f"  Intensity range: [{lidar_bev[1].min():.2f}, {lidar_bev[1].max():.2f}]")
    print(f"  Density sum: {lidar_bev[2].sum():.0f}")
    
    # Test trajectory to BEV
    print("\n2. Testing Trajectory to BEV...")
    trajectory = np.array([
        [-5, -5],
        [-3, 0],
        [0, 5],
        [2, 10],
        [1, 15]
    ])
    history_bev = trajectory_to_bev(trajectory)
    print(f"History BEV shape: {history_bev.shape}")  # Expected: (1, 300, 400)
    print(f"  Mask sum: {history_bev[0].sum():.0f} pixels")
    
    # Test OSM to BEV
    print("\n3. Testing OSM to BEV...")
    osm_coords = np.array([
        [0, 0],
        [0, 5],
        [0, 10],
        [0, 15],
        [0, 20]
    ])
    ego_pose = (0, 0, 0)  # x, y, yaw
    map_bev = osm_to_bev(osm_coords, ego_pose)
    print(f"Map BEV shape: {map_bev.shape}")  # Expected: (1, 300, 400)
    print(f"  Mask sum: {map_bev[0].sum():.0f} pixels")
    
    # Test full input tensor
    print("\n4. Testing full input tensor...")
    input_tensor = create_input_tensor(lidar_bev, history_bev, map_bev)
    print(f"Input tensor shape: {input_tensor.shape}")  # Expected: (5, 300, 400)
    
    # Convert to torch tensor
    input_torch = torch.from_numpy(input_tensor).unsqueeze(0)
    print(f"PyTorch tensor shape: {input_torch.shape}")  # Expected: (1, 5, 300, 400)
    
    print("\nAll tests passed!")
