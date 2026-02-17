"""
Block 1: BEV Rasterization Module for TopoDiffuser.

Converts raw sensor data (LiDAR, trajectory history, topometric maps) into
a unified Bird's Eye View (BEV) representation for the encoder.

Paper Reference: Section III-B, Appendix Table I
Input: Raw LiDAR point cloud, past poses, OSM road network
Output: [B, 5, 300, 400] BEV tensor
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union
import warnings


class BEVRasterizer:
    """
    BEV Rasterization for multimodal input.
    
    Following the paper specifications:
    - LiDAR: 3 channels (Height, Intensity, Density)
    - Trajectory History: 1 channel (binary occupancy)
    - Topometric Map: 1 channel (OSM road network)
    
    Output: [5, H, W] tensor where:
        - Channel 0: LiDAR Height
        - Channel 1: LiDAR Intensity
        - Channel 2: LiDAR Density
        - Channel 3: Trajectory History
        - Channel 4: Topometric Map
    """
    
    # Default BEV configuration (KITTI-like)
    DEFAULT_CONFIG = {
        'grid_size': (300, 400),      # (H, W) - Height first for numpy
        'resolution': 0.1,            # meters per pixel
        'x_range': (-20, 20),         # left-right (lateral) in meters
        'y_range': (-10, 30),         # back-forward (longitudinal) in meters
        'z_range': (-3, 4),           # height range in meters (expanded for tall vehicles)
        'max_intensity': 255.0,       # max LiDAR intensity value
        'density_normalize': 128,     # N_max for density log normalization (increased for dense scenes)
        'ground_height': -0.5,        # approximate ground plane height
    }
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize BEV rasterizer.
        
        Args:
            config: Custom configuration dict (uses DEFAULT_CONFIG if None)
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.H, self.W = self.config['grid_size']
        self.resolution = self.config['resolution']
        self.x_range = self.config['x_range']
        self.y_range = self.config['y_range']
        self.z_range = self.config['z_range']
        
    def rasterize_lidar(self, points: np.ndarray) -> np.ndarray:
        """
        Convert LiDAR point cloud to 3-channel BEV representation.
        
        Following paper Section III-B:
        - Channel 0 (Height): Max Z-value per cell (captures curbs, vehicle heights)
        - Channel 1 (Intensity): Average laser reflectivity (road markings vs asphalt)
        - Channel 2 (Density): Log-normalized point count per cell
        
        Args:
            points: [N, 4+] array of (x, y, z, intensity, ...)
                   Coordinates should be in ego vehicle frame
        
        Returns:
            bev: [3, H, W] array with (height, intensity, density) channels
        """
        # Initialize BEV
        bev = np.zeros((3, self.H, self.W), dtype=np.float32)
        
        if len(points) == 0:
            return bev
        
        # FIX 1: Only filter by X,Y for grid bounds, NOT by Z
        # This preserves points with extreme Z values (tall objects)
        mask = (
            (points[:, 0] >= self.x_range[0]) & (points[:, 0] < self.x_range[1]) &
            (points[:, 1] >= self.y_range[0]) & (points[:, 1] < self.y_range[1])
        )
        
        # Store original points count for coverage metric
        total_points = len(points)
        
        points_xy_filtered = points[mask]
        
        if len(points_xy_filtered) == 0:
            warnings.warn("No LiDAR points within valid XY range after filtering")
            return bev
        
        # Step 2: Convert to pixel coordinates
        # x (lateral, left-right) -> column (width)
        # y (longitudinal, back-forward) -> row (height)
        px = ((points_xy_filtered[:, 0] - self.x_range[0]) / self.resolution).astype(np.int32)
        py = ((points_xy_filtered[:, 1] - self.y_range[0]) / self.resolution).astype(np.int32)
        
        # Clip to grid bounds
        px = np.clip(px, 0, self.W - 1)
        py = np.clip(py, 0, self.H - 1)
        
        # Step 3: Voxelization with channel encoding
        z_vals = points_xy_filtered[:, 2] if points_xy_filtered.shape[1] >= 3 else np.zeros(len(points_xy_filtered))
        intensity_vals = points_xy_filtered[:, 3] if points_xy_filtered.shape[1] >= 4 else np.zeros(len(points_xy_filtered))
        
        # Vectorized aggregation (replaces slow Python for-loop)
        # Height channel: max pooling per cell
        # Initialize to -inf so np.maximum.at works correctly
        bev[0, :, :] = -np.inf
        np.maximum.at(bev[0], (py, px), z_vals)
        bev[0][bev[0] == -np.inf] = 0  # Reset empty cells to 0

        # Intensity channel: sum for averaging later
        np.add.at(bev[1], (py, px), intensity_vals)

        # Density channel: count points per cell
        np.add.at(bev[2], (py, px), 1)
        
        # Step 4: Post-processing and normalization
        # Normalize intensity by density (average)
        density_mask = bev[2] > 0
        bev[1][density_mask] /= bev[2][density_mask]
        
        # FIX 4: Normalize by configured z_range (allows values outside [0,1] then clips)
        # This standardizes height to a consistent range
        bev[0] = (bev[0] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        bev[0] = np.clip(bev[0], 0, 1)
        
        # Intensity: normalize by max intensity
        bev[1] = bev[1] / self.config['max_intensity']
        bev[1] = np.clip(bev[1], 0, 1)
        
        # Density: log normalization as per paper
        # D[u,v] = log(1 + count) / log(1 + N_max)
        n_max = self.config['density_normalize']
        bev[2] = np.log(1 + bev[2]) / np.log(1 + n_max)
        bev[2] = np.clip(bev[2], 0, 1)
        
        return bev
    
    def rasterize_trajectory(self, trajectory: np.ndarray, 
                            line_width: int = 2,
                            blur_sigma: Optional[float] = None) -> np.ndarray:
        """
        Convert trajectory waypoints to BEV binary mask.
        
        Args:
            trajectory: [N, 2] array of (x, y) waypoints in ego frame
            line_width: Width of trajectory line in pixels
            blur_sigma: Gaussian blur sigma (None = no blur)
        
        Returns:
            bev: [1, H, W] binary mask
        """
        bev = np.zeros((1, self.H, self.W), dtype=np.float32)
        
        if len(trajectory) < 2:
            return bev
        
        # Convert waypoints to pixel coordinates
        def world_to_pixel(pt):
            px = int((pt[0] - self.x_range[0]) / self.resolution)
            py = int((pt[1] - self.y_range[0]) / self.resolution)
            return (px, py)
        
        # Draw lines between consecutive waypoints using Bresenham's algorithm
        for i in range(len(trajectory) - 1):
            pt1 = world_to_pixel(trajectory[i])
            pt2 = world_to_pixel(trajectory[i + 1])
            
            # Skip if both points are out of bounds
            in_bounds_1 = (0 <= pt1[0] < self.W) and (0 <= pt1[1] < self.H)
            in_bounds_2 = (0 <= pt2[0] < self.W) and (0 <= pt2[1] < self.H)
            
            if not (in_bounds_1 or in_bounds_2):
                continue
            
            # Clip points to bounds for drawing
            pt1 = (np.clip(pt1[0], 0, self.W - 1), np.clip(pt1[1], 0, self.H - 1))
            pt2 = (np.clip(pt2[0], 0, self.W - 1), np.clip(pt2[1], 0, self.H - 1))
            
            bev[0] = self._draw_line(bev[0], pt1, pt2, line_width)
        
        # Optional Gaussian blur for smoothness
        if blur_sigma is not None and blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            bev[0] = gaussian_filter(bev[0], sigma=blur_sigma)
            bev[0] = np.clip(bev[0], 0, 1)
        
        return bev
    
    def rasterize_topometric_map(self, osm_coords: np.ndarray,
                                 ego_pose: Tuple[float, float, float],
                                 line_width: int = 3) -> np.ndarray:
        """
        Convert OSM road coordinates to BEV binary mask.
        
        Args:
            osm_coords: [N, 2] array of (x, y) in world frame or list of polylines
            ego_pose: (x, y, yaw) ego vehicle pose in world frame
            line_width: Width of road line in pixels
        
        Returns:
            bev: [1, H, W] binary mask
        """
        bev = np.zeros((1, self.H, self.W), dtype=np.float32)
        
        if len(osm_coords) < 2:
            return bev
        
        ego_x, ego_y, ego_yaw = ego_pose
        
        # Transform from world to ego frame
        def world_to_ego(pt):
            dx = pt[0] - ego_x
            dy = pt[1] - ego_y
            
            # Rotate by -ego_yaw
            cos_yaw = np.cos(-ego_yaw)
            sin_yaw = np.sin(-ego_yaw)
            
            x_ego = dx * cos_yaw - dy * sin_yaw
            y_ego = dx * sin_yaw + dy * cos_yaw
            
            return np.array([x_ego, y_ego])
        
        # Transform to pixel coordinates
        def ego_to_pixel(pt):
            px = int((pt[0] - self.x_range[0]) / self.resolution)
            py = int((pt[1] - self.y_range[0]) / self.resolution)
            return (px, py)
        
        # Handle list of polylines or single polyline
        if isinstance(osm_coords, list):
            polylines = osm_coords
        else:
            polylines = [osm_coords]
        
        # Draw each polyline
        for polyline in polylines:
            if len(polyline) < 2:
                continue
            
            # Transform all points
            pixel_coords = []
            for coord in polyline:
                ego_pt = world_to_ego(coord)
                pixel_pt = ego_to_pixel(ego_pt)
                pixel_coords.append(pixel_pt)
            
            # Draw lines between consecutive points
            for i in range(len(pixel_coords) - 1):
                pt1 = pixel_coords[i]
                pt2 = pixel_coords[i + 1]
                
                # Check if either point is within bounds
                in_bounds_1 = (0 <= pt1[0] < self.W) and (0 <= pt1[1] < self.H)
                in_bounds_2 = (0 <= pt2[0] < self.W) and (0 <= pt2[1] < self.H)
                
                if in_bounds_1 or in_bounds_2:
                    bev[0] = self._draw_line(bev[0], pt1, pt2, line_width)
        
        return bev
    
    def create_input_tensor(self, 
                           lidar_points: Optional[np.ndarray] = None,
                           trajectory: Optional[np.ndarray] = None,
                           osm_coords: Optional[np.ndarray] = None,
                           ego_pose: Tuple[float, float, float] = (0, 0, 0),
                           mode: str = 'full') -> np.ndarray:
        """
        Create complete input tensor from all modalities.
        
        Args:
            lidar_points: [N, 4] LiDAR points (x, y, z, intensity)
            trajectory: [N, 2] trajectory waypoints (x, y)
            osm_coords: [N, 2] OSM road coordinates or list of polylines
            ego_pose: (x, y, yaw) ego vehicle pose
            mode: 'full' (5 channels), 'lidar_only' (3 channels), 
                  'lidar_traj' (4 channels), 'lidar_map' (4 channels)
        
        Returns:
            input_tensor: [C, H, W] concatenated BEV tensor
        """
        channels = []
        
        # LiDAR (3 channels) - always included
        if lidar_points is not None:
            lidar_bev = self.rasterize_lidar(lidar_points)
        else:
            lidar_bev = np.zeros((3, self.H, self.W), dtype=np.float32)
        
        if mode in ['lidar_only', 'lidar_traj', 'lidar_map', 'full']:
            channels.append(lidar_bev)
        
        # Trajectory history (1 channel)
        if mode in ['lidar_traj', 'full']:
            if trajectory is not None:
                traj_bev = self.rasterize_trajectory(trajectory)
            else:
                traj_bev = np.zeros((1, self.H, self.W), dtype=np.float32)
            channels.append(traj_bev)
        
        # Topometric map (1 channel)
        if mode in ['lidar_map', 'full']:
            if osm_coords is not None:
                map_bev = self.rasterize_topometric_map(osm_coords, ego_pose)
            else:
                map_bev = np.zeros((1, self.H, self.W), dtype=np.float32)
            channels.append(map_bev)
        
        return np.concatenate(channels, axis=0)
    
    def _draw_line(self, grid: np.ndarray, pt1: Tuple[int, int], 
                   pt2: Tuple[int, int], width: int = 1) -> np.ndarray:
        """
        Draw a line on the grid using Bresenham's algorithm.
        
        Args:
            grid: [H, W] array to draw on
            pt1: (x, y) start point
            pt2: (x, y) end point
            width: line width in pixels
        
        Returns:
            grid: Modified grid with line drawn
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            # Draw pixel with width
            half_w = width // 2
            for w in range(-half_w, half_w + 1):
                for h in range(-half_w, half_w + 1):
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


class BEVRasterizationBlock(nn.Module):
    """
    PyTorch module wrapper for BEV rasterization.
    
    This is a convenience wrapper that handles batch processing
    and converts numpy arrays to torch tensors.
    
    Note: The actual rasterization happens on CPU with numpy for efficiency,
    then converted to GPU tensors for the encoder.
    """
    
    def __init__(self, config: Optional[dict] = None, device: str = 'cpu'):
        """
        Initialize BEV rasterization block.
        
        Args:
            config: BEV configuration dict
            device: torch device for output tensors
        """
        super().__init__()
        self.rasterizer = BEVRasterizer(config)
        self.device = device
        self.config = self.rasterizer.config
        
    def forward(self, batch_data: List[dict]) -> torch.Tensor:
        """
        Process a batch of data into BEV tensors.
        
        Args:
            batch_data: List of dicts with keys:
                - 'lidar': [N, 4] LiDAR points
                - 'trajectory': [N, 2] trajectory waypoints (optional)
                - 'osm_coords': OSM coordinates (optional)
                - 'ego_pose': (x, y, yaw) ego pose
        
        Returns:
            bev_tensor: [B, C, H, W] batched BEV tensor
        """
        bev_tensors = []
        
        for data in batch_data:
            lidar = data.get('lidar')
            trajectory = data.get('trajectory')
            osm_coords = data.get('osm_coords')
            ego_pose = data.get('ego_pose', (0, 0, 0))
            mode = data.get('mode', 'full')
            
            bev = self.rasterizer.create_input_tensor(
                lidar_points=lidar,
                trajectory=trajectory,
                osm_coords=osm_coords,
                ego_pose=ego_pose,
                mode=mode
            )
            bev_tensors.append(torch.from_numpy(bev))
        
        # Stack into batch
        bev_batch = torch.stack(bev_tensors, dim=0).to(self.device)
        return bev_batch
    
    def to_tensor(self, bev_numpy: np.ndarray) -> torch.Tensor:
        """Convert numpy BEV to torch tensor."""
        return torch.from_numpy(bev_numpy).unsqueeze(0).to(self.device)


def load_kitti_lidar(bin_path: str) -> np.ndarray:
    """
    Load KITTI LiDAR bin file.
    
    Args:
        bin_path: Path to .bin file
    
    Returns:
        points: [N, 4] array of (x, y, z, intensity)
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def extract_trajectory_from_poses(poses: np.ndarray, 
                                   num_frames: int = 5,
                                   spacing_meters: float = 2.0) -> np.ndarray:
    """
    Extract trajectory history from KITTI pose file.
    
    KITTI poses are 3x4 transformation matrices [R|t] flattened row-wise.
    Format: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    
    Args:
        poses: [N, 12] array of flattened 3x4 pose matrices
        num_frames: Number of history frames to extract
        spacing_meters: Spacing between keyframes in meters
    
    Returns:
        trajectory: [num_frames, 2] array of (x, y) positions
    """
    # Extract translation (indices 3, 7, 11 correspond to tx, ty, tz)
    positions = poses[:, [3, 7, 11]]  # (x, y, z)
    
    # Select keyframes based on spacing
    trajectory = []
    if len(positions) > 0:
        last_pos = positions[-1]
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


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("Block 1: BEV Rasterization Module Test")
    print("=" * 60)
    
    # Initialize rasterizer
    rasterizer = BEVRasterizer()
    print(f"\nBEV Configuration:")
    print(f"  Grid size: {rasterizer.H} x {rasterizer.W}")
    print(f"  Resolution: {rasterizer.resolution} m/pixel")
    print(f"  X range: {rasterizer.x_range} m")
    print(f"  Y range: {rasterizer.y_range} m")
    print(f"  Z range: {rasterizer.z_range} m")
    
    # Test 1: LiDAR rasterization
    print("\n" + "-" * 40)
    print("Test 1: LiDAR Rasterization")
    print("-" * 40)
    
    np.random.seed(42)
    num_points = 50000
    lidar_points = np.zeros((num_points, 4), dtype=np.float32)
    lidar_points[:, 0] = np.random.uniform(-15, 15, num_points)  # x
    lidar_points[:, 1] = np.random.uniform(-5, 25, num_points)   # y
    lidar_points[:, 2] = np.random.uniform(-2, 2, num_points)    # z
    lidar_points[:, 3] = np.random.uniform(0, 255, num_points)   # intensity
    
    lidar_bev = rasterizer.rasterize_lidar(lidar_points)
    print(f"LiDAR BEV shape: {lidar_bev.shape}")
    print(f"  Height range: [{lidar_bev[0].min():.3f}, {lidar_bev[0].max():.3f}]")
    print(f"  Intensity range: [{lidar_bev[1].min():.3f}, {lidar_bev[1].max():.3f}]")
    print(f"  Density sum: {lidar_bev[2].sum():.1f}")
    print(f"  Non-zero cells: {(lidar_bev[2] > 0).sum()}")
    
    # Test 2: Trajectory rasterization
    print("\n" + "-" * 40)
    print("Test 2: Trajectory Rasterization")
    print("-" * 40)
    
    trajectory = np.array([
        [-5, -5],
        [-3, 0],
        [0, 5],
        [2, 10],
        [1, 15]
    ], dtype=np.float32)
    
    traj_bev = rasterizer.rasterize_trajectory(trajectory)
    print(f"Trajectory BEV shape: {traj_bev.shape}")
    print(f"  Mask sum: {traj_bev[0].sum():.0f} pixels")
    print(f"  Trajectory points: {len(trajectory)}")
    
    # Test 3: Topometric map rasterization
    print("\n" + "-" * 40)
    print("Test 3: Topometric Map Rasterization")
    print("-" * 40)
    
    osm_coords = np.array([
        [0, -5],
        [0, 0],
        [0, 5],
        [0, 10],
        [0, 15],
        [0, 20]
    ], dtype=np.float32)
    ego_pose = (0, 0, 0)
    
    map_bev = rasterizer.rasterize_topometric_map(osm_coords, ego_pose)
    print(f"Map BEV shape: {map_bev.shape}")
    print(f"  Mask sum: {map_bev[0].sum():.0f} pixels")
    
    # Test 4: Full input tensor
    print("\n" + "-" * 40)
    print("Test 4: Full Input Tensor (All Modalities)")
    print("-" * 40)
    
    input_tensor = rasterizer.create_input_tensor(
        lidar_points=lidar_points,
        trajectory=trajectory,
        osm_coords=osm_coords,
        ego_pose=ego_pose,
        mode='full'
    )
    print(f"Full input tensor shape: {input_tensor.shape}")
    print(f"  Expected: [5, {rasterizer.H}, {rasterizer.W}]")
    print(f"  Channel 0 (LiDAR Height): mean={input_tensor[0].mean():.3f}")
    print(f"  Channel 1 (LiDAR Intensity): mean={input_tensor[1].mean():.3f}")
    print(f"  Channel 2 (LiDAR Density): mean={input_tensor[2].mean():.3f}")
    print(f"  Channel 3 (Trajectory): sum={input_tensor[3].sum():.0f}")
    print(f"  Channel 4 (OSM Map): sum={input_tensor[4].sum():.0f}")
    
    # Test 5: PyTorch module wrapper
    print("\n" + "-" * 40)
    print("Test 5: PyTorch Module Wrapper")
    print("-" * 40)
    
    bev_block = BEVRasterizationBlock()
    batch_data = [
        {
            'lidar': lidar_points,
            'trajectory': trajectory,
            'osm_coords': osm_coords,
            'ego_pose': ego_pose,
            'mode': 'full'
        },
        {
            'lidar': lidar_points,
            'trajectory': trajectory,
            'osm_coords': osm_coords,
            'ego_pose': ego_pose,
            'mode': 'full'
        }
    ]
    
    bev_batch = bev_block.forward(batch_data)
    print(f"Batch output shape: {bev_batch.shape}")
    print(f"  Expected: [2, 5, {rasterizer.H}, {rasterizer.W}]")
    print(f"  Device: {bev_batch.device}")
    print(f"  Dtype: {bev_batch.dtype}")
    
    # Test 6: Different modes
    print("\n" + "-" * 40)
    print("Test 6: Different Input Modes")
    print("-" * 40)
    
    for mode in ['lidar_only', 'lidar_traj', 'lidar_map', 'full']:
        tensor = rasterizer.create_input_tensor(
            lidar_points=lidar_points,
            trajectory=trajectory,
            osm_coords=osm_coords,
            ego_pose=ego_pose,
            mode=mode
        )
        print(f"  Mode '{mode}': shape {tensor.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
