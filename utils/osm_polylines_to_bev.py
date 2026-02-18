"""
Convert OSM polylines to BEV masks for 5-channel input (Modality 3).

This module provides functions to:
1. Load aligned OSM polylines from PBF parsing
2. Rasterize polylines to BEV binary mask
3. Extract OSM route within radius of current pose
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional


def load_osm_polylines(seq: str, data_root: str = "data") -> List[np.ndarray]:
    """
    Load aligned OSM polylines for a sequence.
    
    Args:
        seq: Sequence number (e.g., "00")
        data_root: Root data directory
        
    Returns:
        List of polylines, each as Nx2 array of (x, y) points
    """
    pkl_file = Path(f"osm_polylines_aligned_seq{seq}.pkl")
    
    if not pkl_file.exists():
        # Try in data directory
        pkl_file = Path(data_root) / "osm" / f"osm_polylines_aligned_seq{seq}.pkl"
    
    if not pkl_file.exists():
        raise FileNotFoundError(f"OSM polylines not found for sequence {seq}")
    
    with open(pkl_file, 'rb') as f:
        polylines = pickle.load(f)
    
    # Convert to numpy arrays
    polylines = [np.array(pl) for pl in polylines]
    
    return polylines


def rasterize_polylines(
    polylines: List[np.ndarray],
    bounds: Tuple[float, float, float, float],
    grid_size: Tuple[int, int],
    line_width: int = 2
) -> np.ndarray:
    """
    Rasterize polylines to binary mask.
    
    Args:
        polylines: List of Nx2 arrays of (x, y) points
        bounds: (x_min, x_max, y_min, y_max) in meters
        grid_size: (height, width) of output grid
        line_width: Width of lines in pixels
        
    Returns:
        Binary mask of shape (height, width)
    """
    from PIL import Image, ImageDraw
    
    height, width = grid_size
    x_min, x_max, y_min, y_max = bounds
    
    # Create blank image
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Scale factors
    x_scale = width / (x_max - x_min)
    y_scale = height / (y_max - y_min)
    
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        
        # Convert to pixel coordinates
        # Note: y is flipped because image coordinates have y=0 at top
        pixels = []
        for x, y in polyline:
            px = int((x - x_min) * x_scale)
            py = int((y_max - y) * y_scale)  # Flip y
            pixels.append((px, py))
        
        # Draw line
        draw.line(pixels, fill=255, width=line_width)
    
    return np.array(img) > 0


def extract_osm_route_bev(
    polylines: List[np.ndarray],
    center: Tuple[float, float],
    radius: float = 50.0,
    grid_size: Tuple[int, int] = (300, 400),
    line_width: int = 2
) -> np.ndarray:
    """
    Extract OSM route as BEV binary mask centered at given pose.
    
    Args:
        polylines: List of road polylines in local frame
        center: (x, y) center of BEV in meters
        radius: Radius to include around center in meters
        grid_size: (height, width) of output BEV
        line_width: Width of road lines in pixels
        
    Returns:
        Binary mask of shape (height, width)
    """
    cx, cy = center
    
    # Filter polylines to those within radius
    local_polylines = []
    for polyline in polylines:
        # Check if any point is within radius
        distances = np.sqrt((polyline[:, 0] - cx)**2 + (polyline[:, 1] - cy)**2)
        if np.any(distances <= radius):
            local_polylines.append(polyline)
    
    # Define bounds
    x_min, x_max = cx - radius, cx + radius
    y_min, y_max = cy - radius, cy + radius
    
    # Rasterize
    mask = rasterize_polylines(
        local_polylines,
        bounds=(x_min, x_max, y_min, y_max),
        grid_size=grid_size,
        line_width=line_width
    )
    
    return mask.astype(np.float32)


def get_osm_bev_for_frame(
    seq: str,
    frame_idx: int,
    trajectory: np.ndarray,
    radius: float = 50.0,
    grid_size: Tuple[int, int] = (300, 400),
    polylines: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    Get OSM BEV mask for a specific frame.
    
    Args:
        seq: Sequence number
        frame_idx: Frame index
        trajectory: Nx2 array of trajectory positions
        radius: Radius to include
        grid_size: (height, width) of BEV
        polylines: Pre-loaded polylines (will load if not provided)
        
    Returns:
        Binary mask of shape (height, width)
    """
    if polylines is None:
        polylines = load_osm_polylines(seq)
    
    center = trajectory[frame_idx]
    mask = extract_osm_route_bev(polylines, center, radius, grid_size)
    
    return mask


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test for sequence 00
    seq = "00"
    
    print(f"Loading OSM polylines for sequence {seq}...")
    polylines = load_osm_polylines(seq)
    print(f"Loaded {len(polylines)} polylines")
    
    # Load trajectory
    pose_file = f"data/kitti/poses/{seq}.txt"
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    poses = np.array(poses)
    trajectory = np.array([[p[0, 3], p[2, 3]] for p in poses])
    
    # Generate BEV for a few frames
    test_frames = [0, 500, 1000, 2000, 3000]
    
    fig, axes = plt.subplots(2, len(test_frames), figsize=(20, 8))
    
    for i, frame_idx in enumerate(test_frames):
        if frame_idx >= len(trajectory):
            continue
        
        # Get BEV mask
        mask = get_osm_bev_for_frame(seq, frame_idx, trajectory, polylines=polylines)
        
        # Plot
        axes[0, i].imshow(mask, cmap='Blues', origin='lower')
        axes[0, i].set_title(f'Frame {frame_idx}')
        axes[0, i].axis('off')
        
        # Plot trajectory context
        center = trajectory[frame_idx]
        radius = 50.0
        
        # Get nearby trajectory points
        distances = np.linalg.norm(trajectory - center, axis=1)
        nearby = distances <= radius
        
        axes[1, i].plot(trajectory[nearby, 0], trajectory[nearby, 1], 'r-', linewidth=2)
        axes[1, i].scatter(center[0], center[1], c='green', s=100, marker='o')
        axes[1, i].set_title(f'Context {frame_idx}')
        axes[1, i].axis('equal')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('osm_bev_samples.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved sample BEV masks to: osm_bev_samples.png")
