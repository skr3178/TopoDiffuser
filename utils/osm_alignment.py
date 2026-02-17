"""
OSM Road Network Alignment with KITTI Local Frame.

Uses OXTS (GPS/IMU) data to align OSM roads with the ego vehicle's coordinate system.
"""

import numpy as np
from pathlib import Path
import pickle


def load_oxts_data(oxts_dir):
    """
    Load OXTS data from directory.
    
    Each OXTS file contains:
    lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ax, ay, af, al, au,
    wx, wy, wz, wf, wl, wu, pos_accuracy, vel_accuracy, navstat, numsats,
    posmode, velmode, orimode
    
    Returns:
        oxts_data: [N, 30] array of OXTS readings
    """
    oxts_dir = Path(oxts_dir)
    oxts_files = sorted(oxts_dir.glob('*.txt'))
    
    data = []
    for f in oxts_files:
        values = np.loadtxt(f)
        data.append(values)
    
    return np.array(data)


def latlon_to_utm(lat, lon, zone=32):
    """Convert latitude/longitude to UTM coordinates (simplified)."""
    # WGS84 parameters
    a = 6378137.0
    e = 0.0818191908426
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    
    k0 = 0.9996
    N = a / np.sqrt(1 - e**2 * np.sin(lat_rad)**2)
    T = np.tan(lat_rad)**2
    C = e**2 * np.cos(lat_rad)**2 / (1 - e**2)
    A = np.cos(lat_rad) * (lon_rad - lon0)
    
    M = a * ((1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad -
             (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*lat_rad) +
             (15*e**4/256 + 45*e**6/1024) * np.sin(4*lat_rad) -
             (35*e**6/3072) * np.sin(6*lat_rad))
    
    east = k0 * N * (A + (1 - T + C) * A**3 / 6 + 
                     (5 - 18*T + T**2 + 72*C - 58*0.006739497) * A**5 / 120)
    east += 500000
    
    north = k0 * (M + N * np.tan(lat_rad) * (A**2 / 2 + 
                  (5 - T + 9*C + 4*C**2) * A**4 / 24 +
                  (61 - 58*T + T**2 + 600*C - 330*0.006739497) * A**6 / 720))
    
    return east, north


def compute_gps_to_local_transform(oxts_data, poses):
    """
    Compute transformation from GPS (UTM) to KITTI local frame.
    
    Args:
        oxts_data: [N, 30] OXTS readings (lat, lon in first 2 cols)
        poses: [N, 12] KITTI pose matrices (3x4 flattened)
    
    Returns:
        transform: dict with offset and rotation to align GPS with local frame
    """
    # Extract GPS positions and convert to UTM
    lats = oxts_data[:, 0]
    lons = oxts_data[:, 1]
    
    # Convert to UTM
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
    local_y = poses[:, 11]  # tz (forward) — KITTI ground plane is (x, z), not (x, y)
    
    # Compute offset (difference between UTM and local at start)
    # Use first frame as reference
    offset_east = utm_east[0] - local_x[0]
    offset_north = utm_north[0] - local_y[0]
    
    # Compute scale factor (should be close to 1)
    # Compare distances between consecutive frames
    utm_distances = np.sqrt(np.diff(utm_east)**2 + np.diff(utm_north)**2)
    local_distances = np.sqrt(np.diff(local_x)**2 + np.diff(local_y)**2)
    
    # Filter out zero distances
    valid = (utm_distances > 0.1) & (local_distances > 0.1)
    if valid.sum() > 0:
        scale = np.median(local_distances[valid] / utm_distances[valid])
    else:
        scale = 1.0
    
    # Compute rotation (yaw offset)
    # From OXTS yaw (column 5)
    oxts_yaw = oxts_data[:, 5]
    
    # From poses (extract yaw from rotation matrix)
    # R = [r11 r12 r13; r21 r22 r23; r31 r32 r33]
    # yaw = atan2(r21, r11)
    pose_yaw = np.arctan2(poses[:, 4], poses[:, 0])
    
    # Yaw offset
    yaw_offset = pose_yaw[0] - oxts_yaw[0]
    
    transform = {
        'offset_east': offset_east,
        'offset_north': offset_north,
        'scale': scale,
        'yaw_offset': yaw_offset,
        'ref_utm_east': utm_east[0],
        'ref_utm_north': utm_north[0],
        'ref_local_x': local_x[0],
        'ref_local_y': local_y[0]
    }
    
    return transform


def gps_to_local(lat, lon, transform):
    """
    Convert GPS coordinates to KITTI local frame.
    
    Args:
        lat, lon: GPS coordinates
        transform: transformation dict from compute_gps_to_local_transform
    
    Returns:
        x, y: Local frame coordinates
    """
    # Convert to UTM
    east, north = latlon_to_utm(lat, lon)
    
    # Apply transformation
    # 1. Offset to align with local frame origin
    east_adj = east - transform['offset_east']
    north_adj = north - transform['offset_north']
    
    # 2. Scale
    east_scaled = transform['ref_local_x'] + (east_adj - transform['ref_utm_east'] + transform['offset_east']) * transform['scale']
    north_scaled = transform['ref_local_y'] + (north_adj - transform['ref_utm_north'] + transform['offset_north']) * transform['scale']
    
    # Actually, simpler approach: just offset
    x = east - transform['offset_east']
    y = north - transform['offset_north']
    
    return x, y


def transform_osm_to_local(osm_edges, transform):
    """
    Transform OSM road edges from GPS to local frame.
    
    Args:
        osm_edges: List of [(lat1, lon1), (lat2, lon2), ...] segments
        transform: transformation dict
    
    Returns:
        local_edges: List of [(x1, y1), (x2, y2), ...] in local frame
    """
    local_edges = []
    
    for edge in osm_edges:
        local_edge = []
        for lat, lon in edge:
            x, y = gps_to_local(lat, lon, transform)
            local_edge.append((x, y))
        local_edges.append(local_edge)
    
    return local_edges


def world_to_ego(point, ego_pose):
    """
    Transform point from world frame to ego frame.
    
    Args:
        point: (x, y) in world frame
        ego_pose: (x, y, yaw) ego vehicle pose
    
    Returns:
        (x, y) in ego frame
    """
    ego_x, ego_y, ego_yaw = ego_pose
    
    dx = point[0] - ego_x
    dy = point[1] - ego_y
    
    # Rotate by -ego_yaw
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    
    x_ego = dx * cos_yaw - dy * sin_yaw
    y_ego = dx * sin_yaw + dy * cos_yaw
    
    return (x_ego, y_ego)


def create_osm_bev_mask(osm_edges_local, ego_pose, grid_size=(300, 400), 
                        resolution=0.1, x_range=(-20, 20), y_range=(-10, 30)):
    """
    Create BEV binary mask from OSM roads in local frame.
    
    Args:
        osm_edges_local: List of road edges in local/world frame
        ego_pose: (x, y, yaw) ego vehicle pose in world frame
        grid_size: (H, W) BEV grid size
        resolution: meters per pixel
        x_range: (min, max) x coordinates in ego frame
        y_range: (min, max) y coordinates in ego frame
    
    Returns:
        bev_mask: [H, W] binary mask
    """
    H, W = grid_size
    mask = np.zeros((H, W), dtype=np.float32)
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Transform all points to ego frame and draw lines
    for edge in osm_edges_local:
        # Transform edge points to ego frame
        ego_points = []
        for pt in edge:
            ego_pt = world_to_ego(pt, ego_pose)
            ego_points.append(ego_pt)
        
        # Draw lines between consecutive points
        for i in range(len(ego_points) - 1):
            pt1 = ego_points[i]
            pt2 = ego_points[i + 1]
            
            # Check if within BEV range
            if not (x_min - 5 <= pt1[0] <= x_max + 5 and y_min - 5 <= pt1[1] <= y_max + 5):
                continue
            if not (x_min - 5 <= pt2[0] <= x_max + 5 and y_min - 5 <= pt2[1] <= y_max + 5):
                continue
            
            # Convert to pixel coordinates
            def world_to_pixel(pt):
                px = int((pt[0] - x_min) / resolution)
                py = int((pt[1] - y_min) / resolution)
                return (px, py)
            
            p1 = world_to_pixel(pt1)
            p2 = world_to_pixel(pt2)
            
            # Draw line using Bresenham's algorithm
            mask = draw_line(mask, p1, p2)
    
    return mask


def draw_line(grid, pt1, pt2, width=2):
    """Draw a line on the grid using Bresenham's algorithm."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    H, W = grid.shape
    
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
                if 0 <= px < W and 0 <= py < H:
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


def process_sequence(oxts_dir, poses_file, osm_pkl_file, output_dir, seq_name):
    """
    Process a sequence: align OSM with local frame and create BEV masks.
    
    Args:
        oxts_dir: Directory with OXTS .txt files
        poses_file: Path to poses .txt file
        osm_pkl_file: Path to OSM pickle file
        output_dir: Output directory
        seq_name: Sequence name (e.g., '0042')
    """
    print(f"\n{'='*60}")
    print(f"Processing sequence {seq_name}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading OXTS data...")
    oxts_data = load_oxts_data(oxts_dir)
    print(f"  Loaded {len(oxts_data)} OXTS frames")
    
    print("Loading poses...")
    poses = np.loadtxt(poses_file)
    print(f"  Loaded {len(poses)} poses")
    
    print("Loading OSM data...")
    with open(osm_pkl_file, 'rb') as f:
        osm_data = pickle.load(f)
    osm_edges = osm_data['edges']
    print(f"  Loaded {len(osm_edges)} OSM edges")
    
    # Compute transformation
    print("\nComputing GPS to local frame transformation...")
    # Use minimum length
    n_frames = min(len(oxts_data), len(poses))
    transform = compute_gps_to_local_transform(oxts_data[:n_frames], poses[:n_frames])
    print(f"  Offset: East={transform['offset_east']:.2f}, North={transform['offset_north']:.2f}")
    print(f"  Scale: {transform['scale']:.6f}")
    print(f"  Yaw offset: {np.degrees(transform['yaw_offset']):.2f}°")
    
    # Transform OSM edges to local frame
    print("\nTransforming OSM edges to local frame...")
    osm_edges_local = transform_osm_to_local(osm_edges, transform)
    print(f"  Transformed {len(osm_edges_local)} edges")
    
    # Save transformation and aligned OSM
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    align_data = {
        'sequence': seq_name,
        'transform': transform,
        'osm_edges_local': osm_edges_local,
        'osm_edges_gps': osm_edges,
        'n_frames': n_frames
    }
    
    output_pkl = output_dir / f'{seq_name}_aligned_osm.pkl'
    with open(output_pkl, 'wb') as f:
        pickle.dump(align_data, f)
    print(f"  Saved aligned data to {output_pkl}")
    
    # Create BEV masks for sample frames
    print("\nCreating sample BEV masks...")
    sample_frames = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
    
    bev_masks = {}
    for frame_idx in sample_frames:
        # Get ego pose from poses file
        pose = poses[frame_idx].reshape(3, 4)
        ego_x = pose[0, 3]
        ego_y = pose[1, 3]
        ego_yaw = np.arctan2(pose[1, 0], pose[0, 0])
        ego_pose = (ego_x, ego_y, ego_yaw)
        
        # Create BEV mask
        mask = create_osm_bev_mask(osm_edges_local, ego_pose)
        bev_masks[frame_idx] = mask
        
        coverage = mask.sum() / mask.size * 100
        print(f"  Frame {frame_idx}: {mask.sum():.0f} pixels ({coverage:.1f}% coverage)")
    
    # Save BEV masks
    bev_file = output_dir / f'{seq_name}_bev_masks.pkl'
    with open(bev_file, 'wb') as f:
        pickle.dump(bev_masks, f)
    print(f"  Saved BEV masks to {bev_file}")
    
    return align_data, bev_masks


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Align OSM with KITTI local frame')
    parser.add_argument('--oxts-dir', required=True, help='OXTS data directory')
    parser.add_argument('--poses-file', required=True, help='Poses file')
    parser.add_argument('--osm-pkl', required=True, help='OSM pickle file')
    parser.add_argument('--output-dir', default='data/osm_aligned', help='Output directory')
    parser.add_argument('--seq', default='0042', help='Sequence name')
    
    args = parser.parse_args()
    
    process_sequence(args.oxts_dir, args.poses_file, args.osm_pkl, 
                     args.output_dir, args.seq)
