"""
Demo: OSM Road Network Alignment with OXTS/GPS Data.

This demonstrates the full pipeline:
1. Load OXTS (GPS/IMU) data
2. Load OSM road network (GPS coordinates)
3. Align OSM with OXTS trajectory
4. Convert to ego vehicle BEV space
5. Create Imap binary mask
"""

import numpy as np
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_oxts_data(oxts_dir):
    """Load OXTS data from directory."""
    oxts_dir = Path(oxts_dir)
    oxts_files = sorted(oxts_dir.glob('*.txt'))
    
    data = []
    for f in oxts_files:
        values = np.loadtxt(f)
        data.append(values)
    
    return np.array(data)


def latlon_to_utm(lat, lon, zone=32):
    """Convert latitude/longitude to UTM coordinates."""
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


def oxts_to_trajectory(oxts_data):
    """
    Convert OXTS data to local trajectory.
    
    Uses GPS coordinates converted to UTM, then centered at origin.
    Returns trajectory in local frame (meters from start).
    """
    lats = oxts_data[:, 0]
    lons = oxts_data[:, 1]
    yaws = oxts_data[:, 5]  # Yaw angle
    
    # Convert all GPS to UTM
    utm_east = []
    utm_north = []
    for lat, lon in zip(lats, lons):
        e, n = latlon_to_utm(lat, lon)
        utm_east.append(e)
        utm_north.append(n)
    
    utm_east = np.array(utm_east)
    utm_north = np.array(utm_north)
    
    # Center at origin (first frame)
    local_x = utm_east - utm_east[0]
    local_y = utm_north - utm_north[0]
    
    # Create trajectory array [N, 3] = (x, y, yaw)
    trajectory = np.column_stack([local_x, local_y, yaws])
    
    return trajectory


def transform_osm_to_local(osm_edges, oxts_trajectory):
    """
    Transform OSM road edges from GPS to local frame aligned with trajectory.
    
    Args:
        osm_edges: List of [(lat, lon), ...] in GPS
        oxts_trajectory: [N, 3] trajectory in local frame (from OXTS)
    
    Returns:
        local_edges: List of [(x, y), ...] in local frame
    """
    # Get reference GPS from first OXTS frame
    # We'll use this to align OSM with the trajectory
    
    # For simplicity, assume OSM is already roughly in the right area
    # and just convert GPS to UTM and offset to match trajectory
    
    # Actually, we need to find the GPS coordinate of the trajectory origin
    # The OXTS trajectory starts at (0,0) in local frame
    # which corresponds to the first GPS reading
    
    local_edges = []
    for edge in osm_edges:
        local_edge = []
        for lat, lon in edge:
            # Convert GPS to UTM
            east, north = latlon_to_utm(lat, lon)
            
            # We need to offset to align with trajectory
            # For now, use a fixed offset based on Karlsruhe
            # This is approximate - proper alignment would use exact GPS reference
            local_x = east 
            local_y = north
            
            local_edge.append((local_x, local_y))
        local_edges.append(local_edge)
    
    return local_edges


def create_osm_bev_mask(osm_edges_local, ego_pose, grid_size=(300, 400),
                        resolution=0.5, x_range=(-75, 75), y_range=(-75, 75)):
    """
    Create BEV binary mask from OSM roads.
    
    Args:
        osm_edges_local: List of road edges in local frame
        ego_pose: (x, y, yaw) ego vehicle pose
        grid_size: (H, W) BEV grid size
        resolution: meters per pixel
        x_range: (min, max) x coordinates
        y_range: (min, max) y coordinates
    
    Returns:
        bev_mask: [H, W] binary mask
    """
    H, W = grid_size
    mask = np.zeros((H, W), dtype=np.float32)
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    ego_x, ego_y, ego_yaw = ego_pose
    
    # Transform points to ego frame and draw
    for edge in osm_edges_local:
        for i in range(len(edge) - 1):
            pt1_world = edge[i]
            pt2_world = edge[i + 1]
            
            # Transform to ego frame
            dx1 = pt1_world[0] - ego_x
            dy1 = pt1_world[1] - ego_y
            dx2 = pt2_world[0] - ego_x
            dy2 = pt2_world[1] - ego_y
            
            # Rotate by -ego_yaw
            cos_yaw = np.cos(-ego_yaw)
            sin_yaw = np.sin(-ego_yaw)
            
            x1 = dx1 * cos_yaw - dy1 * sin_yaw
            y1 = dx1 * sin_yaw + dy1 * cos_yaw
            x2 = dx2 * cos_yaw - dy2 * sin_yaw
            y2 = dx2 * sin_yaw + dy2 * cos_yaw
            
            # Check if within range
            if not (x_min <= x1 <= x_max and y_min <= y1 <= y_max):
                continue
            if not (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
                continue
            
            # Convert to pixels
            px1 = int((x1 - x_min) / resolution)
            py1 = int((y1 - y_min) / resolution)
            px2 = int((x2 - x_min) / resolution)
            py2 = int((y2 - y_min) / resolution)
            
            # Clip to grid
            px1 = np.clip(px1, 0, W - 1)
            py1 = np.clip(py1, 0, H - 1)
            px2 = np.clip(px2, 0, W - 1)
            py2 = np.clip(py2, 0, H - 1)
            
            # Draw line (simple)
            mask = draw_line(mask, (px1, py1), (px2, py2))
    
    return mask


def draw_line(grid, p1, p2, width=2):
    """Draw line on grid."""
    x1, y1 = p1
    x2, y2 = p2
    H, W = grid.shape
    
    steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
    for t in np.linspace(0, 1, steps):
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        
        for dx in range(-width, width + 1):
            for dy in range(-width, width + 1):
                px, py = x + dx, y + dy
                if 0 <= px < W and 0 <= py < H:
                    grid[py, px] = 1.0
    
    return grid


def visualize_results(trajectory, osm_edges_local, bev_masks, output_dir):
    """Visualize the alignment and BEV masks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Trajectory and OSM roads
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='x', label='End')
    
    # Plot OSM roads (sample)
    sample_edges = osm_edges_local[::10]  # Every 10th edge for clarity
    for edge in sample_edges:
        xs = [p[0] for p in edge]
        ys = [p[1] for p in edge]
        ax.plot(xs, ys, 'gray', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Trajectory and OSM Roads (UTM Coordinates)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'trajectory_osm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved trajectory visualization")
    
    # Plot 2: BEV masks for sample frames
    n_samples = len(bev_masks)
    fig, axes = plt.subplots(1, n_samples, figsize=(4*n_samples, 4))
    if n_samples == 1:
        axes = [axes]
    
    for idx, (frame_idx, mask) in enumerate(sorted(bev_masks.items())):
        ax = axes[idx]
        ax.imshow(mask, origin='lower', cmap='gray')
        ax.set_title(f'Frame {frame_idx}\n({mask.sum():.0f} road pixels)')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bev_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved BEV masks visualization")


def main():
    """Main processing pipeline."""
    
    # Paths
    oxts_dir = Path('data/raw_data/2011_10_03_drive_0042_sync/oxts/data')
    osm_pkl = Path('data/osm/00_osm.pkl')  # Use sequence 00 OSM as demo (same area roughly)
    output_dir = Path('data/osm_aligned')
    
    print("="*60)
    print("OSM Alignment Demo - Drive 0042")
    print("="*60)
    
    # Step 1: Load OXTS data
    print("\n1. Loading OXTS data...")
    oxts_data = load_oxts_data(oxts_dir)
    print(f"   Loaded {len(oxts_data)} OXTS frames")
    print(f"   Sample OXTS: lat={oxts_data[0,0]:.6f}, lon={oxts_data[0,1]:.6f}")
    
    # Step 2: Convert to trajectory
    print("\n2. Converting OXTS to trajectory...")
    trajectory = oxts_to_trajectory(oxts_data)
    print(f"   Trajectory shape: {trajectory.shape}")
    print(f"   Distance traveled: {np.sum(np.sqrt(np.sum(np.diff(trajectory[:, :2], axis=0)**2, axis=1))):.1f} meters")
    
    # Step 3: Load OSM
    print("\n3. Loading OSM data...")
    with open(osm_pkl, 'rb') as f:
        osm_data = pickle.load(f)
    osm_edges = osm_data['edges']
    print(f"   Loaded {len(osm_edges)} OSM edges")
    
    # Step 4: Transform OSM to local frame (aligned with trajectory)
    print("\n4. Transforming OSM to local frame...")
    # For demo: just convert OSM GPS to UTM
    # In practice, would align with trajectory start position
    osm_edges_local = []
    for edge in osm_edges:
        local_edge = []
        for lat, lon in edge:
            east, north = latlon_to_utm(lat, lon)
            local_edge.append((east, north))
        osm_edges_local.append(local_edge)
    print(f"   Transformed {len(osm_edges_local)} edges to UTM")
    
    # Step 5: Create BEV masks
    print("\n5. Creating BEV masks (Imap)...")
    bev_masks = {}
    sample_frames = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4, len(trajectory)-1]
    
    for frame_idx in sample_frames:
        ego_pose = tuple(trajectory[frame_idx])  # (x, y, yaw)
        
        mask = create_osm_bev_mask(osm_edges_local, ego_pose)
        bev_masks[frame_idx] = mask
        
        coverage = mask.sum() / mask.size * 100
        print(f"   Frame {frame_idx:5d}: {mask.sum():6.0f} pixels ({coverage:5.2f}% coverage)")
    
    # Step 6: Save results
    print("\n6. Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aligned data
    aligned_data = {
        'trajectory': trajectory,
        'osm_edges_local': osm_edges_local,
        'bev_masks': bev_masks
    }
    with open(output_dir / 'drive0042_aligned.pkl', 'wb') as f:
        pickle.dump(aligned_data, f)
    print(f"   Saved aligned data to {output_dir / 'drive0042_aligned.pkl'}")
    
    # Step 7: Visualize
    print("\n7. Creating visualizations...")
    visualize_results(trajectory, osm_edges_local, bev_masks, output_dir)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
