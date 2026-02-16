"""
Final OSM Alignment Demo - Drive 0042 with correct OSM.

Demonstrates:
1. Load OXTS (GPS) data
2. Convert to local trajectory (UTM centered at origin)
3. Load OSM road network (downloaded for drive 0042's area)
4. Align OSM with trajectory (same coordinate origin)
5. Convert to ego vehicle BEV space
6. Create Imap binary mask
"""

import numpy as np
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def latlon_to_utm(lat, lon, zone=32):
    """Convert GPS to UTM."""
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
                     (5 - 18*T + T**2 + 72*C - 58*0.006739497) * A**5 / 120) + 500000
    
    north = k0 * (M + N * np.tan(lat_rad) * (A**2 / 2 + 
                  (5 - T + 9*C + 4*C**2) * A**4 / 24 +
                  (61 - 58*T + T**2 + 600*C - 330*0.006739497) * A**6 / 720))
    
    return east, north


def create_bev_mask(osm_edges_local, ego_pose, grid_size=(300, 400),
                    resolution=0.5, x_range=(-75, 75), y_range=(-75, 75)):
    """Create BEV binary mask from OSM roads (Imap)."""
    H, W = grid_size
    mask = np.zeros((H, W), dtype=np.float32)
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    ego_x, ego_y, ego_yaw = ego_pose
    
    for edge in osm_edges_local:
        for i in range(len(edge) - 1):
            # World to ego frame
            dx1, dy1 = edge[i][0] - ego_x, edge[i][1] - ego_y
            dx2, dy2 = edge[i+1][0] - ego_x, edge[i+1][1] - ego_y
            
            cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw)
            x1 = dx1 * cos_yaw - dy1 * sin_yaw
            y1 = dx1 * sin_yaw + dy1 * cos_yaw
            x2 = dx2 * cos_yaw - dy2 * sin_yaw
            y2 = dx2 * sin_yaw + dy2 * cos_yaw
            
            # Check bounds
            if not (x_min <= x1 <= x_max and y_min <= y1 <= y_max):
                continue
            if not (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
                continue
            
            # To pixels
            px1 = int((x1 - x_min) / resolution)
            py1 = int((y1 - y_min) / resolution)
            px2 = int((x2 - x_min) / resolution)
            py2 = int((y2 - y_min) / resolution)
            
            # Draw line
            steps = max(abs(px2-px1), abs(py2-py1)) + 1
            for t in np.linspace(0, 1, int(steps)):
                px = int(px1 + t * (px2 - px1))
                py = int(py1 + t * (py2 - py1))
                if 0 <= px < W and 0 <= py < H:
                    mask[py, px] = 1.0
    
    return mask


def main():
    print("="*70)
    print("OSM ALIGNMENT PIPELINE - Drive 0042")
    print("Creating Imap (topometric map BEV mask)")
    print("="*70)
    
    # Step 1: Load OXTS data
    print("\n[Step 1] Loading OXTS (GPS/IMU) data...")
    oxts_dir = Path('data/raw_data/2011_10_03_drive_0042_sync/oxts/data')
    oxts_files = sorted(oxts_dir.glob('*.txt'))
    oxts_data = np.array([np.loadtxt(f) for f in oxts_files])
    print(f"  ✓ Loaded {len(oxts_data)} OXTS frames")
    
    # Step 2: Convert to trajectory (UTM, centered)
    print("\n[Step 2] Converting OXTS to local trajectory...")
    lats, lons, yaws = oxts_data[:, 0], oxts_data[:, 1], oxts_data[:, 5]
    
    utm_east = np.array([latlon_to_utm(lat, lon)[0] for lat, lon in zip(lats, lons)])
    utm_north = np.array([latlon_to_utm(lat, lon)[1] for lat, lon in zip(lats, lons)])
    
    # Center at origin
    traj_x = utm_east - utm_east[0]
    traj_y = utm_north - utm_north[0]
    trajectory = np.column_stack([traj_x, traj_y, yaws])
    print(f"  ✓ Trajectory: {len(trajectory)} frames")
    print(f"     Distance: {np.sum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2)):.1f}m")
    
    # Step 3: Load OSM for drive 0042
    print("\n[Step 3] Loading OSM road network...")
    with open('data/osm/0042_osm.pkl', 'rb') as f:
        osm_data = pickle.load(f)
    osm_edges_gps = osm_data['edges']
    print(f"  ✓ Loaded {len(osm_edges_gps)} road segments")
    
    # Step 4: Transform OSM to local frame (same origin as trajectory)
    print("\n[Step 4] Aligning OSM with trajectory...")
    osm_edges_local = []
    for edge in osm_edges_gps:
        local_edge = [(latlon_to_utm(lat, lon)[0] - utm_east[0],
                       latlon_to_utm(lat, lon)[1] - utm_north[0]) for lat, lon in edge]
        osm_edges_local.append(local_edge)
    
    # Verify overlap
    all_osm_x = [p[0] for edge in osm_edges_local for p in edge]
    all_osm_y = [p[1] for edge in osm_edges_local for p in edge]
    print(f"  ✓ OSM X range: [{min(all_osm_x):.1f}, {max(all_osm_x):.1f}]")
    print(f"     OSM Y range: [{min(all_osm_y):.1f}, {max(all_osm_y):.1f}]")
    print(f"     Traj X range: [{traj_x.min():.1f}, {traj_x.max():.1f}]")
    print(f"     Traj Y range: [{traj_y.min():.1f}, {traj_y.max():.1f}]")
    
    # Check overlap
    overlap_x = max(0, min(max(all_osm_x), traj_x.max()) - max(min(all_osm_x), traj_x.min()))
    overlap_y = max(0, min(max(all_osm_y), traj_y.max()) - max(min(all_osm_y), traj_y.min()))
    if overlap_x > 0 and overlap_y > 0:
        print(f"  ✓ Overlap confirmed: {overlap_x:.1f}m x {overlap_y:.1f}m")
    
    # Step 5: Create BEV masks (Imap)
    print("\n[Step 5] Creating Imap (BEV road masks)...")
    output_dir = Path('data/osm_aligned')
    output_dir.mkdir(exist_ok=True)
    
    bev_masks = {}
    sample_frames = [0, 300, 600, 900, 1169]
    
    for frame_idx in sample_frames:
        ego_pose = tuple(trajectory[frame_idx])  # (x, y, yaw)
        mask = create_bev_mask(osm_edges_local, ego_pose)
        bev_masks[frame_idx] = mask
        coverage = mask.sum() / mask.size * 100
        print(f"  Frame {frame_idx:4d}: {mask.sum():5.0f} pixels ({coverage:.2f}% coverage)")
    
    # Step 6: Save results
    print("\n[Step 6] Saving results...")
    results = {
        'trajectory': trajectory,
        'osm_edges_local': osm_edges_local,
        'bev_masks': bev_masks,
        'utm_origin': (utm_east[0], utm_north[0])
    }
    with open(output_dir / 'drive0042_complete.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✓ Saved to {output_dir / 'drive0042_complete.pkl'}")
    
    # Step 7: Visualize
    print("\n[Step 7] Creating visualizations...")
    
    # Plot 1: Full overview
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Trajectory + OSM
    ax = axes[0]
    for edge in osm_edges_local[::5]:  # Sample for clarity
        xs, ys = zip(*edge)
        ax.plot(xs, ys, 'gray', linewidth=0.5, alpha=0.5)
    ax.plot(traj_x, traj_y, 'b-', linewidth=2, label='Trajectory')
    ax.scatter(traj_x[0], traj_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(traj_x[-1], traj_y[-1], c='red', s=100, marker='x', label='End', zorder=5)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('Drive 0042: Trajectory and OSM Roads (Aligned)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Right: Sample BEV masks
    ax = axes[1]
    # Show frame 600 mask with trajectory context
    frame_idx = 600
    mask = bev_masks[frame_idx]
    ego_x, ego_y, ego_yaw = trajectory[frame_idx]
    
    im = ax.imshow(mask, origin='lower', cmap='Greens', extent=(-75, 75, -75, 75))
    ax.set_title(f'Imap (BEV Road Mask) - Frame {frame_idx}\nEgo at origin, yaw={np.degrees(ego_yaw):.1f}°')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    plt.colorbar(im, ax=ax, label='Road present')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drive0042_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved overview plot")
    
    # Plot 2: All BEV masks
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(sample_frames):
        ax = axes[idx]
        mask = bev_masks[frame_idx]
        ax.imshow(mask, origin='lower', cmap='Greens')
        ax.set_title(f'Frame {frame_idx}\n{mask.sum():.0f} road pixels ({mask.sum()/mask.size*100:.1f}%)')
        ax.axis('off')
    
    # Last subplot: combined view
    axes[-1].axis('off')
    axes[-1].text(0.5, 0.5, 'Imap = Binary mask\nshowing drivable roads\nin BEV space\n\nShape: [H, W] = [300, 400]\nResolution: 0.5m/pixel\nRange: ±75m', 
                  transform=axes[-1].transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drive0042_bev_masks.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved BEV masks plot")
    
    print("\n" + "="*70)
    print("✓ OSM ALIGNMENT COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print(f"  • Aligned data: {output_dir / 'drive0042_complete.pkl'}")
    print(f"  • Overview plot: {output_dir / 'drive0042_overview.png'}")
    print(f"  • BEV masks plot: {output_dir / 'drive0042_bev_masks.png'}")
    print("\nThe Imap (BEV road mask) is ready to use in TopoDiffuser!")
    print("Shape: [300, 400] binary mask showing drivable road corridors")


if __name__ == '__main__':
    main()
