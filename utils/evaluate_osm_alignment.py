"""
Evaluate OSM Alignment Accuracy.

Metrics:
1. Trajectory-to-road distance (should be small)
2. Road coverage (trajectory should be on roads)
3. Heading alignment (trajectory direction vs road direction)
4. Visual validation
"""

import numpy as np
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


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


def load_oxts_data(oxts_dir):
    """Load OXTS data."""
    oxts_files = sorted(Path(oxts_dir).glob('*.txt'))
    return np.array([np.loadtxt(f) for f in oxts_files])


def extract_osm_points(osm_edges):
    """Extract all OSM road points as a flat list."""
    points = []
    for edge in osm_edges:
        for pt in edge:
            points.append(pt)
    return np.array(points)


def compute_nearest_road_distances(trajectory_xy, osm_edges):
    """
    Compute distance from each trajectory point to nearest OSM road.
    
    Returns:
        distances: [N] array of minimum distances
        stats: dict with statistics
    """
    # Extract all OSM road points
    osm_points = extract_osm_points(osm_edges)
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(osm_points)
    
    # Query nearest neighbor for each trajectory point
    distances, indices = tree.query(trajectory_xy, k=1)
    
    stats = {
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'percentile_95': np.percentile(distances, 95),
        'percentile_99': np.percentile(distances, 99),
        'on_road_ratio': np.mean(distances < 5.0),  # Within 5m
        'close_to_road_ratio': np.mean(distances < 10.0),  # Within 10m
    }
    
    return distances, stats


def compute_heading_alignment(trajectory, osm_edges, window=5):
    """
    Compute alignment between trajectory heading and road direction.
    
    Returns:
        alignment_scores: [N] array of heading differences (radians)
        stats: dict with statistics
    """
    traj_x = trajectory[:, 0]
    traj_y = trajectory[:, 1]
    traj_yaw = trajectory[:, 2]
    
    alignments = []
    
    for i in range(len(trajectory)):
        # Get trajectory heading at this point
        if i < window:
            dx = traj_x[i+window] - traj_x[i]
            dy = traj_y[i+window] - traj_y[i]
        elif i >= len(trajectory) - window:
            dx = traj_x[i] - traj_x[i-window]
            dy = traj_y[i] - traj_y[i-window]
        else:
            dx = traj_x[i+window] - traj_x[i-window]
            dy = traj_y[i+window] - traj_y[i-window]
        
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            alignments.append(0)
            continue
            
        traj_heading = np.arctan2(dy, dx)
        
        # Find nearest OSM road segment and its direction
        min_dist = float('inf')
        road_heading = traj_heading  # default
        
        for edge in osm_edges:
            for j in range(len(edge) - 1):
                # Distance from point to line segment
                p1 = np.array(edge[j])
                p2 = np.array(edge[j+1])
                p = np.array([traj_x[i], traj_y[i]])
                
                # Project point onto line segment
                line_vec = p2 - p1
                line_len_sq = np.dot(line_vec, line_vec)
                
                if line_len_sq < 0.01:
                    continue
                    
                t = max(0, min(1, np.dot(p - p1, line_vec) / line_len_sq))
                projection = p1 + t * line_vec
                dist = np.linalg.norm(p - projection)
                
                if dist < min_dist:
                    min_dist = dist
                    road_heading = np.arctan2(line_vec[1], line_vec[0])
        
        # Compute heading difference
        diff = abs(np.arctan2(np.sin(traj_heading - road_heading), 
                              np.cos(traj_heading - road_heading)))
        alignments.append(diff)
    
    alignments = np.array(alignments)
    
    stats = {
        'mean_alignment_rad': np.mean(alignments),
        'mean_alignment_deg': np.degrees(np.mean(alignments)),
        'median_alignment_deg': np.degrees(np.median(alignments)),
        'well_aligned_ratio': np.mean(alignments < np.radians(15)),  # Within 15°
        'moderately_aligned_ratio': np.mean(alignments < np.radians(30)),  # Within 30°
    }
    
    return alignments, stats


def evaluate_coverage(trajectory_xy, osm_edges, grid_size=50):
    """
    Evaluate spatial coverage of OSM roads relative to trajectory.
    
    Returns:
        coverage_stats: dict with coverage metrics
    """
    # Get bounding box of trajectory
    x_min, x_max = trajectory_xy[:, 0].min(), trajectory_xy[:, 0].max()
    y_min, y_max = trajectory_xy[:, 1].min(), trajectory_xy[:, 1].max()
    
    # Create grid
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)
    
    # Digitize trajectory points
    traj_x_bins = np.digitize(trajectory_xy[:, 0], x_bins)
    traj_y_bins = np.digitize(trajectory_xy[:, 1], y_bins)
    
    # Digitize OSM points
    osm_points = extract_osm_points(osm_edges)
    osm_x_bins = np.digitize(osm_points[:, 0], x_bins)
    osm_y_bins = np.digitize(osm_points[:, 1], y_bins)
    
    # Count cells with trajectory and OSM
    traj_cells = set(zip(traj_x_bins, traj_y_bins))
    osm_cells = set(zip(osm_x_bins, osm_y_bins))
    
    covered_cells = traj_cells.intersection(osm_cells)
    
    coverage_stats = {
        'trajectory_cells': len(traj_cells),
        'osm_cells': len(osm_cells),
        'covered_cells': len(covered_cells),
        'coverage_ratio': len(covered_cells) / len(traj_cells) if traj_cells else 0,
        'trajectory_area_m2': (x_max - x_min) * (y_max - y_min),
    }
    
    return coverage_stats


def visualize_evaluation(trajectory, osm_edges, distances, alignments, output_dir):
    """Create evaluation visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Trajectory with distance heatmap
    ax = axes[0, 0]
    scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], 
                         c=distances, cmap='RdYlGn_r', s=10, 
                         vmin=0, vmax=20)
    plt.colorbar(scatter, ax=ax, label='Distance to nearest road (m)')
    ax.set_title('Trajectory Colored by Distance to OSM Road')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distance histogram
    ax = axes[0, 1]
    ax.hist(distances, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(distances), color='red', linestyle='--', 
               label=f'Mean: {np.mean(distances):.2f}m')
    ax.axvline(np.median(distances), color='green', linestyle='--', 
               label=f'Median: {np.median(distances):.2f}m')
    ax.set_xlabel('Distance to nearest road (m)')
    ax.set_ylabel('Number of frames')
    ax.set_title('Distribution of Road Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Heading alignment
    ax = axes[1, 0]
    alignment_deg = np.degrees(alignments)
    ax.hist(alignment_deg, bins=50, color='orange', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(alignment_deg), color='red', linestyle='--',
               label=f'Mean: {np.mean(alignment_deg):.1f}°')
    ax.axvline(15, color='green', linestyle=':', label='15° threshold')
    ax.axvline(30, color='orange', linestyle=':', label='30° threshold')
    ax.set_xlabel('Heading difference (degrees)')
    ax.set_ylabel('Number of frames')
    ax.set_title('Trajectory vs Road Heading Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution of distances
    ax = axes[1, 1]
    sorted_dist = np.sort(distances)
    cumsum = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist) * 100
    ax.plot(sorted_dist, cumsum, linewidth=2, color='purple')
    ax.axvline(5, color='green', linestyle='--', alpha=0.7, label='5m')
    ax.axvline(10, color='orange', linestyle='--', alpha=0.7, label='10m')
    ax.axhline(95, color='red', linestyle=':', alpha=0.7, label='95%')
    ax.set_xlabel('Distance to nearest road (m)')
    ax.set_ylabel('Cumulative percentage (%)')
    ax.set_title('Cumulative Distribution of Road Distances')
    ax.set_xlim(0, 50)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Detailed trajectory with OSM overlay
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    
    # Plot OSM roads
    for edge in osm_edges[::5]:  # Sample for clarity
        xs, ys = zip(*edge)
        ax.plot(xs, ys, 'lightgray', linewidth=0.5, alpha=0.5)
    
    # Plot trajectory colored by distance
    scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], 
                         c=distances, cmap='RdYlGn_r', s=15,
                         vmin=0, vmax=15)
    plt.colorbar(scatter, ax=ax, label='Distance to road (m)')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('OSM Road Network with Trajectory (colored by distance)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'trajectory_detail.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("="*70)
    print("OSM ALIGNMENT EVALUATION")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    oxts_dir = Path('data/raw_data/2011_10_03_drive_0042_sync/oxts/data')
    oxts_data = load_oxts_data(oxts_dir)
    
    lats, lons, yaws = oxts_data[:, 0], oxts_data[:, 1], oxts_data[:, 2]
    utm_east = np.array([latlon_to_utm(lat, lon)[0] for lat, lon in zip(lats, lons)])
    utm_north = np.array([latlon_to_utm(lat, lon)[1] for lat, lon in zip(lats, lons)])
    
    traj_x = utm_east - utm_east[0]
    traj_y = utm_north - utm_north[0]
    trajectory = np.column_stack([traj_x, traj_y, yaws])
    trajectory_xy = trajectory[:, :2]
    
    with open('data/osm/0042_osm.pkl', 'rb') as f:
        osm_data = pickle.load(f)
    osm_edges_gps = osm_data['edges']
    
    # Transform OSM to local frame
    osm_edges_local = []
    for edge in osm_edges_gps:
        local_edge = [(latlon_to_utm(lat, lon)[0] - utm_east[0],
                       latlon_to_utm(lat, lon)[1] - utm_north[0]) 
                      for lat, lon in edge]
        osm_edges_local.append(local_edge)
    
    print(f"  Trajectory: {len(trajectory)} frames")
    print(f"  OSM edges: {len(osm_edges_local)} segments")
    
    # Compute metrics
    print("\n[2/5] Computing trajectory-to-road distances...")
    distances, distance_stats = compute_nearest_road_distances(trajectory_xy, osm_edges_local)
    
    print("\n[3/5] Computing heading alignment...")
    alignments, alignment_stats = compute_heading_alignment(trajectory, osm_edges_local)
    
    print("\n[4/5] Computing coverage statistics...")
    coverage_stats = evaluate_coverage(trajectory_xy, osm_edges_local)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📏 DISTANCE METRICS:")
    print(f"  Mean distance to road:     {distance_stats['mean_distance']:.2f} m")
    print(f"  Median distance:           {distance_stats['median_distance']:.2f} m")
    print(f"  Std deviation:             {distance_stats['std_distance']:.2f} m")
    print(f"  95th percentile:           {distance_stats['percentile_95']:.2f} m")
    print(f"  99th percentile:           {distance_stats['percentile_99']:.2f} m")
    print(f"  Max distance:              {distance_stats['max_distance']:.2f} m")
    
    print("\n🎯 COVERAGE METRICS:")
    print(f"  On road (< 5m):            {distance_stats['on_road_ratio']*100:.1f}%")
    print(f"  Close to road (< 10m):     {distance_stats['close_to_road_ratio']*100:.1f}%")
    print(f"  Spatial coverage:          {coverage_stats['coverage_ratio']*100:.1f}%")
    
    print("\n🧭 HEADING ALIGNMENT:")
    print(f"  Mean heading diff:         {alignment_stats['mean_alignment_deg']:.1f}°")
    print(f"  Median heading diff:       {alignment_stats['median_alignment_deg']:.1f}°")
    print(f"  Well aligned (< 15°):      {alignment_stats['well_aligned_ratio']*100:.1f}%")
    print(f"  Moderately aligned (< 30°): {alignment_stats['moderately_aligned_ratio']*100:.1f}%")
    
    # Qualitative assessment
    print("\n📊 QUALITY ASSESSMENT:")
    
    score = 0
    if distance_stats['median_distance'] < 5:
        print("  ✅ Median distance < 5m: EXCELLENT")
        score += 2
    elif distance_stats['median_distance'] < 10:
        print("  ⚠️  Median distance 5-10m: GOOD")
        score += 1
    else:
        print("  ❌ Median distance > 10m: POOR")
    
    if distance_stats['on_road_ratio'] > 0.8:
        print("  ✅ >80% on road: EXCELLENT")
        score += 2
    elif distance_stats['on_road_ratio'] > 0.6:
        print("  ⚠️  60-80% on road: GOOD")
        score += 1
    else:
        print("  ❌ <60% on road: POOR")
    
    if alignment_stats['well_aligned_ratio'] > 0.7:
        print("  ✅ >70% well aligned: EXCELLENT")
        score += 2
    elif alignment_stats['well_aligned_ratio'] > 0.5:
        print("  ⚠️  50-70% well aligned: GOOD")
        score += 1
    else:
        print("  ❌ <50% well aligned: POOR")
    
    print(f"\n  OVERALL SCORE: {score}/6")
    if score >= 5:
        print("  🌟 ALIGNMENT QUALITY: EXCELLENT")
    elif score >= 3:
        print("  ✅ ALIGNMENT QUALITY: GOOD")
    else:
        print("  ⚠️  ALIGNMENT QUALITY: NEEDS IMPROVEMENT")
    
    # Visualize
    print("\n[5/5] Creating visualizations...")
    visualize_evaluation(trajectory, osm_edges_local, distances, alignments, 
                         'data/osm_aligned')
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nOutput files:")
    print("  • data/osm_aligned/evaluation_metrics.png")
    print("  • data/osm_aligned/trajectory_detail.png")
    
    # Save metrics
    metrics = {
        'distance': distance_stats,
        'alignment': alignment_stats,
        'coverage': coverage_stats,
        'score': score
    }
    
    with open('data/osm_aligned/evaluation_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("  • data/osm_aligned/evaluation_metrics.pkl")


if __name__ == '__main__':
    main()
