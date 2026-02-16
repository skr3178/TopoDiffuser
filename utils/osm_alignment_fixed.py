"""
Fixed OSM Alignment Demo.

Properly aligns OSM roads with OXTS trajectory by using the same coordinate origin.
"""

import numpy as np
from pathlib import Path
import pickle


def load_oxts_data(oxts_dir):
    """Load OXTS data."""
    oxts_files = sorted(Path(oxts_dir).glob('*.txt'))
    return np.array([np.loadtxt(f) for f in oxts_files])


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


def main():
    print("="*60)
    print("OSM Alignment - Fixed Version")
    print("="*60)
    
    # Load data
    oxts_dir = Path('data/raw_data/2011_10_03_drive_0042_sync/oxts/data')
    oxts_data = load_oxts_data(oxts_dir)
    print(f"\n1. Loaded {len(oxts_data)} OXTS frames")
    
    # Convert OXTS trajectory to UTM
    lats = oxts_data[:, 0]
    lons = oxts_data[:, 1]
    yaws = oxts_data[:, 5]
    
    # Get UTM coordinates
    utm_east = []
    utm_north = []
    for lat, lon in zip(lats, lons):
        e, n = latlon_to_utm(lat, lon)
        utm_east.append(e)
        utm_north.append(n)
    
    utm_east = np.array(utm_east)
    utm_north = np.array(utm_north)
    
    print(f"   UTM range: East [{utm_east.min():.0f}, {utm_east.max():.0f}]")
    print(f"             North [{utm_north.min():.0f}, {utm_north.max():.0f}]")
    
    # Center trajectory at origin
    traj_x = utm_east - utm_east[0]
    traj_y = utm_north - utm_north[0]
    trajectory = np.column_stack([traj_x, traj_y, yaws])
    
    print(f"\n2. Trajectory (centered at origin):")
    print(f"   Distance traveled: {np.sum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2)):.1f} m")
    print(f"   Start: ({traj_x[0]:.1f}, {traj_y[0]:.1f})")
    print(f"   End: ({traj_x[-1]:.1f}, {traj_y[-1]:.1f})")
    
    # Load OSM
    with open('data/osm/00_osm.pkl', 'rb') as f:
        osm_data = pickle.load(f)
    osm_edges_gps = osm_data['edges']
    print(f"\n3. Loaded {len(osm_edges_gps)} OSM edges")
    
    # Convert OSM to UTM using SAME origin as trajectory
    osm_edges_local = []
    for edge in osm_edges_gps:
        local_edge = []
        for lat, lon in edge:
            east, north = latlon_to_utm(lat, lon)
            # Subtract the SAME origin as trajectory
            x = east - utm_east[0]
            y = north - utm_north[0]
            local_edge.append((x, y))
        osm_edges_local.append(local_edge)
    
    # Check overlap
    all_osm_x = [p[0] for edge in osm_edges_local for p in edge]
    all_osm_y = [p[1] for edge in osm_edges_local for p in edge]
    print(f"\n4. OSM in local frame:")
    print(f"   X range: [{min(all_osm_x):.1f}, {max(all_osm_x):.1f}]")
    print(f"   Y range: [{min(all_osm_y):.1f}, {max(all_osm_y):.1f}]")
    print(f"   Trajectory X range: [{traj_x.min():.1f}, {traj_x.max():.1f}]")
    print(f"   Trajectory Y range: [{traj_y.min():.1f}, {traj_y.max():.1f}]")
    
    # Check if they overlap
    osm_bounds_x = (min(all_osm_x), max(all_osm_x))
    osm_bounds_y = (min(all_osm_y), max(all_osm_y))
    traj_bounds_x = (traj_x.min(), traj_x.max())
    traj_bounds_y = (traj_y.min(), traj_y.max())
    
    overlap_x = max(0, min(osm_bounds_x[1], traj_bounds_x[1]) - max(osm_bounds_x[0], traj_bounds_x[0]))
    overlap_y = max(0, min(osm_bounds_y[1], traj_bounds_y[1]) - max(osm_bounds_y[0], traj_bounds_y[0]))
    
    print(f"\n5. Overlap check:")
    print(f"   X overlap: {overlap_x:.1f} m")
    print(f"   Y overlap: {overlap_y:.1f} m")
    
    if overlap_x > 0 and overlap_y > 0:
        print("   ✓ Trajectory and OSM overlap!")
    else:
        print("   ✗ No overlap - different areas of Karlsruhe")
        print(f"\n   This is expected: Drive 0042 is in a different location")
        print(f"   than sequence 00's OSM data.")
        print(f"\n   To get proper OSM for drive 0042, you would need to:")
        print(f"   1. Get the GPS bounds of drive 0042")
        print(f"   2. Download OSM for that specific area")
        print(f"   3. Or use sequence-matched OSM (if drive 0042 corresponds to a sequence)")


if __name__ == '__main__':
    main()
