"""Download OSM for drive 0042's specific GPS area."""

import numpy as np
from pathlib import Path
try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    print("Installing osmnx...")
    import subprocess
    subprocess.run(['pip', 'install', 'osmnx', '-q'])
    import osmnx as ox


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
    print("Download OSM for Drive 0042")
    print("="*60)
    
    # Load OXTS to get GPS bounds
    oxts_dir = Path('data/raw_data/2011_10_03_drive_0042_sync/oxts/data')
    oxts_files = sorted(oxts_dir.glob('*.txt'))
    oxts_data = np.array([np.loadtxt(f) for f in oxts_files])
    
    lats = oxts_data[:, 0]
    lons = oxts_data[:, 1]
    
    print(f"\nDrive 0042 GPS bounds:")
    print(f"  Latitude:  [{lats.min():.6f}, {lats.max():.6f}]")
    print(f"  Longitude: [{lons.min():.6f}, {lons.max():.6f}]")
    
    # Center point
    center_lat = (lats.min() + lats.max()) / 2
    center_lon = (lons.min() + lons.max()) / 2
    
    # Radius to cover entire drive (with margin)
    # Approximate: 1 degree lat ~ 111km, 1 degree lon ~ 78km at 49°N
    lat_range = lats.max() - lats.min()
    lon_range = lons.max() - lons.min()
    lat_dist = lat_range * 111000  # meters
    lon_dist = lon_range * 78000   # meters
    radius = int(max(lat_dist, lon_dist) / 2 + 500)  # half range + 500m margin
    
    print(f"\nDownloading OSM from OpenStreetMap...")
    print(f"  Center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"  Radius: {radius} meters")
    
    try:
        # Download road network
        graph = ox.graph_from_point(
            (center_lat, center_lon),
            dist=radius,
            network_type='drive',
            simplify=True
        )
        
        print(f"  Downloaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Extract edges
        edges = []
        for u, v, data in graph.edges(data=True):
            if 'geometry' in data:
                coords = list(data['geometry'].coords)
                segment = [(lat, lon) for lon, lat in coords]
            else:
                node_u = graph.nodes[u]
                node_v = graph.nodes[v]
                segment = [(node_u['y'], node_u['x']), (node_v['y'], node_v['x'])]
            edges.append(segment)
        
        print(f"  Extracted {len(edges)} road segments")
        
        # Save
        import pickle
        output_dir = Path('data/osm')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / '0042_osm.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump({
                'drive': '0042',
                'center': (center_lat, center_lon),
                'radius': radius,
                'edges': edges,
                'n_edges': len(edges)
            }, f)
        
        print(f"\n✓ Saved OSM data to {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("  Make sure you have internet connection.")


if __name__ == '__main__':
    main()
