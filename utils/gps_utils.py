"""
GPS/UTM coordinate conversion utilities.

Converts between GPS (lat/lon), UTM (metric), and KITTI local coordinates.
"""

import numpy as np

# UTM zone for Karlsruhe, Germany is 32U
UTM_ZONE = 32
UTM_HEMISPHERE = 'N'


def latlon_to_utm(lat, lon, zone=32):
    """
    Convert latitude/longitude to UTM coordinates.
    
    This is a simplified conversion for Karlsruhe area (UTM zone 32U).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zone: UTM zone number
    
    Returns:
        (east, north): UTM coordinates in meters
    """
    # WGS84 parameters
    a = 6378137.0  # Semi-major axis
    e = 0.0818191908426  # Eccentricity
    
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # UTM zone central meridian
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    
    # UTM parameters
    k0 = 0.9996  # Scale factor
    
    # Calculate UTM coordinates
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
    east += 500000  # UTM easting offset
    
    north = k0 * (M + N * np.tan(lat_rad) * (A**2 / 2 + 
                  (5 - T + 9*C + 4*C**2) * A**4 / 24 +
                  (61 - 58*T + T**2 + 600*C - 330*0.006739497) * A**6 / 720))
    
    # Add northing offset for southern hemisphere
    if lat < 0:
        north += 10000000
    
    return east, north


def utm_to_latlon(east, north, zone=32, northern=True):
    """
    Convert UTM coordinates to latitude/longitude.
    
    Args:
        east: UTM easting in meters
        north: UTM northing in meters
        zone: UTM zone number
        northern: True if in northern hemisphere
    
    Returns:
        (lat, lon): Latitude and longitude in degrees
    """
    # WGS84 parameters
    a = 6378137.0
    e = 0.0818191908426
    
    # Adjust for UTM offsets
    x = east - 500000
    y = north
    if not northern:
        y -= 10000000
    
    # UTM zone central meridian
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    
    # Scale factor
    k0 = 0.9996
    
    # Calculate footprint latitude
    M = y / k0
    mu = M / (a * (1 - e**2/4 - 3*e**4/64 - 5*e**6/256))
    
    e1 = (1 - np.sqrt(1 - e**2)) / (1 + np.sqrt(1 - e**2))
    
    J1 = 3*e1/2 - 27*e1**3/32
    J2 = 21*e1**2/16 - 55*e1**4/32
    J3 = 151*e1**3/96
    J4 = 1097*e1**4/512
    
    fp = mu + J1*np.sin(2*mu) + J2*np.sin(4*mu) + J3*np.sin(6*mu) + J4*np.sin(8*mu)
    
    e2 = e**2 / (1 - e**2)
    C1 = e2 * np.cos(fp)**2
    T1 = np.tan(fp)**2
    R1 = a * (1 - e**2) / (1 - e**2 * np.sin(fp)**2)**1.5
    N1 = a / np.sqrt(1 - e**2 * np.sin(fp)**2)
    D = x / (N1 * k0)
    
    Q1 = N1 * np.tan(fp) / R1
    Q2 = D**2 / 2
    Q3 = (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e2) * D**4 / 24
    Q4 = (61 + 90*T1 + 298*C1 + 45*T1**2 - 3*C1**2 - 252*e2) * D**6 / 720
    
    lat = fp - Q1*(Q2 - Q3 + Q4)
    
    Q5 = D
    Q6 = (1 + 2*T1 + C1) * D**3 / 6
    Q7 = (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e2 + 24*T1**2) * D**5 / 120
    
    lon = lon0 + (Q5 - Q6 + Q7) / np.cos(fp)
    
    return np.degrees(lat), np.degrees(lon)


def gps_edges_to_kitti_frame(edges, reference_pose):
    """
    Convert GPS road edges to KITTI local coordinate frame.
    
    Args:
        edges: List of [(lat1, lon1), (lat2, lon2), ...] segments
        reference_pose: (x, y, yaw) reference pose in KITTI frame
    
    Returns:
        kitti_edges: List of [(x1, y1), (x2, y2), ...] in KITTI frame
    """
    ref_x, ref_y, ref_yaw = reference_pose
    
    # Convert reference GPS to UTM
    ref_east, ref_north = latlon_to_utm(ref_y, ref_x)
    
    kitti_edges = []
    for edge in edges:
        kitti_edge = []
        for lat, lon in edge:
            # Convert to UTM
            east, north = latlon_to_utm(lat, lon)
            
            # Convert to KITTI frame (relative to reference)
            # KITTI x is forward (north), y is left (west)
            dx = east - ref_east
            dy = north - ref_north
            
            # Rotate by -reference yaw
            cos_yaw = np.cos(-ref_yaw)
            sin_yaw = np.sin(-ref_yaw)
            
            x = dx * cos_yaw - dy * sin_yaw
            y = dx * sin_yaw + dy * cos_yaw
            
            kitti_edge.append((x, y))
        
        kitti_edges.append(kitti_edge)
    
    return kitti_edges


def load_osm_for_sequence(osm_file):
    """
    Load OSM data for a sequence.
    
    Args:
        osm_file: Path to OSM pickle file
    
    Returns:
        edges: List of road edges in GPS coordinates
        bounds: Dict with lat/lon bounds
    """
    import pickle
    
    with open(osm_file, 'rb') as f:
        data = pickle.load(f)
    
    return data['edges'], data.get('bounds', None)


if __name__ == '__main__':
    # Test coordinate conversion
    print("Testing GPS/UTM conversion...")
    
    # Karlsruhe coordinates
    lat, lon = 49.0095, 8.4267
    
    east, north = latlon_to_utm(lat, lon)
    print(f"\nKarlsruhe: lat={lat}, lon={lon}")
    print(f"UTM: east={east:.2f}, north={north:.2f}")
    
    # Round-trip test
    lat2, lon2 = utm_to_latlon(east, north)
    print(f"\nRound-trip: lat={lat2:.6f}, lon={lon2:.6f}")
    print(f"Error: lat={abs(lat-lat2):.8f}, lon={abs(lon-lon2):.8f}")
    
    # Test OSM loading
    print("\n\nTesting OSM data loading...")
    import os
    osm_file = 'data/osm/00_osm.pkl'
    
    if os.path.exists(osm_file):
        edges, bounds = load_osm_for_sequence(osm_file)
        print(f"Loaded {len(edges)} road edges")
        print(f"Bounds: {bounds}")
        print(f"First edge (GPS): {edges[0]}")
    else:
        print(f"OSM file not found: {osm_file}")
