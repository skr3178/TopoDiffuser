"""
OSM Edges Processing Utilities.

Handles loading, splitting, and processing of OSM road edge data.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pickle


def split_osm_edges_to_polylines(edges_array: np.ndarray, 
                                  gap_threshold: float = 0.001) -> List[List[Tuple[float, float]]]:
    """
    Split flat OSM edges array into individual road polylines.
    
    The input array is a flat (N, 2) array of (lat, lon) points where consecutive
    points belong to the same road segment unless there's a large gap.
    
    Args:
        edges_array: [N, 2] array of (lat, lon) points
        gap_threshold: Gap threshold in degrees. Default 0.001 degrees ≈ 100m
    
    Returns:
        List of polylines, where each polyline is [(lat, lon), ...]
    """
    if len(edges_array) == 0:
        return []
    
    polylines = []
    current_polyline = [tuple(edges_array[0])]
    
    for i in range(1, len(edges_array)):
        prev_point = edges_array[i-1]
        curr_point = edges_array[i]
        
        # Check gap between consecutive points
        gap = np.linalg.norm(curr_point - prev_point)
        
        if gap > gap_threshold:
            # Large gap indicates start of new road segment
            if len(current_polyline) > 1:
                polylines.append(current_polyline)
            current_polyline = [tuple(curr_point)]
        else:
            current_polyline.append(tuple(curr_point))
    
    # Don't forget the last polyline
    if len(current_polyline) > 1:
        polylines.append(current_polyline)
    
    return polylines


def load_and_split_osm_edges(edges_path: Path, 
                              gap_threshold: float = 0.001) -> Dict:
    """
    Load OSM edges from .npy file and split into polylines.
    
    Args:
        edges_path: Path to {seq}_edges.npy file
        gap_threshold: Gap threshold in degrees
    
    Returns:
        Dictionary with polylines and statistics
    """
    if not edges_path.exists():
        return {
            'polylines': [],
            'num_polylines': 0,
            'total_points': 0,
            'error': f'File not found: {edges_path}'
        }
    
    # Load edges
    edges_array = np.load(edges_path)
    
    # Split into polylines
    polylines = split_osm_edges_to_polylines(edges_array, gap_threshold)
    
    # Compute statistics
    total_points = sum(len(p) for p in polylines)
    avg_length = np.mean([len(p) for p in polylines]) if polylines else 0
    
    # Compute bounding box
    all_points = np.array([p for polyline in polylines for p in polyline])
    if len(all_points) > 0:
        lat_min, lon_min = all_points.min(axis=0)
        lat_max, lon_max = all_points.max(axis=0)
        bounds = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
    else:
        bounds = {}
    
    return {
        'polylines': polylines,
        'num_polylines': len(polylines),
        'total_points': total_points,
        'avg_points_per_polyline': avg_length,
        'bounds': bounds,
        'gap_threshold': gap_threshold
    }


def save_polylines(polylines: List[List[Tuple[float, float]]], 
                   output_path: Path):
    """
    Save polylines to pickle file.
    
    Args:
        polylines: List of polylines
        output_path: Output path (.pkl)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(polylines, f)
    print(f"Saved {len(polylines)} polylines to {output_path}")


def load_polylines(input_path: Path) -> List[List[Tuple[float, float]]]:
    """
    Load polylines from pickle file.
    
    Args:
        input_path: Path to .pkl file
    
    Returns:
        List of polylines
    """
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def filter_polylines_by_bounds(polylines: List[List[Tuple[float, float]]],
                                lat_min: float, lat_max: float,
                                lon_min: float, lon_max: float) -> List[List[Tuple[float, float]]]:
    """
    Filter polylines to only include those within bounds.
    
    Args:
        polylines: List of polylines
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds
    
    Returns:
        Filtered polylines
    """
    filtered = []
    
    for polyline in polylines:
        # Check if any point is within bounds
        for lat, lon in polyline:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                filtered.append(polyline)
                break
    
    return filtered


def compute_polyline_lengths(polylines: List[List[Tuple[float, float]]]) -> List[float]:
    """
    Compute approximate length of each polyline in meters.
    
    Args:
        polylines: List of polylines with (lat, lon) points
    
    Returns:
        List of lengths in meters
    """
    from utils.bev_multimodal import latlon_to_utm
    
    lengths = []
    
    for polyline in polylines:
        if len(polyline) < 2:
            lengths.append(0.0)
            continue
        
        # Convert to UTM and compute segment lengths
        utm_points = []
        for lat, lon in polyline:
            east, north = latlon_to_utm(lat, lon)
            utm_points.append([east, north])
        
        utm_points = np.array(utm_points)
        segments = np.diff(utm_points, axis=0)
        length = np.sum(np.linalg.norm(segments, axis=1))
        lengths.append(length)
    
    return lengths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process OSM edges files'
    )
    parser.add_argument('--osm_dir', type=str, default='data/osm',
                       help='Directory containing {seq}_edges.npy files')
    parser.add_argument('--sequences', nargs='+',
                       default=['00', '02', '05', '07', '08', '09', '10'],
                       help='Sequences to process')
    parser.add_argument('--gap_threshold', type=float, default=0.001,
                       help='Gap threshold in degrees (~0.001 = 100m)')
    parser.add_argument('--output_dir', type=str, default='data/osm_polylines',
                       help='Output directory for polyline .pkl files')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OSM Edges Processing")
    print("=" * 70)
    
    osm_dir = Path(args.osm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for seq in args.sequences:
        edges_path = osm_dir / f'{seq}_edges.npy'
        
        print(f"\nSequence {seq}:")
        print(f"  Loading {edges_path}")
        
        result = load_and_split_osm_edges(edges_path, args.gap_threshold)
        
        if 'error' in result:
            print(f"  ❌ {result['error']}")
            continue
        
        print(f"  ✓ Split into {result['num_polylines']} polylines")
        print(f"  Total points: {result['total_points']}")
        print(f"  Avg points/polyline: {result['avg_points_per_polyline']:.1f}")
        
        if result['bounds']:
            b = result['bounds']
            print(f"  Bounds: lat [{b['lat_min']:.4f}, {b['lat_max']:.4f}], "
                  f"lon [{b['lon_min']:.4f}, {b['lon_max']:.4f}]")
        
        # Save polylines
        output_path = output_dir / f'{seq}_polylines.pkl'
        save_polylines(result['polylines'], output_path)
        
        # Compute and display length statistics
        lengths = compute_polyline_lengths(result['polylines'])
        if lengths:
            print(f"  Road length stats: min={min(lengths):.1f}m, "
                  f"max={max(lengths):.1f}m, "
                  f"total={sum(lengths)/1000:.1f}km")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
