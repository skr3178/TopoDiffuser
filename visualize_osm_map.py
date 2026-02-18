#!/usr/bin/env python3
"""
Visualize Karlsruhe OSM map data.

Requires: pip install osmnx pyrosm matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def visualize_osm_from_polylines(polylines_file: str, output_file: str = "osm_karlsruhe.png"):
    """Visualize OSM from extracted polylines."""
    
    with open(polylines_file, 'rb') as f:
        polylines = pickle.load(f)
    
    print(f"Loaded {len(polylines)} road polylines")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot each road polyline
    for i, polyline in enumerate(polylines):
        if len(polyline) < 2:
            continue
        
        lats = [p[0] for p in polyline]
        lons = [p[1] for p in polyline]
        
        # Plot with thin gray lines
        ax.plot(lons, lats, 'b-', linewidth=0.5, alpha=0.6)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Karlsruhe OSM Road Network\n{len(polylines)} road segments', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    
    return fig, ax


def visualize_osm_from_edges(edges_file: str, output_file: str = "osm_edges.png"):
    """Visualize OSM from edges file."""
    
    edges = np.load(edges_file)
    print(f"Loaded {len(edges)} road edges")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot edges as scatter (each edge is a point, connect consecutive points)
    lats = edges[:, 0]
    lons = edges[:, 1]
    
    # Plot as scatter with small points
    ax.scatter(lons, lats, s=0.1, c='blue', alpha=0.5)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Karlsruhe OSM Road Network (Edges)\n{len(edges)} points', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    
    return fig, ax


def visualize_osm_with_trajectory_overlay(osm_file: str, trajectory_file: str, 
                                          output_file: str = "osm_with_trajectory.png"):
    """Visualize OSM with KITTI trajectory overlaid."""
    
    # Load OSM edges
    edges = np.load(osm_file)
    
    # Load trajectory (poses file)
    # Extract trajectory from poses
    poses = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    
    # Extract x, z positions
    trajectory = np.array([[p[0, 3], p[2, 3]] for p in poses])
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: OSM only
    ax = axes[0]
    ax.scatter(edges[:, 1], edges[:, 0], s=0.1, c='blue', alpha=0.5)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Karlsruhe OSM Road Network', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Right: Trajectory only (in local frame)
    ax = axes[1]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='KITTI Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='*', label='End')
    ax.set_xlabel('X (m) - Local Frame', fontsize=11)
    ax.set_ylabel('Z (m) - Local Frame', fontsize=11)
    ax.set_title('KITTI Sequence Trajectory (Local Frame)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.suptitle('OSM (GPS) vs KITTI Trajectory (Local) - Different Coordinate Systems!', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    
    return fig


def visualize_osm_with_osmnx(place: str = "Karlsruhe, Germany", 
                              output_file: str = "osm_osmnx.png"):
    """Visualize OSM using osmnx (fetches from online if needed)."""
    
    try:
        import osmnx as ox
        
        print(f"Fetching OSM data for {place}...")
        # Download graph
        G = ox.graph_from_place(place, network_type="drive")
        
        print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Plot
        fig, ax = ox.plot_graph(G, figsize=(16, 12), 
                                node_size=0, 
                                edge_color='blue',
                                edge_linewidth=0.5,
                                bgcolor='white',
                                show=False,
                                close=False)
        
        plt.title(f'OSM Road Network: {place}', fontsize=14, fontweight='bold')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_file}")
        
        return fig, ax
        
    except ImportError:
        print("osmnx not installed. Install with: pip install osmnx")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Karlsruhe OSM data")
    parser.add_argument("--type", choices=['polylines', 'edges', 'trajectory', 'osmnx'], 
                       default='polylines',
                       help="Type of visualization")
    parser.add_argument("--seq", type=str, default='00',
                       help="Sequence number for trajectory overlay")
    
    args = parser.parse_args()
    
    if args.type == 'polylines':
        # Visualize from polylines pickle file
        polylines_file = f"data/osm_polylines/{args.seq}_polylines.pkl"
        if Path(polylines_file).exists():
            visualize_osm_from_polylines(polylines_file, 
                                         f"osm_polylines_seq{args.seq}.png")
        else:
            print(f"Polylines file not found: {polylines_file}")
    
    elif args.type == 'edges':
        # Visualize from edges numpy file
        edges_file = f"data/osm/{args.seq}_edges.npy"
        if Path(edges_file).exists():
            visualize_osm_from_edges(edges_file, 
                                    f"osm_edges_seq{args.seq}.png")
        else:
            print(f"Edges file not found: {edges_file}")
    
    elif args.type == 'trajectory':
        # Overlay OSM with trajectory
        edges_file = f"data/osm/{args.seq}_edges.npy"
        trajectory_file = f"data/kitti/poses/{args.seq}.txt"
        if Path(edges_file).exists() and Path(trajectory_file).exists():
            visualize_osm_with_trajectory_overlay(edges_file, trajectory_file,
                                                  f"osm_trajectory_seq{args.seq}.png")
        else:
            print(f"Required files not found")
    
    elif args.type == 'osmnx':
        # Use osmnx to fetch and visualize
        visualize_osm_with_osmnx()
