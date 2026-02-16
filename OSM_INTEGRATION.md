# OSM Integration Guide for TopoDiffuser

## Overview

TopoDiffuser uses a topometric map (OSM road network) as one of its three input modalities:
- **LiDAR BEV** (3 channels): Height, Intensity, Density
- **History Trajectory** (1 channel): Past motion
- **OSM Route** (1 channel): Road network from OpenStreetMap

## Current Status

### Option 1: Trajectory as Proxy (✅ Working)
**Current implementation**: Uses the trajectory history as the road mask.

**Pros:**
- Always aligned with ego frame
- No additional data needed
- Indicates where the vehicle has been driving

**Cons:**
- Doesn't show future road topology
- May miss intersections or lane changes

### Option 2: Approximate OSM (⚠️ Partial)
**Status**: OSM data downloaded (`data/osm/XX_osm.pkl`) but coordinate alignment needs work.

The OSM edges are in GPS coordinates (lat/lon). To use them:
1. Convert GPS → UTM
2. Align UTM with KITTI local frame
3. Transform to ego frame

**Challenge**: KITTI poses are in a local coordinate frame, not GPS. Alignment requires OXTS data.

### Option 3: GPS-Matched OSM (⬜ Not Implemented)
**Requirements**: Download 22GB KITTI raw data + OXTS for GPS coordinates.

**Steps:**
1. Download KITTI raw data for sequences 00, 02, 05, 07, 08, 09, 10
2. Extract OXTS data (GPS/IMU readings)
3. Use OXTS to compute GPS → local frame transformation
4. Transform OSM roads to ego frame

## Recommendation

For initial training and testing, **Option 1 (trajectory proxy)** is sufficient. The model can learn road-following behavior from the trajectory history.

To upgrade to proper OSM integration:
```bash
# Download KITTI raw data (22GB total)
python data/download_kitti_raw_gps.py

# This provides OXTS data for GPS alignment
# Then update utils/dataset.py to use proper coordinate transformation
```

## Input Tensor Structure

```
Input BEV: [5, H, W]
├── Channel 0: LiDAR Maximum Height
├── Channel 1: LiDAR Intensity  
├── Channel 2: LiDAR Density
├── Channel 3: Trajectory History (or OSM proxy)
└── Channel 4: OSM Road Mask (currently same as history)
```

## Paper Reference

From the TopoDiffuser paper:
> "The sparse topometric route derived from OpenStreetMap (OSM) is converted 
> into a binary mask Imap ∈ RH0×W0×1 that indicates the feasible driving 
> corridor in the local BEV space."
