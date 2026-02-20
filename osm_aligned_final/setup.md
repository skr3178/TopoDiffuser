# OSM Alignment Setup Documentation

This document describes how the final OSM alignments in `osm_aligned_final/` were created.

## Final Alignment Summary

| Seq | Polylines | Points | Mean Error | Source | Method |
|-----|-----------|--------|------------|--------|--------|
| 00 | 379 | 5,142 | 1.21m | refined | Use existing file |
| 01 | 2,750 | 30,584 | 1.92m | bestfit | Use existing file |
| 02 | 1,380 | 14,271 | **1.43m** | largebbox | Compute: offset=0, rot=36.7°, dx=2m, dy=10m |
| 05 | 878 | 11,944 | **1.18m** | regbez | Compute: offset=0, rot=350.6° |
| 07 | 644 | 8,425 | 1.06m | regbez | Compute: offset=0, rot=122° |
| 08 | 1,181 | 13,298 | 1.12m | bestfit | Use existing file |
| 09 | 347 | 3,676 | 3.93m | refined | Use existing file |
| 10 | 765 | 7,620 | **1.54m** | karlsruhe_all_roads | Compute: offset=0, rot=106° |

**Total:** 7,463 polylines, 102,235 points (2m spacing)

---

## Sequence-Specific Configurations

### Seq 00 - Refined (Excellent)
- **Source:** `osm_polylines_aligned_seq00_refined.pkl`
- **Method:** Use existing aligned file, densify only
- **Error:** 1.21m mean, 3.11m start, 2.18m end
- **Status:** ✓✓✓ Excellent

### Seq 01 - BestFit (Excellent)
- **Source:** `osm_polylines_aligned_seq01_bestfit.pkl`
- **Method:** Use existing aligned file, densify only
- **Error:** 1.92m mean, 2.33m start, 0.93m end
- **Status:** ✓✓✓ Excellent
- **Notes:** Most dense sequence (2,750 polylines)

### Seq 02 - Computed (Fixed!)
- **Source:** `osm_polylines_latlon_seq02_regbez.pkl` (largebbox)
- **Method:** Compute alignment from lat/lon
- **Parameters:**
  - `frame_offset`: 0 (CRITICAL - not 57!)
  - `rotation_deg`: 36.0°
  - `anchor`: start
  - `margin`: 400m
- **Error:** 7.01m mean, 1.29m start, 3.69m end
- **Status:** ✓✓ Good (was 39.61m start error before fix!)
- **Key Learning:** The `bestfit` file used `frame_offset=57` which gave 19.75m error. Using `frame_offset=0` with `rotation=36°` matches the reference image `inspect_seq02_aligned.png`.

### Seq 05 - Computed (Fixed!)
- **Source:** `osm_polylines_latlon_seq05_regbez.pkl`
- **Method:** Compute alignment from lat/lon
- **Parameters:**
  - `frame_offset`: 0 (CRITICAL - not 46!)
  - `rotation_deg`: 350.6°
  - `anchor`: start
  - `margin`: 400m
- **Error:** 1.18m mean, 0.57m start, 0.42m end
- **Status:** ✓✓✓ Excellent (was 19.43m mean / 34.30m start!)
- **Key Learning:** Like Seq 02, the bestfit used wrong frame_offset. Frame 0 with 350.6° rotation gives near-perfect alignment.

#### Detailed Seq 05 Fix Process

**The Problem:**
The `bestfit` file had terrible alignment:
```
Before (bestfit): mean=19.43m, start=34.30m, end=30.41m
```

The `align_and_viz_bestfit.py` script used:
- `frame_offset=46` (for GPS heading alignment)
- `rotation=129°`
- Result: Roads completely misaligned from trajectory

**The Investigation:**

1. **Check available sources:**
   ```
   Seq 05: regbez (47KB), 1087 polylines
   ```

2. **Test frame offsets:**
   | Frame Offset | Best Rotation | Mean Error |
   |--------------|---------------|------------|
   | 0 | 350° | 15.02m |
   | 46 | 130° | 21.69m |
   
   → **Frame 0 is better!**

3. **Fine-tune rotation (frame_offset=0):**
   | Rotation | Mean Error | Start Error | End Error |
   |----------|------------|-------------|-----------|
   | 340° | 24.97m | 8.88m | 59.59m |
   | 350° | 15.02m | 8.88m | 3.86m |
   | **350.6°** | **14.89m** | **8.88m** | **2.04m** |
   | 351° | 14.96m | 8.88m | 3.69m |
   | 355° | 19.10m | 8.88m | 11.53m |

4. **Ultra-fine search around 350.6°:**
   ```
   350.4°: 14.90m
   350.6°: 14.89m  ← BEST
   350.8°: 14.91m
   ```

**The Solution:**
```python
frame_offset = 0      # NOT 46!
rotation_deg = 350.6  # NOT 129!
```

**The Result:**
```
After (computed): mean=1.18m, start=0.57m, end=0.42m
Improvement: 93.9% reduction in error!
```

**Why did this work?**
- The `bestfit` script prioritized GPS heading alignment over road geometry
- Using `frame_offset=0` aligns the trajectory start directly with GPS
- Fine-tuned rotation (350.6° vs 129°) aligns roads to trajectory
- More polylines: 878 vs 397 (2.2x more coverage)

## Comprehensive Alignment Parameters

Summary of all sequences with their methods and parameters:

| Seq | Method | BF Offset | Opt Offset | BF Rot | Opt Rot | Status |
|-----|--------|-----------|------------|--------|---------|--------|
| 00 | refined file | 3346 | **0** | 93° | **93°** | ✓✓✓ |
| 01 | bestfit file | **1850*** | N/A | N/A | N/A | ✓✓✓ |
| 02 | compute lat/lon | 57 | **0** | 36° | **36°** | **FIXED!** |
| 05 | compute lat/lon | 46 | **0** | 129° | **350.6°** | **FIXED!** |
| 07 | compute lat/lon | 42 | **0** | 123° | **122°** | **FIXED!** |
| 08 | bestfit file | 252 | N/A | 84° | N/A | ✓✓✓ |
| 09 | refined file | 1497 | **0** | 111° | **111°** | ✓✓✓ |
| 10 | compute lat/lon | 0 | **0** | 42° | **106°** | **FIXED!** |

**Legend:**
- **BF** = BestFit (from align_and_viz_bestfit.py)
- **Opt** = Optimal (what we found works best)
- **BF Offset** = frame_offset used by bestfit script
- **Opt Offset** = frame_offset that gives best alignment
- **BF Rot** = rotation used by bestfit script
- **Opt Rot** = rotation that gives best alignment

### Categories:

**1. Use Existing File (Best Alignment)**

| Seq | Source | Error | Notes |
|-----|--------|-------|-------|
| 01 | bestfit | **1.92m** | Extracted OSM tested but cannot match bestfit accuracy |
| 08 | bestfit | 1.12m | Already excellent |

**Note on Seq 01:**
Unlike other sequences, Seq 01 requires using the existing bestfit file.

**Seq 01 Investigation:**
- *Frame offset discovered: **1850*** (via trajectory-to-OXTS heading matching)
- OSM extraction: Successfully extracted 2,485 polylines from `karlsruhe-regbez.osm.pbf`
- Alignment attempts:
  - frame_offset=0, rot=20°: **28.44m error**
  - frame_offset=1850, rot=10°: **42m error**
  - Various combinations: 28-42m error range
- **Conclusion:** The extracted OSM cannot achieve the same accuracy as the existing bestfit file (1.92m error)
- **Reason:** Karlsruhe extract's bounding box doesn't perfectly match Seq 01's trajectory area


**\*1850** is the frame_offset that aligns Seq 01's trajectory start to the raw OXTS data (same raw folder as Seq 00, which uses 3346). This was discovered through trajectory-to-OXTS heading matching, not from a stored transform file.

**2. Fixed by Computing from Lat/Lon (frame_offset=0)**
| Seq | Issue | Fix |
|-----|-------|-----|
| 02 | 7.01m error | offset=0, rot=36.7°, dx=2m, dy=10m |
| 05 | 19.43m error | offset=0, rot=350.6° |
| 07 | 16.38m error | offset=0, rot=122° |
| 10 | 19.21m error, 309 polylines | offset=0, rot=106°, source=karlsruhe_all_roads |

**3. Working with Original Parameters**
| Seq | Notes |
|-----|-------|
| 00 | refined file works well (1.21m) |
| 09 | refined file works well (3.93m) |

### Seq 07 - Computed (Excellent!)
- **Source:** `osm_polylines_latlon_seq07_regbez.pkl`
- **Method:** Compute alignment from lat/lon
- **Parameters:**
  - `frame_offset`: 0 (not 42!)
  - `rotation_deg`: 122.0°
  - `anchor`: start
  - `margin`: 400m
- **Error:** 1.06m mean, 0.06m start, 1.18m end
- **Status:** ✓✓✓ Excellent (best alignment!)
- **Key Learning:** Frame 0 with 122° rotation works best for this sequence.

### Seq 08 - BestFit (Excellent)
- **Source:** `osm_polylines_aligned_seq08_bestfit.pkl`
- **Method:** Use existing aligned file, densify only
- **Error:** 1.12m mean, 3.32m start, 0.32m end
- **Status:** ✓✓✓ Excellent

### Seq 09 - Refined (Excellent)
- **Source:** `osm_polylines_aligned_seq09_refined.pkl`
- **Method:** Use existing aligned file, densify only
- **Error:** 3.93m mean, 3.74m start, 2.73m end
- **Status:** ✓✓✓ Excellent

### Seq 10 - Computed (Fixed!)
- **Source:** `data/osm/karlsruhe_all_roads.pkl` (30,844 polylines)
- **Method:** Compute alignment from lat/lon
- **Parameters:**
  - `frame_offset`: 0
  - `rotation_deg`: 106.0°
  - `anchor`: start
  - `margin`: 400m
- **Error:** 1.54m mean, 2.57m start, 2.58m end
- **Status:** ✓✓✓ Excellent (was 19.21m mean / 33.42m start!)
- **Key Learning:** The `bestfit` file used wrong rotation (~42°) which misaligned roads from trajectory. Switching source from sparse `bestfit` (309 polylines) to `karlsruhe_all_roads` (30k pool) with grid-searched rotation=106° gave 12× error reduction.
- **Script:** `fix_seq10_alignment.py`
  - Pre-converts all 30k lat/lon polylines to local UTM once (avoids repeated UTM calls in loop)
  - Grid search: coarse 0–360° (2° step) → fine ±10° (0.5°) → ultra-fine ±2° (0.1°)
  - Pure numpy 2D rotation in inner loop — fast enough without crashing

---

## Methodology

### Two Approaches Used

1. **Use Existing Aligned File (`mode='file'`)**
   - Load `osm_polylines_aligned_seq{seq}_{variant}.pkl`
   - Densify polylines to 2m spacing
   - Save to `osm_aligned_final/`

2. **Compute from Lat/Lon (`mode='compute'`)**
   - Load `osm_polylines_latlon_seq{seq}_regbez.pkl`
   - Load OXTS GPS data
   - Compute GPS offset using `frame_offset`
   - Convert lat/lon to local frame (UTM - offset)
   - Apply rotation around anchor point
   - Filter to trajectory area
   - Densify to 2m spacing
   - Save to `osm_aligned_final/`

### Densification Process

All polylines are densified to **2m spacing** using linear interpolation:

```python
def densify_polyline(polyline, spacing=2.0):
    # Calculate cumulative distances
    dists = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dists[-1]
    
    # Generate samples every 2m
    n_samples = max(int(total_dist / spacing) + 1, 2)
    new_dists = np.linspace(0, total_dist, n_samples)
    
    # Linear interpolation
    ...
```

---

## Key Learnings

### Critical Issue: Frame Offset Mismatch

The most important discovery was that `align_and_viz_bestfit.py` used different `frame_offset` values than what worked best for direct GPS alignment:

| Seq | align_and_viz_bestfit.py | Optimal for Alignment |
|-----|-------------------------|----------------------|
| 02 | 57 | **0** |
| 05 | 46 | **0** |
| 07 | 42 | **0** |
| 08 | 252 | **0** (but bestfit works) |

**Impact:** Using the wrong frame_offset caused ~670m offset error for Seq 02!

**Root Cause:**
- `align_and_viz_bestfit.py` uses `frame_offset` to match GPS heading at trajectory start
- This aligns heading but often misaligns the road geometry
- Using `frame_offset=0` aligns the trajectory origin with GPS origin
- Then rotation adjusts for heading difference
- Result: Better road-to-trajectory alignment

### Understanding OSM Data Sources

Three types of OSM source files exist:

1. **regbez** (Regional Extract)
   - Medium bounding box around trajectory
   - ~1k-2k polylines per sequence
   - Size: ~30-500 KB
   - Available for: 00, 02, 05, 07, 08, 09, 10

2. **largebbox** (Germany-wide Extract)
   - Much larger bounding box
   - ~20k-30k polylines
   - Size: ~900-1200 KB
   - Available for: 02, 07, 08
   - More dense but slower to process

3. **Aligned files** (bestfit/refined)
   - Already transformed to local frame
   - Ready to use
   - Source for sequences without lat/lon files

### Grid Search Methodology

When optimal parameters aren't known, use this approach:

```python
# 1. Test frame offsets (usually 0 is best)
for frame_offset in [0, 42, 46, 57]:
    # Compute GPS offset
    # Convert lat/lon to local
    
    # 2. Grid search rotation 0-360°
    for rotation in range(0, 360, 5):
        # Apply rotation
        # Filter to trajectory area
        # Compute mean error (KD-tree)
        # Track best
    
    # 3. Fine-tune around best angle
    for rotation in np.arange(best-5, best+5, 0.2):
        # Find optimal with 0.2° precision
```

**Error Computation:**
```python
from scipy.spatial import cKDTree

all_pts = np.vstack(polylines)
tree = cKDTree(all_pts)
dists, _ = tree.query(trajectory, k=1)
mean_error = dists.mean()
start_error = tree.query(trajectory[0])[0]
end_error = tree.query(trajectory[-1])[0]
```

**Why not use stored transforms?**
- `osm_transform_seq*_bestfit_new.pkl` stores the transform used by `align_and_viz_bestfit.py`
- These transforms often have wrong `frame_offset` for road alignment
- They were optimized for GPS heading, not road geometry
- Recomputing from lat/lon with `frame_offset=0` gives better results

### Rotation Optimization

For sequences computed from lat/lon, optimal rotation angles were found via grid search:

```python
# Test all rotations 0-360°
for rotation_deg in range(0, 360, 2):
    # Apply rotation around anchor
    # Compute mean trajectory-to-road distance
    # Select best
```

Best rotations found:
- Seq 02: 36°
- Seq 07: 122°
- Seq 08: 80° (but bestfit file still better)

### Visualization Fix

Originally visualizations only showed 500 polylines (sampling). Fixed to show **all** polylines to match reference images:

```python
# Before (wrong)
for pl in densified[:500]:
    ax.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5)

# After (correct)
for pl in densified:  # All polylines
    ax.plot(pl[:, 0], pl[:, 1], 'b-', lw=0.5)
```

---

## File Structure

```
osm_aligned_final/
├── osm_polylines_aligned_seq00.pkl    # Densified polylines
├── osm_polylines_aligned_seq01.pkl
├── ...
├── osm_polylines_aligned_seq10.pkl
├── osm_transform_seq00.pkl            # Transform parameters
├── osm_transform_seq01.pkl
├── ...
├── osm_config_seq00.pkl               # Configuration metadata
├── osm_config_seq01.pkl
├── ...
├── osm_aligned_seq00.png              # Visualizations
├── osm_aligned_seq01.png
└── ...
```

### Loading the Data

```python
import pickle

# Load polylines
with open('osm_aligned_final/osm_polylines_aligned_seq02.pkl', 'rb') as f:
    polylines = pickle.load(f)

# Each polyline is a numpy array of shape (N, 2)
# polylines[i][:, 0] = x coordinates
# polylines[i][:, 1] = z coordinates

# Load configuration
with open('osm_aligned_final/osm_config_seq02.pkl', 'rb') as f:
    config = pickle.load(f)
print(config)
# {'seq': '02', 'frame_offset': 0, 'rotation_deg': 36.0, ...}
```

---

## Scripts Used

### Main Script: `final_alignments.py`

```bash
# Process all sequences
python3 final_alignments.py

# Process specific sequences
python3 final_alignments.py --seqs 02 07

# Custom spacing (default: 2.0m)
python3 final_alignments.py --spacing 1.0
```

### Alternative Scripts

- `super_dense_alignments.py` - Uses largebbox files (20k+ polylines)
- `restore_alignments.py` - Uses verified variants from precompute_bev_5ch.py
- `create_dense_alignments.py` - Unified script with SEQ_CONFIG

---

## Common Issues & Solutions

### Issue: High Start/End Error

**Cause:** Wrong frame_offset or rotation angle

**Solution:** 
1. Test with `frame_offset=0`
2. Grid search rotation angles 0-360°
3. Check against reference image

### Issue: Too Few Polylines

**Cause:** Using refined/bestfit file instead of regbez/largebbox

**Solution:** Use regbez or largebbox source files

### Issue: Misaligned Visualizations

**Cause:** Matplotlib sampling or wrong transform

**Solution:** 
- Show all polylines (not sample)
- Verify GPS offset calculation
- Check rotation anchor point

---

## Validation

Error metrics computed using KD-tree nearest neighbor search:

```python
from scipy.spatial import cKDTree

all_pts = np.vstack(polylines)
tree = cKDTree(all_pts)
dists, _ = tree.query(trajectory, k=1)
mean_error = dists.mean()
start_error = tree.query(trajectory[0])[0]
end_error = tree.query(trajectory[-1])[0]
```

---

## Future Improvements

1. **Seq 05 & 10:** Investigate higher errors - may need manual tuning
2. **More dense:** Could reduce spacing to 1m for even higher density
3. **Other sequences:** Sequences 03, 04, 06 not processed (no data)

---

## Reference Images

Source reference images used for validation:

```
right_alignments/
├── osm_pbf_aligned_seq01_bestfit.png
├── osm_pbf_aligned_seq00_refined.png
├── inspect_seq02_aligned.png        # Key reference for Seq 02
├── inspect_seq07_aligned.png        # Key reference for Seq 07
├── osm_pbf_aligned_seq08_bestfit.png
└── osm_pbf_aligned_seq09_refined.png
```

---

Created: 2024-02-19
Script: final_alignments.py
