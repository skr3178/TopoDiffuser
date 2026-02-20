# OSM Alignment Report — Seq 02, 07 & 08

## Result

| Seq | Polylines | Mean Error | Start Error | End Error | Status |
|-----|-----------|------------|-------------|-----------|--------|
| 02  | 1,112     | **9.4 m**  | 5.4 m       | 12.3 m    | ✅ Aligned |
| 07  | 444       | **14.8 m** | 5.7 m       | 3.8 m     | ✅ Aligned |
| 08  | 1,181     | **17.3 m** | 3.3 m       | 20.9 m    | ✅ Aligned |

All sequences are visually confirmed as correctly aligned (trajectory follows OSM road centerlines).

---

## Files

### Input files

| File | Description |
|------|-------------|
| `osm_polylines_latlon_seq02_regbez.pkl` | OSM road polylines for seq 02 in lat/lon (WGS84), 8,696 polylines |
| `osm_polylines_latlon_seq07_regbez.pkl` | OSM road polylines for seq 07 in lat/lon (WGS84), 8,907 polylines |
| `osm_polylines_latlon_seq08_regbez.pkl` | OSM road polylines for seq 08 in lat/lon (WGS84), 11,199 polylines |
| `data/kitti/poses/02.txt` | KITTI odometry ground-truth poses for seq 02 (4,661 frames) |
| `data/kitti/poses/07.txt` | KITTI odometry ground-truth poses for seq 07 (1,101 frames) |
| `data/kitti/poses/08.txt` | KITTI odometry ground-truth poses for seq 08 (4,071 frames) |
| `data/raw_data/2011_10_03_drive_0034_sync/.../oxts/data/` | OXTS GPS/IMU data for seq 02 (258 frames) |
| `data/raw_data/2011_09_30_drive_0027_sync/.../oxts/data/` | OXTS GPS/IMU data for seq 07 (75 frames) |
| `data/raw_data/2011_09_30_drive_0028_sync/.../oxts/data/` | OXTS GPS/IMU data for seq 08 (1,259 frames) |

### Output files

| File | Description |
|------|-------------|
| `osm_polylines_aligned_seq02_bestfit.pkl` | Final aligned polylines for seq 02 in KITTI local frame (metres) |
| `osm_polylines_aligned_seq07_bestfit.pkl` | Final aligned polylines for seq 07 in KITTI local frame (metres) |
| `osm_polylines_aligned_seq08_bestfit.pkl` | Final aligned polylines for seq 08 in KITTI local frame (metres) |
| `osm_transform_seq02_bestfit_new.pkl` | Transform record: GPS offset, coarse rotation, fine correction |
| `osm_transform_seq07_bestfit_new.pkl` | Transform record: GPS offset, coarse rotation, fine correction |
| `osm_transform_seq08_bestfit_new.pkl` | Transform record: GPS offset, coarse rotation, fine correction |
| `osm_pbf_aligned_seq02_bestfit.png` | Overlay visualisation — main view + start/end zooms |
| `osm_pbf_aligned_seq07_bestfit.png` | Overlay visualisation — main view + start/end zooms |
| `osm_pbf_aligned_seq08_bestfit.png` | Overlay visualisation — main view + start/end zooms |

### Scripts

| Script | Role |
|--------|------|
| `align_and_viz_bestfit.py` | Main pipeline: lat/lon → local frame → rotation search → bestfit → visualisation |
| `inspect_alignment.py` | Generates 3-panel inspection images at any rotation for visual debugging |
| `utils/osm_alignment.py` | Helpers: `load_oxts_data()`, `latlon_to_utm()` |

---

## Critical Issue: Seq 08 Frame Offset Correction

### The Problem

Seq 08 alignment initially failed because the **OXTS frame offset was incorrect**. The pipeline assumed OXTS frame 1100 corresponded to odometry pose frame 0, but this was wrong.

**Why this happened:**
- KITTI odometry sequences are **subsets** of raw drive recordings
- Seq 08 uses `2011_09_30_drive_0028_sync` starting at raw frame 1100
- The original assumption was that OXTS frame 1100 → pose frame 0
- **However**, OXTS data and odometry poses are not synchronized by frame number alone
- The correct approach is to match by **heading direction**, not frame index

### The Solution

Found the correct frame by comparing trajectory heading with OXTS yaw:

```python
# Trajectory start heading (from first 50 frames)
traj_heading = 90.2°  # Moving north

# Search OXTS frames for matching heading
OXTS frame 252: yaw = 90.3°  ← MATCH!
OXTS frame 1100: yaw = 9.4°   ← Wrong direction!
```

**Result:** Frame offset changed from `1100` → **`252`**

### Verification

After correction, OSM and trajectory coordinate ranges overlap properly:

| | Before (frame 1100) | After (frame 252) |
|---|---------------------|-------------------|
| Trajectory x range | [-390, 418] | [-390, 418] |
| Trajectory z range | [0, 390] | [0, 390] |
| OSM x range | [-1194, 964] | **[-1803, 2076]** |
| OSM z range | [897, 2052] | **[-2660, 2263]** |
| Overlap | ❌ None | ✅ Yes |

---

## Pipeline

### Step 1 — Determine OXTS Frame Offset

**For each new sequence, follow this procedure:**

```python
import numpy as np
from utils.osm_alignment import load_oxts_data, latlon_to_utm

def find_correct_oxts_frame(trajectory, oxts_data):
    """
    Find OXTS frame that corresponds to odometry pose frame 0.
    Match by heading direction, not frame index.
    """
    # Compute trajectory heading from first N frames
    N = 50
    dx = trajectory[N, 0] - trajectory[0, 0]
    dz = trajectory[N, 1] - trajectory[0, 1]
    traj_heading = np.arctan2(dz, dx)
    
    # Search OXTS frames for matching yaw
    best_frame = 0
    best_diff = float('inf')
    
    for frame in range(min(len(oxts_data), len(trajectory))):
        oxts_yaw = oxts_data[frame, 5]  # OXTS yaw column
        # Normalize angle difference to [-pi, pi]
        diff = abs(np.arctan2(np.sin(oxts_yaw - traj_heading), 
                              np.cos(oxts_yaw - traj_heading)))
        if diff < best_diff:
            best_diff = diff
            best_frame = frame
    
    print(f"Best match: OXTS frame {best_frame}, heading diff = {np.degrees(best_diff):.1f}°")
    return best_frame
```

**Update `SEQ_FRAME_OFFSET` in scripts:**

```python
SEQ_FRAME_OFFSET = {
    '02': 0,      # drive_0034 starts at frame 0
    '07': 0,      # drive_0027 starts at frame 0  
    '08': 252,    # drive_0028 - frame 252 matches heading (NOT 1100!)
    'XX': ??,     # Use find_correct_oxts_frame() for new sequences
}
```

### Step 2 — Verify GPS Offset Alignment

Run `inspect_alignment.py` with **zero rotation** to verify the GPS offset is correct:

```bash
python inspect_alignment.py --seqs 08 --rotation 0
```

**Check:** OSM roads and trajectory should be in the same coordinate range (overlap visible). If they are in completely different areas, the frame offset is wrong.

### Step 3 — Determine Anchor Point and Rotation

**Visual inspection approach:**

1. Generate inspection images at different angles:
```bash
python inspect_alignment.py --seqs 08 --rotation 0
python inspect_alignment.py --seqs 08 --rotation 45
python inspect_alignment.py --seqs 08 --rotation 90
```

2. Identify which angle makes the OSM road grid align with the trajectory shape

3. Determine anchor point:
   - If **start point** of trajectory is on/near OSM → use `anchor='start'`
   - If **end point** of trajectory is on/near OSM → use `anchor='end'`

**Numerical optimization approach:**

Test angles numerically and pick the one with lowest mean error:

```python
from scipy.spatial import cKDTree

def test_rotation_angles(trajectory, osm_polylines, anchor='start'):
    """Find optimal rotation angle by brute force search."""
    pivot = trajectory[0] if anchor == 'start' else trajectory[-1]
    
    best_angle = 0
    best_error = float('inf')
    
    for angle_deg in range(0, 360, 1):
        # Rotate OSM around pivot
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        # ... rotation code ...
        
        # Compute mean distance to nearest road
        tree = cKDTree(rotated_osm_pts)
        dists, _ = tree.query(trajectory, k=1)
        mean_error = dists.mean()
        
        if mean_error < best_error:
            best_error = mean_error
            best_angle = angle_deg
    
    return best_angle, best_error
```

### Step 4 — Update Configuration

Update `align_and_viz_bestfit.py`:

```python
SEQ_ANCHOR = {
    '02': 'start',
    '07': 'end',
    '08': 'start',  # determined by visual inspection
    'XX': '??',     # fill in for new sequences
}

SEQ_ROTATION_HINT = {
    '08': (84.0, 'force'),  # determined by numerical optimization
    'XX': (angle, 'force'), # fill in for new sequences
}
```

### Step 5 — Run Full Pipeline

```bash
python align_and_viz_bestfit.py --seqs 08
```

Check that:
- Bestfit rotation is **NOT** hitting bounds (±30° or ±200m)
- Start and end errors are reasonable (< 30m)
- Visual overlay shows trajectory following roads

---

## Alignment Parameters Summary

| Seq | Frame Offset | Anchor | Coarse Rotation | Fine Rotation | Translation | Mean Error |
|-----|--------------|--------|-----------------|---------------|-------------|------------|
| 02  | 0 | start | +36.7° | −0.3° | (+3.0, +9.6) m | 9.4 m |
| 07  | 0 | end | +127.6° | −5.4° | (−7.0, −8.2) m | 14.8 m |
| 08  | **252** | **start** | **+84.0°** | **−0.4°** | **(−6.1, +17.3) m** | 17.3 m |

---

## Key Lessons

1. **OXTS frame offset ≠ raw drive frame number.** KITTI odometry sequences are subsets of raw drives. The odometry pose frame 0 corresponds to the OXTS frame with **matching heading**, not necessarily the same index.

2. **Always verify GPS offset before rotation search.** Run `inspect_alignment.py --rotation 0` first. If OSM and trajectory are in different coordinate ranges, the frame offset is wrong.

3. **Match heading to find correct OXTS frame.** Use `atan2()` to compute trajectory heading from first N poses, then search OXTS data for the frame with closest yaw angle.

4. **Grid search over rotation is more reliable than GPS heading.** GPS/IMU heading can be noisy, especially with sparse OXTS data. A full 360° grid search (or narrow search around visual estimate) is more robust.

5. **Anchor point selection matters.** Choose the point (start or end) that is already closest to OSM roads. This fixes that point and lets rotation adjust the rest.

6. **Widen bestfit bounds if hitting limits.** If fine-tuning hits ±15° or ±150m bounds, widen to ±30° and ±200m. Hitting bounds indicates coarse rotation is still off.

---

## Quick Checklist for New Sequences

- [ ] Identify raw drive folder (e.g., `2011_09_30_drive_0028_sync`)
- [ ] Load OXTS data and count frames
- [ ] Load trajectory and compute start heading
- [ ] **Find matching OXTS frame by heading comparison**
- [ ] Update `SEQ_FRAME_OFFSET` with correct frame number
- [ ] Run `inspect_alignment.py --rotation 0` to verify GPS offset
- [ ] Determine anchor point (start or end)
- [ ] Test rotation angles visually or numerically
- [ ] Update `SEQ_ANCHOR` and `SEQ_ROTATION_HINT`
- [ ] Run `align_and_viz_bestfit.py` and verify results
