#!/usr/bin/env python3
"""
Comprehensive diagnostic for all 5 BEV channels.
Saves probe_modalities.png with visual grid + prints quantitative stats.
"""
import pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import ndimage

META_TRAIN = 'data/paper_split/train_meta.pkl'
OUT_PNG    = 'probe_modalities.png'

CH_NAMES = ['ch0 height', 'ch1 intensity', 'ch2 density', 'ch3 history', 'ch4 OSM']
CH_CMAPS = ['plasma',     'hot',           'viridis',     'Greens',      'Blues']

# ── helpers ──────────────────────────────────────────────────────────────────
def load(s):
    return np.load(s['npy_path']).astype(np.float32)   # [5,300,400]

def connected_components(ch):
    labeled, n = ndimage.label(ch > 0)
    return n, labeled

# ── load metadata ─────────────────────────────────────────────────────────────
with open(META_TRAIN, 'rb') as f:
    samples = pickle.load(f)

by_seq = {}
for s in samples:
    by_seq.setdefault(s['seq'], []).append(s)
for seq in by_seq:
    by_seq[seq].sort(key=lambda x: x['frame_idx'])

print("="*70)
print("  MODALITY DIAGNOSTIC")
print("="*70)

# ═══════════════════════════════════════════════════════════════════════════
# 1. VISUAL GRID — 8 diverse frames (different seqs + positions)
# ═══════════════════════════════════════════════════════════════════════════
viz_samples = []
for seq in ['00','02','05','07']:
    seq_s = by_seq[seq]
    for frac in [0.15, 0.5]:
        idx = int(frac * len(seq_s))
        viz_samples.append(seq_s[idx])

n_frames = len(viz_samples)
n_ch     = 5
fig = plt.figure(figsize=(n_frames * 2.8, n_ch * 2.4))
gs  = gridspec.GridSpec(n_ch, n_frames, hspace=0.05, wspace=0.05)

for col, s in enumerate(viz_samples):
    bev = load(s)
    for row in range(n_ch):
        ax = fig.add_subplot(gs[row, col])
        ch = bev[row]
        vmin = 0
        # ch0: show range; ch1/ch2 show as-is
        im = ax.imshow(ch, cmap=CH_CMAPS[row], vmin=vmin, vmax=1,
                       origin='lower', aspect='auto', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(CH_NAMES[row], fontsize=8)
        if row == 0:
            ax.set_title(f"seq{s['seq']}\nf{s['frame_idx']}", fontsize=7)

plt.suptitle('BEV Channel Visual Inspection (rows=channels, cols=frames)', fontsize=10, y=1.01)
plt.savefig(OUT_PNG, dpi=110, bbox_inches='tight')
plt.close()
print(f"\n  Saved visual grid → {OUT_PNG}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. QUANTITATIVE SPARSITY across all 3860 train samples
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Quantitative sparsity (all 3860 train samples) ──")
total = 300 * 400
N = len(samples)
ch_nz  = np.zeros((N, 5))
ch_min = np.full((N, 5),  np.inf)
ch_max = np.full((N, 5), -np.inf)

for i, s in enumerate(samples):
    bev = load(s)
    for c in range(5):
        ch = bev[c]
        nz = ch[ch > 0]
        ch_nz[i, c]  = len(nz) / total * 100
        if len(nz):
            ch_min[i, c] = nz.min()
            ch_max[i, c] = nz.max()
    if i % 500 == 0:
        print(f"  {i}/{N}", end='\r')

print(f"  {N}/{N} done        ")
print(f"\n  {'Channel':<16} {'nz% mean':>9} {'nz% std':>8} {'val min':>8} {'val max':>8}")
print("  " + "-"*52)
for c, name in enumerate(CH_NAMES):
    m   = ch_nz[:, c].mean()
    sd  = ch_nz[:, c].std()
    vmi = ch_min[ch_min[:, c] < np.inf, c].min() if (ch_min[:, c] < np.inf).any() else 0
    vmx = ch_max[ch_max[:, c] > -np.inf, c].max() if (ch_max[:, c] > -np.inf).any() else 0
    print(f"  {name:<16} {m:>8.2f}%  {sd:>7.2f}%  {vmi:>8.4f}  {vmx:>8.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. CONNECTED COMPONENTS — history & OSM
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Connected components (50 random samples) ──")
rng = np.random.default_rng(42)
idx50 = rng.choice(N, 50, replace=False)
hist_cc, osm_cc = [], []
for i in idx50:
    bev = load(samples[i])
    n_hist, _ = connected_components(bev[3])
    n_osm,  _ = connected_components(bev[4])
    hist_cc.append(n_hist)
    osm_cc.append(n_osm)

print(f"  History (ch3): mean={np.mean(hist_cc):.1f}  median={np.median(hist_cc):.0f}  "
      f"min={np.min(hist_cc)}  max={np.max(hist_cc)}")
print(f"  OSM     (ch4): mean={np.mean(osm_cc):.1f}  median={np.median(osm_cc):.0f}  "
      f"min={np.min(osm_cc)}  max={np.max(osm_cc)}")
print(f"  (History=1 → single connected line ✓;  OSM>1 → road network with junctions ✓)")

# ═══════════════════════════════════════════════════════════════════════════
# 4. VALUE DISTRIBUTION — binary check
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Value distribution: binary check (ch3, ch4) ──")
s0   = samples[0]
bev0 = load(s0)
for c, name in [(3,'ch3 history'), (4,'ch4 OSM')]:
    ch  = bev0[c]
    nz  = ch[ch > 0]
    is_binary = np.all(np.isin(np.round(nz, 4), [0.0, 1.0]))
    unique_vals = np.unique(np.round(nz, 3))
    print(f"  {name}: unique nonzero values = {unique_vals[:10]}  "
          f"{'BINARY ✓' if is_binary else 'GRADED ✗'}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL CONSISTENCY — 5 consecutive frames
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Temporal consistency (5 consecutive frames, seq00) ──")
seq00 = by_seq['00']
mid   = len(seq00) // 2
consec = seq00[mid:mid+5]

print(f"  Frame indices: {[s['frame_idx'] for s in consec]}")
hist_nz = []
osm_nz  = []
for s in consec:
    bev = load(s)
    hist_nz.append((bev[3] > 0).sum())
    osm_nz.append((bev[4] > 0).sum())

# History should shift across frames (different pixel counts expected)
hist_identical = all(h == hist_nz[0] for h in hist_nz)
print(f"  History nonzero pixel counts: {hist_nz}")
print(f"  History identical across frames: {hist_identical}  "
      f"{'✗ broken (static, not real history)' if hist_identical else '✓ varies as expected'}")
print(f"  OSM    nonzero pixel counts: {osm_nz}")
osm_identical = all(o == osm_nz[0] for o in osm_nz)
print(f"  OSM identical across frames: {osm_identical}  "
      f"{'→ same map crop (ok if ego barely moved)' if osm_identical else '✓ varies'}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. CROSS-CHANNEL ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Cross-channel alignment (50 samples) ──")
osm_on_hist, hist_on_osm = [], []
osm_on_lidar_h = []

for i in idx50:
    bev = load(samples[i])
    h0, h1, h2, h3, h4 = bev

    osm_mask  = h4 > 0
    hist_mask = h3 > 0

    # History pixels that fall on OSM road
    if hist_mask.sum() > 0:
        hist_on_osm.append(osm_mask[hist_mask].mean())

    # OSM pixels that have LiDAR height return
    if osm_mask.sum() > 0:
        osm_on_lidar_h.append((h0[osm_mask] > 0).mean())

print(f"  History pixels that land on OSM road : {np.mean(hist_on_osm)*100:.1f}%  "
      f"(expect >60% if ego trajectory follows road)")
print(f"  OSM pixels with any LiDAR height     : {np.mean(osm_on_lidar_h)*100:.1f}%  "
      f"(expect >30% — dense scan should hit road cells)")

# ═══════════════════════════════════════════════════════════════════════════
# 7. COORDINATE FRAME VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Coordinate frame check ──")
print("  BEV grid: x∈[-20,20]m lateral,  y∈[-10,30]m forward,  0.1m/px")
print("  Ego at pixel col=200 (x=0), row=100 (y=0)")
# Find centroid of history in first frame
for s in seq00[:3]:
    bev = load(s)
    h3  = bev[3]
    if h3.sum() > 0:
        rows, cols = np.where(h3 > 0)
        crow, ccol = rows.mean(), cols.mean()
        # Convert pixel → world
        cx_m = ccol * 0.1 + (-20)
        cy_m = crow * 0.1 + (-10)
        print(f"  seq00 f{s['frame_idx']:04d}: history centroid at pixel "
              f"({ccol:.0f},{crow:.0f}) = world ({cx_m:.1f}m, {cy_m:.1f}m)  "
              f"{'✓ behind ego (negative y = past)' if cy_m < 0 else '? check'}")
        break

print("\n" + "="*70)
print("  Done. See probe_modalities.png for visual grid.")
print("="*70)
