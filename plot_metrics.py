#!/usr/bin/env python3
"""
Plot metrics from training log file.
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# Read log file
with open('/media/skr/storage/self_driving/TopoDiffuser/train_diffusion_paper.log', 'r') as f:
    log_content = f.read()

# Parse epoch data
epochs = []
minADE_values = []
minFDE_values = []
hitrate_values = []
hd_values = []
train_loss = []
val_loss = []

# Pattern to match epoch summary
epoch_pattern = r'Epoch \[(\d+)/(\d+)\].*?\n.*?Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)\n.*?minADE: ([\d.]+)m \| minFDE: ([\d.]+)m\n.*?HitRate: ([\d.]+) \| HD: ([\d.]+)m'

matches = re.findall(epoch_pattern, log_content, re.DOTALL)

for match in matches:
    epoch, total_epochs, tr_loss, v_loss, minade, minfde, hitrate, hd = match
    epochs.append(int(epoch))
    train_loss.append(float(tr_loss))
    val_loss.append(float(v_loss))
    minADE_values.append(float(minade))
    minFDE_values.append(float(minfde))
    hitrate_values.append(float(hitrate))
    hd_values.append(float(hd))

print(f"Found {len(epochs)} epochs of data")
print(f"Epochs: {epochs}")
print(f"minADE: {minADE_values}")
print(f"minFDE: {minFDE_values}")
print(f"HitRate: {hitrate_values}")
print(f"HD: {hd_values}")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Diffusion Training Metrics (Frozen Encoder)', fontsize=14, fontweight='bold')

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot 1: minADE
ax = axes[0, 0]
ax.plot(epochs, minADE_values, 'o-', color=colors[0], linewidth=2, markersize=6)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('minADE (m)', fontsize=11)
ax.set_title('Minimum Average Displacement Error', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, max(epochs) + 0.5)
# Annotate best value
best_idx = np.argmin(minADE_values)
ax.annotate(f'Best: {minADE_values[best_idx]:.2f}m', 
            xy=(epochs[best_idx], minADE_values[best_idx]),
            xytext=(epochs[best_idx]+0.3, minADE_values[best_idx]+3),
            fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

# Plot 2: minFDE
ax = axes[0, 1]
ax.plot(epochs, minFDE_values, 'o-', color=colors[1], linewidth=2, markersize=6)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('minFDE (m)', fontsize=11)
ax.set_title('Minimum Final Displacement Error', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, max(epochs) + 0.5)
# Annotate best value
best_idx = np.argmin(minFDE_values)
ax.annotate(f'Best: {minFDE_values[best_idx]:.2f}m', 
            xy=(epochs[best_idx], minFDE_values[best_idx]),
            xytext=(epochs[best_idx]+0.3, minFDE_values[best_idx]+3),
            fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

# Plot 3: HitRate
ax = axes[1, 0]
ax.plot(epochs, hitrate_values, 'o-', color=colors[2], linewidth=2, markersize=6)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('HitRate', fontsize=11)
ax.set_title('HitRate@2m (Success Rate)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, max(epochs) + 0.5)
ax.set_ylim(-0.001, max(hitrate_values) * 1.5 + 0.001)
# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

# Plot 4: Hausdorff Distance
ax = axes[1, 1]
ax.plot(epochs, hd_values, 'o-', color=colors[3], linewidth=2, markersize=6)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('HD (m)', fontsize=11)
ax.set_title('Hausdorff Distance (Shape Similarity)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, max(epochs) + 0.5)
# Annotate best value
best_idx = np.argmin(hd_values)
ax.annotate(f'Best: {hd_values[best_idx]:.2f}m', 
            xy=(epochs[best_idx], hd_values[best_idx]),
            xytext=(epochs[best_idx]+0.3, hd_values[best_idx]+3),
            fontsize=9, color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

plt.tight_layout()
plt.savefig('/media/skr/storage/self_driving/TopoDiffuser/metrics_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('/media/skr/storage/self_driving/TopoDiffuser/metrics_plot.pdf', bbox_inches='tight')
print("\nPlots saved to:")
print("  - metrics_plot.png")
print("  - metrics_plot.pdf")

# Also create a combined plot with all metrics
fig2, ax1 = plt.subplots(figsize=(12, 6))

# Plot distance metrics on left axis
line1, = ax1.plot(epochs, minADE_values, 'o-', color=colors[0], linewidth=2, label='minADE', markersize=6)
line2, = ax1.plot(epochs, minFDE_values, 's-', color=colors[1], linewidth=2, label='minFDE', markersize=6)
line3, = ax1.plot(epochs, hd_values, '^-', color=colors[3], linewidth=2, label='Hausdorff', markersize=6)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Distance (meters)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, max(epochs) + 0.5)

# Plot HitRate on right axis
ax2 = ax1.twinx()
line4, = ax2.plot(epochs, hitrate_values, 'd-', color=colors[2], linewidth=2, label='HitRate@2m', markersize=6)
ax2.set_ylabel('HitRate@2m', fontsize=12, color=colors[2])
ax2.tick_params(axis='y', labelcolor=colors[2])
ax2.set_ylim(-0.001, max(max(hitrate_values) * 2, 0.01))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

# Combined legend
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', fontsize=10)

plt.title('Diffusion Training Metrics Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/media/skr/storage/self_driving/TopoDiffuser/metrics_combined.png', dpi=150, bbox_inches='tight')
print("  - metrics_combined.png")

# plt.show()  # Commented out for non-interactive mode
print("\nDone!")
