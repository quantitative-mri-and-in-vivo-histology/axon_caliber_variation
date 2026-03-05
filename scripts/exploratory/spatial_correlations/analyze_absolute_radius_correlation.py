#!/usr/bin/env python3
"""Analyze spatial correlation of absolute vs relative axon radii."""

import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
import json
from pathlib import Path
from skimage.measure import regionprops
import matplotlib.pyplot as plt

from axonometry.io import load_volume_with_metadata

print("Loading data...")
volume, voxel_size_tuple, _ = load_volume_with_metadata(
    Path("data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat")
)
voxel_size = voxel_size_tuple[0]

profiles = np.load("data/processed/LM/sham_25_ipsi_axon_profiles.npz", allow_pickle=True)
with open("data/processed/LM/sham_25_ipsi_populations.json") as f:
    pop_data = json.load(f)

cc_labels = set()
for pop in pop_data["populations"]:
    if pop["name"].upper() == "CC":
        cc_labels = set(pop["axon_labels"])
        slice_axis = pop["dominant_axis"]
        break

label_to_mean = {
    int(lab): rad for lab, rad in
    zip(profiles["labels"], profiles["mean_radii_um"])
    if rad > 0 and int(lab) in cc_labels
}

# Analyze one slice
slice_idx = volume.shape[slice_axis] // 2
if slice_axis == 2:
    slice_2d = volume[:, :, slice_idx]
else:
    slice_2d = volume[slice_idx, :, :]

pop_mask = np.isin(slice_2d, list(cc_labels))
slice_2d = np.where(pop_mask, slice_2d, 0)

props = regionprops(slice_2d)
data = []
for prop in props:
    if prop.label in label_to_mean and label_to_mean[prop.label] > 0:
        area_um2 = prop.area * voxel_size * voxel_size
        slice_r = np.sqrt(area_um2 / np.pi)
        mean_r = label_to_mean[prop.label]
        rel_r = slice_r / mean_r
        cy, cx = prop.centroid
        data.append([cx * voxel_size, cy * voxel_size, slice_r, mean_r, rel_r])

data = np.array(data)
positions = data[:, :2]
slice_radii = data[:, 2]
mean_radii = data[:, 3]  # 3D mean radius - the true axon size
rel_radii = data[:, 4]

print(f"Slice {slice_idx}: {len(data)} axons")
print(f"Mean radius range: {mean_radii.min():.3f} - {mean_radii.max():.3f} μm")
print()

tree = cKDTree(positions)

# Compare correlations: absolute (mean_r) vs relative (rel_r)
print("=== Spatial Correlation: Absolute vs Relative Radius ===")
print()
print("Kernel | Abs (mean_r) | Rel (r/⟨r⟩) | Slice_r")
print("-" * 55)

results = []
for kernel_r in [2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]:
    neighbors = tree.query_ball_point(positions, r=kernel_r)

    nbr_mean_abs = []
    nbr_mean_rel = []
    nbr_mean_slice = []

    for i, nbrs in enumerate(neighbors):
        nbrs_excl = [n for n in nbrs if n != i]
        if len(nbrs_excl) > 0:
            nbr_mean_abs.append(np.mean(mean_radii[nbrs_excl]))
            nbr_mean_rel.append(np.mean(rel_radii[nbrs_excl]))
            nbr_mean_slice.append(np.mean(slice_radii[nbrs_excl]))
        else:
            nbr_mean_abs.append(np.nan)
            nbr_mean_rel.append(np.nan)
            nbr_mean_slice.append(np.nan)

    nbr_mean_abs = np.array(nbr_mean_abs)
    nbr_mean_rel = np.array(nbr_mean_rel)
    nbr_mean_slice = np.array(nbr_mean_slice)
    valid = ~np.isnan(nbr_mean_abs)

    r_abs, p_abs = stats.pearsonr(mean_radii[valid], nbr_mean_abs[valid])
    r_rel, p_rel = stats.pearsonr(rel_radii[valid], nbr_mean_rel[valid])
    r_slice, p_slice = stats.pearsonr(slice_radii[valid], nbr_mean_slice[valid])

    results.append((kernel_r, r_abs, r_rel, r_slice))
    print(f"{kernel_r:5.0f}μm | {r_abs:+.3f}        | {r_rel:+.3f}       | {r_slice:+.3f}")

print()
print("Legend:")
print("  Abs (mean_r): Correlation of 3D mean radius with neighbors")
print("  Rel (r/⟨r⟩): Correlation of relative radius with neighbors")
print("  Slice_r: Correlation of slice radius with neighbors")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Spatial map colored by mean radius
ax1 = axes[0]
scatter = ax1.scatter(positions[:, 0], positions[:, 1], c=mean_radii,
                      cmap='viridis', s=5, alpha=0.7)
plt.colorbar(scatter, ax=ax1, label='Mean radius (μm)')
ax1.set_xlabel('X (μm)')
ax1.set_ylabel('Y (μm)')
ax1.set_title('Spatial Distribution of Axon Size')
ax1.set_aspect('equal')

# Panel 2: Correlation vs kernel radius
ax2 = axes[1]
kernel_radii = [r[0] for r in results]
corr_abs = [r[1] for r in results]
corr_rel = [r[2] for r in results]
corr_slice = [r[3] for r in results]

ax2.plot(kernel_radii, corr_abs, 'o-', label='Absolute (mean_r)', linewidth=2, markersize=8)
ax2.plot(kernel_radii, corr_rel, 's--', label='Relative (r/⟨r⟩)', linewidth=2, markersize=8)
ax2.plot(kernel_radii, corr_slice, '^:', label='Slice radius', linewidth=2, markersize=8)
ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Kernel radius (μm)')
ax2.set_ylabel('Correlation')
ax2.set_title('Spatial Correlation vs Scale')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel 3: Histogram of mean radii
ax3 = axes[2]
ax3.hist(mean_radii, bins=50, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Mean radius (μm)')
ax3.set_ylabel('Count')
ax3.set_title(f'Axon Size Distribution\nmean={mean_radii.mean():.3f}, std={mean_radii.std():.3f}')
ax3.axvline(mean_radii.mean(), color='red', linestyle='--', label='Mean')
ax3.legend()

plt.tight_layout()
plt.savefig('fig/spatial_correlation_2d/absolute_vs_relative_correlation.png', dpi=150, bbox_inches='tight')
print()
print("Saved figure to fig/spatial_correlation_2d/absolute_vs_relative_correlation.png")
