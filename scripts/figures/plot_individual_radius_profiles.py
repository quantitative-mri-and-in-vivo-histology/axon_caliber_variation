#!/usr/bin/env python3
"""
Plot 3D rendering and radius profiles for 3 representative axons.

Loads pre-computed NPZ from create_representative_axons.py and creates a
two-panel figure: 3D volume rendering (left) + radius profiles (right).

Usage:
    python scripts/figures/plot_individual_radius_profiles.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy import ndimage

from axonometry import add_panel_labels, get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()


def load_data(npz_path: Path) -> dict:
    """Load representative axon NPZ."""
    data = np.load(npz_path, allow_pickle=True)

    arc_lengths = data["arc_lengths"]
    radii = data["radii"]
    cv_values = data["cv"]
    mean_radii = data["mean_radii"]
    lengths = data["lengths"]
    vol_labels = data["volume_labels"] if "volume_labels" in data else np.arange(1, len(cv_values) + 1)

    rep_axons = []
    for i in range(len(cv_values)):
        rep_axons.append({
            "arc_lengths": np.array(arc_lengths[i]),
            "radii": np.array(radii[i]),
            "cv": float(cv_values[i]),
            "label": int(vol_labels[i]),
            "mean_radius": float(mean_radii[i]),
            "length": float(lengths[i]),
        })

    return {
        "rep_axons": rep_axons,
        "volume": np.array(data["volume"]) if "volume" in data else None,
        "volume_labels": list(vol_labels),
        "voxel_size": float(data["voxel_size"]),
    }


def render_axons_3d(volume, labels, colors, voxel_size, ax, rep_axons,
                    arc_interval=10.0):
    """Render pre-aligned axons as 3D surface scatter."""
    # Global radius range for color scaling
    all_radii = []
    for axon in rep_axons:
        all_radii.extend(axon["radii"])
    global_r_min = np.percentile(all_radii, 5)
    global_r_max = np.percentile(all_radii, 95)

    all_points = []

    for i, (label, color) in enumerate(zip(labels, colors)):
        mask = volume == label
        if not mask.any():
            continue

        # Surface voxels via erosion
        eroded = ndimage.binary_erosion(mask)
        surface = mask & ~eroded
        coords_vox = np.argwhere(surface)
        if len(coords_vox) == 0:
            continue

        # Volume axes: (Z, X, Y) → plot as (x=X, y=Y, z=Z)
        z_um = coords_vox[:, 0] * voxel_size
        x_um = coords_vox[:, 1] * voxel_size
        y_um = coords_vox[:, 2] * voxel_size

        # Subsample
        if len(coords_vox) > 20000:
            idx = np.random.choice(len(coords_vox), size=20000, replace=False)
            coords_vox = coords_vox[idx]
            z_um, x_um, y_um = z_um[idx], x_um[idx], y_um[idx]

        # Color by local radius (distance transform)
        dist = ndimage.distance_transform_edt(mask)
        local_radii = dist[coords_vox[:, 0], coords_vox[:, 1], coords_vox[:, 2]] * voxel_size

        if global_r_max > global_r_min:
            norm = np.clip((local_radii - global_r_min) / (global_r_max - global_r_min), 0, 1)
        else:
            norm = np.ones(len(coords_vox)) * 0.5

        intensities = 0.25 + 0.75 * norm
        point_sizes = 2 + 8 * norm
        base_color = np.array(to_rgb(color))
        point_colors = np.outer(intensities, base_color)

        ax.scatter(x_um, y_um, z_um, c=point_colors, s=point_sizes,
                   alpha=0.9, rasterized=True)

        pts = np.column_stack([x_um, y_um, z_um])
        all_points.append(pts)

        # Arc length tick marks
        z_min_ax, z_max_ax = z_um.min(), z_um.max()
        x_center = x_um.mean()
        y_center = y_um.mean()
        z_range = z_max_ax - z_min_ax
        if z_range > 0:
            for arc_val in np.arange(arc_interval, z_range, arc_interval):
                z_pos = z_min_ax + arc_val
                tick_len = 1.0
                ax.plot([x_center - tick_len, x_center + tick_len],
                        [y_center, y_center], [z_pos, z_pos],
                        color="black", linewidth=0.8, zorder=10)

        # CV label
        if i < len(rep_axons):
            cv_val = rep_axons[i]["cv"]
            ax.text(x_center, y_center, z_max_ax + 2,
                    f"CoV = {cv_val:.2f}", fontsize=10, ha="center", va="bottom",
                    color=color, fontweight="bold", zorder=10)

        logger.info(f"  Axon {label}: {len(coords_vox)} surface pts, Z-extent: {z_range:.1f} μm")

    if not all_points:
        return

    all_pts = np.vstack(all_points)
    pt_min = all_pts.min(axis=0)
    pt_max = all_pts.max(axis=0)
    extent = pt_max - pt_min
    pad = 1.0
    pad_top = 8.0

    ax.set_xlim(pt_min[0] - pad, pt_max[0] + pad)
    ax.set_ylim(pt_min[1] - pad, pt_max[1] + pad)
    ax.set_zlim(pt_min[2] - pad, pt_max[2] + pad_top)

    extent_padded = extent.copy()
    extent_padded[2] += pad_top - pad
    ax.set_box_aspect(extent_padded + 2 * pad)

    # Arc length arrow
    leftmost_x = pt_min[0] - 4
    y_mid = (pt_min[1] + pt_max[1]) / 2
    ax.plot([leftmost_x, leftmost_x], [y_mid, y_mid],
            [pt_min[2], pt_max[2] + 3], color="black", linewidth=1.5, zorder=10)
    ax.plot([leftmost_x - 0.5, leftmost_x, leftmost_x + 0.5],
            [y_mid, y_mid, y_mid],
            [pt_max[2] + 1, pt_max[2] + 3, pt_max[2] + 1],
            color="black", linewidth=1.5, zorder=10)
    ax.text2D(0.265, 0.5, "Arc length [μm]", fontsize=14,
              ha="center", va="center", color="black",
              transform=ax.transAxes, rotation=90)

    ax.view_init(elev=5, azim=-85)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description="Plot 3D axon rendering + profiles")
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/rat/lm/representative_axons.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("fig/main/individual_radius_profiles.svg"))
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return

    data = load_data(args.input)
    rep_axons = data["rep_axons"]
    vol = data["volume"]
    vol_labels = data["volume_labels"]
    voxel_size = data["voxel_size"]

    colors = [settings.colors["example_3"],   # Purple (low CV)
              settings.colors["example_2"],   # Orange (mid CV)
              settings.colors["example_1"]]   # Green (high CV)

    # Figure: 3D (left, tall) + profiles (right)
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.6, 1], wspace=0.05)

    ax_vol = fig.add_subplot(gs[0, 0], projection="3d")
    ax_vol.set_position([-0.35, -0.5, 0.55, 2.0])

    ax_prof = fig.add_subplot(gs[0, 1])

    # 3D rendering
    if vol is not None:
        render_axons_3d(vol, vol_labels, colors, voxel_size, ax_vol,
                        rep_axons, arc_interval=10.0)
    else:
        ax_vol.text2D(0.5, 0.5, "No volume data", ha="center", va="center",
                      transform=ax_vol.transAxes)

    # Radius profiles
    for i, axon in enumerate(rep_axons):
        ax_prof.plot(axon["arc_lengths"], axon["radii"], color=colors[i],
                     linewidth=1.5, label=f'CoV = {axon["cv"]:.2f}')

    style_axis(ax_prof, xlabel="Arc length [μm]", ylabel="Axon radius [μm]")
    ax_prof.legend(loc="upper right", fontsize=settings.fonts["legend_size"])
    max_arc = max(a["arc_lengths"].max() for a in rep_axons)
    x_max = int(np.ceil(max_arc / 10) * 10)
    ax_prof.set_xticks(range(0, x_max + 1, 10))
    ax_prof.set_xlim(0, x_max)
    ymin, ymax = ax_prof.get_ylim()
    ax_prof.set_ylim(ymin, ymax * 1.15)
    ax_prof.set_box_aspect(1)

    fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.08)
    ax_vol.set_position([-0.35, -0.5, 0.55, 2.0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.savefig(args.output.with_suffix(".png"), dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
