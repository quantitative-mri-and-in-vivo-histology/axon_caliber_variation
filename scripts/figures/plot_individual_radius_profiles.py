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

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

settings = get_plot_settings()

VOLUME_KEYS = ["low_cv_axon", "mid_cv_axon", "high_cv_axon"]


def load_data(npz_path: Path) -> dict:
    """Load representative axon NPZ."""
    data = np.load(npz_path, allow_pickle=True)

    arc_lengths = data["arc_lengths"]
    radii = data["radii"]
    cv_values = data["cv"]
    mean_radii = data["mean_radii"]
    lengths = data["lengths"]

    rep_axons = []
    for i in range(len(cv_values)):
        rep_axons.append({
            "arc_lengths": np.array(arc_lengths[i]),
            "radii": np.array(radii[i]),
            "cv": float(cv_values[i]),
            "mean_radius": float(mean_radii[i]),
            "length": float(lengths[i]),
        })

    volumes = [np.array(data[k]) for k in VOLUME_KEYS]

    # Rotate high-CV axon 90° around Z (swap X↔Y) so diameter
    # variations are visible from the default viewing angle
    volumes[2] = np.swapaxes(volumes[2], 1, 2)

    return {
        "rep_axons": rep_axons,
        "volumes": volumes,
        "voxel_size": float(data["voxel_size"]),
    }


def render_axon_surface(vol, color, voxel_size, ax, x_offset=0.0,
                        max_points=20000):
    """Render a single axon volume as 3D surface scatter, colored by local radius."""
    mask = vol > 0
    if not mask.any():
        return None

    # Surface voxels via erosion
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    coords_vox = np.argwhere(surface)
    if len(coords_vox) == 0:
        return None

    # Color by local radius (distance transform on this small volume)
    dist = ndimage.distance_transform_edt(mask)
    local_radii = dist[coords_vox[:, 0], coords_vox[:, 1], coords_vox[:, 2]] * voxel_size

    # Subsample
    if len(coords_vox) > max_points:
        idx = np.random.choice(len(coords_vox), size=max_points, replace=False)
        coords_vox = coords_vox[idx]
        local_radii = local_radii[idx]

    # Volume axes: (Z, X, Y) → plot as (x=X, y=Y, z=Z)
    z_um = coords_vox[:, 0] * voxel_size
    x_um = coords_vox[:, 1] * voxel_size + x_offset
    y_um = coords_vox[:, 2] * voxel_size

    r_min, r_max = np.percentile(local_radii, [5, 95])
    if r_max > r_min:
        norm = np.clip((local_radii - r_min) / (r_max - r_min), 0, 1)
    else:
        norm = np.ones(len(coords_vox)) * 0.5

    intensities = 0.25 + 0.75 * norm
    point_sizes = 2 + 8 * norm
    base_color = np.array(to_rgb(color))
    point_colors = np.outer(intensities, base_color)

    ax.scatter(x_um, y_um, z_um, c=point_colors, s=point_sizes,
               alpha=0.9, rasterized=True)

    logger.info(f"  {len(coords_vox)} surface pts, Z-extent: {z_um.max()-z_um.min():.1f} μm")

    return np.column_stack([x_um, y_um, z_um])


def render_axons_3d(volumes, colors, voxel_size, ax, rep_axons,
                    arc_interval=10.0):
    """Render individual axon volumes side by side along X in 3D."""
    all_points = []
    x_offset = 0.0
    spacing_um = 8.0

    axon_extents = []  # (x_lo, x_hi, y_center, z_min, z_max) per axon

    for i, (vol, color) in enumerate(zip(volumes, colors)):
        pts = render_axon_surface(vol, color, voxel_size, ax, x_offset=x_offset)
        if pts is not None:
            all_points.append(pts)

            x_lo = pts[:, 0].min() - 1.0
            x_hi = pts[:, 0].max() + 1.0
            y_center = pts[:, 1].mean()
            z_min_ax = pts[:, 2].min()
            z_top = pts[:, 2].max()

            axon_extents.append((x_lo, x_hi, y_center, z_min_ax, z_top))

            # CV label
            x_center = pts[:, 0].mean()
            ax.text(x_center, y_center, z_top + 2,
                    f"CoV = {rep_axons[i]['cv']:.2f}", fontsize=10,
                    ha="center", va="bottom", color=color,
                    fontweight="bold", zorder=10)
        else:
            axon_extents.append(None)

        x_offset += vol.shape[1] * voxel_size + spacing_um

    # Arc length tick lines + connecting lines between axons
    if axon_extents:
        z_min_all = min(e[3] for e in axon_extents if e)
        z_max_all = max(e[4] for e in axon_extents if e)
        first_ext = next(e for e in axon_extents if e is not None)
        tick_count = 0
        for z_tick in np.arange(z_min_all + arc_interval, z_max_all, arc_interval):
            tick_count += 1
            arc_um = tick_count * arc_interval
            for j, ext in enumerate(axon_extents):
                if ext is None:
                    continue
                x_lo, x_hi, y_c, z_lo, z_hi = ext
                if z_lo <= z_tick <= z_hi:
                    ax.plot([x_lo, x_hi], [y_c, y_c],
                            [z_tick, z_tick], color="black", linewidth=0.8,
                            alpha=0.6, zorder=10)
                    # Connect to next axon
                    if j + 1 < len(axon_extents) and axon_extents[j + 1] is not None:
                        ext_next = axon_extents[j + 1]
                        if ext_next[3] <= z_tick <= ext_next[4]:
                            ax.plot([x_hi, ext_next[0]],
                                    [y_c, ext_next[2]],
                                    [z_tick, z_tick], color="black",
                                    linewidth=0.5, alpha=0.3, zorder=5)
            # Label left of first axon
            ax.text(first_ext[0] - 1.5, first_ext[2], z_tick,
                    f"{arc_um:.0f}", fontsize=7, ha="right", va="center",
                    color="black", zorder=10)

    if not all_points:
        return

    all_pts = np.vstack(all_points)
    pt_min = all_pts.min(axis=0)
    pt_max = all_pts.max(axis=0)
    extent = pt_max - pt_min
    pad = 1.0
    pad_top = 8.0

    ax.set_xlim(pt_min[0] - pad - 15, pt_max[0] + pad)
    ax.set_ylim(pt_min[1] - pad, pt_max[1] + pad)
    ax.set_zlim(pt_min[2] - pad, pt_max[2] + pad_top)

    extent_padded = extent.copy()
    extent_padded[0] += 15
    extent_padded[2] += pad_top - pad
    ax.set_box_aspect(extent_padded + 2 * pad)

    # Vertical arrow with label to the left of tick numbers
    if axon_extents:
        arrow_x = first_ext[0] - 8.0
        arrow_y = first_ext[2]
        ax.plot([arrow_x, arrow_x], [arrow_y, arrow_y],
                [z_min_all, z_max_all], color="black", linewidth=1.2, zorder=10)
        # Arrowhead
        head_len = 1.5
        ax.plot([arrow_x - 0.4, arrow_x, arrow_x + 0.4],
                [arrow_y, arrow_y, arrow_y],
                [z_max_all - head_len, z_max_all, z_max_all - head_len],
                color="black", linewidth=1.2, zorder=10)
        # Label
        ax.text2D(0.27, 0.5, r"Arc length [$\mu$m]", fontsize=12,
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
    volumes = data["volumes"]
    voxel_size = data["voxel_size"]

    colors = [settings.colors["example_1"],   # Green (low CV)
              settings.colors["example_2"],   # Orange (mid CV)
              settings.colors["example_3"]]   # Purple (high CV)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    width = 6

    # Figure 1: 3D rendering (tall)
    fig_vol = plt.figure(figsize=(width, width * 2))
    ax_vol = fig_vol.add_subplot(111, projection="3d")
    render_axons_3d(volumes, colors, voxel_size, ax_vol, rep_axons,
                    arc_interval=10.0)
    vol_path = args.output.with_stem(args.output.stem + "_rendering")
    plt.savefig(vol_path, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.savefig(vol_path.with_suffix(".png"), dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"Saved rendering to {vol_path}")

    # Figure 2: Radius profiles
    fig_prof, ax_prof = plt.subplots(figsize=(width, width))
    for i, axon in enumerate(rep_axons):
        ax_prof.plot(axon["arc_lengths"], axon["radii"], color=colors[i],
                     linewidth=1.5, label=f'CoV = {axon["cv"]:.2f}')

    style_axis(ax_prof, xlabel=r"Arc length [$\mu$m]", ylabel=r"Axon radius [$\mu$m]")
    ax_prof.legend(loc="upper right", fontsize=settings.fonts["legend_size"], ncol=2)
    max_arc = max(a["arc_lengths"].max() for a in rep_axons)
    x_max = int(np.ceil(max_arc / 10) * 10)
    ax_prof.set_xticks(range(0, x_max + 1, 10))
    ax_prof.set_xlim(0, x_max)
    ymin, ymax = ax_prof.get_ylim()
    ax_prof.set_ylim(ymin, ymax * 1.15)
    ax_prof.set_box_aspect(1)

    plt.tight_layout()
    prof_path = args.output.with_stem(args.output.stem + "_profiles")
    plt.savefig(prof_path, dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.savefig(prof_path.with_suffix(".png"), dpi=settings.figure["dpi"], bbox_inches="tight")
    plt.close()
    logger.info(f"Saved profiles to {prof_path}")


if __name__ == "__main__":
    main()
