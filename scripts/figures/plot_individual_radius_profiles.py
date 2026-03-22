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

    aligned_skeletons = data["aligned_skeletons"] if "aligned_skeletons" in data else [None] * len(cv_values)

    rep_axons = []
    for i in range(len(cv_values)):
        rep_axons.append({
            "arc_lengths": np.array(arc_lengths[i]),
            "radii": np.array(radii[i]),
            "cv": float(cv_values[i]),
            "mean_radius": float(mean_radii[i]),
            "length": float(lengths[i]),
            "aligned_skeleton": np.asarray(aligned_skeletons[i], dtype=np.float64) if aligned_skeletons[i] is not None else None,
        })

    volumes = [np.array(data[k]) for k in VOLUME_KEYS]

    return {
        "rep_axons": rep_axons,
        "volumes": volumes,
        "voxel_size": float(data["voxel_size"]),
    }


def render_axon_surface(vol, color, voxel_size, ax, x_offset=0.0,
                        max_points=20000):
    """Render a single axon volume as 3D surface scatter, colored by per-slice radius."""
    mask = vol > 0
    if not mask.any():
        return None

    # Surface voxels via erosion
    eroded = ndimage.binary_erosion(mask)
    surface = mask & ~eroded
    coords_vox = np.argwhere(surface)
    if len(coords_vox) == 0:
        return None

    # Per-Z-slice equivalent radius (= local axon thickness)
    slice_radius = np.zeros(vol.shape[0])
    for z in range(vol.shape[0]):
        n = np.count_nonzero(mask[z])
        if n > 0:
            slice_radius[z] = np.sqrt(n * voxel_size**2 / np.pi)

    # Map each surface voxel to its slice's radius
    local_radii = slice_radius[coords_vox[:, 0]]

    # Subsample
    if len(coords_vox) > max_points:
        idx = np.random.choice(len(coords_vox), size=max_points, replace=False)
        coords_vox = coords_vox[idx]
        local_radii = local_radii[idx]

    # Volume axes: (Z, X, Y) → plot as (x=X, y=Y, z=Z)
    z_um = coords_vox[:, 0] * voxel_size
    x_um = coords_vox[:, 1] * voxel_size + x_offset
    y_um = coords_vox[:, 2] * voxel_size

    r_min, r_max = np.percentile(local_radii[local_radii > 0], [2, 98])
    if r_max > r_min:
        norm = np.clip((local_radii - r_min) / (r_max - r_min), 0, 1)
    else:
        norm = np.ones(len(coords_vox)) * 0.5

    intensities = 0.35 + 0.65 * norm
    point_sizes = 0.05 + 0.2 * norm
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
    spacing_um = 2.0

    axon_extents = []  # (x_lo, x_hi, y_center, z_min, z_max) per axon
    x_offsets = []

    for i, (vol, color) in enumerate(zip(volumes, colors)):
        x_offsets.append(x_offset)
        pts = render_axon_surface(vol, color, voxel_size, ax, x_offset=x_offset)
        if pts is not None:
            all_points.append(pts)

            x_lo = pts[:, 0].min() - 1.0
            x_hi = pts[:, 0].max() + 1.0
            y_center = pts[:, 1].mean()
            z_min_ax = pts[:, 2].min()
            z_top = pts[:, 2].max()

            axon_extents.append((x_lo, x_hi, y_center, z_min_ax, z_top))
        else:
            axon_extents.append(None)

        x_offset += vol.shape[1] * voxel_size + spacing_um

    # Arc length tick lines using transformed skeleton coords
    if axon_extents:
        first_ext = next(e for e in axon_extents if e is not None)

        max_arc = max(a["length"] for a in rep_axons)
        tick_arcs = np.arange(arc_interval, max_arc, arc_interval)

        for arc_um in tick_arcs:
            z_positions = []
            for j, (ext, axon) in enumerate(zip(axon_extents, rep_axons)):
                if ext is None:
                    z_positions.append(None)
                    continue
                skel = axon.get("aligned_skeleton")
                if skel is None:
                    z_positions.append(None)
                    continue
                skel = np.asarray(skel, dtype=np.float64)
                diffs = np.diff(skel, axis=0)
                seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
                cum_arc = np.concatenate([[0], np.cumsum(seg_lens)])
                if arc_um > cum_arc[-1]:
                    z_positions.append(None)
                    continue
                z_tick = float(np.interp(arc_um, cum_arc, skel[:, 0]))
                z_positions.append(z_tick)

            # Draw short tick lines centered on skeleton X at this arc length
            tick_half = 1.5  # μm half-width
            tick_centers = []
            for j, ext in enumerate(axon_extents):
                if ext is None or z_positions[j] is None:
                    tick_centers.append(None)
                    continue
                _, _, y_c, _, _ = ext
                skel = np.asarray(rep_axons[j].get("aligned_skeleton"), dtype=np.float64)
                diffs = np.diff(skel, axis=0)
                seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
                cum_arc = np.concatenate([[0], np.cumsum(seg_lens)])
                # Skeleton X (axis 1 in volume) at this arc length + render x_offset
                x_center = float(np.interp(arc_um, cum_arc, skel[:, 1])) + x_offsets[j]
                tick_centers.append(x_center)
                z_tick = z_positions[j]
                ax.plot([x_center - tick_half, x_center + tick_half], [y_c, y_c],
                        [z_tick, z_tick], color="black", linewidth=0.5,
                        alpha=0.5, zorder=10)
                # Connect to next axon
                if j + 1 < len(axon_extents) and axon_extents[j + 1] is not None and z_positions[j + 1] is not None:
                    # Will draw connection after computing next tick_center
                    pass

            # Draw connections between axons
            for j in range(len(tick_centers) - 1):
                if tick_centers[j] is not None and tick_centers[j + 1] is not None:
                    y_c = axon_extents[j][2]
                    y_c_next = axon_extents[j + 1][2]
                    ax.plot([tick_centers[j] + tick_half, tick_centers[j + 1] - tick_half],
                            [y_c, y_c_next],
                            [z_positions[j], z_positions[j + 1]], color="black",
                            linewidth=0.3, alpha=0.2, zorder=5)

            # Tick label left of first axon
            if z_positions[0] is not None:
                ax.text(first_ext[0] - 1.5, first_ext[2], z_positions[0],
                        f"{arc_um:.0f}", fontsize=settings.fonts["tick_size"] / 2,
                        ha="right", va="center", color="black", zorder=10)

    if not all_points:
        return

    all_pts = np.vstack(all_points)
    pt_min = all_pts.min(axis=0)
    pt_max = all_pts.max(axis=0)
    extent = pt_max - pt_min
    pad = 1.0

    ax.set_xlim(pt_min[0] - pad, pt_max[0] + pad)
    ax.set_ylim(pt_min[1] - pad, pt_max[1] + pad)
    ax.set_zlim(pt_min[2] - pad, pt_max[2] + pad)
    # Shrink Y in box aspect since elev≈0 makes it nearly invisible
    box = extent + 2 * pad
    box[1] = box[0]  # make Y same as X so it doesn't inflate the projection
    ax.set_box_aspect(box)

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
    # Figure 1: 3D rendering (tall)
    fig_vol = plt.figure(figsize=(5, 9))
    ax_vol = fig_vol.add_axes([0, 0, 1, 1], projection="3d")
    render_axons_3d(volumes, colors, voxel_size, ax_vol, rep_axons,
                    arc_interval=10.0)
    vol_path = args.output.with_stem(args.output.stem + "_rendering")
    plt.savefig(vol_path, dpi=800)
    plt.savefig(vol_path.with_suffix(".png"), dpi=800)
    plt.close()
    logger.info(f"Saved rendering to {vol_path}")

    # Figure 2: Radius profiles
    fig_prof, ax_prof = plt.subplots(figsize=(5, 4.5))
    for i, axon in enumerate(rep_axons):
        ax_prof.plot(axon["arc_lengths"], axon["radii"], color=colors[i],
                     linewidth=settings.line["linewidth"], label=f'CoV = {axon["cv"]:.2f}')

    style_axis(ax_prof, xlabel=r"Arc length [$\mu$m]", ylabel=r"Axon radius [$\mu$m]")
    ax_prof.legend(loc="upper right", fontsize=settings.fonts["legend_size"], handlelength=1.5)
    max_arc = max(a["arc_lengths"].max() for a in rep_axons)
    x_max = int(np.ceil(max_arc / 10) * 10)
    ax_prof.set_xticks(range(0, x_max + 1, 10))
    ax_prof.set_xlim(0, x_max)
    ymin, ymax = ax_prof.get_ylim()
    ax_prof.set_ylim(ymin, ymax * 1.15)
    ax_prof.set_box_aspect(1)
    plt.tight_layout()
    prof_path = args.output.with_stem(args.output.stem + "_profiles")
    plt.savefig(prof_path, dpi=settings.figure["dpi"], bbox_inches="tight", pad_inches=0.02)
    plt.savefig(prof_path.with_suffix(".png"), dpi=settings.figure["dpi"], bbox_inches="tight", pad_inches=0.02)
    plt.close()
    logger.info(f"Saved profiles to {prof_path}")


if __name__ == "__main__":
    main()
