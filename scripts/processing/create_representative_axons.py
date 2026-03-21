#!/usr/bin/env python3
"""
Extract representative axons for the individual axon stats figure.

Selects 3 axons with low/mid/high CV from 3D axon profile data, extracts each
from its zarr volume, PCA-aligns to Z axis, and places them side by side in a
synthetic volume. Saves everything to a single NPZ file.

Usage:
    python scripts/processing/create_representative_axons.py
"""

import argparse
import glob as glob_module
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from axonometry.io import load_zarr_volume

# Find repo root (contains pyproject.toml)
_root = Path(__file__).resolve().parent
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VOXEL_SIZE = 0.05  # μm per voxel (isotropic)
G_BAR = 0.6  # g-ratio for conduction velocity


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_axon_data(npz_file: Path, min_length: float = 20.0):
    """Load axon profiles from NPZ and compute CV + slowdown."""
    data = np.load(npz_file, allow_pickle=True)

    mean_radii = data["mean_radii_um"]
    std_radii = data["std_radii_um"]
    lengths = data["lengths_um"]
    radii_profiles = data["radii_profiles_um"]
    skeleton_coords = data["skeleton_coords_um"]
    labels = data["labels"]

    # CV = std / mean
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean_radii > 0, std_radii / mean_radii, np.nan)

    # Conduction velocity slowdown per axon
    slowdown = np.full(len(radii_profiles), np.nan)
    for i, profile in enumerate(radii_profiles):
        r = np.asarray(profile)
        r = r[r > 0]
        if len(r) < 2:
            continue
        r_bar = np.mean(r)
        d_m = r_bar * (1 - G_BAR) / G_BAR
        v = r * np.sqrt(np.log(1.0 + d_m / r))
        v_eff = len(v) / np.sum(1.0 / v)  # harmonic mean
        v_ideal = r_bar * np.sqrt(np.log(1.0 + d_m / r_bar))
        if v_ideal > 0:
            slowdown[i] = v_eff / v_ideal

    # Valid mask for pooled stats
    valid = (
        np.isfinite(cv) & np.isfinite(slowdown)
        & (mean_radii > 0) & (lengths >= min_length)
    )

    sample = npz_file.stem.replace("_axon_profiles", "")
    logger.info(f"Loaded {npz_file.name}: {valid.sum()}/{len(labels)} valid axons")

    return {
        "mean_radii": mean_radii,
        "std_radii": std_radii,
        "lengths": lengths,
        "cv": cv,
        "slowdown": slowdown,
        "labels": labels,
        "radii_profiles": radii_profiles,
        "skeleton_coords": skeleton_coords,
        "valid_mask": valid,
        "sample": sample,
    }


# ---------------------------------------------------------------------------
# Axon selection
# ---------------------------------------------------------------------------

def select_representative_axons(all_data, files, min_arc=50.0, max_arc=70.0):
    """
    Pick 3 representative axons at the 10th, 50th, and 90th CV percentile.

    All must have arc length in [min_arc, max_arc] μm.
    """
    candidates = []
    for file_idx, d in enumerate(all_data):
        for i in range(len(d["labels"])):
            skel = d["skeleton_coords"][i]
            if skel is None or len(skel) < 3:
                continue
            if not (min_arc <= d["lengths"][i] <= max_arc
                    and np.isfinite(d["cv"][i])):
                continue
            # Only single-segment skeletons (no big jumps between points)
            diffs = np.diff(np.asarray(skel), axis=0)
            max_jump = np.sqrt(np.sum(diffs**2, axis=1)).max()
            if max_jump > 1.0:
                continue
            candidates.append({
                "file_idx": file_idx,
                "axon_idx": i,
                "cv": d["cv"][i],
                "mean_radius": d["mean_radii"][i],
                "length": d["lengths"][i],
            })

    if len(candidates) < 3:
        raise ValueError(f"Only {len(candidates)} candidates found — need at least 3")

    logger.info(f"  {len(candidates)} candidates with arc length {min_arc}-{max_arc} μm")

    candidates.sort(key=lambda x: x["cv"])
    n = len(candidates)

    selected = [
        candidates[int(n * 0.001)],   # 10th percentile CV
        candidates[int(n * 0.5)],   # 50th percentile CV
        candidates[int(n * 0.99)],   # 90th percentile CV
    ]

    for s in selected:
        fi = s["file_idx"]
        logger.info(
            f"  Selected: file={files[fi].name} label={_get_label(all_data, s)} "
            f"CV={s['cv']:.3f} r={s['mean_radius']:.3f}μm len={s['length']:.1f}μm"
        )
    return selected


def _get_label(all_data, sel):
    return int(all_data[sel["file_idx"]]["labels"][sel["axon_idx"]])


# ---------------------------------------------------------------------------
# Volume extraction and PCA alignment
# ---------------------------------------------------------------------------

def extract_axon_crop(zarr_path: Path, label: int, skeleton_um, padding: int = 60):
    """
    Read a small bounding-box crop around one axon from a zarr volume.

    Uses skeleton coordinates (in μm) to compute the bounding box so we
    never load the full volume.

    Returns (crop, bbox_min) where crop has only `label` nonzero.
    """
    volume, _voxel_size = load_zarr_volume(zarr_path)
    vol_shape = np.array(volume.shape)

    skel_vox = (np.asarray(skeleton_um) / VOXEL_SIZE).astype(int)
    bbox_min = np.maximum(skel_vox.min(axis=0) - padding, 0)
    bbox_max = np.minimum(skel_vox.max(axis=0) + padding, vol_shape)

    crop = volume[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]
    crop = crop.copy()
    crop[crop != label] = 0

    sz = tuple(bbox_max - bbox_min)
    logger.info(f"    Crop {sz} from {zarr_path.name} ({crop.nbytes / 1e6:.1f} MB)")
    return crop, bbox_min


def align_axon_to_z(crop: np.ndarray, label: int):
    """
    PCA-align an axon so its principal (longest) axis maps to Z (axis 0).

    The key insight: scipy.ndimage.affine_transform maps each OUTPUT
    coordinate to an INPUT coordinate via:  in = matrix @ out + offset.
    So the matrix must be the INVERSE of the desired forward rotation.

    Forward rotation: Vt (maps PC1 → axis 0).
    Inverse (for scipy): V = Vt.T.
    """
    coords = np.argwhere(crop == label).astype(np.float64)
    if len(coords) < 50:
        logger.warning(f"    Only {len(coords)} voxels for label {label}, skipping alignment")
        return crop

    centroid = coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(coords - centroid, full_matrices=False)

    # Ensure right-handed rotation (det = +1)
    if np.linalg.det(Vt) < 0:
        Vt[2] *= -1

    V = Vt.T  # inverse rotation (for scipy)

    # Compute output bounding box by rotating the 8 corners of the input
    shape = np.array(crop.shape, dtype=float)
    corners = np.array(
        [[i, j, k] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])]
    )
    # Forward transform of corners: out = Vt @ (in - centroid)
    rot_corners = (corners - centroid) @ Vt.T  # same as Vt @ (corner - c) for each row
    out_min = rot_corners.min(axis=0)
    out_max = rot_corners.max(axis=0)
    out_shape = (np.ceil(out_max - out_min) + 2).astype(int)  # +2 padding

    # scipy: in = V @ out + offset
    # We want: out = Vt @ (in - centroid) - out_min
    #    => in = V @ (out + out_min) + centroid = V @ out + V @ out_min + centroid
    offset = V @ out_min + centroid

    aligned = affine_transform(
        crop.astype(np.float64), V, offset=offset,
        output_shape=tuple(out_shape), order=0, mode="constant", cval=0,
    )
    aligned = np.round(aligned).astype(crop.dtype)
    aligned[aligned != label] = 0

    n_before = np.count_nonzero(crop == label)
    n_after = np.count_nonzero(aligned == label)
    logger.info(f"    Aligned: {crop.shape} → {tuple(out_shape)}  "
                f"voxels: {n_before} → {n_after} ({100*n_after/max(n_before,1):.0f}%)")

    return aligned


def extract_aligned_axons(selected, all_data, files, zarr_dir):
    """
    Extract each selected axon, PCA-align to Z, and return as individual
    binary volumes (tight bounding box, label=1).
    """
    volumes = []

    for i, sel in enumerate(selected):
        d = all_data[sel["file_idx"]]
        label = int(d["labels"][sel["axon_idx"]])
        skeleton_um = d["skeleton_coords"][sel["axon_idx"]]

        # Map NPZ file → zarr path
        zarr_name = files[sel["file_idx"]].stem.replace("_axon_profiles", "") + ".zarr"
        zarr_path = zarr_dir / zarr_name

        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr not found: {zarr_path}")

        logger.info(f"  Axon {i+1}: label={label} from {zarr_path.name}")

        crop, _ = extract_axon_crop(zarr_path, label, skeleton_um)
        aligned = align_axon_to_z(crop, label)

        # Binary: set axon to 1
        aligned[aligned == label] = 1
        aligned[aligned != 1] = 0
        aligned = aligned.astype(np.uint8)

        # Trim to tight bounding box
        nz = np.argwhere(aligned)
        if len(nz) > 0:
            bbox_min = nz.min(axis=0)
            bbox_max = nz.max(axis=0) + 1
            aligned = aligned[bbox_min[0]:bbox_max[0],
                              bbox_min[1]:bbox_max[1],
                              bbox_min[2]:bbox_max[2]]

        logger.info(f"    Final: {aligned.shape} ({aligned.nbytes / 1e6:.1f} MB)")
        volumes.append(aligned)

    return volumes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create representative axon volume")
    parser.add_argument("--input", type=str,
                        default="data/processed/rat/lm/sham_25_contra_cc_myelin_axon_profiles.npz")
    parser.add_argument("--zarr-dir", type=Path, default=Path("data/raw/rat/lm"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/rat/lm/representative_axons.npz"))
    parser.add_argument("--min-length", type=float, default=20.0,
                        help="Minimum length for pooled stats")
    args = parser.parse_args()

    files = sorted(Path(f) for f in glob_module.glob(args.input, recursive=True))
    if not files:
        logger.error(f"No files found: {args.input}")
        return
    logger.info(f"Found {len(files)} axon profile files")

    # Load all data
    all_data = [load_axon_data(f, args.min_length) for f in files]

    # Select 3 representative axons
    logger.info("Selecting representative axons...")
    selected = select_representative_axons(all_data, files)

    # Extract and align axons
    logger.info("Extracting and aligning axons...")
    volumes = extract_aligned_axons(selected, all_data, files, args.zarr_dir)

    # Collect per-axon profile data from precomputed profiles
    rep_axons = []
    for sel in selected:
        d = all_data[sel["file_idx"]]
        idx = sel["axon_idx"]
        profile = np.asarray(d["radii_profiles"][idx])
        arc_lengths = np.linspace(0, d["lengths"][idx], len(profile))
        rep_axons.append({
            "arc_lengths": arc_lengths,
            "radii": profile,
            "cv": d["cv"][idx],
            "mean_radius": d["mean_radii"][idx],
            "length": d["lengths"][idx],
        })

    # Save
    save_dict = {
        # Per-axon volumes (PCA-aligned, binary, tight bbox)
        "low_cv_axon": volumes[0],
        "mid_cv_axon": volumes[1],
        "high_cv_axon": volumes[2],
        # Per-axon profiles (re-profiled on aligned volumes)
        "arc_lengths": np.array([a["arc_lengths"] for a in rep_axons], dtype=object),
        "radii": np.array([a["radii"] for a in rep_axons], dtype=object),
        "cv": np.array([a["cv"] for a in rep_axons]),
        "mean_radii": np.array([a["mean_radius"] for a in rep_axons]),
        "lengths": np.array([a["length"] for a in rep_axons]),
        "voxel_size": np.array(VOXEL_SIZE),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **save_dict)
    sz = args.output.stat().st_size / 1e6
    logger.info(f"Saved to {args.output} ({sz:.1f} MB)")

    # Quick sanity-check visualization
    visualize_representative_axons(volumes, rep_axons)


def visualize_representative_axons(volumes, rep_axons):
    """Save a quick sanity-check PNG: ZY projections + radius profiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#6B9E6B", "#D9864A", "#8B6BAE"]  # green, orange, purple
    fig, axes = plt.subplots(1, len(volumes) + 1, figsize=(16, 4),
                             gridspec_kw={"width_ratios": [1] * len(volumes) + [1.5]})

    for i, (vol, rep) in enumerate(zip(volumes, rep_axons)):
        ax = axes[i]
        proj = np.any(vol > 0, axis=1)  # ZY projection (collapse X)
        ax.imshow(proj.T, cmap="gray_r", aspect="auto",
                  interpolation="nearest", origin="lower")
        ax.set_xlabel("Z [voxels]")
        ax.set_ylabel("Y [voxels]")
        ax.set_title(f"CV={rep['cv']:.2f} r={rep['mean_radius']:.2f}", fontsize=9)

    ax = axes[-1]
    for i, rep in enumerate(rep_axons):
        ax.plot(rep["arc_lengths"], rep["radii"], color=colors[i],
                label=f"CV={rep['cv']:.2f} r={rep['mean_radius']:.2f}")
    ax.set_xlabel("Arc length [μm]")
    ax.set_ylabel("Radius [μm]")
    ax.legend(fontsize=8)

    plt.tight_layout()
    debug_dir = Path("fig/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    fig_path = debug_dir / "representative_axons.png"
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Sanity check figure: {fig_path}")


if __name__ == "__main__":
    main()
