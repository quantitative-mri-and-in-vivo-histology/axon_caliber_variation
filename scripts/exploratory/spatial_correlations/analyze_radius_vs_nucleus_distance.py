#!/usr/bin/env python3
"""
Analyze normalized axon radius as a function of distance to nearest nucleus.

For each skeleton point along an axon:
1. Compute normalized radius = radius / mean_radius_along_axon
2. Compute distance to nearest nucleus centroid

Output: Binned statistics plot showing mean normalized radius vs distance to nucleus.
"""

import argparse
import glob as glob_module
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from tqdm import tqdm

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()


def derive_nucleus_path(npz_file: Path, nucleus_base_dir: Path) -> Path:
    """
    Derive nucleus .mat file path from axon profiles NPZ filename.

    Mapping:
        NPZ: data/processed/LM/sham_25_ipsi_axon_profiles.npz
        Nucleus: data/raw/Sham_25_ipsi/LM_25_ipsi_nucleus.mat

        NPZ: data/processed/HM/sham_25_ipsi_axon_profiles.npz
        Nucleus: data/raw/Sham_25_ipsi/HM_25_ipsi_nucleus.mat

    Args:
        npz_file: Path to axon profiles NPZ file
        nucleus_base_dir: Base directory for raw nucleus files (e.g., data/raw)

    Returns:
        Path to corresponding nucleus .mat file
    """
    stem = npz_file.stem.replace('_axon_profiles', '')  # "sham_25_ipsi"
    parts = stem.split('_')  # ["sham", "25", "ipsi"]

    if len(parts) < 3:
        raise ValueError(f"Cannot parse sample info from filename: {npz_file.name}")

    condition = parts[0].capitalize()  # "Sham" or "Tbi" -> needs TBI fix
    rat_id = parts[1]  # "25"
    hemisphere = parts[2]  # "ipsi"

    # Handle TBI capitalization
    if condition.lower() == 'tbi':
        condition = 'TBI'

    # Detect microscopy type from parent directory (HM or LM)
    parent_dir = npz_file.parent.name
    if parent_dir in ['HM', 'LM']:
        microscopy = parent_dir
    else:
        # Fallback: default to LM with warning
        microscopy = 'LM'
        logger.warning(f"Could not detect microscopy type from path {npz_file}, defaulting to LM")

    # Construct path: Sham_25_ipsi/LM_25_ipsi_nucleus.mat
    sample_dir = f"{condition}_{rat_id}_{hemisphere}"
    mat_name = f"{microscopy}_{rat_id}_{hemisphere}_nucleus.mat"

    return nucleus_base_dir / sample_dir / mat_name


def load_nucleus_volume(mat_file: Path) -> np.ndarray:
    """
    Load nucleus labeled volume from HDF5 .mat file.

    Args:
        mat_file: Path to nucleus .mat file

    Returns:
        3D numpy array of nucleus labels
    """
    if not mat_file.exists():
        raise FileNotFoundError(f"Nucleus file not found: {mat_file}")

    with h5py.File(str(mat_file), 'r') as f:
        # Find the labeled volume key
        volume_key = None
        for key in f.keys():
            if not key.startswith('#') and not key.startswith('_'):
                volume_key = key
                break

        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")

        volume = f[volume_key][:]
        logger.info(f"Loaded nucleus volume: shape={volume.shape}, dtype={volume.dtype}")

    return volume


def extract_nucleus_centroids(nucleus_volume: np.ndarray, voxel_size: Tuple[float, float, float]) -> np.ndarray:
    """
    Extract centroids of all nuclei in physical coordinates (micrometers).

    Uses skimage.regionprops which is memory-efficient (processes labels one at a time
    internally) unlike scipy.ndimage.center_of_mass which creates large intermediate arrays.

    Args:
        nucleus_volume: 3D labeled volume (0=background, positive=nucleus IDs)
        voxel_size: Tuple of (vz, vy, vx) in micrometers

    Returns:
        Array of shape (n_nuclei, 3) with centroid coordinates in micrometers [z, y, x]
    """
    vz, vy, vx = voxel_size

    logger.info("Computing nucleus centroids with regionprops...")

    # regionprops is memory-efficient: iterates through labels without creating
    # large intermediate arrays (unlike scipy.ndimage.center_of_mass)
    props = regionprops(nucleus_volume)

    if len(props) == 0:
        raise ValueError("No nuclei found in nucleus volume")

    logger.info(f"Found {len(props)} nuclei, extracting centroids...")

    # Extract centroids and convert to physical coordinates
    centroids_um = np.array([p.centroid for p in props]) * np.array([vz, vy, vx])

    logger.info(f"Extracted {len(centroids_um)} nucleus centroids")

    return centroids_um


def compute_distances_and_radii(
    axon_profiles_file: Path,
    nucleus_centroids: np.ndarray,
    nucleus_tree: cKDTree
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute distance to nearest nucleus and radius metrics for all skeleton points.

    Args:
        axon_profiles_file: Path to axon profiles NPZ file
        nucleus_centroids: Array of nucleus centroid coordinates in micrometers
        nucleus_tree: KDTree built from nucleus_centroids for fast nearest-neighbor lookup

    Returns:
        Tuple of (distances, normalized_radii, absolute_deviations) arrays:
        - distances: distance to nearest nucleus centroid
        - normalized_radii: r / mean_r (dimensionless)
        - absolute_deviations: r - mean_r (in micrometers)
    """
    data = np.load(axon_profiles_file, allow_pickle=True)

    skeleton_coords = data['skeleton_coords_um']  # Object array of (n_points, 3) arrays
    radii_profiles = data['radii_profiles_um']    # Object array of (n_points,) arrays
    mean_radii = data['mean_radii_um']            # (n_axons,) array

    n_axons = len(skeleton_coords)
    logger.info(f"Processing {n_axons} axons from {axon_profiles_file.name}")

    # Sanity check: verify coordinate system overlap between skeleton and nucleus data
    all_skel_points = np.vstack([s for s in skeleton_coords if len(s) > 0])
    skel_min, skel_max = all_skel_points.min(axis=0), all_skel_points.max(axis=0)
    nuc_min, nuc_max = nucleus_centroids.min(axis=0), nucleus_centroids.max(axis=0)

    # Check if bounding boxes overlap
    overlap = np.all(skel_min <= nuc_max) and np.all(nuc_min <= skel_max)
    if not overlap:
        logger.warning(
            f"Skeleton bounds [{skel_min} - {skel_max}] do not overlap with "
            f"nucleus bounds [{nuc_min} - {nuc_max}]. "
            "This may indicate a coordinate system mismatch."
        )
    else:
        logger.debug(f"Skeleton bounds: {skel_min} to {skel_max}")
        logger.debug(f"Nucleus bounds: {nuc_min} to {nuc_max}")

    all_distances = []
    all_normalized_radii = []
    all_absolute_deviations = []

    skipped_zero_radius = 0
    skipped_empty_skeleton = 0

    for i in tqdm(range(n_axons), desc="Processing axons", disable=n_axons < 1000):
        skel = skeleton_coords[i]  # (n_points, 3) in micrometers
        radii = radii_profiles[i]  # (n_points,)
        mean_r = mean_radii[i]

        if mean_r <= 0:
            skipped_zero_radius += 1
            continue
        if len(skel) == 0:
            skipped_empty_skeleton += 1
            continue

        # Compute normalized radius and absolute deviation at each point
        normalized_r = radii / mean_r
        absolute_dev = radii - mean_r

        # Query KDTree for distance to nearest nucleus centroid
        distances, _ = nucleus_tree.query(skel)

        all_distances.append(distances)
        all_normalized_radii.append(normalized_r)
        all_absolute_deviations.append(absolute_dev)

    if skipped_zero_radius > 0 or skipped_empty_skeleton > 0:
        logger.debug(f"Skipped {skipped_zero_radius} axons with zero/negative mean radius, "
                     f"{skipped_empty_skeleton} with empty skeletons")

    if len(all_distances) == 0:
        return np.array([]), np.array([]), np.array([])

    distances = np.concatenate(all_distances)
    normalized_radii = np.concatenate(all_normalized_radii)
    absolute_deviations = np.concatenate(all_absolute_deviations)

    return distances, normalized_radii, absolute_deviations


def bin_data(
    distances: np.ndarray,
    normalized_radii: np.ndarray,
    bin_size: float,
    max_distance: Optional[float],
    min_points_per_bin: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin data by distance and compute statistics per bin.

    Args:
        distances: Distance to nearest nucleus for each point
        normalized_radii: Normalized radius for each point
        bin_size: Width of each distance bin in micrometers
        max_distance: Maximum distance to include (None for auto)
        min_points_per_bin: Minimum points required to include a bin

    Returns:
        Tuple of (bin_centers, mean_values, std_values, counts)
    """
    if max_distance is None:
        max_distance = np.percentile(distances, 99)  # Avoid extreme outliers

    bin_edges = np.arange(0, max_distance + bin_size, bin_size)
    n_bins = len(bin_edges) - 1

    bin_centers = []
    mean_values = []
    std_values = []
    counts = []

    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        count = np.sum(mask)

        if count >= min_points_per_bin:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            mean_values.append(np.mean(normalized_radii[mask]))
            std_values.append(np.std(normalized_radii[mask]))
            counts.append(count)

    return (np.array(bin_centers), np.array(mean_values),
            np.array(std_values), np.array(counts))


def plot_radius_vs_distance(
    bin_centers: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    abs_mean: np.ndarray,
    abs_std: np.ndarray,
    output_file: Path,
    title: str,
    total_points: int
) -> None:
    """
    Create plot of radius metrics vs distance to nucleus.

    Args:
        bin_centers: Center of each distance bin
        norm_mean: Mean normalized radius (r/mean_r) per bin
        norm_std: Std of normalized radius per bin
        abs_mean: Mean absolute deviation (r - mean_r) per bin in μm
        abs_std: Std of absolute deviation per bin
        output_file: Output PNG file path
        title: Plot title
        total_points: Total number of data points
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    font_settings = settings.fonts
    line_settings = settings.line
    grid_settings = settings.grid
    main_color = settings.get_group_color('sham')

    # Panel 1: Normalized radius (r / mean_r)
    ax1.fill_between(bin_centers, norm_mean - norm_std, norm_mean + norm_std,
                     color=main_color, alpha=0.3, label='±1 std')
    ax1.plot(bin_centers, norm_mean, color=main_color, linestyle='-',
             linewidth=line_settings['linewidth'], marker='o',
             markersize=line_settings['marker_size'], label='Mean')

    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Distance to Nearest Nucleus (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax1.set_ylabel('Normalized Radius (r / $\\bar{r}$)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax1.set_title(f'Normalized (n={total_points:,})', fontsize=font_settings['title_size'],
                  fontweight=font_settings['weight'])
    ax1.legend(loc='best', fontsize=font_settings['legend_size'])
    ax1.grid(grid_settings['enabled'], alpha=grid_settings['alpha'])
    ax1.set_xlim(0, bin_centers.max() * 1.05)

    # Panel 2: Absolute deviation (r - mean_r) in μm
    ax2.fill_between(bin_centers, abs_mean - abs_std, abs_mean + abs_std,
                     color=main_color, alpha=0.3, label='±1 std')
    ax2.plot(bin_centers, abs_mean, color=main_color, linestyle='-',
             linewidth=line_settings['linewidth'], marker='o',
             markersize=line_settings['marker_size'], label='Mean')

    ax2.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    ax2.set_xlabel('Distance to Nearest Nucleus (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax2.set_ylabel('Absolute Deviation (r - $\\bar{r}$) (μm)', fontsize=font_settings['label_size'],
                   fontweight=font_settings['weight'])
    ax2.set_title('Absolute', fontsize=font_settings['title_size'],
                  fontweight=font_settings['weight'])
    ax2.legend(loc='best', fontsize=font_settings['legend_size'])
    ax2.grid(grid_settings['enabled'], alpha=grid_settings['alpha'])
    ax2.set_xlim(0, bin_centers.max() * 1.05)

    fig.suptitle(title, fontsize=font_settings['title_size'] + 2,
                 fontweight=font_settings['weight'], y=1.02)

    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")


def collect_data_from_file(
    npz_file: Path,
    nucleus_base_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect distance and radius data from a single file.

    Args:
        npz_file: Path to axon profiles NPZ file
        nucleus_base_dir: Base directory for nucleus .mat files

    Returns:
        Tuple of (distances, normalized_radii, absolute_deviations) arrays
    """
    logger.info(f"\nProcessing: {npz_file.name}")

    # Derive nucleus file path
    nucleus_file = derive_nucleus_path(npz_file, nucleus_base_dir)
    logger.info(f"Nucleus file: {nucleus_file}")

    # Load nucleus volume and extract centroids
    nucleus_volume = load_nucleus_volume(nucleus_file)

    # Get voxel size from axon profiles
    data = np.load(npz_file, allow_pickle=True)
    voxel_size = tuple(data['voxel_size_um'])
    data.close()

    # Extract centroids and immediately free volume memory
    nucleus_centroids = extract_nucleus_centroids(nucleus_volume, voxel_size)
    del nucleus_volume

    # Build KDTree for fast nearest-neighbor queries
    logger.info("Building KDTree for nucleus centroids...")
    nucleus_tree = cKDTree(nucleus_centroids)

    # Compute distances and radii metrics
    distances, normalized_radii, absolute_deviations = compute_distances_and_radii(
        npz_file, nucleus_centroids, nucleus_tree
    )

    if len(distances) > 0:
        logger.info(f"Total skeleton points: {len(distances):,}")
        logger.info(f"Distance range: {distances.min():.2f} - {distances.max():.2f} μm")

    return distances, normalized_radii, absolute_deviations


def process_single_file(
    npz_file: Path,
    nucleus_base_dir: Path,
    output_file: Path,
    bin_size: float,
    max_distance: Optional[float],
    min_points_per_bin: int,
    title: Optional[str]
) -> None:
    """
    Process a single axon profiles file and create plot.

    Args:
        npz_file: Path to axon profiles NPZ file
        nucleus_base_dir: Base directory for nucleus .mat files
        output_file: Output PNG file path
        bin_size: Distance bin size in micrometers
        max_distance: Maximum distance to consider
        min_points_per_bin: Minimum points per bin
        title: Custom plot title (auto-generated if None)
    """
    distances, normalized_radii, absolute_deviations = collect_data_from_file(
        npz_file, nucleus_base_dir
    )

    if len(distances) == 0:
        logger.warning(f"No valid data points for {npz_file.name}")
        return

    # Bin normalized radii
    norm_centers, norm_mean, norm_std, _ = bin_data(
        distances, normalized_radii, bin_size, max_distance, min_points_per_bin
    )

    # Bin absolute deviations
    abs_centers, abs_mean, abs_std, _ = bin_data(
        distances, absolute_deviations, bin_size, max_distance, min_points_per_bin
    )

    if len(norm_centers) == 0:
        logger.warning(f"No bins with enough points for {npz_file.name}")
        return

    # Generate title if not provided
    if title is None:
        sample_name = npz_file.stem.replace('_axon_profiles', '')
        title = f"Radius vs Distance to Nucleus\n{sample_name}"

    # Create plot
    plot_radius_vs_distance(
        norm_centers, norm_mean, norm_std,
        abs_mean, abs_std,
        output_file, title, len(distances)
    )


def find_profile_files(pattern: str) -> List[Path]:
    """
    Find all axon profile NPZ files matching the pattern.

    Args:
        pattern: Glob pattern for axon profile files

    Returns:
        List of matching file paths
    """
    files = sorted([Path(f) for f in glob_module.glob(pattern, recursive=True)])
    return files


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Analyze normalized axon radius vs distance to nearest nucleus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  python analyze_radius_vs_nucleus_distance.py \\
      data/processed/LM/sham_25_ipsi_axon_profiles.npz \\
      fig/sham_25_ipsi_radius_vs_nucleus.png

  # Batch processing all LM files
  python analyze_radius_vs_nucleus_distance.py \\
      "data/processed/LM/*_axon_profiles.npz" \\
      fig/radius_vs_nucleus/

  # Custom bin size
  python analyze_radius_vs_nucleus_distance.py \\
      data/processed/LM/sham_25_ipsi_axon_profiles.npz \\
      fig/output.png --distance-bin-size 5.0
        """
    )

    parser.add_argument('input', type=str,
                        help='Input: single NPZ file or glob pattern for 3D axon profiles')
    parser.add_argument('output', type=Path,
                        help='Output PNG file (single) or directory (batch)')
    parser.add_argument('--nucleus-dir', type=Path, default=Path('data/raw'),
                        help='Directory containing nucleus .mat files (default: data/raw)')
    parser.add_argument('--distance-bin-size', type=float, default=2.0,
                        help='Distance bin size in micrometers (default: 2.0)')
    parser.add_argument('--min-points-per-bin', type=int, default=100,
                        help='Minimum points per bin to include in plot (default: 100)')
    parser.add_argument('--max-distance', type=float, default=10.0,
                        help='Maximum distance to consider in micrometers (default: 10.0)')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title (auto-generated if not provided)')
    parser.add_argument('--pool', action='store_true',
                        help='Pool data from all input files into a single plot')

    args = parser.parse_args()

    # Find matching files
    files = find_profile_files(args.input)

    if not files:
        logger.error(f"No files found matching pattern: {args.input}")
        return 1

    logger.info(f"Found {len(files)} axon profile file(s)")

    if args.pool and len(files) > 1:
        # Pooled mode - collect data from all files, create single plot
        all_distances = []
        all_normalized_radii = []
        all_absolute_deviations = []
        successful = []
        failed = []

        for npz_file in files:
            try:
                distances, normalized_radii, absolute_deviations = collect_data_from_file(
                    npz_file, args.nucleus_dir
                )
                if len(distances) > 0:
                    all_distances.append(distances)
                    all_normalized_radii.append(normalized_radii)
                    all_absolute_deviations.append(absolute_deviations)
                    successful.append(npz_file.name)
            except Exception as e:
                failed.append((npz_file.name, str(e)))
                logger.error(f"Failed to process {npz_file.name}: {e}")

        if not all_distances:
            logger.error("No valid data collected from any files")
            return 1

        # Pool all data
        distances = np.concatenate(all_distances)
        normalized_radii = np.concatenate(all_normalized_radii)
        absolute_deviations = np.concatenate(all_absolute_deviations)

        logger.info(f"\n{'='*60}")
        logger.info(f"Pooled Data Summary")
        logger.info(f"{'='*60}")
        logger.info(f"Files processed: {len(successful)}/{len(files)}")
        logger.info(f"Total skeleton points: {len(distances):,}")
        logger.info(f"Distance range: {distances.min():.2f} - {distances.max():.2f} μm")

        # Bin pooled data
        norm_centers, norm_mean, norm_std, _ = bin_data(
            distances, normalized_radii, args.distance_bin_size,
            args.max_distance, args.min_points_per_bin
        )
        _, abs_mean, abs_std, _ = bin_data(
            distances, absolute_deviations, args.distance_bin_size,
            args.max_distance, args.min_points_per_bin
        )

        if len(norm_centers) == 0:
            logger.error("No bins with enough points in pooled data")
            return 1

        # Generate title
        title = args.title or f"Radius vs Distance to Nucleus\nPooled ({len(successful)} samples)"

        # Create plot
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plot_radius_vs_distance(
            norm_centers, norm_mean, norm_std,
            abs_mean, abs_std,
            args.output, title, len(distances)
        )

        if failed:
            logger.info(f"Failed: {len(failed)} files")
            for filename, error in failed:
                logger.info(f"  - {filename}: {error}")

    elif len(files) == 1:
        # Single file mode
        process_single_file(
            files[0],
            args.nucleus_dir,
            args.output,
            args.distance_bin_size,
            args.max_distance,
            args.min_points_per_bin,
            args.title
        )
    else:
        # Batch mode - output is directory
        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)

        successful = []
        failed = []

        for npz_file in files:
            sample_name = npz_file.stem.replace('_axon_profiles', '')
            output_file = output_dir / f"{sample_name}_radius_vs_nucleus.png"

            try:
                process_single_file(
                    npz_file,
                    args.nucleus_dir,
                    output_file,
                    args.distance_bin_size,
                    args.max_distance,
                    args.min_points_per_bin,
                    None  # Auto-generate title per file
                )
                successful.append(npz_file.name)
            except Exception as e:
                failed.append((npz_file.name, str(e)))
                logger.error(f"Failed to process {npz_file.name}: {e}")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Processing Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully processed: {len(successful)}/{len(files)} files")

        if failed:
            logger.info(f"Failed: {len(failed)} files")
            for filename, error in failed:
                logger.info(f"  - {filename}: {error}")

    return 0


if __name__ == '__main__':
    exit(main())
