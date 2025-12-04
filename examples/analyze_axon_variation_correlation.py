#!/usr/bin/env python3
"""
Analyze correlation of axon radius variation profiles between axon pairs.

For each axon, we compute its per-slice radius deviation from its own mean.
Then for each pair of axons that share common slices, we correlate their
deviation profiles and plot correlation vs mean inter-axon distance.

This differs from analyze_spatial_correlation_ensemble.py which computes
per-slice pairwise correlations. Here we correlate ENTIRE profiles.

Algorithm:
1. Load labeled volume from .mat file
2. Pass 1: Compute per-axon mean radius and build per-slice deviation profile
3. For each axon pair with sufficient shared slices:
   - Extract deviation profiles over common slice range
   - Compute Pearson correlation
   - Compute mean XY centroid distance
4. Bin correlations by distance and plot

Usage:
    python analyze_axon_variation_correlation.py \\
        data/raw/LM/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        fig/axon_variation_correlation.png
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob as glob_module

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import pearsonr, ttest_1samp
from skimage.measure import regionprops
from tqdm import tqdm

# Import from axonometry library
sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry import get_plot_settings
from axonometry.io import load_volume_with_metadata, resample_to_isotropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()

# Global variables for worker processes
_volume_data = None
_slice_axis = None
_voxel_size_um = None


def find_populations_json(mat_file: Path) -> Path:
    """Find the populations JSON file for a given .mat file."""
    stem = mat_file.stem.replace('_myelinated_axons', '')
    parts = stem.split('_')
    if len(parts) >= 3:
        rat_id = parts[1]
        hemisphere = parts[2]
        parent_name = mat_file.parent.name.lower()
        if 'sham' in parent_name:
            condition = 'sham'
        elif 'tbi' in parent_name:
            condition = 'tbi'
        else:
            condition = 'sham'
        json_name = f"{condition}_{rat_id}_{hemisphere}_populations.json"
        json_path = Path("data/processed/LM") / json_name
        if json_path.exists():
            return json_path
    raise FileNotFoundError(f"Could not find populations JSON for {mat_file}")


def get_dominant_axis_from_json(json_file: Path) -> int:
    """Get the most common dominant axis from populations JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    populations = data.get('populations', [])
    if not populations:
        return 0
    best_pop = max(populations, key=lambda p: p.get('n_axons', 0))
    return best_pop.get('dominant_axis', 0)


# =============================================================================
# Pass 1: Build per-axon deviation profiles
# =============================================================================

def _init_worker(volume_data: np.ndarray, slice_axis: int, voxel_size_um: float):
    """Initialize worker process."""
    global _volume_data, _slice_axis, _voxel_size_um
    _volume_data = volume_data
    _slice_axis = slice_axis
    _voxel_size_um = voxel_size_um


def _process_slice(z: int) -> Dict[int, Tuple[float, float, float]]:
    """
    Extract radius and centroid for each axon in a slice.

    Returns: {label: (radius_um, centroid_y_um, centroid_x_um)}
    """
    pixel_area_um2 = _voxel_size_um * _voxel_size_um

    if _slice_axis == 0:
        slice_2d = _volume_data[z, :, :]
    elif _slice_axis == 1:
        slice_2d = _volume_data[:, z, :]
    else:
        slice_2d = _volume_data[:, :, z]

    regions = regionprops(slice_2d.astype(np.int32))

    result = {}
    for region in regions:
        area_um2 = region.area * pixel_area_um2
        radius_um = np.sqrt(area_um2 / np.pi)
        cy, cx = region.centroid
        cy_um = cy * _voxel_size_um
        cx_um = cx * _voxel_size_um
        result[region.label] = (radius_um, cy_um, cx_um)

    return result


def build_axon_profiles(volume: np.ndarray, slice_axis: int,
                        voxel_size_um: float, n_jobs: int = -1,
                        min_slices: int = 10) -> Dict[int, Dict]:
    """
    Build per-axon profiles: radius at each slice, mean, deviation profile.

    Returns:
        Dict mapping label to {
            'slice_radii': {slice_idx: radius},
            'slice_centroids': {slice_idx: (cy, cx)},
            'mean_radius': float,
            'mean_centroid': (cy, cx),
            'slices': sorted list of slice indices
        }
    """
    n_slices = volume.shape[slice_axis]

    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    # Process all slices
    if n_workers > 1:
        with Pool(processes=n_workers,
                  initializer=_init_worker,
                  initargs=(volume, slice_axis, voxel_size_um)) as pool:
            slice_results = list(tqdm(
                pool.imap(_process_slice, range(n_slices)),
                total=n_slices,
                desc="  Processing slices",
                leave=False
            ))
    else:
        _init_worker(volume, slice_axis, voxel_size_um)
        slice_results = [_process_slice(z) for z in tqdm(range(n_slices),
                                                          desc="  Processing slices",
                                                          leave=False)]

    # Aggregate per-axon data
    axon_data = defaultdict(lambda: {'slice_radii': {}, 'slice_centroids': {}})

    for slice_idx, slice_result in enumerate(slice_results):
        for label, (radius, cy, cx) in slice_result.items():
            axon_data[label]['slice_radii'][slice_idx] = radius
            axon_data[label]['slice_centroids'][slice_idx] = (cy, cx)

    # Compute statistics and filter by min_slices
    profiles = {}
    for label, data in axon_data.items():
        slices = sorted(data['slice_radii'].keys())
        if len(slices) < min_slices:
            continue

        radii = [data['slice_radii'][s] for s in slices]
        centroids = [data['slice_centroids'][s] for s in slices]

        mean_radius = np.mean(radii)
        mean_cy = np.mean([c[0] for c in centroids])
        mean_cx = np.mean([c[1] for c in centroids])

        profiles[label] = {
            'slice_radii': data['slice_radii'],
            'slice_centroids': data['slice_centroids'],
            'mean_radius': mean_radius,
            'mean_centroid': (mean_cy, mean_cx),
            'slices': slices
        }

    return profiles


# =============================================================================
# Compute pairwise correlations
# =============================================================================

def compute_pair_correlation(profile_i: Dict, profile_j: Dict,
                              min_shared_slices: int = 10) -> Optional[Tuple[float, float, float, int]]:
    """
    Compute correlation between two axon deviation profiles.

    Returns: (correlation, p_value, mean_distance, n_shared) or None
    """
    # Find shared slices
    slices_i = set(profile_i['slices'])
    slices_j = set(profile_j['slices'])
    shared_slices = sorted(slices_i & slices_j)

    if len(shared_slices) < min_shared_slices:
        return None

    # Extract radii at shared slices
    radii_i = np.array([profile_i['slice_radii'][s] for s in shared_slices])
    radii_j = np.array([profile_j['slice_radii'][s] for s in shared_slices])

    # Compute relative deviations
    mean_i = profile_i['mean_radius']
    mean_j = profile_j['mean_radius']

    if mean_i <= 0 or mean_j <= 0:
        return None

    delta_i = (radii_i - mean_i) / mean_i
    delta_j = (radii_j - mean_j) / mean_j

    # Compute correlation
    try:
        r, p = pearsonr(delta_i, delta_j)
    except Exception:
        return None

    if np.isnan(r):
        return None

    # Compute surface-to-surface distance (centroid distance minus sum of radii)
    cy_i, cx_i = profile_i['mean_centroid']
    cy_j, cx_j = profile_j['mean_centroid']
    centroid_dist = np.sqrt((cy_i - cy_j)**2 + (cx_i - cx_j)**2)
    surface_dist = max(0, centroid_dist - mean_i - mean_j)

    return (r, p, surface_dist, len(shared_slices))


def compute_all_pair_correlations(profiles: Dict[int, Dict],
                                   max_distance: float = 50.0,
                                   min_shared_slices: int = 10,
                                   max_pairs: int = 500000) -> List[Tuple[float, float, int, int]]:
    """
    Compute correlations for all axon pairs within max_distance.

    Returns: List of (correlation, mean_distance, label_i, label_j)
    """
    labels = list(profiles.keys())
    n_axons = len(labels)

    if n_axons < 2:
        return []

    # Build KDTree from mean centroids for efficient neighbor search
    centroids = np.array([profiles[l]['mean_centroid'] for l in labels])
    tree = KDTree(centroids)

    # Find candidate pairs within max_distance
    candidate_pairs = tree.query_pairs(max_distance)
    n_candidates = len(candidate_pairs)
    logger.info(f"  Found {n_candidates:,} candidate pairs within {max_distance} um")

    # Subsample if too many
    pairs_list = list(candidate_pairs)
    if n_candidates > max_pairs:
        logger.info(f"  Subsampling to {max_pairs:,} pairs")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pairs_list), size=max_pairs, replace=False)
        pairs_list = [pairs_list[i] for i in indices]

    # Compute correlations
    results = []
    for idx_i, idx_j in tqdm(pairs_list, desc="  Computing correlations", leave=False):
        label_i = labels[idx_i]
        label_j = labels[idx_j]

        result = compute_pair_correlation(
            profiles[label_i], profiles[label_j],
            min_shared_slices=min_shared_slices
        )

        if result is not None:
            r, p, dist, n_shared = result
            results.append((r, dist, label_i, label_j))

    return results


# =============================================================================
# Pooling and Statistics
# =============================================================================

def pool_correlations_by_distance(results: List[Tuple[float, float, int, int]],
                                   distance_bins: np.ndarray,
                                   n_bootstrap: int = 1000) -> Dict:
    """Pool pair correlations by mean inter-axon distance."""
    n_bins = len(distance_bins) - 1
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2

    # Bin results by distance
    binned_correlations = [[] for _ in range(n_bins)]

    for r, dist, _, _ in results:
        bin_idx = np.searchsorted(distance_bins, dist, side='right') - 1
        if 0 <= bin_idx < n_bins:
            binned_correlations[bin_idx].append(r)

    # Compute statistics per bin
    mean_correlations = []
    ci_low = []
    ci_high = []
    n_pairs = []
    p_values = []

    rng = np.random.default_rng(42)
    min_for_stats = 30

    for correlations in binned_correlations:
        n_pairs.append(len(correlations))

        if len(correlations) < min_for_stats:
            mean_correlations.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            p_values.append(np.nan)
            continue

        correlations = np.array(correlations)
        mean_correlations.append(np.mean(correlations))

        # Bootstrap CI
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(correlations, size=len(correlations), replace=True)
            boot_means.append(np.mean(sample))
        ci_low.append(np.percentile(boot_means, 2.5))
        ci_high.append(np.percentile(boot_means, 97.5))

        # One-sample t-test against 0
        _, p = ttest_1samp(correlations, 0)
        p_values.append(p)

    return {
        'bin_centers': bin_centers.tolist(),
        'bin_edges': distance_bins.tolist(),
        'mean_correlations': mean_correlations,
        'n_pairs': n_pairs,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_values': p_values
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_correlation_by_distance(results: Dict, output_file: Path,
                                  n_samples: int, total_axons: int,
                                  total_pairs: int):
    """Plot pooled correlation vs distance."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08}
    )

    font_settings = settings.fonts

    bin_centers = np.array(results['bin_centers'])
    bin_edges = np.array(results['bin_edges'])
    correlations = np.array(results['mean_correlations'])
    ci_low = np.array(results['ci_low'])
    ci_high = np.array(results['ci_high'])
    n_pairs = np.array(results['n_pairs'])
    p_values = np.array(results['p_values'])
    bin_width = bin_edges[1] - bin_edges[0]

    valid = ~np.isnan(correlations)

    # Colors
    main_color = '#2E86AB'
    sig_color = '#E94F37'
    nonsig_color = '#A1A1A1'

    # ===== Top plot: Correlation vs distance =====
    ax1.axhline(0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axhspan(-0.05, 0.05, color='#E8E8E8', alpha=0.5, zorder=0)

    if np.any(valid):
        sig_mask = valid & (np.array(p_values) < 0.05)
        nonsig_mask = valid & (np.array(p_values) >= 0.05)

        if np.any(nonsig_mask):
            yerr_low_ns = correlations[nonsig_mask] - ci_low[nonsig_mask]
            yerr_high_ns = ci_high[nonsig_mask] - correlations[nonsig_mask]
            ax1.errorbar(bin_centers[nonsig_mask], correlations[nonsig_mask],
                        yerr=[yerr_low_ns, yerr_high_ns],
                        fmt='o', color=nonsig_color, capsize=3, capthick=1,
                        markersize=8, markerfacecolor='white', markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p >= 0.05')

        if np.any(sig_mask):
            yerr_low_s = correlations[sig_mask] - ci_low[sig_mask]
            yerr_high_s = ci_high[sig_mask] - correlations[sig_mask]
            ax1.errorbar(bin_centers[sig_mask], correlations[sig_mask],
                        yerr=[yerr_low_s, yerr_high_s],
                        fmt='o', color=sig_color, capsize=3, capthick=1,
                        markersize=8, markerfacecolor=sig_color, markeredgewidth=1.5,
                        linewidth=0, elinewidth=1.2, label='p < 0.05')

        ax1.plot(bin_centers[valid], correlations[valid],
                '-', color=main_color, alpha=0.4, linewidth=1, zorder=1)

    ax1.set_ylabel('Mean Profile Correlation',
                   fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax1.set_title('Axon Radius Profile Correlation vs Distance\n'
                  r'$r = \mathrm{corr}(\Delta r_i(z), \Delta r_j(z))$ over shared slices',
                  fontsize=font_settings['title_size'], fontweight=font_settings['weight'])
    ax1.legend(loc='upper right', fontsize=font_settings['legend_size'], framealpha=0.9)

    if np.any(valid):
        max_abs = max(abs(np.nanmin(correlations)), abs(np.nanmax(correlations)))
        y_limit = max(0.15, min(0.5, max_abs * 1.5))
    else:
        y_limit = 0.3
    ax1.set_ylim(-y_limit, y_limit)
    ax1.grid(True, alpha=0.3)

    # Summary stats
    valid_corrs = [c for c in correlations if not np.isnan(c)]
    nn_corr = valid_corrs[0] if valid_corrs else np.nan

    short_range_corrs = [c for c, d in zip(correlations, bin_centers)
                         if d < 10 and not np.isnan(c)]
    sr_mean = np.mean(short_range_corrs) if short_range_corrs else np.nan

    summary_text = f'n = {n_samples} samples\n{total_axons:,} axons\n{total_pairs:,} pairs'
    if not np.isnan(nn_corr):
        summary_text += f'\nNN corr: {nn_corr:.3f}'
    if not np.isnan(sr_mean):
        summary_text += f'\n<10um mean: {sr_mean:.3f}'

    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.9))

    # ===== Bottom plot: Number of pairs =====
    colors = [sig_color if p < 0.05 and not np.isnan(p) else nonsig_color for p in p_values]
    ax2.bar(bin_centers, n_pairs, width=bin_width * 0.85,
            color=colors, edgecolor='white', linewidth=0.5, alpha=0.8)

    ax2.set_xlabel('Surface-to-Surface Distance (um)',
                   fontsize=font_settings['label_size'], fontweight=font_settings['weight'])
    ax2.set_ylabel('N Pairs', fontsize=font_settings['label_size'])
    ax2.set_yscale('log')
    ax2.set_xlim(bin_edges[0], bin_edges[-1])
    ax2.grid(True, alpha=0.3)

    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved figure to {output_file}")


# =============================================================================
# Sample Processing
# =============================================================================

def process_sample(mat_file: Path, voxel_size_um: Optional[float],
                   max_distance: float, min_shared_slices: int,
                   max_pairs: int, n_jobs: int) -> Tuple[str, List, int]:
    """Process a single .mat file."""
    sample_name = mat_file.stem.replace('_myelinated_axons', '')
    logger.info(f"Processing {sample_name}...")

    # Load volume
    volume, voxel_tuple, metadata = load_volume_with_metadata(mat_file, voxel_size_um)
    logger.info(f"  Volume shape: {volume.shape}")

    # Get slice axis
    slice_axis = 0
    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}

    if metadata is not None and 'dominant_axis' in metadata:
        slice_axis = metadata['dominant_axis']
        logger.info(f"  Using dominant axis from metadata: {slice_axis} ({axis_names[slice_axis]})")
    else:
        try:
            json_file = find_populations_json(mat_file)
            slice_axis = get_dominant_axis_from_json(json_file)
            logger.info(f"  Using dominant axis from populations JSON: {slice_axis} ({axis_names[slice_axis]})")
        except FileNotFoundError:
            logger.warning(f"  No dominant axis found, defaulting to axis 0 (Z)")

    # Resample to isotropic
    volume, iso_voxel_size = resample_to_isotropic(volume, voxel_tuple)
    n_slices = volume.shape[slice_axis]
    logger.info(f"  {n_slices} slices along axis {slice_axis}, voxel: {iso_voxel_size:.3f} um")

    # Build per-axon profiles
    profiles = build_axon_profiles(volume, slice_axis, iso_voxel_size, n_jobs,
                                    min_slices=min_shared_slices)
    n_axons = len(profiles)
    logger.info(f"  Found {n_axons} axons with >= {min_shared_slices} slices")

    # Compute pairwise correlations
    results = compute_all_pair_correlations(
        profiles,
        max_distance=max_distance,
        min_shared_slices=min_shared_slices,
        max_pairs=max_pairs
    )
    logger.info(f"  Computed {len(results):,} valid correlations")

    return sample_name, results, n_axons


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze correlation of axon radius profiles between pairs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single .mat file
  python analyze_axon_variation_correlation.py \\
      "data/raw/LM/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat" \\
      fig/axon_variation_correlation.png

  # Multiple files
  python analyze_axon_variation_correlation.py \\
      "data/raw/LM/*/LM_*_myelinated_axons.mat" \\
      fig/axon_variation_correlation_pooled.png
        """
    )

    parser.add_argument('input_pattern', type=str,
                        help='Glob pattern for .mat files')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--voxel-size', type=float, default=None,
                        help='Override voxel size in um (default: from metadata)')
    parser.add_argument('--max-distance', type=float, default=50.0,
                        help='Maximum centroid distance to consider in um (default: 50)')
    parser.add_argument('--min-shared-slices', type=int, default=10,
                        help='Minimum shared slices for correlation (default: 10)')
    parser.add_argument('--distance-bin-width', type=float, default=5.0,
                        help='Width of distance bins in um (default: 5.0)')
    parser.add_argument('--max-pairs', type=int, default=500000,
                        help='Maximum pairs to process per sample (default: 500000)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all cores)')

    args = parser.parse_args()

    # Find matching files
    mat_files = sorted([Path(f) for f in glob_module.glob(args.input_pattern, recursive=True)])

    if not mat_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return

    logger.info(f"Found {len(mat_files)} .mat files")

    # Setup distance bins
    distance_bins = np.arange(0, args.max_distance + args.distance_bin_width,
                              args.distance_bin_width)

    # Process each sample
    all_results = []
    sample_names = []
    total_axons = 0

    for mat_file in mat_files:
        try:
            sample_name, results, n_axons = process_sample(
                mat_file,
                voxel_size_um=args.voxel_size,
                max_distance=args.max_distance,
                min_shared_slices=args.min_shared_slices,
                max_pairs=args.max_pairs,
                n_jobs=args.n_jobs
            )
            sample_names.append(sample_name)
            total_axons += n_axons
            all_results.extend(results)

        except Exception as e:
            logger.error(f"Failed to process {mat_file.name}: {e}")
            import traceback
            traceback.print_exc()

    if not sample_names:
        logger.error("No samples processed successfully")
        return

    total_pairs = len(all_results)
    logger.info(f"\nTotal: {total_pairs:,} pair correlations from {len(sample_names)} samples")

    # Pool correlations
    logger.info("Pooling correlations by distance...")
    pooled = pool_correlations_by_distance(all_results, distance_bins)

    # Add metadata
    pooled['n_samples'] = len(sample_names)
    pooled['total_axons'] = total_axons
    pooled['total_pairs'] = total_pairs
    pooled['sample_names'] = sample_names
    pooled['parameters'] = {
        'max_distance_um': args.max_distance,
        'min_shared_slices': args.min_shared_slices,
        'distance_bin_width_um': args.distance_bin_width
    }

    # Plot
    plot_correlation_by_distance(pooled, args.output,
                                  len(sample_names), total_axons, total_pairs)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Samples processed: {len(sample_names)}")
    logger.info(f"Total axons: {total_axons:,}")
    logger.info(f"Total pairs: {total_pairs:,}")

    valid_corrs = [c for c in pooled['mean_correlations'] if not np.isnan(c)]
    if valid_corrs:
        nn_corr = valid_corrs[0]
        short_range = [c for c, d in zip(pooled['mean_correlations'], pooled['bin_centers'])
                       if d < 10 and not np.isnan(c)]
        sr_mean = np.mean(short_range) if short_range else np.nan

        logger.info(f"Nearest-neighbor mean correlation: {nn_corr:.4f}")
        if not np.isnan(sr_mean):
            logger.info(f"Short-range (<10um) mean: {sr_mean:.4f}")

    # Save JSON results
    json_file = args.output.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(pooled, f, indent=2)
    logger.info(f"Saved results to {json_file}")


if __name__ == '__main__':
    main()
