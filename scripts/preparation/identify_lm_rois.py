"""
Identify CC and CG population ROIs using orientation-based classification and spatial separation.

This script:
1. Identifies CC and CG populations using dominant-axis orientation classification
2. Finds an optimal separation plane along one image axis
3. Outputs ROI bounding boxes as JSON
4. Outputs a Neuroglancer-compatible segmentation for visualization

The separation plane is placed at the midpoint between population centroids,
with a configurable margin to create non-overlapping ROIs.

Example usage:
    python scripts/processing/identify_population_rois.py \\
        data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        data/processed/LM \\
        --min-length 50.0 \\
        --margin 10.0
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from axonometry.populations import (classify_by_dominant_axis,
                                    compute_all_orientations,
                                    create_populations,
                                    load_volume_downsampled,
                                    precompute_axon_voxels)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_population_centroids(
    populations: List[Dict],
    axon_data: Dict[int, Dict],
) -> Dict[str, np.ndarray]:
    """
    Compute mean centroid for each population.

    Args:
        populations: List of population dicts with 'name' and 'axon_labels'
        axon_data: Dict mapping axon label to data including 'centroid_um'

    Returns:
        Dict mapping population name to centroid array [z, y, x] in micrometers
    """
    centroids = {}
    for pop in populations:
        name = pop['name']
        labels = pop['axon_labels']

        pop_centroids = []
        for label in labels:
            if label in axon_data:
                pop_centroids.append(axon_data[label]['centroid_um'])

        if pop_centroids:
            centroids[name] = np.mean(pop_centroids, axis=0)
            logger.info(f"  {name.upper()} centroid: {centroids[name]}")
        else:
            logger.warning(f"  {name.upper()}: no valid centroids found")

    return centroids


def find_separation_plane(
    populations: List[Dict],
    axon_data: Dict[int, Dict],
    centroids: Dict[str, np.ndarray],
) -> Tuple[int, float, Dict]:
    """
    Find the axis and position that best separates two populations.

    Uses the axis with the largest centroid difference, then finds the
    optimal plane position that minimizes overlap (misclassification).

    Args:
        populations: List of population dicts with 'name' and 'axon_labels'
        axon_data: Dict mapping axon label to data including 'centroid_um'
        centroids: Dict mapping population name to mean centroid [z, y, x] in um

    Returns:
        Tuple of (axis_index, position_um, stats_dict)
    """
    if 'cc' not in centroids or 'cg' not in centroids:
        raise ValueError("Both CC and CG centroids required for separation")

    cc_centroid = np.array(centroids['cc'])
    cg_centroid = np.array(centroids['cg'])

    # Find axis with largest difference
    diff = np.abs(cc_centroid - cg_centroid)
    best_axis = int(np.argmax(diff))

    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    logger.info(f"  Centroid differences: Z={diff[0]:.1f}, Y={diff[1]:.1f}, X={diff[2]:.1f} um")
    logger.info(f"  Best separation axis: {axis_names[best_axis]} (axis {best_axis})")

    # Collect axon positions along separation axis for each population
    pop_positions = {}
    for pop in populations[:2]:
        name = pop['name']
        positions = []
        for label in pop['axon_labels']:
            if label in axon_data:
                positions.append(axon_data[label]['centroid_um'][best_axis])
        pop_positions[name] = np.array(positions)

    cc_positions = pop_positions['cc']
    cg_positions = pop_positions['cg']

    # Determine which population is on which side (based on mean)
    cc_mean = np.mean(cc_positions)
    cg_mean = np.mean(cg_positions)
    cc_on_positive = cc_mean > cg_mean  # CC is on the "high" side

    logger.info(f"  CC mean position: {cc_mean:.1f} um, CG mean position: {cg_mean:.1f} um")
    logger.info(f"  CC on {'positive' if cc_on_positive else 'negative'} side")

    # Find optimal separation plane by minimizing overlap
    # Combine all positions with labels (1 = CC, 0 = CG)
    all_positions = np.concatenate([cc_positions, cg_positions])
    all_labels = np.concatenate([np.ones(len(cc_positions)), np.zeros(len(cg_positions))])

    # Sort by position
    sort_idx = np.argsort(all_positions)
    sorted_positions = all_positions[sort_idx]
    sorted_labels = all_labels[sort_idx]

    # Sweep through possible thresholds and count misclassifications
    # If CC is on positive side: CC below threshold = error, CG above threshold = error
    # If CC is on negative side: CC above threshold = error, CG below threshold = error
    n_total = len(all_positions)
    n_cc = len(cc_positions)
    n_cg = len(cg_positions)

    best_threshold = None
    min_errors = n_total

    # Count cumulative CC and CG below each threshold
    cc_below = np.cumsum(sorted_labels)  # CC count at or below this index
    cg_below = np.cumsum(1 - sorted_labels)  # CG count at or below this index

    for i in range(n_total - 1):
        # Threshold between position[i] and position[i+1]
        threshold = (sorted_positions[i] + sorted_positions[i + 1]) / 2

        if cc_on_positive:
            # CC should be above threshold, CG should be below
            cc_errors = cc_below[i]  # CC below threshold (wrong side)
            cg_errors = n_cg - cg_below[i]  # CG above threshold (wrong side)
        else:
            # CC should be below threshold, CG should be above
            cc_errors = n_cc - cc_below[i]  # CC above threshold (wrong side)
            cg_errors = cg_below[i]  # CG below threshold (wrong side)

        total_errors = cc_errors + cg_errors

        if total_errors < min_errors:
            min_errors = total_errors
            best_threshold = threshold

    # Fallback to midpoint if no improvement found
    if best_threshold is None:
        best_threshold = (cc_mean + cg_mean) / 2

    position_um = float(best_threshold)

    # Compute final stats
    if cc_on_positive:
        cc_correct = np.sum(cc_positions > position_um)
        cg_correct = np.sum(cg_positions < position_um)
    else:
        cc_correct = np.sum(cc_positions < position_um)
        cg_correct = np.sum(cg_positions > position_um)

    cc_accuracy = cc_correct / len(cc_positions) * 100
    cg_accuracy = cg_correct / len(cg_positions) * 100

    logger.info(f"  Optimal separation plane: {position_um:.1f} um")
    logger.info(f"  CC accuracy: {cc_accuracy:.1f}% ({cc_correct}/{len(cc_positions)} correct)")
    logger.info(f"  CG accuracy: {cg_accuracy:.1f}% ({cg_correct}/{len(cg_positions)} correct)")

    stats = {
        'cc_accuracy': cc_accuracy,
        'cg_accuracy': cg_accuracy,
        'cc_on_positive': cc_on_positive,
    }

    return best_axis, position_um, stats


def compute_rois(
    separation_axis: int,
    separation_position_um: float,
    margin_um: float,
    volume_shape: Tuple[int, int, int],
    voxel_size_um: float,
    centroids: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    """
    Compute ROI bounding boxes from separation plane.

    Args:
        separation_axis: Axis index (0=Z, 1=Y, 2=X)
        separation_position_um: Position of separation plane in micrometers
        margin_um: Margin on each side of the plane
        volume_shape: Shape of volume (z, y, x)
        voxel_size_um: Voxel size in micrometers
        centroids: Population centroids to determine which side is CC/CG

    Returns:
        Dict with 'cc' and 'cg' ROI definitions
    """
    # Volume bounds in micrometers
    volume_max_um = [s * voxel_size_um for s in volume_shape]

    # Determine which population is on which side of the plane
    cc_pos = centroids['cc'][separation_axis]
    cg_pos = centroids['cg'][separation_axis]

    # CC is on the side where its centroid is located
    if cc_pos < separation_position_um:
        cc_side = 'negative'  # CC is below the plane
        cg_side = 'positive'  # CG is above the plane
    else:
        cc_side = 'positive'
        cg_side = 'negative'

    logger.info(f"  CC on {cc_side} side, CG on {cg_side} side")

    # Compute ROI bounds
    rois = {}

    # Base bounds (full volume extent on non-separation axes)
    base_min = [0.0, 0.0, 0.0]
    base_max = volume_max_um.copy()

    # CC ROI
    cc_min = base_min.copy()
    cc_max = base_max.copy()
    if cc_side == 'negative':
        cc_max[separation_axis] = separation_position_um - margin_um
    else:
        cc_min[separation_axis] = separation_position_um + margin_um

    rois['cc'] = {
        'min_um': cc_min,
        'max_um': cc_max,
    }

    # CG ROI
    cg_min = base_min.copy()
    cg_max = base_max.copy()
    if cg_side == 'negative':
        cg_max[separation_axis] = separation_position_um - margin_um
    else:
        cg_min[separation_axis] = separation_position_um + margin_um

    rois['cg'] = {
        'min_um': cg_min,
        'max_um': cg_max,
    }

    # Validate ROI extents
    for name, roi in list(rois.items()):
        extent = [roi['max_um'][i] - roi['min_um'][i] for i in range(3)]
        if any(e <= 0 for e in extent):
            logger.warning(
                f"  {name.upper()} ROI has zero/negative extent: {extent} um "
                f"(separation at {separation_position_um:.1f} um with {margin_um:.1f} um margin)"
            )

    return rois



def identify_population_rois(
    mat_file: Path,
    output_dir: Path,
    voxel_size_um: float = 0.05,
    downsample: int = 4,
    max_angle_deg: float = 30.0,
    min_length_um: float = 50.0,
    k_neighbors: int = 10,
    max_neighbor_distance_um: float = 30.0,
    margin_um: float = 10.0,
) -> Dict:
    """
    Identify CC and CG population ROIs.

    Args:
        mat_file: Input .mat file with labeled volume
        output_dir: Output directory for results
        voxel_size_um: Voxel size in micrometers
        downsample: Downsampling factor for population identification
        max_angle_deg: Maximum angle from dominant axis for classification
        min_length_um: Minimum axon length for inclusion
        k_neighbors: K for KNN sparse filtering
        max_neighbor_distance_um: Max distance for KNN
        margin_um: Margin on each side of separation plane

    Returns:
        Dictionary with ROI metadata
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Identifying population ROIs: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Step 1: Load and identify populations
    logger.info("Step 1: Loading volume and identifying populations...")
    volume_ds, vol_metadata = load_volume_downsampled(mat_file, downsample)
    voxel_size_ds = voxel_size_um * downsample

    axon_voxels_ds = precompute_axon_voxels(volume_ds)
    axon_data = compute_all_orientations(
        axon_voxels_ds, voxel_size_ds, min_voxels=3, min_length_um=min_length_um
    )

    if not axon_data:
        raise ValueError("No valid axons found after filtering")

    label_to_bundle, axis_to_bundle = classify_by_dominant_axis(axon_data, max_angle_deg)

    if not label_to_bundle:
        raise ValueError("No axons remain after classification")

    populations = create_populations(
        axon_data, label_to_bundle, axis_to_bundle, axon_voxels_ds, voxel_size_ds,
        k_neighbors, max_neighbor_distance_um
    )

    if len(populations) < 2:
        raise ValueError(f"Only {len(populations)} population(s) found, expected 2 (CC and CG)")

    # Step 2: Compute population centroids
    logger.info("\nStep 2: Computing population centroids...")
    centroids = compute_population_centroids(populations, axon_data)

    # Step 3: Find separation plane
    logger.info("\nStep 3: Finding optimal separation plane...")
    volume_shape = vol_metadata['original_shape']
    separation_axis, separation_position, _separation_stats = find_separation_plane(
        populations, axon_data, centroids
    )

    # Step 4: Compute ROI bounding boxes
    logger.info("\nStep 4: Computing ROI bounding boxes...")
    rois = compute_rois(
        separation_axis, separation_position, margin_um,
        volume_shape, voxel_size_um, centroids
    )

    for name, roi in rois.items():
        logger.info(f"  {name.upper()} ROI: {roi['min_um']} to {roi['max_um']}")

    # Construct output path
    # Derive sample name: LM_25_ipsi_myelinated_axons -> lm_25_ipsi
    # Parent dir encodes condition: Sham_25_ipsi -> sham
    parent_name = mat_file.parent.name.lower()  # e.g. "sham_25_ipsi"
    condition = parent_name.split('_')[0]  # e.g. "sham"
    stem = mat_file.stem.replace('_myelinated_axons', '')  # e.g. "LM_25_ipsi"
    parts = stem.split('_', 1)  # ["LM", "25_ipsi"]
    sample_name = f"{condition}_{parts[1]}" if len(parts) > 1 else stem.lower()
    sample_output_dir = output_dir

    # Step 5: Save per-population ROI JSONs (simplified format)
    logger.info("\nStep 6: Saving ROI JSONs...")

    # Build a lookup for dominant_axis per population
    pop_dominant_axis = {
        pop['name']: pop['dominant_axis']
        for pop in populations[:2]
    }

    roi_jsons = {}
    for pop_name, roi in rois.items():
        # Convert um bounds to voxel coordinates
        min_vox = [max(0, int(roi['min_um'][i] / voxel_size_um)) for i in range(3)]
        max_vox = [min(volume_shape[i], int(np.ceil(roi['max_um'][i] / voxel_size_um))) for i in range(3)]

        roi_data = {
            "fiber_dir_axis": pop_dominant_axis[pop_name],
            "min": min_vox,
            "max": max_vox,
        }

        json_path = sample_output_dir / f"{sample_name}_{pop_name}_roi.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(roi_data, f, indent=2)

        roi_jsons[pop_name] = str(json_path)
        logger.info(f"  {pop_name.upper()}: {json_path}")
        logger.info(f"    fiber_dir_axis={pop_dominant_axis[pop_name]}, "
                     f"min={min_vox}, max={max_vox}")

    logger.info(f"\n{'='*80}")
    logger.info("ROI identification complete!")
    for pop_name, path in roi_jsons.items():
        logger.info(f"  {pop_name.upper()}: {path}")
    logger.info(f"{'='*80}\n")

    return roi_jsons


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Identify CC and CG population ROIs for all LM volumes in a source directory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/preparation/identify_lm_rois.py \\
        data/source/rat \\
        data/raw/rat/LM

Finds all LM_*_myelinated_axons.mat files under source_dir and writes
per-population ROI JSONs to output_dir.
        """
    )
    parser.add_argument(
        'source_dir', type=Path, nargs='?', default=Path('data/source/rat'),
        help='Root directory containing LM source .mat files (default: data/source/rat)'
    )
    parser.add_argument(
        'output_dir', type=Path, nargs='?', default=Path('data/raw/rat/lm'),
        help='Output directory for ROI JSONs and Neuroglancer volumes (default: data/raw/rat/lm)'
    )
    parser.add_argument(
        '--voxel-size', type=float, default=0.05,
        help='Voxel size in um (default: 0.05)'
    )
    parser.add_argument(
        '--downsample', type=int, default=4,
        help='Downsampling factor for population identification (default: 4)'
    )
    parser.add_argument(
        '--max-angle', type=float, default=30.0,
        help='Max deviation from dominant axis in degrees (default: 30.0)'
    )
    parser.add_argument(
        '--min-length', type=float, default=50.0,
        help='Minimum axon length in um (default: 50.0)'
    )
    parser.add_argument(
        '--k-neighbors', type=int, default=10,
        help='K for sparse axon filtering (default: 10)'
    )
    parser.add_argument(
        '--max-neighbor-distance', type=float, default=30.0,
        help='Max distance to k-th neighbor in um (default: 30.0)'
    )
    parser.add_argument(
        '--margin', type=float, default=10.0,
        help='Margin on each side of separation plane in um (default: 10.0)'
    )
    args = parser.parse_args()

    # Find all LM .mat files under source_dir
    input_files = sorted(args.source_dir.rglob('LM_*_myelinated_axons.mat'))

    if not input_files:
        logger.error(f"No LM_*_myelinated_axons.mat files found under {args.source_dir}")
        return

    logger.info(f"Found {len(input_files)} LM volume(s)")

    for mat_path in input_files:
        try:
            identify_population_rois(
                mat_path,
                args.output_dir,
                voxel_size_um=args.voxel_size,
                downsample=args.downsample,
                max_angle_deg=args.max_angle,
                min_length_um=args.min_length,
                k_neighbors=args.k_neighbors,
                max_neighbor_distance_um=args.max_neighbor_distance,
                margin_um=args.margin,
            )
        except Exception as e:
            logger.error(f"Failed to process {mat_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
