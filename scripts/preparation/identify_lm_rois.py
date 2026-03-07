#!/usr/bin/env python3
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
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from axonometry.io import construct_output_path
from axonometry.populations import (
    load_volume_downsampled,
    precompute_axon_voxels,
    compute_all_orientations,
    classify_by_dominant_axis,
    create_populations,
)

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

    return rois


def create_roi_volume(
    volume_shape: Tuple[int, int, int],
    separation_axis: int,
    separation_position_um: float,
    margin_um: float,
    voxel_size_um: float,
    centroids: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Create a labeled volume with ROI masks.

    Labels:
        0 = margin/excluded zone
        1 = CC ROI
        2 = CG ROI

    Args:
        volume_shape: Shape of output volume (z, y, x)
        separation_axis: Axis index for separation
        separation_position_um: Position of separation plane in um
        margin_um: Margin on each side
        voxel_size_um: Voxel size in micrometers
        centroids: Population centroids

    Returns:
        Labeled volume array (uint8)
    """
    roi_volume = np.zeros(volume_shape, dtype=np.uint8)

    # Convert positions to voxel indices
    sep_pos_voxels = separation_position_um / voxel_size_um
    margin_voxels = margin_um / voxel_size_um

    # Determine CC/CG sides
    cc_pos = centroids['cc'][separation_axis]
    cc_on_negative = cc_pos < separation_position_um

    # Create coordinate array for the separation axis
    coords = np.arange(volume_shape[separation_axis])

    # Create masks for each region
    negative_mask = coords < (sep_pos_voxels - margin_voxels)
    positive_mask = coords >= (sep_pos_voxels + margin_voxels)

    # Assign labels based on which side each population is on
    if cc_on_negative:
        cc_mask = negative_mask
        cg_mask = positive_mask
    else:
        cc_mask = positive_mask
        cg_mask = negative_mask

    # Apply masks along the separation axis
    if separation_axis == 0:  # Z
        for z in range(volume_shape[0]):
            if cc_mask[z]:
                roi_volume[z, :, :] = 1
            elif cg_mask[z]:
                roi_volume[z, :, :] = 2
    elif separation_axis == 1:  # Y
        for y in range(volume_shape[1]):
            if cc_mask[y]:
                roi_volume[:, y, :] = 1
            elif cg_mask[y]:
                roi_volume[:, y, :] = 2
    else:  # X (axis 2)
        for x in range(volume_shape[2]):
            if cc_mask[x]:
                roi_volume[:, :, x] = 1
            elif cg_mask[x]:
                roi_volume[:, :, x] = 2

    # Count voxels per region
    n_cc = np.sum(roi_volume == 1)
    n_cg = np.sum(roi_volume == 2)
    n_margin = np.sum(roi_volume == 0)
    total = roi_volume.size

    logger.info(f"  ROI volume: CC={n_cc/total*100:.1f}%, CG={n_cg/total*100:.1f}%, margin={n_margin/total*100:.1f}%")

    return roi_volume


def write_neuroglancer_volume(
    roi_volume: np.ndarray,
    output_dir: Path,
    voxel_size_um: float,
    chunk_size: int = 64,
    target_resolution_um: float = 0.8,
) -> None:
    """
    Write ROI volume as Neuroglancer Precomputed format (single low-res level).

    For simple ROI masks, we only need one coarse resolution level to save
    memory and disk space.

    Args:
        roi_volume: Labeled volume (z, y, x order)
        output_dir: Output directory
        voxel_size_um: Original voxel size in micrometers
        chunk_size: Chunk size for Neuroglancer
        target_resolution_um: Target resolution in micrometers (default: 0.8)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Downsample to target resolution
    downsample_factor = max(1, int(target_resolution_um / voxel_size_um))
    if downsample_factor > 1:
        downsampled = roi_volume[::downsample_factor, ::downsample_factor, ::downsample_factor]
        actual_resolution_um = voxel_size_um * downsample_factor
    else:
        downsampled = roi_volume
        actual_resolution_um = voxel_size_um

    logger.info(f"  ROI volume downsampled {downsample_factor}x: {roi_volume.shape} -> {downsampled.shape}")

    # Create info with single scale
    voxel_size_nm = actual_resolution_um * 1000
    size_xyz = [int(downsampled.shape[2]), int(downsampled.shape[1]), int(downsampled.shape[0])]
    adjusted_chunk = [min(chunk_size, size_xyz[i]) for i in range(3)]

    info = {
        "@type": "neuroglancer_multiscale_volume",
        "type": "segmentation",
        "data_type": "uint8",
        "num_channels": 1,
        "scales": [{
            "key": f"{int(voxel_size_nm)}_{int(voxel_size_nm)}_{int(voxel_size_nm)}",
            "size": size_xyz,
            "resolution": [voxel_size_nm, voxel_size_nm, voxel_size_nm],
            "chunk_sizes": [adjusted_chunk],
            "encoding": "raw",
        }],
    }

    # Write info file
    with open(output_dir / "info", 'w') as f:
        json.dump(info, f, indent=2)

    # Write chunks for single scale
    scale = info["scales"][0]
    scale_dir = output_dir / scale["key"]
    scale_dir.mkdir(parents=True, exist_ok=True)

    vol_size = scale["size"]
    chunk_sizes = scale["chunk_sizes"][0]
    n_chunks = [(vol_size[i] + chunk_sizes[i] - 1) // chunk_sizes[i] for i in range(3)]

    for cz in range(n_chunks[2]):
        z_start = cz * chunk_sizes[2]
        z_end = min(z_start + chunk_sizes[2], vol_size[2])

        for cy in range(n_chunks[1]):
            y_start = cy * chunk_sizes[1]
            y_end = min(y_start + chunk_sizes[1], vol_size[1])

            for cx in range(n_chunks[0]):
                x_start = cx * chunk_sizes[0]
                x_end = min(x_start + chunk_sizes[0], vol_size[0])

                chunk_data = downsampled[z_start:z_end, y_start:y_end, x_start:x_end]
                chunk_data = chunk_data.transpose(2, 1, 0).astype(np.uint8)

                chunk_filename = f"{x_start}-{x_end}_{y_start}-{y_end}_{z_start}-{z_end}"
                with open(scale_dir / chunk_filename, 'wb') as f:
                    f.write(chunk_data.tobytes(order='F'))

    logger.info(f"  Wrote Neuroglancer volume to {output_dir}")


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
    chunk_size: int = 64,
    organize_by_microscopy: bool = False,
) -> Dict:
    """
    Identify CC and CG population ROIs and export for visualization.

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
        chunk_size: Chunk size for Neuroglancer output
        organize_by_microscopy: Organize output by HM/LM subdirectories

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
    separation_axis, separation_position, separation_stats = find_separation_plane(
        populations, axon_data, centroids
    )

    # Step 4: Compute ROI bounding boxes
    logger.info("\nStep 4: Computing ROI bounding boxes...")
    rois = compute_rois(
        separation_axis, separation_position, margin_um,
        volume_shape, voxel_size_um, centroids
    )

    axis_names = {0: 'Z', 1: 'Y', 2: 'X'}
    for name, roi in rois.items():
        logger.info(f"  {name.upper()} ROI: {roi['min_um']} to {roi['max_um']}")

    # Step 5: Create and export ROI volume
    logger.info("\nStep 5: Creating ROI volume for Neuroglancer...")
    roi_volume = create_roi_volume(
        volume_shape, separation_axis, separation_position, margin_um,
        voxel_size_um, centroids
    )

    # Construct output path
    if organize_by_microscopy:
        base_output_path = construct_output_path(
            mat_file, output_dir,
            output_suffix="_population_rois",
            organize_by_microscopy=True
        )
        sample_output_dir = base_output_path.parent
        sample_name = base_output_path.stem.replace('_population_rois', '')
        # Neuroglancer dir matches the JSON name (without extension)
        neuroglancer_dir = base_output_path.with_suffix('')
    else:
        sample_name = mat_file.stem.replace('_myelinated_axons', '')
        sample_output_dir = output_dir / sample_name
        neuroglancer_dir = sample_output_dir / "population_rois"

    # Write Neuroglancer volume
    write_neuroglancer_volume(
        roi_volume, neuroglancer_dir, voxel_size_um, chunk_size
    )

    # Step 6: Save JSON metadata
    logger.info("\nStep 6: Saving ROI metadata...")

    metadata = {
        "source_file": str(mat_file),
        "voxel_size_um": voxel_size_um,
        "volume_shape_zyx": list(volume_shape),
        "parameters": {
            "downsample": downsample,
            "max_angle_deg": max_angle_deg,
            "min_length_um": min_length_um,
            "k_neighbors": k_neighbors,
            "max_neighbor_distance_um": max_neighbor_distance_um,
        },
        "separation": {
            "axis": separation_axis,
            "axis_name": axis_names[separation_axis],
            "position_um": float(separation_position),
            "margin_um": margin_um,
            "cc_accuracy_pct": separation_stats['cc_accuracy'],
            "cg_accuracy_pct": separation_stats['cg_accuracy'],
        },
        "populations": {
            pop['name']: {
                "n_axons": pop['n_axons'],
                "centroid_um": centroids[pop['name']].tolist(),
                "mean_orientation": pop['mean_orientation'],
                "dominant_axis": pop['dominant_axis'],
                "dominant_axis_name": pop['dominant_axis_name'],
            }
            for pop in populations[:2]
        },
        "rois": rois,
        "neuroglancer_path": str(neuroglancer_dir),
    }

    json_path = sample_output_dir / f"{sample_name}_population_rois.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Saved: {json_path}")

    logger.info(f"\n{'='*80}")
    logger.info("ROI identification complete!")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  Neuroglancer: {neuroglancer_dir}")
    logger.info(f"{'='*80}\n")

    return metadata


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Identify CC and CG population ROIs using spatial separation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file
    python scripts/processing/identify_population_rois.py \\
        data/raw/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat \\
        data/processed/LM

    # Batch processing
    python scripts/processing/identify_population_rois.py \\
        "data/raw/**/LM*_myelinated_axons.mat" \\
        data/processed/LM

Outputs:
    - <sample>_population_rois.json: ROI bounding boxes and metadata
    - <sample>/population_rois/: Neuroglancer precomputed volume (1=CC, 2=CG)
        """
    )
    parser.add_argument(
        'input_files', type=str,
        help='Input .mat file(s) with labeled volumes (glob pattern supported)'
    )
    parser.add_argument(
        'output_dir', type=Path,
        help='Output directory for results'
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
    parser.add_argument(
        '--chunk-size', type=int, default=64,
        help='Chunk size for Neuroglancer output (default: 64)'
    )
    parser.add_argument(
        '--organize-by-microscopy', action='store_true',
        help='Organize output by HM/LM subdirectories'
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_pattern = args.input_files
    if '*' in input_pattern:
        input_files = sorted(glob(input_pattern, recursive=True))
    else:
        input_files = [input_pattern]

    if not input_files:
        logger.error(f"No files found matching: {args.input_files}")
        return

    logger.info(f"Found {len(input_files)} input file(s)")

    # Process each file
    for mat_file in input_files:
        mat_path = Path(mat_file)
        if not mat_path.exists():
            logger.warning(f"File not found: {mat_file}")
            continue

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
                chunk_size=args.chunk_size,
                organize_by_microscopy=args.organize_by_microscopy,
            )
        except Exception as e:
            logger.error(f"Failed to process {mat_file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
