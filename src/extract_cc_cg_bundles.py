#!/usr/bin/env python3
"""
Extract CC and CG fiber bundles to a single OME-Zarr file.

Uses dominant-axis classification: axons are grouped by which coordinate axis
(Z, Y, X) they align with most strongly. CC is assigned to the axis with most
axons, CG to the second-most. Axons aligned with the third axis or exceeding
the angular threshold are excluded.

Output structure:
    output.zarr/
    ├── cc/0, cc/1, ...  (pyramid levels for corpus callosum)
    ├── cg/0, cg/1, ...  (pyramid levels for cingulum)
    └── .zattrs          (combined metadata)
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import h5py
import numcodecs
import numpy as np
import zarr
from scipy.spatial import KDTree
from tqdm import tqdm

# Import population identification functions from axonometry
sys.path.insert(0, str(Path(__file__).parent.parent))
from axonometry.populations import (
    load_volume_downsampled,
    precompute_axon_voxels,
    compute_all_orientations,
    classify_by_dominant_axis,
    filter_sparse_axons,
    create_populations,
)

# Alias for compatibility (src uses 'bundles' terminology)
create_bundles = create_populations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Volume Extraction Functions (from extract_bundle_volumes_simple.py)
# =============================================================================

def create_label_lut(axon_labels: Set[int], max_label: int) -> np.ndarray:
    """Create lookup table for fast label filtering."""
    lut = np.zeros(max_label + 1, dtype=bool)
    for label in axon_labels:
        if label <= max_label:
            lut[label] = True
    return lut


def filter_with_lut(data: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Filter array using lookup table."""
    clamped = np.minimum(data, len(lut) - 1)
    mask = lut[clamped]
    return np.where(mask, data, 0).astype(data.dtype)


def downsample_segmentation(data: np.ndarray, factor: int) -> np.ndarray:
    """Fast downsampling using center voxel."""
    if factor == 1:
        return data.copy()
    offset = factor // 2
    return data[offset::factor, offset::factor, offset::factor].copy()


def create_group_ome_metadata(group_name: str, voxel_size_um: float, n_levels: int) -> Dict:
    """Create OME-NGFF metadata for a single group."""
    datasets = []
    for level in range(n_levels):
        scale = voxel_size_um * (2 ** level)
        datasets.append({
            "path": str(level),
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale, scale, scale]}
            ]
        })

    return {
        "multiscales": [{
            "version": "0.4",
            "name": group_name,
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": datasets
        }]
    }


def write_bundle_to_group(volume: np.ndarray,
                          bundle: Dict,
                          group: zarr.Group,
                          lut: np.ndarray,
                          voxel_size_um: float,
                          n_levels: int = 5) -> np.ndarray:
    """Write bundle to Zarr group with pyramid levels and OME-NGFF metadata.

    Keeps original volume orientation [Z, Y, X] for spatial alignment of all bundles.
    """
    # Use original volume shape - no permutation for spatial alignment
    output_shape = volume.shape  # [Z, Y, X]
    n_slices = output_shape[0]

    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Write level 0 with (1, Y, X) chunking for efficient slice access
    chunk_shape_0 = (1, output_shape[1], output_shape[2])
    level_0 = group.create_array(
        '0', shape=output_shape, chunks=chunk_shape_0,
        dtype=np.uint16, compressor=compressor
    )

    segment_id_set = set()
    for z_out in tqdm(range(n_slices), desc=f"Writing {group.name}", unit="slice", leave=False):
        # Direct slice extraction - no permutation
        input_slice = volume[z_out, :, :]
        filtered_slice = filter_with_lut(input_slice, lut)
        level_0[z_out, :, :] = filtered_slice
        segment_id_set.update(np.unique(filtered_slice).tolist())

    segment_id_set.discard(0)
    segment_ids = np.array(sorted(segment_id_set), dtype=np.uint16)

    # Generate pyramid
    current_data = level_0[:]
    for level in range(1, n_levels):
        downsampled = downsample_segmentation(current_data, 2)
        chunk_size = min(64, min(downsampled.shape))
        level_ds = group.create_array(
            str(level), shape=downsampled.shape,
            chunks=(chunk_size, chunk_size, chunk_size),
            dtype=np.uint16, compressor=compressor
        )
        level_ds[:] = downsampled
        current_data = downsampled

    # Store segment IDs
    labels_group = group.create_group('labels')
    labels_group.create_array('segment_ids', data=segment_ids)

    # Write OME-NGFF metadata to group
    group_name = group.name.strip('/')
    group.attrs.update(create_group_ome_metadata(group_name, voxel_size_um, n_levels))

    return segment_ids


def create_ome_metadata(bundles: List[Dict], voxel_size_um: float, n_levels: int) -> Dict:
    """Create OME-NGFF metadata for combined Zarr."""
    def make_datasets(n_levels, voxel_size):
        datasets = []
        for level in range(n_levels):
            scale = voxel_size * (2 ** level)
            datasets.append({
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [scale, scale, scale]}
                ]
            })
        return datasets

    axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}
    ]

    # CC is largest (index 0), CG is second (index 1)
    cc_bundle = bundles[0] if len(bundles) > 0 else None
    cg_bundle = bundles[1] if len(bundles) > 1 else None

    metadata = {
        "multiscales": [
            {
                "version": "0.4",
                "name": "cc",
                "axes": axes,
                "datasets": make_datasets(n_levels, voxel_size_um)
            },
            {
                "version": "0.4",
                "name": "cg",
                "axes": axes,
                "datasets": make_datasets(n_levels, voxel_size_um)
            }
        ],
        "cc": {
            "n_axons": cc_bundle['n_axons'] if cc_bundle else 0,
            "mean_orientation": cc_bundle['mean_orientation'] if cc_bundle else [0, 0, 0],
            "mean_length_um": cc_bundle['mean_length_um'] if cc_bundle else 0
        },
        "cg": {
            "n_axons": cg_bundle['n_axons'] if cg_bundle else 0,
            "mean_orientation": cg_bundle['mean_orientation'] if cg_bundle else [0, 0, 0],
            "mean_length_um": cg_bundle['mean_length_um'] if cg_bundle else 0
        },
        "voxel_size_um": voxel_size_um,
        "n_levels": n_levels
    }

    return metadata


# =============================================================================
# Main Pipeline
# =============================================================================

def extract_cc_cg_bundles(mat_file: Path,
                          output_zarr: Path,
                          metadata_file: Path,
                          downsample: int = 4,
                          voxel_size_um: float = 0.05,
                          max_angle_deg: float = 30.0,
                          min_length_um: float = 50.0,
                          min_voxels: int = 10,
                          k_neighbors: int = 10,
                          max_neighbor_distance_um: float = 30.0,
                          n_levels: int = 5):
    """
    Complete pipeline to identify and extract CC/CG bundles to single Zarr.

    Uses dominant-axis classification: axons are grouped by which coordinate axis
    they align with most strongly. CC is assigned to the axis with most axons,
    CG to the second-most.

    Args:
        mat_file: Input .mat file
        output_zarr: Output Zarr store path
        metadata_file: Output JSON metadata path
        downsample: Downsampling factor for identification
        voxel_size_um: Physical voxel size
        max_angle_deg: Maximum deviation from dominant axis (default 45°)
        min_length_um: Minimum axon length
        min_voxels: Minimum voxels per axon
        k_neighbors: K for KNN filtering
        max_neighbor_distance_um: Max distance for KNN
        n_levels: Pyramid levels
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Extracting CC and CG bundles: {mat_file.name}")
    logger.info(f"{'='*80}\n")

    # Phase 1: Identify bundles (on downsampled volume)
    logger.info("Phase 1: Identifying bundles...")
    volume_ds, vol_metadata = load_volume_downsampled(mat_file, downsample)
    voxel_size_ds = voxel_size_um * downsample

    axon_voxels_ds = precompute_axon_voxels(volume_ds)
    axon_data = compute_all_orientations(axon_voxels_ds, voxel_size_ds, min_voxels, min_length_um)

    if not axon_data:
        raise ValueError("No valid axons found after filtering")

    label_to_bundle, axis_to_bundle = classify_by_dominant_axis(axon_data, max_angle_deg)

    if not label_to_bundle:
        raise ValueError("No axons remain after classification")

    bundles = create_bundles(
        axon_data, label_to_bundle, axon_voxels_ds, voxel_size_ds,
        k_neighbors, max_neighbor_distance_um
    )

    if len(bundles) < 2:
        logger.warning(f"Only {len(bundles)} bundle(s) found, expected 2")

    # Free downsampled data
    del volume_ds, axon_voxels_ds

    # Phase 2: Load full volume for extraction
    logger.info("\nPhase 2: Extracting bundles...")
    logger.info("Loading full volume...")
    with h5py.File(mat_file, 'r') as f:
        volume_full = f['final_lbl'][:]
    max_label = int(volume_full.max())
    logger.info(f"Full volume: {volume_full.shape}, max label: {max_label}")

    # Create output Zarr
    root = zarr.open_group(str(output_zarr), mode='w', zarr_format=2)

    # Extract CC (largest bundle) - keep original orientation for spatial alignment
    if len(bundles) >= 1:
        cc_bundle = bundles[0]
        logger.info(f"\nExtracting CC (largest): {cc_bundle['n_axons']} axons")
        cc_group = root.create_group('cc')
        lut = create_label_lut(set(cc_bundle['axon_labels']), max_label)
        write_bundle_to_group(volume_full, cc_bundle, cc_group, lut, voxel_size_um, n_levels)

    # Extract CG (second largest) - keep original orientation for spatial alignment
    if len(bundles) >= 2:
        cg_bundle = bundles[1]
        logger.info(f"\nExtracting CG (second): {cg_bundle['n_axons']} axons")
        cg_group = root.create_group('cg')
        lut = create_label_lut(set(cg_bundle['axon_labels']), max_label)
        write_bundle_to_group(volume_full, cg_bundle, cg_group, lut, voxel_size_um, n_levels)

    # Write metadata
    root.attrs.update(create_ome_metadata(bundles, voxel_size_um, n_levels))

    # Save JSON metadata
    output_metadata = {
        'volume_metadata': vol_metadata,
        'n_bundles': len(bundles),
        'bundles': [
            {**bundles[0], 'name': 'cc'} if len(bundles) > 0 else {},
            {**bundles[1], 'name': 'cg'} if len(bundles) > 1 else {}
        ]
    }
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(output_metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_file}")

    # Summary
    total_size = sum(f.stat().st_size for f in output_zarr.rglob('*') if f.is_file())
    logger.info(f"Saved Zarr: {output_zarr} ({total_size / 1024**2:.1f} MB)")

    logger.info(f"\n{'='*80}")
    logger.info("Extraction complete!")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract CC and CG bundles to single OME-Zarr file'
    )
    parser.add_argument('mat_file', type=Path,
                       help='Input .mat file with labeled volume')
    parser.add_argument('output_zarr', type=Path,
                       help='Output Zarr store path')
    parser.add_argument('--metadata', type=Path, required=True,
                       help='Output JSON metadata file')
    parser.add_argument('--downsample', type=int, default=4,
                       help='Downsampling factor for identification (default: 4)')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel size in μm (default: 0.05)')
    parser.add_argument('--max-angle', type=float, default=30.0,
                       help='Max deviation from dominant axis in degrees (default: 30.0)')
    parser.add_argument('--min-length', type=float, default=50.0,
                       help='Minimum axon length in μm (default: 50.0)')
    parser.add_argument('--min-voxels', type=int, default=10,
                       help='Minimum voxels per axon (default: 10)')
    parser.add_argument('--k-neighbors', type=int, default=10,
                       help='K for sparse axon filtering (default: 10)')
    parser.add_argument('--max-neighbor-distance', type=float, default=30.0,
                       help='Max distance to k-th neighbor in μm (default: 30.0)')
    parser.add_argument('--n-levels', type=int, default=5,
                       help='Pyramid levels (default: 5)')

    args = parser.parse_args()

    extract_cc_cg_bundles(
        args.mat_file,
        args.output_zarr,
        args.metadata,
        downsample=args.downsample,
        voxel_size_um=args.voxel_size,
        max_angle_deg=args.max_angle,
        min_length_um=args.min_length,
        min_voxels=args.min_voxels,
        k_neighbors=args.k_neighbors,
        max_neighbor_distance_um=args.max_neighbor_distance,
        n_levels=args.n_levels
    )
