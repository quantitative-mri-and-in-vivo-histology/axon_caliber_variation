#!/usr/bin/env python3
"""
Run tubular decomposition (skeletonization + shape decomposition) on labeled axon data.

Uses DeepACSON's CSD module to:
1. Extract skeleton for each axon
2. Perform cylindrical shape decomposition

Since the data already contains labeled axons, we skip the segmentation step
and directly process each axon for morphometric analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
import multiprocessing as mp

import numpy as np
import scipy.io as sio

# Global variable for volume access in worker processes (fork method shares memory)
_shared_volume = None

# Add external DeepACSON to path
DEEPACSON_PATH = Path(__file__).parent.parent / 'external' / 'DeepACSON' / 'CSD'
sys.path.insert(0, str(DEEPACSON_PATH))

from skeleton3D import skeleton
from shape_decomposition import object_analysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_worker(volume):
    """Initialize worker process with shared volume reference."""
    global _shared_volume
    _shared_volume = volume


def extract_axon_binary(volume: np.ndarray, axon_label: int) -> np.ndarray:
    """
    Extract binary volume for a single axon.

    Args:
        volume: Full labeled volume (uint16)
        axon_label: Axon label to extract

    Returns:
        Binary array (uint8) where 1 = axon voxels
    """
    return (volume == axon_label).astype(np.uint8)


def process_single_axon(args):
    """
    Process a single axon through skeletonization and shape decomposition.

    Args:
        args: Tuple of (axon_label, voxel_size_um)

    Returns:
        Dict with morphometric results or None if processing failed
    """
    axon_label, voxel_size_um = args

    try:
        # Extract binary volume from shared memory
        axon_binary = extract_axon_binary(_shared_volume, axon_label)

        # Get axon coordinates and bounding box
        coords = np.argwhere(axon_binary)
        if len(coords) < 100:  # Skip very small axons
            return None

        # Crop to bounding box with padding
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        padding = 5
        min_padded = np.maximum(min_coords - padding, 0)
        max_padded = np.minimum(max_coords + padding, np.array(axon_binary.shape))

        cropped = axon_binary[
            min_padded[0]:max_padded[0],
            min_padded[1]:max_padded[1],
            min_padded[2]:max_padded[2]
        ].copy()

        # Run skeletonization
        skel_segments = skeleton(cropped)

        if len(skel_segments) == 0:
            logger.debug(f"Axon {axon_label}: No skeleton segments found")
            return None

        # Compute total skeleton length
        total_length = 0
        for seg in skel_segments:
            if len(seg) > 1:
                seg_length = np.sum(np.sqrt(np.sum(np.diff(seg, axis=0)**2, axis=1)))
                total_length += seg_length

        total_length_um = total_length * voxel_size_um

        # Run shape decomposition
        decomposed_objs, decomposed_skels = object_analysis(cropped, skel_segments)

        n_segments = len(decomposed_objs)

        # Compute per-segment radii from decomposed components
        # Assuming cylindrical shape: V = π * r² * L, so r = sqrt(V / (π * L))
        segment_radii_um = []
        segment_lengths_um = []
        segment_volumes_um3 = []

        for dec_obj, dec_skel in zip(decomposed_objs, decomposed_skels):
            # Segment volume
            vol_voxels = np.sum(dec_obj)
            vol_um3 = vol_voxels * (voxel_size_um ** 3)

            # Segment skeleton length
            if len(dec_skel) > 1:
                seg_len = np.sum(np.sqrt(np.sum(np.diff(dec_skel, axis=0)**2, axis=1)))
                seg_len_um = seg_len * voxel_size_um
            else:
                seg_len_um = voxel_size_um  # Minimum 1 voxel length

            # Compute equivalent cylindrical radius: r = sqrt(V / (π * L))
            if seg_len_um > 0:
                radius_um = np.sqrt(vol_um3 / (np.pi * seg_len_um))
            else:
                radius_um = 0.0

            segment_radii_um.append(radius_um)
            segment_lengths_um.append(seg_len_um)
            segment_volumes_um3.append(vol_um3)

        # Total volume
        total_volume_voxels = np.sum(axon_binary)
        total_volume_um3 = total_volume_voxels * (voxel_size_um ** 3)

        result = {
            'label': axon_label,
            'volume_um3': total_volume_um3,
            'skeleton_length_um': total_length_um,
            'n_skeleton_segments': len(skel_segments),
            'n_decomposed_segments': n_segments,
            'segment_radii_um': segment_radii_um,
            'segment_lengths_um': segment_lengths_um,
            'segment_volumes_um3': segment_volumes_um3,
            'bounding_box': {
                'min': min_coords.tolist(),
                'max': max_coords.tolist()
            }
        }

        logger.info(f"Axon {axon_label}: length={total_length_um:.1f} μm, "
                   f"segments={n_segments}, volume={total_volume_um3:.1f} μm³")

        return result

    except Exception as e:
        import traceback
        logger.error(f"Axon {axon_label}: Processing failed - {e}\n{traceback.format_exc()}")
        return None


def run_tubular_decomposition(mat_file: Path,
                              output_file: Path,
                              voxel_size_um: float = 0.05,
                              n_jobs: int = -1,
                              max_axons: int = 0):
    """
    Run tubular decomposition on all axons in a labeled volume.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
    """
    # Load labeled volume
    logger.info(f"Loading {mat_file}")
    mat = sio.loadmat(str(mat_file))

    # Find the labeled volume key
    volume_key = None
    for key in mat.keys():
        if not key.startswith('_'):
            volume_key = key
            break

    if volume_key is None:
        raise ValueError(f"No data found in {mat_file}")

    volume = mat[volume_key]
    logger.info(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

    # Get unique axon labels
    axon_labels = np.unique(volume)
    axon_labels = axon_labels[axon_labels > 0]  # Remove background

    logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    # Prepare arguments - only pass labels, not binary volumes
    args_list = [(label, voxel_size_um) for label in axon_labels]

    # Process axons
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    logger.info(f"Processing {len(args_list)} axons with {n_jobs} workers")

    results = []
    if n_jobs == 1:
        # Sequential processing - initialize globals directly
        global _shared_volume
        _shared_volume = volume
        for args in args_list:
            result = process_single_axon(args)
            if result is not None:
                results.append(result)
    else:
        # Parallel processing - use fork to share memory (default on Linux)
        # The volume is shared via copy-on-write with fork
        with mp.Pool(n_jobs, initializer=init_worker, initargs=(volume,)) as pool:
            for result in pool.imap_unordered(process_single_axon, args_list):
                if result is not None:
                    results.append(result)

    logger.info(f"Successfully processed {len(results)}/{len(args_list)} axons")

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to arrays for saving
    labels = np.array([r['label'] for r in results])
    volumes = np.array([r['volume_um3'] for r in results])
    lengths = np.array([r['skeleton_length_um'] for r in results])
    n_skel_segs = np.array([r['n_skeleton_segments'] for r in results])
    n_dec_segs = np.array([r['n_decomposed_segments'] for r in results])

    # Segment data (variable length per axon, stored as object arrays)
    seg_radii = np.array([r['segment_radii_um'] for r in results], dtype=object)
    seg_lengths = np.array([r['segment_lengths_um'] for r in results], dtype=object)
    seg_volumes = np.array([r['segment_volumes_um3'] for r in results], dtype=object)

    # Also create flattened arrays for easier analysis
    all_seg_radii = np.concatenate([r['segment_radii_um'] for r in results if len(r['segment_radii_um']) > 0]) if any(len(r['segment_radii_um']) > 0 for r in results) else np.array([])
    all_seg_lengths = np.concatenate([r['segment_lengths_um'] for r in results if len(r['segment_lengths_um']) > 0]) if any(len(r['segment_lengths_um']) > 0 for r in results) else np.array([])

    np.savez(
        output_file,
        labels=labels,
        volumes_um3=volumes,
        skeleton_lengths_um=lengths,
        n_skeleton_segments=n_skel_segs,
        n_decomposed_segments=n_dec_segs,
        segment_radii_um=seg_radii,
        segment_lengths_um=seg_lengths,
        segment_volumes_um3=seg_volumes,
        all_segment_radii_um=all_seg_radii,
        all_segment_lengths_um=all_seg_lengths,
        voxel_size_um=voxel_size_um,
        source_file=str(mat_file)
    )

    logger.info(f"Saved results to {output_file}")

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Mean skeleton length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean volume: {np.mean(volumes):.2f} ± {np.std(volumes):.2f} μm³")
    logger.info(f"  Mean decomposed segments: {np.mean(n_dec_segs):.1f}")
    if len(all_seg_radii) > 0:
        logger.info(f"  Total segments: {len(all_seg_radii)}")
        logger.info(f"  Mean segment radius: {np.mean(all_seg_radii):.3f} ± {np.std(all_seg_radii):.3f} μm")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run tubular decomposition (skeletonization + shape decomposition) on labeled axon data'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Path to .mat file with labeled axons')
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file for results')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                        help='Voxel size in micrometers (default: 0.05)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all, default: 0)')

    args = parser.parse_args()

    run_tubular_decomposition(
        args.mat_file,
        args.output_file,
        voxel_size_um=args.voxel_size,
        n_jobs=args.n_jobs,
        max_axons=args.max_axons
    )
