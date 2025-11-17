#!/usr/bin/env python3
"""
Extract 3D skeletons from labeled axon volumes using true 3D skeletonization.

This script:
1. Loads the entire labeled volume into memory
2. Identifies all unique axon labels
3. For each axon, extracts a binary mask and computes 3D skeleton
4. Saves skeleton coordinates for each axon

Author: Generated for axon morphology analysis
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SkeletonStats:
    """Statistics for skeleton extraction"""
    axon_id: int
    n_voxels_original: int
    n_skeleton_points: int
    extraction_time_sec: float


class AxonSkeletonExtractor:
    """Extract 3D skeletons from labeled volumes"""

    def __init__(self, voxel_size_um: float = 0.05, downsample_factor: int = 1):
        self.voxel_size_um = voxel_size_um
        self.downsample_factor = downsample_factor

    def load_volume(self, input_file: Path) -> np.ndarray:
        """
        Load the entire labeled volume into memory with optional downsampling.

        Args:
            input_file: Path to .mat file with 'final_lbl' dataset

        Returns:
            3D numpy array of labels
        """
        logger.info(f"Loading volume from {input_file}")
        start_time = time.time()

        with h5py.File(input_file, 'r') as f:
            labels = f['final_lbl'][:]  # Load entire volume

        load_time = time.time() - start_time
        logger.info(f"Loaded volume shape {labels.shape} in {load_time:.2f}s")
        logger.info(f"Memory usage: ~{labels.nbytes / 1e9:.2f} GB")

        # Downsample if requested
        if self.downsample_factor > 1:
            logger.info(f"Downsampling by factor {self.downsample_factor}...")
            labels = self.downsample_labels(labels, self.downsample_factor)
            logger.info(f"Downsampled volume shape: {labels.shape}")
            logger.info(f"Memory usage after downsampling: ~{labels.nbytes / 1e9:.2f} GB")

        return labels

    def downsample_labels(self, labels: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsample labeled volume using nearest neighbor sampling.

        Args:
            labels: Original labeled volume
            factor: Downsampling factor (e.g., 4 means sample every 4th voxel)

        Returns:
            Downsampled labeled volume
        """
        # Simple nearest neighbor: just sample every Nth voxel
        downsampled = labels[::factor, ::factor, ::factor]

        logger.info(f"Downsampled from {labels.shape} to {downsampled.shape} using nearest neighbor")

        return downsampled

    def identify_axons(self, labels: np.ndarray) -> np.ndarray:
        """
        Find all unique axon IDs in the volume.

        Args:
            labels: 3D labeled volume

        Returns:
            Array of unique axon IDs (excluding background=0)
        """
        logger.info("Identifying unique axons...")
        start_time = time.time()

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background

        identify_time = time.time() - start_time
        logger.info(f"Found {len(unique_labels)} axons in {identify_time:.2f}s")

        return unique_labels

    def extract_skeleton(self, labels: np.ndarray, axon_id: int) -> np.ndarray:
        """
        Extract 3D skeleton for a single axon using distance transform.

        This is much faster than morphological skeletonization.
        The approach:
        1. Extract binary mask for the axon
        2. Compute distance transform (distance to nearest boundary)
        3. Find local maxima of distance transform = medial axis points

        Args:
            labels: Full 3D labeled volume
            axon_id: ID of the axon to skeletonize

        Returns:
            Array of skeleton coordinates [N, 3] where each row is [x, y, z]
        """
        # Create binary mask for this axon
        binary_mask = (labels == axon_id)
        n_voxels = binary_mask.sum()

        if n_voxels == 0:
            return np.array([])

        # Get bounding box to work with smaller array
        coords = np.argwhere(binary_mask)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        # Add small padding
        mins = np.maximum(mins - 2, 0)
        maxs = np.minimum(maxs + 3, binary_mask.shape)

        # Extract cropped region
        cropped_mask = binary_mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

        # Distance transform: each voxel = distance to nearest boundary
        dist = distance_transform_edt(cropped_mask)

        # Find skeleton points: local maxima along each axis
        # A point is on the skeleton if it's a local maximum in the distance map

        # Use small neighborhood for local maxima detection
        local_max = maximum_filter(dist, size=3)
        skeleton_mask = (dist == local_max) & (dist > 0)

        # Extract skeleton coordinates (in cropped space)
        skel_coords_cropped = np.argwhere(skeleton_mask)

        if len(skel_coords_cropped) == 0:
            # Fallback: just take the centerline
            # Find points with maximum distance values
            threshold = dist.max() * 0.5
            skeleton_mask = dist >= threshold
            skel_coords_cropped = np.argwhere(skeleton_mask)

        # Convert back to full volume coordinates
        skel_coords = skel_coords_cropped + mins

        # Reorder to [x, y, z]
        skel_coords = skel_coords[:, [2, 1, 0]]

        return skel_coords

    def extract_all_skeletons(self, labels: np.ndarray,
                              axon_ids: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract skeletons for all axons.

        Args:
            labels: Full 3D labeled volume
            axon_ids: Array of axon IDs to process

        Returns:
            Dict mapping axon_id -> skeleton coordinates [N, 3]
        """
        logger.info(f"Extracting skeletons for {len(axon_ids)} axons...")

        skeletons = {}
        stats = []

        total_start = time.time()

        # Use tqdm for progress bar
        for axon_id in tqdm(axon_ids, desc="Extracting skeletons", unit="axon",
                           bar_format='{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]'):
            start_time = time.time()

            # Get binary mask
            binary_mask = (labels == axon_id)
            n_voxels = binary_mask.sum()

            if n_voxels == 0:
                continue

            # Extract skeleton
            skeleton_coords = self.extract_skeleton(labels, axon_id)

            extraction_time = time.time() - start_time

            if len(skeleton_coords) > 0:
                skeletons[axon_id] = skeleton_coords

                stats.append(SkeletonStats(
                    axon_id=int(axon_id),
                    n_voxels_original=int(n_voxels),
                    n_skeleton_points=len(skeleton_coords),
                    extraction_time_sec=extraction_time
                ))

        total_time = time.time() - total_start
        logger.info(f"Extracted {len(skeletons)} skeletons in {total_time:.2f}s")
        logger.info(f"Average: {total_time/len(axon_ids):.3f}s per axon")

        return skeletons, stats

    def save_skeletons(self, output_file: Path,
                       skeletons: Dict[int, np.ndarray],
                       stats: List[SkeletonStats],
                       metadata: dict):
        """
        Save skeletons to HDF5 file.

        Args:
            output_file: Path to output HDF5 file
            skeletons: Dict mapping axon_id -> skeleton coordinates
            stats: List of extraction statistics
            metadata: Additional metadata to save
        """
        logger.info(f"Saving skeletons to {output_file}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_file, 'w') as f:
            # Create group for skeletons
            skeletons_group = f.create_group('skeletons')

            # Save each skeleton as a separate dataset
            for axon_id, coords in skeletons.items():
                skeletons_group.create_dataset(
                    str(axon_id),
                    data=coords,
                    compression='gzip',
                    compression_opts=4
                )

            # Save axon IDs as a list
            f.create_dataset('axon_ids', data=np.array(list(skeletons.keys()), dtype=np.uint32))

            # Save statistics
            stats_group = f.create_group('statistics')
            stats_dict = {
                'axon_ids': [s.axon_id for s in stats],
                'n_voxels_original': [s.n_voxels_original for s in stats],
                'n_skeleton_points': [s.n_skeleton_points for s in stats],
                'extraction_time_sec': [s.extraction_time_sec for s in stats]
            }
            for key, val in stats_dict.items():
                stats_group.create_dataset(key, data=np.array(val))

            # Save metadata as attributes
            for key, val in metadata.items():
                if isinstance(val, (list, tuple, dict)):
                    val = json.dumps(val)
                f.attrs[key] = val

        logger.info(f"Saved {len(skeletons)} skeletons to {output_file}")

        # Print summary statistics
        avg_skeleton_length = np.mean([len(coords) for coords in skeletons.values()])
        avg_compression = np.mean([s.n_voxels_original / s.n_skeleton_points
                                   for s in stats if s.n_skeleton_points > 0])

        logger.info(f"Summary:")
        logger.info(f"  Average skeleton points: {avg_skeleton_length:.1f}")
        logger.info(f"  Average compression ratio: {avg_compression:.1f}x")


def process_volume(input_file: Path, output_dir: Path,
                   voxel_size_um: float = 0.05,
                   downsample_factor: int = 1):
    """
    Extract skeletons from a single volume.

    Args:
        input_file: Path to input .mat file
        output_dir: Directory for output files
        voxel_size_um: Voxel size in micrometers
        downsample_factor: Factor to downsample volume (1 = no downsampling)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {input_file.name}")
    logger.info(f"{'='*80}\n")

    extractor = AxonSkeletonExtractor(
        voxel_size_um=voxel_size_um,
        downsample_factor=downsample_factor
    )

    # Load volume
    labels = extractor.load_volume(input_file)

    # Identify axons
    axon_ids = extractor.identify_axons(labels)

    # Extract skeletons
    skeletons, stats = extractor.extract_all_skeletons(labels, axon_ids)

    # Prepare metadata
    effective_voxel_size = voxel_size_um * downsample_factor
    metadata = {
        'source_file': input_file.name,
        'volume_shape': labels.shape,
        'voxel_size_um': voxel_size_um,
        'downsample_factor': downsample_factor,
        'effective_voxel_size_um': effective_voxel_size,
        'n_axons_total': len(axon_ids),
        'n_axons_with_skeletons': len(skeletons)
    }

    # Save results
    volume_name = input_file.stem.replace('_myelinated_axons', '')
    output_file = output_dir / volume_name / 'skeletons.h5'

    extractor.save_skeletons(output_file, skeletons, stats, metadata)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed: {input_file.name}")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    # Example usage
    input_file = Path('/workspace/data/raw/LM/LM_25_ipsi_myelinated_axons.mat')
    output_dir = Path('/workspace/data/processed')

    # Downsample by factor 4 for faster processing
    # Original voxel: 0.05 um, downsampled: 0.2 um
    process_volume(input_file, output_dir, voxel_size_um=0.05, downsample_factor=4)
