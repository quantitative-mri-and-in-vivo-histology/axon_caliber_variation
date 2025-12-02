"""
Morphometry utilities for axon and myelin analysis.

Includes:
- Myelin-to-axon instance assignment
- G-ratio computation
- Outer fiber diameter calculation
"""

import logging
from typing import Tuple, Optional

import numpy as np
from scipy import ndimage
from skimage.measure import label

logger = logging.getLogger(__name__)


def assign_myelin_to_axons(
    axon_volume: np.ndarray,
    myelin_volume: np.ndarray,
    max_distance: Optional[float] = None,
    connectivity: int = 1
) -> np.ndarray:
    """
    Assign binary myelin segmentation to axon instances using watershed.

    Uses axon instances as seeds and propagates labels into myelin regions
    via watershed algorithm based on Euclidean distance transform.

    Strategy for overlap regions (axon AND myelin voxels):
    - Keep axon labels (prioritize axoplasm interior)
    - Only assign myelin labels to pure myelin voxels

    Args:
        axon_volume: Instance-labeled axon volume (0=background, n=axon n)
        myelin_volume: Binary myelin segmentation (0=background, 1=myelin)
        max_distance: Optional max distance in voxels for assignment
        connectivity: Connectivity for myelin component labeling (1 or 2)

    Returns:
        Instance-labeled myelin volume (0=background, n=myelin for axon n)
        Same dtype as axon_volume to preserve label range
    """
    logger.info("Starting myelin-to-axon assignment")
    logger.info(f"  Axon volume: {axon_volume.shape}, {axon_volume.dtype}")
    logger.info(f"  Myelin volume: {myelin_volume.shape}, {myelin_volume.dtype}")

    # Validate inputs
    if axon_volume.shape != myelin_volume.shape:
        raise ValueError(
            f"Shape mismatch: axon {axon_volume.shape} vs myelin {myelin_volume.shape}"
        )

    # Create masks
    axon_mask = axon_volume > 0
    myelin_mask = myelin_volume > 0

    # Identify pure myelin regions (myelin NOT in axon)
    pure_myelin_mask = myelin_mask & ~axon_mask

    logger.info(f"  Axon voxels: {axon_mask.sum():,}")
    logger.info(f"  Myelin voxels: {myelin_mask.sum():,}")
    logger.info(f"  Overlap voxels: {(axon_mask & myelin_mask).sum():,}")
    logger.info(f"  Pure myelin voxels: {pure_myelin_mask.sum():,}")

    # Label connected components in pure myelin
    myelin_components, n_components = label(
        pure_myelin_mask, connectivity=connectivity, return_num=True
    )
    logger.info(f"  Myelin connected components: {n_components}")

    if n_components == 0:
        logger.warning("No myelin components found - returning empty volume")
        return np.zeros_like(axon_volume)

    # Compute distance transform for watershed
    # Distance from each myelin voxel to nearest axon
    distance = ndimage.distance_transform_edt(~axon_mask)

    # Mask distance to only pure myelin regions
    distance_masked = np.where(pure_myelin_mask, distance, 0)

    # Apply max distance threshold if specified
    if max_distance is not None:
        logger.info(f"  Applying max distance threshold: {max_distance} voxels")
        distance_masked = np.where(distance_masked <= max_distance, distance_masked, 0)
        pure_myelin_mask = pure_myelin_mask & (distance_masked > 0)

    # Use watershed to assign myelin to nearest axon
    # Seeds are the axon instances, propagate into pure myelin regions
    logger.info("  Running watershed assignment...")

    # Create markers: axon instances + background
    # Pure myelin regions are unlabeled (0) and will be assigned by watershed
    # Ensure markers are int32 (required by watershed_ift)
    markers = axon_volume.astype(np.int32)

    # Watershed propagates from markers into regions with label 0
    # We want to assign pure myelin, so invert distance (watershed flows downhill)
    # Scale distance to uint8 range for watershed_ift
    max_dist = distance_masked.max()
    if max_dist > 0:
        # Scale and invert: closer to axon = higher value
        distance_scaled = ((max_dist - distance_masked) / max_dist * 255).astype(np.uint8)
    else:
        distance_scaled = np.zeros_like(distance_masked, dtype=np.uint8)

    myelin_assigned = ndimage.watershed_ift(
        distance_scaled,
        markers
    )

    # Extract only the myelin labels (keep only pure myelin voxels)
    myelin_instances = np.where(pure_myelin_mask, myelin_assigned, 0)
    myelin_instances = myelin_instances.astype(axon_volume.dtype)

    n_assigned = (myelin_instances > 0).sum()
    logger.info(f"  Assigned myelin voxels: {n_assigned:,} / {pure_myelin_mask.sum():,}")
    logger.info(f"  Unique myelin instances: {len(np.unique(myelin_instances)) - 1}")

    return myelin_instances


def compute_fiber_metrics(
    axon_labels: np.ndarray,
    myelin_labels: np.ndarray,
    voxel_size_um: float = 0.05
) -> dict:
    """
    Compute fiber metrics including outer diameter and g-ratio.

    For each axon instance:
    - Inner diameter: diameter of axon (axoplasm)
    - Outer diameter: diameter of axon + myelin (total fiber)
    - G-ratio: inner diameter / outer diameter

    Args:
        axon_labels: Instance-labeled axon volume
        myelin_labels: Instance-labeled myelin volume (same instances)
        voxel_size_um: Voxel size in micrometers

    Returns:
        Dictionary with per-instance metrics:
        - instance_ids: array of instance labels
        - inner_diameter_um: array of axon diameters
        - outer_diameter_um: array of total fiber diameters
        - g_ratio: array of g-ratios
        - axon_area_um2: array of axon cross-section areas
        - fiber_area_um2: array of total fiber cross-section areas
    """
    from skimage.measure import regionprops

    # Get unique instances (excluding background)
    instance_ids = np.unique(axon_labels)
    instance_ids = instance_ids[instance_ids > 0]

    if len(instance_ids) == 0:
        return {
            'instance_ids': np.array([], dtype=int),
            'inner_diameter_um': np.array([]),
            'outer_diameter_um': np.array([]),
            'g_ratio': np.array([]),
            'axon_area_um2': np.array([]),
            'fiber_area_um2': np.array([])
        }

    inner_diameters = []
    outer_diameters = []
    axon_areas = []
    fiber_areas = []

    for instance_id in instance_ids:
        # Get axon area
        axon_area_voxels = (axon_labels == instance_id).sum()

        # Get myelin area for this instance
        myelin_area_voxels = (myelin_labels == instance_id).sum()

        # Total fiber area (axon + myelin)
        fiber_area_voxels = axon_area_voxels + myelin_area_voxels

        # Convert to physical area
        axon_area_um2 = axon_area_voxels * (voxel_size_um ** 2)
        fiber_area_um2 = fiber_area_voxels * (voxel_size_um ** 2)

        # Compute equivalent circular diameters
        inner_diameter = 2 * np.sqrt(axon_area_um2 / np.pi)
        outer_diameter = 2 * np.sqrt(fiber_area_um2 / np.pi)

        inner_diameters.append(inner_diameter)
        outer_diameters.append(outer_diameter)
        axon_areas.append(axon_area_um2)
        fiber_areas.append(fiber_area_um2)

    inner_diameters = np.array(inner_diameters)
    outer_diameters = np.array(outer_diameters)
    axon_areas = np.array(axon_areas)
    fiber_areas = np.array(fiber_areas)

    # Compute g-ratio (inner / outer)
    # Avoid division by zero
    g_ratios = np.divide(
        inner_diameters,
        outer_diameters,
        out=np.zeros_like(inner_diameters),
        where=outer_diameters > 0
    )

    return {
        'instance_ids': instance_ids,
        'inner_diameter_um': inner_diameters,
        'outer_diameter_um': outer_diameters,
        'g_ratio': g_ratios,
        'axon_area_um2': axon_areas,
        'fiber_area_um2': fiber_areas
    }
