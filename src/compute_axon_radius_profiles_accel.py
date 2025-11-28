#!/usr/bin/env python3
"""
Compute axon radius profiles by sampling perpendicular cross-sections along skeletons.

ACCELERATED IMPLEMENTATION using Numba-optimized skeletonization.
Uses skeleton_tools.py with JIT-compiled inner loops for faster performance
than the original DeepACSON implementation.

For each axon:
1. Extract skeleton using optimized fast-marching + Euler path tracing
2. Walk along skeleton points
3. Sample perpendicular cross-section at each point
4. Compute equivalent circular radius from cross-section area

Output format matches cc_morph.mat structure: per-axon arrays of (radius_profile, skeleton_coords)
"""

import argparse
import logging
import traceback
from pathlib import Path
import multiprocessing as mp
from typing import Tuple, Union

import numpy as np
import h5py
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import zoom, find_objects
from skimage.measure import label, regionprops
from scipy.ndimage import median_filter
from tqdm import tqdm

# Import optimized skeleton functions
from skeleton_tools import skeleton, warmup as skeleton_warmup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_voxel_size(voxel_size_um: Union[float, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """
    Parse voxel size to a (vz, vy, vx) tuple matching array axis order (Z, Y, X).

    Args:
        voxel_size_um: Scalar (isotropic) or (vz, vy, vx) tuple (anisotropic)

    Returns:
        Tuple of (vz, vy, vx) in micrometers, matching array axes (Z, Y, X)
    """
    if isinstance(voxel_size_um, (tuple, list)):
        if len(voxel_size_um) == 3:
            return tuple(float(v) for v in voxel_size_um)
        elif len(voxel_size_um) == 1:
            v = float(voxel_size_um[0])
            return (v, v, v)
        else:
            raise ValueError(f"Expected 1 or 3 voxel size values, got {len(voxel_size_um)}")
    # Scalar - isotropic
    v = float(voxel_size_um)
    return (v, v, v)


def parse_voxel_size_arg(value: str) -> Union[float, Tuple[float, float, float]]:
    """Parse voxel size CLI argument: single float or comma-separated triple."""
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Expected 1 or 3 values, got {len(parts)}: '{value}'"
            )
        return tuple(float(p.strip()) for p in parts)
    return float(value)


def resample_to_isotropic(volume: np.ndarray,
                          voxel_size: Tuple[float, float, float]
                          ) -> Tuple[np.ndarray, float]:
    """
    Resample anisotropic volume to isotropic voxels.

    Downsamples to the coarsest voxel dimension to avoid upsampling artifacts.
    Uses nearest-neighbor interpolation (order=0) to preserve label integrity.

    Args:
        volume: 3D labeled volume with shape (Z, Y, X)
        voxel_size: Tuple of (vz, vy, vx) voxel sizes in micrometers, matching array axes

    Returns:
        Tuple of (resampled_volume, isotropic_voxel_size)
    """
    vz, vy, vx = voxel_size

    # Check if already isotropic
    if np.allclose([vz, vy, vx], vz):
        logger.info("Volume is already isotropic, no resampling needed")
        return volume, vz

    # Target voxel size is the coarsest dimension
    target_size = max(vz, vy, vx)

    # Compute zoom factors (< 1 means downsampling)
    # Volume axes are (Z, Y, X), voxel_size is (vz, vy, vx)
    zoom_factors = (vz / target_size, vy / target_size, vx / target_size)

    logger.info(f"Resampling anisotropic volume to isotropic:")
    logger.info(f"  Original voxel size (Z, Y, X): vz={vz:.4f}, vy={vy:.4f}, vx={vx:.4f} μm")
    logger.info(f"  Target voxel size: {target_size:.4f} μm (isotropic)")
    logger.info(f"  Zoom factors (Z, Y, X): {zoom_factors}")
    logger.info(f"  Original shape: {volume.shape}")

    # Use order=0 (nearest-neighbor) to preserve label integrity
    resampled = zoom(volume, zoom_factors, order=0, mode='nearest')

    logger.info(f"  Resampled shape: {resampled.shape}")

    return resampled, target_size


# Global variables for worker processes
_shared_volume = None
_shared_bboxes = None  # Dict mapping label -> (min_coords, max_coords)


def init_worker(volume, bboxes):
    """Initialize worker process with shared volume and bounding boxes."""
    global _shared_volume, _shared_bboxes
    _shared_volume = volume
    _shared_bboxes = bboxes
    # Warmup Numba JIT in each worker
    skeleton_warmup()


def compute_bounding_boxes(volume: np.ndarray) -> dict:
    """
    Compute bounding boxes for all labels using scipy.ndimage.find_objects.

    This is O(V) for a single pass over the volume, much faster than
    calling np.argwhere for each label separately.

    Args:
        volume: 3D labeled volume

    Returns:
        Dict mapping label -> (min_coords, max_coords) as numpy arrays
    """
    logger.info("Computing bounding boxes with find_objects (single pass)...")
    slices = find_objects(volume)

    bboxes = {}
    for label_idx, bbox_slices in enumerate(slices):
        if bbox_slices is None:
            continue
        label = label_idx + 1  # find_objects uses 0-indexed, labels are 1-indexed
        min_coords = np.array([s.start for s in bbox_slices])
        max_coords = np.array([s.stop for s in bbox_slices])
        bboxes[label] = (min_coords, max_coords)

    logger.info(f"Computed bounding boxes for {len(bboxes)} labels")
    return bboxes


def unit_tangent_vector(curve):
    """
    Compute unit tangent vectors along a curve.

    Args:
        curve: (N, 3) array of points

    Returns:
        (N, 3) array of unit tangent vectors
    """
    d_curve = np.gradient(curve, axis=0)
    ds = np.sqrt(np.sum(d_curve**2, axis=1, keepdims=True))
    ds[ds == 0] = 1e-5
    return d_curve / ds


def rotation_matrix_3D(vector, theta):
    """
    Create rotation matrix for counterclockwise rotation about a unit vector.
    Uses Euler-Rodrigues formula.
    """
    a = np.cos(theta / 2.0)
    b, c, d = -vector * np.sin(theta / 2.0)
    aa, bb, cc, dd = a**2, b**2, c**2, d**2
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
    ])


def unit_normal_vector(vec1, vec2):
    """Compute unit normal vector from cross product."""
    n = np.cross(vec1, vec2)
    if np.allclose(n, 0):
        n = vec1.copy()
    s = max(np.linalg.norm(n), 1e-5)
    return n / s


def angle_between(vec1, vec2):
    """Compute angle between two vectors."""
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)


def validate_skeleton_points(skeleton_points, binary_volume):
    """
    Validate that skeleton points are inside the binary volume.

    This catches cases where the Euler path tracing escapes the actual
    object boundaries (e.g., skeleton Z=13μm when axon only extends to Z=6μm).

    Args:
        skeleton_points: (N, 3) array of skeleton coordinates in voxel units
        binary_volume: 3D binary array of the axon

    Returns:
        Boolean mask of valid points (True = inside the axon)
    """
    n_points = len(skeleton_points)
    valid = np.ones(n_points, dtype=bool)
    shape = np.array(binary_volume.shape)

    for i, point in enumerate(skeleton_points):
        # Round to nearest voxel
        voxel = np.round(point).astype(int)

        # Check bounds
        if np.any(voxel < 0) or np.any(voxel >= shape):
            valid[i] = False
            continue

        # Check if this voxel is inside the axon
        if binary_volume[voxel[0], voxel[1], voxel[2]] == 0:
            valid[i] = False

    return valid


def filter_radius_outliers(radii, window_size=5, threshold=3.0):
    """
    Filter outlier radius measurements using local median comparison.

    Replaces values that are significantly larger than the local median
    with the local median value.

    Args:
        radii: 1D array of radius measurements
        window_size: Size of median filter window (must be odd)
        threshold: Multiplier for outlier detection (value > threshold * median)

    Returns:
        Filtered radii array with outliers replaced
    """
    if len(radii) < window_size:
        return radii

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Compute local median
    local_median = median_filter(radii, size=window_size, mode='reflect')

    # Find outliers: values significantly larger than local median
    # Use max to avoid division issues with zero median
    outlier_mask = radii > threshold * np.maximum(local_median, 0.01)

    # Replace outliers with local median
    radii_filtered = radii.copy()
    radii_filtered[outlier_mask] = local_median[outlier_mask]

    n_replaced = np.sum(outlier_mask)
    if n_replaced > 0:
        logger.debug(f"  Replaced {n_replaced}/{len(radii)} outlier radius measurements")

    return radii_filtered


def sample_perpendicular_cross_section(binary_volume, point, tangent_vec,
                                       plane_radius=5.0, plane_resolution=0.5,
                                       interpolator=None):
    """
    Sample a perpendicular cross-section at a skeleton point.

    Args:
        binary_volume: 3D binary array of the axon
        point: (3,) skeleton point coordinates
        tangent_vec: (3,) unit tangent vector at the point
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        interpolator: Optional pre-built RegularGridInterpolator

    Returns:
        area: Cross-section area in voxels^2, or None if sampling failed
    """
    # Create sampling grid in XY plane
    coords = np.arange(-plane_radius, plane_radius + plane_resolution, plane_resolution)
    x, y = np.meshgrid(coords, coords)
    z = np.zeros_like(x)

    # Flatten for rotation
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Compute rotation to align Z-axis with tangent vector
    z_axis = np.array([0, 0, 1])

    if np.allclose(tangent_vec, z_axis) or np.allclose(tangent_vec, -z_axis):
        # Already aligned or opposite
        if np.dot(tangent_vec, z_axis) < 0:
            rotated_plane = xyz * np.array([1, 1, -1])
        else:
            rotated_plane = xyz
    else:
        rot_axis = unit_normal_vector(z_axis, tangent_vec)
        theta = angle_between(z_axis, tangent_vec)
        rot_mat = rotation_matrix_3D(rot_axis, theta)
        rotated_plane = xyz @ rot_mat.T

    # Translate to skeleton point
    sample_coords = rotated_plane + point

    # Set up interpolator if not provided
    if interpolator is None:
        sz = binary_volume.shape
        interpolator = rgi(
            (range(sz[0]), range(sz[1]), range(sz[2])),
            binary_volume.astype(float),
            bounds_error=False,
            fill_value=0
        )

    # Sample cross-section
    cross_section = interpolator(sample_coords)
    bw_cross_section = (cross_section >= 0.5).reshape(x.shape)

    # Label connected components
    labeled, n_components = label(bw_cross_section, connectivity=1, return_num=True)

    if n_components == 0:
        return None

    # Find the component at the center (should be the axon)
    center_idx = bw_cross_section.shape[0] // 2
    center_label = labeled[center_idx, center_idx]

    if center_label == 0:
        # No component at center - find largest
        props = regionprops(labeled)
        if not props:
            return None
        areas = [p.area for p in props]
        center_label = props[np.argmax(areas)].label

    # Get area of the main component
    main_component = (labeled == center_label)
    area = np.sum(main_component)

    # Convert from grid squares to actual area
    area_voxels = area * (plane_resolution ** 2)

    return area_voxels


def process_single_axon(args):
    """
    Process a single axon to extract radius profile along skeleton.

    Args:
        args: Tuple of (axon_label, voxel_size_um, plane_radius, plane_resolution, step_size, path_method)
              voxel_size_um: (vx, vy, vz) tuple in micrometers
              path_method: 'discrete' (fast) or 'euler' (subvoxel accuracy)

    Returns:
        Dict with radius profile and skeleton coords, or None if failed
    """
    axon_label, voxel_size_um, plane_radius, plane_resolution, step_size, path_method = args

    # Parse voxel size to ensure we have (vz, vy, vx) matching array axes (Z, Y, X)
    vz, vy, vx = parse_voxel_size(voxel_size_um)
    # Geometric mean for approximate isotropic conversions
    voxel_geom_mean = (vz * vy * vx) ** (1/3)

    try:
        # Get precomputed bounding box (much faster than scanning full volume)
        bbox = _shared_bboxes.get(axon_label)
        if bbox is None:
            return None

        min_coords, max_coords = bbox

        # Add padding for cross-section sampling
        padding = int(plane_radius) + 5
        vol_shape = np.array(_shared_volume.shape)
        min_padded = np.maximum(min_coords - padding, 0)
        max_padded = np.minimum(max_coords + padding, vol_shape)

        # Extract subvolume and create binary mask (only scans small region)
        subvol = _shared_volume[
            min_padded[0]:max_padded[0],
            min_padded[1]:max_padded[1],
            min_padded[2]:max_padded[2]
        ]

        # Create binary mask within subvolume
        cropped = (subvol == axon_label).astype(np.uint8)

        # Get local coordinates within cropped region
        local_coords = np.argwhere(cropped)
        if len(local_coords) < 100:  # Skip very small axons
            return None

        # Run ACCELERATED skeletonization (using skeleton_tools.py)
        skel_segments = skeleton(cropped, verbose=False, path_method=path_method)

        if len(skel_segments) == 0:
            return None

        # Use the longest skeleton segment
        if len(skel_segments) == 1:
            main_skel = skel_segments[0]
        else:
            lengths = [len(seg) for seg in skel_segments]
            main_skel = skel_segments[np.argmax(lengths)]

        if len(main_skel) < 3:
            return None

        # Validate skeleton points are inside the axon (catches Euler escapes)
        valid_mask = validate_skeleton_points(main_skel, cropped)
        if not np.any(valid_mask):
            return None

        # Keep only valid skeleton points (find longest contiguous segment)
        if not np.all(valid_mask):
            # Find contiguous valid segments
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < 3:
                return None

            # Find gaps in valid indices
            gaps = np.diff(valid_indices)
            segment_starts = np.concatenate([[0], np.where(gaps > 1)[0] + 1])
            segment_ends = np.concatenate([np.where(gaps > 1)[0] + 1, [len(valid_indices)]])

            # Find longest contiguous segment
            segment_lengths = segment_ends - segment_starts
            longest_seg_idx = np.argmax(segment_lengths)
            start_idx = segment_starts[longest_seg_idx]
            end_idx = segment_ends[longest_seg_idx]

            valid_segment_indices = valid_indices[start_idx:end_idx]
            main_skel = main_skel[valid_segment_indices]

            if len(main_skel) < 3:
                return None

        # Subsample skeleton to step_size intervals
        # Compute cumulative arc length in physical units (accounting for anisotropy)
        diffs = np.diff(main_skel, axis=0)
        # Scale by voxel sizes: axis 0 = z, axis 1 = y, axis 2 = x
        diffs_physical = diffs * np.array([vz, vy, vx])
        segment_lengths = np.sqrt(np.sum(diffs_physical**2, axis=1))
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_length[-1]  # Now in physical units (μm)

        if total_length < step_size:
            # Too short - use all points
            sampled_skel = main_skel
        else:
            # Sample at regular intervals
            sample_positions = np.arange(0, total_length, step_size)
            sampled_skel = []
            for pos in sample_positions:
                # Find the segment containing this position
                idx = np.searchsorted(cumulative_length, pos) - 1
                idx = max(0, min(idx, len(main_skel) - 2))

                # Interpolate within segment
                t = (pos - cumulative_length[idx]) / (segment_lengths[idx] + 1e-10)
                t = np.clip(t, 0, 1)
                point = main_skel[idx] + t * (main_skel[idx + 1] - main_skel[idx])
                sampled_skel.append(point)

            sampled_skel = np.array(sampled_skel)

        if len(sampled_skel) < 2:
            return None

        # Compute tangent vectors
        tangent_vecs = unit_tangent_vector(sampled_skel)

        # Create interpolator once for all cross-sections
        sz = cropped.shape
        interpolator = rgi(
            (range(sz[0]), range(sz[1]), range(sz[2])),
            cropped.astype(float),
            bounds_error=False,
            fill_value=0
        )

        # Sample cross-sections and compute radii
        radii = []
        valid_skel_points = []

        for point, tangent in zip(sampled_skel, tangent_vecs):
            # Skip if tangent is zero
            if np.allclose(tangent, 0):
                continue

            area = sample_perpendicular_cross_section(
                cropped, point, tangent,
                plane_radius=plane_radius,
                plane_resolution=plane_resolution,
                interpolator=interpolator
            )

            if area is not None and area > 0:
                # Convert area to equivalent circular radius
                # A = π * r², so r = sqrt(A / π)
                # For anisotropic voxels, use geometric mean
                radius_voxels = np.sqrt(area / np.pi)
                radius_um = radius_voxels * voxel_geom_mean

                radii.append(radius_um)
                # Convert skeleton point back to original coordinates
                original_point = point + min_padded
                valid_skel_points.append(original_point * np.array([vz, vy, vx]))

        if len(radii) < 2:
            return None

        radii = np.array(radii)
        skel_coords = np.array(valid_skel_points)

        # Apply local consistency filter to remove outliers from self-intersection artifacts
        radii = filter_radius_outliers(radii, window_size=5, threshold=3.0)

        result = {
            'label': axon_label,
            'radii_um': radii,
            'skeleton_um': skel_coords,
            'n_points': len(radii),
            'mean_radius_um': np.mean(radii),
            'std_radius_um': np.std(radii),
            'length_um': total_length  # Already in physical units (μm)
        }

        return result

    except Exception as e:
        logger.error(f"Axon {axon_label}: Processing failed - {e}\n{traceback.format_exc()}")
        return None


def compute_radius_profiles(mat_file: Path,
                            output_file: Path,
                            voxel_size_um: Union[float, Tuple[float, float, float]] = 0.05,
                            plane_radius: float = 10.0,
                            plane_resolution: float = 0.5,
                            step_size: float = 2.0,
                            n_jobs: int = -1,
                            max_axons: int = 0,
                            anisotropy_mode: str = 'simple',
                            path_method: str = 'discrete'):
    """
    Compute radius profiles for all axons in a labeled volume.

    Args:
        mat_file: Path to .mat file with labeled axons
        output_file: Path to save results (.npz)
        voxel_size_um: Voxel size in micrometers. Can be:
                      - float: isotropic voxel size (e.g., 0.05)
                      - tuple: anisotropic (vx, vy, vz) (e.g., (0.015, 0.015, 0.05))
        plane_radius: Radius of sampling plane in voxels
        plane_resolution: Resolution of sampling plane
        step_size: Step size along skeleton in physical units (μm)
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        max_axons: Maximum number of axons to process (0 = all)
        anisotropy_mode: How to handle anisotropic voxels:
                        - 'simple': Resample to isotropic (coarsest dimension)
                        - 'none': Use geometric mean (legacy, less accurate)
        path_method: Skeleton path tracing method:
                    - 'discrete': Fast voxel-level gradient descent (default, ~2.8x faster)
                    - 'euler': Subvoxel Euler integration (slower but smoother)
    """
    # Warmup Numba JIT in main process
    logger.info("Warming up Numba JIT compilation...")
    skeleton_warmup()

    # Parse voxel size to (vx, vy, vz) tuple
    voxel_size_tuple = parse_voxel_size(voxel_size_um)

    # Validate input file
    if not mat_file.exists():
        raise FileNotFoundError(f"Input file not found: {mat_file}")

    # Load labeled volume - try HDF5 first, then scipy.io for older formats
    logger.info(f"Loading {mat_file}")
    volume = None

    # Try HDF5 format first (MATLAB v7.3)
    try:
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
            logger.info(f"Loaded HDF5 format, volume shape: {volume.shape}, dtype: {volume.dtype}")
    except OSError as e:
        # Try scipy.io for older MATLAB formats (v5/v6/v7)
        logger.info(f"HDF5 failed ({e}), trying scipy.io for older MATLAB format...")
        mat_data = sio.loadmat(str(mat_file))

        # Find the labeled volume key (skip MATLAB metadata keys)
        volume_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                volume_key = key
                break

        if volume_key is None:
            raise ValueError(f"No data found in {mat_file}")

        volume = mat_data[volume_key]
        logger.info(f"Loaded scipy.io format, volume shape: {volume.shape}, dtype: {volume.dtype}")

    # Handle anisotropic voxels
    if anisotropy_mode == 'simple':
        volume, iso_voxel_size = resample_to_isotropic(volume, voxel_size_tuple)
        voxel_size_tuple = (iso_voxel_size, iso_voxel_size, iso_voxel_size)
    elif anisotropy_mode != 'none':
        raise ValueError(f"Unknown anisotropy_mode: {anisotropy_mode}. Use 'simple' or 'none'.")

    # Compute bounding boxes for all labels (single pass - much faster than per-axon argwhere)
    bboxes = compute_bounding_boxes(volume)

    # Get unique axon labels
    axon_labels = np.unique(volume)
    axon_labels = axon_labels[axon_labels > 0]  # Remove background

    logger.info(f"Found {len(axon_labels)} axons")

    if max_axons > 0:
        axon_labels = axon_labels[:max_axons]
        logger.info(f"Processing first {max_axons} axons")

    # Handle n_jobs
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Prepare arguments for parallel processing
    args_list = [
        (label, voxel_size_tuple, plane_radius, plane_resolution, step_size, path_method)
        for label in axon_labels
    ]

    # Log voxel size info
    vz, vy, vx = voxel_size_tuple
    if vz == vy == vx:
        logger.info(f"Voxel size (isotropic): {vz:.4f} μm")
    else:
        logger.info(f"Voxel size (anisotropic, Z,Y,X): vz={vz:.4f}, vy={vy:.4f}, vx={vx:.4f} μm")

    logger.info(f"Processing {len(args_list)} axons with {n_jobs} workers")
    logger.info(f"Parameters: plane_radius={plane_radius}, resolution={plane_resolution}, step={step_size} μm")

    results = []
    if n_jobs == 1:
        # Sequential processing
        global _shared_volume, _shared_bboxes
        _shared_volume = volume
        _shared_bboxes = bboxes
        for args in tqdm(args_list, desc="Processing axons"):
            result = process_single_axon(args)
            if result is not None:
                results.append(result)
    else:
        # Parallel processing using fork
        with mp.Pool(n_jobs, initializer=init_worker, initargs=(volume, bboxes)) as pool:
            for result in tqdm(pool.imap_unordered(process_single_axon, args_list),
                               total=len(args_list), desc="Processing axons"):
                if result is not None:
                    results.append(result)

    logger.info(f"Successfully processed {len(results)}/{len(args_list)} axons")

    if len(results) == 0:
        logger.error("No axons were successfully processed!")
        return

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to arrays
    labels = np.array([r['label'] for r in results])
    n_points = np.array([r['n_points'] for r in results])
    mean_radii = np.array([r['mean_radius_um'] for r in results])
    std_radii = np.array([r['std_radius_um'] for r in results])
    lengths = np.array([r['length_um'] for r in results])

    # Per-axon profiles (variable length, stored as object arrays)
    radii_profiles = np.array([r['radii_um'] for r in results], dtype=object)
    skeleton_coords = np.array([r['skeleton_um'] for r in results], dtype=object)

    # Flatten all radii for distribution analysis
    all_radii = np.concatenate([r['radii_um'] for r in results])

    np.savez(
        output_file,
        labels=labels,
        n_points=n_points,
        mean_radii_um=mean_radii,
        std_radii_um=std_radii,
        lengths_um=lengths,
        radii_profiles_um=radii_profiles,
        skeleton_coords_um=skeleton_coords,
        all_radii_um=all_radii,
        voxel_size_um=np.array(voxel_size_tuple),  # Store as (vz, vy, vx) array matching axes
        plane_radius=plane_radius,
        plane_resolution=plane_resolution,
        step_size=step_size,
        source_file=str(mat_file)
    )

    logger.info(f"Saved results to {output_file}")

    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total axons processed: {len(results)}")
    logger.info(f"  Total radius samples: {len(all_radii)}")
    logger.info(f"  Mean axon length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} μm")
    logger.info(f"  Mean points per axon: {np.mean(n_points):.1f}")
    logger.info(f"  Mean radius: {np.mean(all_radii):.3f} ± {np.std(all_radii):.3f} μm")
    logger.info(f"  Median radius: {np.median(all_radii):.3f} μm")

    # Compute effective radius
    r_eff = (np.mean(all_radii**6) / np.mean(all_radii**2)) ** 0.25
    logger.info(f"  Effective radius (r_eff): {r_eff:.3f} μm")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute axon radius profiles by sampling perpendicular cross-sections (ACCELERATED with Numba)'
    )
    parser.add_argument('mat_file', type=Path,
                        help='Path to .mat file with labeled axons')
    parser.add_argument('output_file', type=Path,
                        help='Output .npz file for results')
    parser.add_argument('--voxel-size', type=parse_voxel_size_arg, default=0.05,
                        help='Voxel size in micrometers: single value for isotropic (e.g., 0.05) '
                             'or vz,vy,vx for anisotropic matching array axes Z,Y,X '
                             '(e.g., 0.05,0.015,0.015). Default: 0.05')
    parser.add_argument('--plane-radius', type=float, default=10.0,
                        help='Radius of sampling plane in voxels (default: 10.0)')
    parser.add_argument('--plane-resolution', type=float, default=0.5,
                        help='Resolution of sampling plane (default: 0.5)')
    parser.add_argument('--step-size', type=float, default=2.0,
                        help='Step size along skeleton in physical units μm (default: 2.0)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (default: -1 = all CPUs)')
    parser.add_argument('--max-axons', type=int, default=0,
                        help='Maximum axons to process (0 = all, default: 0)')
    parser.add_argument('--anisotropy-mode', type=str, default='simple',
                        choices=['simple', 'none'],
                        help="How to handle anisotropic voxels: 'simple' resamples to isotropic "
                             "(using coarsest dimension), 'none' uses geometric mean (legacy). "
                             "Default: 'simple'")
    parser.add_argument('--path-method', type=str, default='discrete',
                        choices=['discrete', 'euler'],
                        help="Skeleton path tracing method: 'discrete' uses fast voxel-level "
                             "gradient descent (~2.8x faster), 'euler' uses subvoxel Euler "
                             "integration (slower but smoother). Default: 'discrete'")

    args = parser.parse_args()

    compute_radius_profiles(
        args.mat_file,
        args.output_file,
        voxel_size_um=args.voxel_size,
        plane_radius=args.plane_radius,
        plane_resolution=args.plane_resolution,
        step_size=args.step_size,
        n_jobs=args.n_jobs,
        max_axons=args.max_axons,
        anisotropy_mode=args.anisotropy_mode,
        path_method=args.path_method
    )
