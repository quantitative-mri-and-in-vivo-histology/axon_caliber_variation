#!/usr/bin/env python3
"""
Plot coefficient of variation (CV) of axon caliber vs mean axon radius.

This script reads 3D skeleton-based axon profiles and creates a scatter plot showing
the relationship between along-axon caliber variability (CV = std/mean) and mean
axon radius.

CV (coefficient of variation) quantifies how much the axon caliber varies along
its length relative to the mean caliber. Higher CV indicates more variable caliber.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from axonometry import get_plot_settings, style_axis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load plot settings
settings = get_plot_settings()


def extract_group(sample_name: str) -> str:
    """
    Extract group (TBI/Sham) from sample name.

    Args:
        sample_name: Sample name like "sham_25_ipsi", "tbi_2_contra", etc.

    Returns:
        "TBI" or "Sham"
    """
    name_lower = sample_name.lower()
    if 'tbi' in name_lower:
        return "TBI"
    elif 'sham' in name_lower:
        return "Sham"
    return "Unknown"


def load_axon_cv_data(npz_file: Path, min_length: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Load axon data and compute CV for each axon.

    Args:
        npz_file: Path to 3D axon profiles NPZ file
        min_length: Minimum axon length in μm (default 20.0)

    Returns:
        Tuple of (mean_radii, cv_values, std_values, slowdown_factors, sample_name, profiles_dict)
        slowdown_factor = v_eff / v(r_bar) for each axon, where v(r) = r*sqrt(ln(1 + d_m/r)),
        d_m = 2*r_bar/3 (constant myelin thickness with g_bar=0.6)
        profiles_dict contains 'radii_profiles', 'lengths', 'cv', 'skeleton_coords' for representative axon selection
    """
    data = np.load(npz_file, allow_pickle=True)

    mean_radii = data['mean_radii_um']
    std_radii = data['std_radii_um']
    lengths = data['lengths_um']
    radii_profiles = data['radii_profiles_um']
    skeleton_coords = data.get('skeleton_coords_um', None)

    # Compute CV = std / mean (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std_radii / mean_radii
        cv = np.where(np.isfinite(cv) & (mean_radii > 0), cv, np.nan)

    # Compute conduction velocity slowdown factor per axon
    # v(r) ∝ r * sqrt(ln(1 + d_m/r)) where d_m = 2*r_bar/3 (constant myelin thickness, g_bar=0.6)
    # v_eff = <1/v(r)>^{-1}  (harmonic mean of velocity along axon)
    # slowdown = v_eff / v(r_bar)  (reduction relative to ideal uniform axon)
    g_bar = 0.6
    slowdown = np.zeros(len(radii_profiles))
    for i, profile in enumerate(radii_profiles):
        profile = np.array(profile)
        valid = profile > 0
        if np.sum(valid) > 0:
            r = profile[valid]
            r_bar = np.mean(r)
            d_m = r_bar * (1 - g_bar) / g_bar  # = 2*r_bar/3 for g_bar=0.6
            # v(r) = r * sqrt(ln(1 + d_m/r))
            v = r * np.sqrt(np.log(1.0 + d_m / r))
            v_eff = len(v) / np.sum(1.0 / v)  # harmonic mean of v
            v_ideal = r_bar * np.sqrt(np.log(1.0 + d_m / r_bar))
            slowdown[i] = v_eff / v_ideal if v_ideal > 0 else np.nan
        else:
            slowdown[i] = np.nan

    # Extract sample name from filename
    sample_name = npz_file.stem.replace('_axon_profiles', '')

    # Store profiles for representative axon selection (before filtering)
    profiles_dict = {
        'radii_profiles': radii_profiles,
        'lengths': lengths,
        'cv': cv.copy(),
        'mean_radii': mean_radii.copy(),
        'labels': data['labels'].copy(),
        'skeleton_coords': skeleton_coords,  # 3D positions for centroid computation
    }

    # Filter: finite values AND minimum length
    valid = (np.isfinite(cv) & np.isfinite(mean_radii) & np.isfinite(std_radii) &
             np.isfinite(slowdown) & (lengths >= min_length))
    mean_radii = mean_radii[valid]
    cv = cv[valid]
    std_radii = std_radii[valid]
    slowdown = slowdown[valid]

    logger.info(f"Loaded {len(mean_radii)} axons from {npz_file.name} (length >= {min_length} μm)")
    logger.info(f"  Mean radius range: {mean_radii.min():.3f} - {mean_radii.max():.3f} um")
    logger.info(f"  CV range: {cv.min():.3f} - {cv.max():.3f}")
    logger.info(f"  Slowdown factor range: {slowdown.min():.3f} - {slowdown.max():.3f}")

    return mean_radii, cv, std_radii, slowdown, sample_name, profiles_dict


def find_axon_profile_files(pattern: str) -> List[Path]:
    """
    Find all axon profile NPZ files matching the pattern.

    Args:
        pattern: Glob pattern for axon profile files

    Returns:
        List of matching file paths
    """
    import glob as glob_module
    files = sorted([Path(f) for f in glob_module.glob(pattern, recursive=True)])
    return files


def select_representative_axons(
    all_profiles: List[dict],
    min_length: float = 50.0,
    seed: int = 42,
    use_spatial_selection: bool = True
) -> List[dict]:
    """
    Select 3 representative axons with low, mid, and high CV from the same file.

    Uses skeleton_coords from NPZ data (if available) to select axons that are
    spatially close to each other for better visualization.

    Args:
        all_profiles: List of profile dicts from each file
        min_length: Minimum axon length in μm
        seed: Random seed for reproducibility
        use_spatial_selection: If True, use skeleton coords for spatial selection

    Returns:
        List of dicts with keys: arc_lengths, radii, cv, label, file_idx, mean_radius
    """
    np.random.seed(seed)

    # Group candidates by file
    candidates_by_file = {}
    skeleton_coords_by_file = {}
    for file_idx, profiles_dict in enumerate(all_profiles):
        radii_profiles = profiles_dict['radii_profiles']
        lengths = profiles_dict['lengths']
        cv_values = profiles_dict['cv']
        mean_radii = profiles_dict['mean_radii']
        labels = profiles_dict.get('labels', np.arange(len(lengths)))
        skeleton_coords = profiles_dict.get('skeleton_coords', None)

        file_candidates = []
        for i in range(len(lengths)):
            if lengths[i] >= min_length and np.isfinite(cv_values[i]) and mean_radii[i] >= 0.1:
                profile = radii_profiles[i]
                if len(profile) > 10:  # Need enough points
                    candidate = {
                        'profile': profile,
                        'length': lengths[i],
                        'cv': cv_values[i],
                        'mean_radius': mean_radii[i],
                        'label': int(labels[i]) if hasattr(labels, '__getitem__') else i,
                        'file_idx': file_idx,
                        'idx': i,  # Store original index for skeleton lookup
                    }
                    # Compute centroid from skeleton coords (memory efficient)
                    if skeleton_coords is not None:
                        coords = skeleton_coords[i]
                        if coords is not None and len(coords) > 0:
                            candidate['centroid'] = np.mean(coords, axis=0)
                    file_candidates.append(candidate)

        if len(file_candidates) >= 3:
            candidates_by_file[file_idx] = file_candidates
            skeleton_coords_by_file[file_idx] = skeleton_coords

    if not candidates_by_file:
        return []

    # Pick the file with the most candidates for best percentile coverage
    best_file_idx = max(candidates_by_file, key=lambda k: len(candidates_by_file[k]))
    candidates = candidates_by_file[best_file_idx]

    # Sort by CV
    candidates.sort(key=lambda x: x['cv'])

    # Define CV ranges: low (0-20%), mid (40-60%), high (80-100%)
    n = len(candidates)
    low_range = candidates[int(n * 0.0):int(n * 0.20)]
    mid_range = candidates[int(n * 0.40):int(n * 0.60)]
    high_range = candidates[int(n * 0.80):int(n * 1.0)]

    # Check if we have skeleton coords for spatial selection
    has_centroids = all(
        ax.get('centroid') is not None
        for ax in (low_range[:5] + mid_range[:5] + high_range[:5])
    ) if use_spatial_selection else False

    if has_centroids and len(low_range) > 0 and len(mid_range) > 0 and len(high_range) > 0:
        # Find triplet with minimum total pairwise distance
        best_triplet = None
        best_distance = float('inf')

        # Sample to avoid combinatorial explosion
        low_sample = [ax for ax in low_range[:20] if ax.get('centroid') is not None]
        mid_sample = [ax for ax in mid_range[:20] if ax.get('centroid') is not None]
        high_sample = [ax for ax in high_range[:20] if ax.get('centroid') is not None]

        for low_ax in low_sample:
            for mid_ax in mid_sample:
                for high_ax in high_sample:
                    # Compute total pairwise distance (in μm from skeleton coords)
                    d1 = np.linalg.norm(low_ax['centroid'] - mid_ax['centroid'])
                    d2 = np.linalg.norm(mid_ax['centroid'] - high_ax['centroid'])
                    d3 = np.linalg.norm(low_ax['centroid'] - high_ax['centroid'])
                    total_dist = d1 + d2 + d3

                    if total_dist < best_distance:
                        best_distance = total_dist
                        best_triplet = [low_ax, mid_ax, high_ax]

        if best_triplet:
            selected = best_triplet
            logger.info(f"Selected spatially close axons (total distance: {best_distance:.1f} μm)")
        else:
            # Fallback to percentile selection
            selected = [candidates[int(n * 0.05)], candidates[int(n * 0.50)], candidates[int(n * 0.95)]]
    else:
        # Select low (5th percentile), mid (50th), high (95th) CV
        idx_low = int(n * 0.05)
        idx_mid = int(n * 0.50)
        idx_high = int(n * 0.95)
        selected = [candidates[idx_low], candidates[idx_mid], candidates[idx_high]]

    result = []
    for axon in selected:
        profile = np.array(axon['profile'])
        length = axon['length']
        n_points = len(profile)
        arc_lengths = np.linspace(0, length, n_points)

        # Get full skeleton coordinates for this axon
        skeleton_coords = None
        file_idx = axon['file_idx']
        if file_idx in skeleton_coords_by_file and skeleton_coords_by_file[file_idx] is not None:
            skeleton_coords = skeleton_coords_by_file[file_idx][axon['idx']]

        result.append({
            'arc_lengths': arc_lengths,
            'radii': profile,
            'cv': axon['cv'],
            'label': axon['label'],
            'file_idx': file_idx,
            'mean_radius': axon['mean_radius'],
            'length': length,
            'centroid': axon.get('centroid'),
            'skeleton_coords': skeleton_coords,  # Full skeleton for arc length annotation
        })

    return result


def render_multiple_axons_3d(volume: np.ndarray, labels: List[int], colors: List[str],
                             voxel_size: float, ax, rep_axons: List[dict] = None,
                             arc_interval: float = 10.0, subsample: int = 4,
                             padding_um: float = 2.0, volume_offset: np.ndarray = None):
    """
    Render multiple axons in 3D with local radius shown as color intensity.

    Uses distance transform to compute local thickness at each voxel,
    displays surface voxels in 3D scatter plot.

    Args:
        volume: 3D labeled volume (can be cropped region)
        labels: List of axon labels to extract
        colors: List of colors for each axon
        voxel_size: Voxel size in μm
        ax: Matplotlib 3D axis
        rep_axons: List of representative axon dicts with skeleton_coords
        arc_interval: Interval in μm for arc length markers (default 10)
        subsample: Subsampling factor for surface points (default 4)
        padding_um: Padding around bounding box in μm (default 2.0)
        volume_offset: Offset of volume origin in voxels (for cropped volumes)
    """
    if volume_offset is None:
        volume_offset = np.array([0, 0, 0])
    from matplotlib.colors import to_rgb
    from scipy import ndimage

    # Find combined bounding box for all axons
    all_coords = []
    for label in labels:
        mask = volume == label
        if mask.any():
            coords = np.argwhere(mask)
            all_coords.append(coords)

    if not all_coords:
        ax.text2D(0.5, 0.5, 'No axons found', ha='center', va='center', transform=ax.transAxes)
        return

    all_coords = np.vstack(all_coords)
    min_coords = all_coords.min(axis=0)
    max_coords = all_coords.max(axis=0)

    # Add padding in voxels (convert from μm)
    pad_voxels = int(np.ceil(padding_um / voxel_size))
    min_coords = np.maximum(min_coords - pad_voxels, 0)
    max_coords = np.minimum(max_coords + pad_voxels, np.array(volume.shape))

    # Extract subvolume
    slices = tuple(slice(mi, ma) for mi, ma in zip(min_coords, max_coords))
    sub_volume = volume[slices]

    # First pass: collect all surface points to compute principal axis
    all_surface_points = []
    axon_data = []  # Store (coords_um, point_colors, point_sizes) for each axon

    # Build lookup from label to rep_axon data (skeleton coords + radii)
    from scipy.spatial import cKDTree
    label_to_axon = {}
    if rep_axons:
        for axon in rep_axons:
            label_to_axon[axon['label']] = axon

    # Get global radius range for consistent color scaling across all axons
    # Use 5th and 95th percentiles for robustness to outliers
    all_radii = []
    for axon in (rep_axons or []):
        all_radii.extend(axon['radii'])
    if all_radii:
        global_r_min = np.percentile(all_radii, 5)
        global_r_max = np.percentile(all_radii, 95)
    else:
        global_r_min, global_r_max = 0, 1

    # Per-axon trim offsets: extra trim for green (High CV, index 2) to match lengths
    axon_trim_offsets = {2: 15.0}  # Index 2 (High CV/green) gets 15 μm extra trim

    for axon_idx, (label, color) in enumerate(zip(labels, colors)):
        mask = sub_volume == label
        if not mask.any():
            continue

        # Find surface voxels (erode and subtract)
        eroded = ndimage.binary_erosion(mask)
        surface = mask & ~eroded

        # Get surface coordinates
        surface_coords = np.argwhere(surface)
        if len(surface_coords) == 0:
            continue

        # Convert to physical coordinates (μm) - account for both volume_offset and subvolume offset
        coords_um = (surface_coords + min_coords + volume_offset) * voxel_size

        # Per-axon trim offset
        base_trim = 2.0  # Base trim for all axons
        extra_trim = axon_trim_offsets.get(axon_idx, 0.0)
        trim_start_um = base_trim + extra_trim

        # Use skeleton for arc-length based trimming
        axon_data_dict = label_to_axon.get(label)
        if axon_data_dict is not None and axon_data_dict.get('skeleton_coords') is not None:
            skeleton_coords = np.array(axon_data_dict['skeleton_coords'])

            # Compute arc length along skeleton
            skel_diffs = np.diff(skeleton_coords, axis=0)
            skel_dists = np.linalg.norm(skel_diffs, axis=1)
            arc_lengths = np.concatenate([[0], np.cumsum(skel_dists)])

            # Build KDTree for fast nearest-neighbor lookup
            tree = cKDTree(skeleton_coords)
            distances, nearest_idx = tree.query(coords_um, k=1)
            point_arc_lengths = arc_lengths[nearest_idx]

            # Filter by arc length (start trim) and distance from skeleton
            max_distance = 3.0  # μm
            keep_mask = (point_arc_lengths >= trim_start_um) & (distances < max_distance)

            coords_um = coords_um[keep_mask]
            surface_coords = surface_coords[keep_mask]

        # Subsample for performance (keep more points for denser appearance)
        n_original = len(surface_coords)
        if len(surface_coords) > 20000:
            indices = np.random.choice(len(surface_coords),
                                       size=min(len(surface_coords) // 2, 20000),
                                       replace=False)
            surface_coords = surface_coords[indices]
            coords_um = coords_um[indices]
            # Also subsample the nearest_idx if we have skeleton data
            if 'nearest_idx' in dir():
                nearest_idx = nearest_idx[indices]

        # Log extent in Z (after rotation this will be the vertical extent)
        z_extent = (coords_um[:, 0].max() - coords_um[:, 0].min()) if len(coords_um) > 0 else 0
        logging.info(f"  Axon {label}: {n_original} surface pts -> {len(surface_coords)} rendered, Z-extent: {z_extent:.1f} μm")

        # Use skeleton-based radius for color coding
        axon_data_dict = label_to_axon.get(label)
        if axon_data_dict is not None and axon_data_dict.get('skeleton_coords') is not None:
            skeleton_coords = np.array(axon_data_dict['skeleton_coords'])
            skeleton_radii = np.array(axon_data_dict['radii'])

            # Build KDTree for radius lookup
            tree = cKDTree(skeleton_coords)
            _, nearest_idx = tree.query(coords_um, k=1)
            local_radii = skeleton_radii[nearest_idx]

            # Normalize using global radius range for consistent coloring
            if global_r_max > global_r_min:
                norm = np.clip((local_radii - global_r_min) / (global_r_max - global_r_min), 0, 1)
                intensities = 0.25 + 0.75 * norm  # More contrast: darker min, brighter max
                point_sizes = 2 + 8 * norm
            else:
                intensities = np.ones(len(coords_um)) * 0.5
                point_sizes = np.ones(len(coords_um)) * 3
        else:
            # Fallback to distance transform if skeleton not available
            dist = ndimage.distance_transform_edt(mask)
            from scipy.ndimage import maximum_filter
            filter_size = 21
            dist_local_max = maximum_filter(dist, size=filter_size)
            dist_values = dist_local_max[surface_coords[:, 0], surface_coords[:, 1], surface_coords[:, 2]]
            dist_values_um = dist_values * voxel_size

            if global_r_max > global_r_min:
                norm = np.clip((dist_values_um - global_r_min) / (global_r_max - global_r_min), 0, 1)
                intensities = 0.25 + 0.75 * norm  # More contrast: darker min, brighter max
                point_sizes = 2 + 8 * norm
            else:
                intensities = np.ones(len(surface_coords)) * 0.5
                point_sizes = np.ones(len(surface_coords)) * 3

        # Create colors with varying intensity
        base_color = np.array(to_rgb(color))
        point_colors = np.outer(intensities, base_color)

        all_surface_points.append(coords_um)
        axon_data.append((coords_um, point_colors, point_sizes, trim_start_um))

    if not all_surface_points:
        return

    # Compute principal axes using PCA on all points
    all_points = np.vstack(all_surface_points)
    centroid = all_points.mean(axis=0)
    centered = all_points - centroid

    # SVD to get principal directions
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = Vt[0]  # First principal component (axon direction)
    pc2 = Vt[1]  # Second principal component (spread direction)

    # Step 1: Rotate to align PC1 with Z (vertical)
    target_z = np.array([0, 0, 1])
    v = np.cross(pc1, target_z)
    c = np.dot(pc1, target_z)

    if np.linalg.norm(v) < 1e-6:
        if c < 0:
            R1 = np.diag([-1, -1, 1])
        else:
            R1 = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R1 = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    # Step 2: Rotate around Z to align PC2 with X (spread axons horizontally)
    pc2_rotated = R1 @ pc2
    pc2_xy = np.array([pc2_rotated[0], pc2_rotated[1], 0])
    pc2_xy_norm = np.linalg.norm(pc2_xy)

    if pc2_xy_norm > 1e-6:
        pc2_xy = pc2_xy / pc2_xy_norm
        # Angle to rotate around Z to align with X-axis
        angle = np.arctan2(pc2_xy[1], pc2_xy[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R2 = np.array([[cos_a, -sin_a, 0],
                       [sin_a, cos_a, 0],
                       [0, 0, 1]])
    else:
        R2 = np.eye(3)

    # Combined rotation
    R = R2 @ R1

    # Apply rotation and flip each axon individually so arc length 0 is at bottom
    rotated_axon_data = []
    for i, (coords_um, point_colors, point_sizes, trim_start_um) in enumerate(axon_data):
        # Center, rotate, then shift back
        rotated = (coords_um - centroid) @ R.T + centroid
        cv = rep_axons[i]['cv'] if rep_axons and i < len(rep_axons) else 0
        z_flip = False
        z_flip_center = None

        # Check this axon's skeleton orientation and flip Z if needed
        if rep_axons and i < len(rep_axons):
            skel = rep_axons[i].get('skeleton_coords')
            if skel is not None and len(skel) > 1:
                skel = np.array(skel)
                # Transform full skeleton to rotated coords
                skel_rot = (skel - centroid) @ R.T + centroid

                # If start has higher Z than end, flip this axon's Z around its center
                if skel_rot[0, 2] > skel_rot[-1, 2]:
                    z_flip_center = rotated[:, 2].mean()
                    rotated[:, 2] = 2 * z_flip_center - rotated[:, 2]
                    # Also flip skeleton for Z extent calculation
                    skel_rot[:, 2] = 2 * z_flip_center - skel_rot[:, 2]
                    z_flip = True

        # Compute actual Z extent after rotation for marker scaling
        surface_z_extent = rotated[:, 2].max() - rotated[:, 2].min() if len(rotated) > 0 else 0

        # Store axon index, z_flip_center, surface Z extent, and trim offset for marker placement
        rotated_axon_data.append((rotated, point_colors, point_sizes, cv, i, z_flip, z_flip_center, surface_z_extent, trim_start_um))

    # Sort by CV (low to high)
    rotated_axon_data.sort(key=lambda x: x[3])

    # Compute X spacing based on axon widths
    x_spacing = 8.0  # μm between axon centers

    all_rotated = []
    axon_plot_info = []  # Store info for arc length markers
    # Per-axon vertical offsets to stagger axons visually
    z_offsets = {1: -2.5}  # Middle axon (orange) shifted down 2.5 μm

    for rank, (rotated, point_colors, point_sizes, cv, orig_idx, z_flip, z_flip_center, surface_z_extent, trim_start_um) in enumerate(rotated_axon_data):
        # Shift X position so axons are arranged left-to-right by CV rank
        # Center the group around 0
        n_axons = len(rotated_axon_data)
        x_offset = (rank - (n_axons - 1) / 2) * x_spacing
        x_center = rotated[:, 0].mean()
        rotated_shifted = rotated.copy()
        rotated_shifted[:, 0] = rotated[:, 0] - x_center + x_offset

        # Apply vertical offset if specified
        z_offset = z_offsets.get(rank, 0.0)
        rotated_shifted[:, 2] = rotated_shifted[:, 2] + z_offset

        all_rotated.append(rotated_shifted)
        axon_plot_info.append({
            'orig_idx': orig_idx,
            'x_offset': x_offset,
            'x_center': x_center,
            'z_flip': z_flip,
            'z_flip_center': z_flip_center,  # Store surface's flip center for skeleton alignment
            'z_center': rotated[:, 2].mean(),
            'z_offset': z_offset,  # Store vertical offset
            'surface_z_extent': surface_z_extent,  # Actual rendered Z extent
            'surface_z_min': rotated_shifted[:, 2].min() if len(rotated_shifted) > 0 else 0,
            'surface_z_max': rotated_shifted[:, 2].max() if len(rotated_shifted) > 0 else 0,
            'surface_y_mean': rotated_shifted[:, 1].mean() if len(rotated_shifted) > 0 else 0,
            'trim_start_um': trim_start_um,  # Per-axon trim offset
            'color': colors[rank] if rank < len(colors) else 'gray'
        })

        # Plot 3D scatter (x, y, z -> for vertical alignment, z is up)
        # Point size varies with local radius for visible thickness variation
        # rasterized=True ensures points are rendered as bitmap in SVG (not 60k vector elements)
        ax.scatter(rotated_shifted[:, 0], rotated_shifted[:, 1], rotated_shifted[:, 2],
                   c=point_colors, s=point_sizes, alpha=0.9, rasterized=True)

    # Compute extent from rotated points
    all_rotated = np.vstack(all_rotated)
    rot_min = all_rotated.min(axis=0)
    rot_max = all_rotated.max(axis=0)

    # Add arc length markers (10, 20, 30, ... μm) along each axon
    # Use surface Z positions directly (skeleton coords are in different frame)
    if rep_axons:
        # First pass: compute Z positions for each arc label for each axon
        arc_label_positions = {}  # {arc_label: [(x, y, z, rank), ...]}

        for rank, info in enumerate(axon_plot_info):
            axon = rep_axons[info['orig_idx']]
            stored_length = axon.get('length', 60)
            trim_start_um = info['trim_start_um']  # Per-axon trim offset

            # The surface Z range represents arc from trim_start to stored_length
            z_min = info['surface_z_min']
            z_max = info['surface_z_max']
            z_range = z_max - z_min
            if z_range <= 0:
                continue
            visible_arc = stored_length - trim_start_um  # Profile visible length

            # Compute positions for arc lengths 10, 20, 30, ...
            for arc_label in range(10, int(visible_arc) + 1, 10):
                # Map relative arc length to Z position
                z_pos = z_min + (arc_label / visible_arc) * z_range
                x_pos = info['x_offset']
                y_pos = info['surface_y_mean']

                if arc_label not in arc_label_positions:
                    arc_label_positions[arc_label] = []
                arc_label_positions[arc_label].append((x_pos, y_pos, z_pos, rank))

        # Second pass: draw tick marks and connecting lines between adjacent axons
        n_axons = len(axon_plot_info)
        tick_len = 1.5  # μm

        for arc_label, positions in sorted(arc_label_positions.items()):
            # Sort positions by rank (left to right)
            positions = sorted(positions, key=lambda p: p[3])

            # Draw tick marks at each axon position
            for x_pos, y_pos, z_pos, rank in positions:
                ax.plot([x_pos - tick_len, x_pos + tick_len],
                       [y_pos, y_pos], [z_pos, z_pos],
                       color='black', linewidth=1.0, zorder=10)

            # Connect tick ends between adjacent axons
            for i in range(len(positions) - 1):
                x1, y1, z1, _ = positions[i]
                x2, y2, z2, _ = positions[i + 1]
                # Connect right end of left tick to left end of right tick
                ax.plot([x1 + tick_len, x2 - tick_len],
                       [y1, y2], [z1, z2],
                       color='black', linewidth=0.8, zorder=9)

            # Add label on the left side of leftmost axon (no μm unit - shown on arrow)
            x_pos, y_pos, z_pos, _ = positions[0]
            ax.text(x_pos - tick_len - 1.5, y_pos, z_pos,
                   f'{arc_label}', fontsize=11, ha='right', va='center',
                   color='black', zorder=10)

        # Add vertical arrow with "Arc length [μm]" label on far left
        # Get z range from surface positions (start from bottom)
        all_surface_z_min = min(info['surface_z_min'] for info in axon_plot_info)
        all_surface_z_max = max(info['surface_z_max'] for info in axon_plot_info)

        # Collect all Z positions from arc labels
        all_z_positions = []
        for positions in arc_label_positions.values():
            for _, _, z_pos, _ in positions:
                all_z_positions.append(z_pos)

        # Arrow starts from bottom of axons, extends to top of tick marks
        z_min_arrow = all_surface_z_min + 1
        z_max_arrow = max(all_z_positions) + 3 if all_z_positions else all_surface_z_max

        # Position arrow to the left of all axons and labels
        leftmost_x = min(info['x_offset'] for info in axon_plot_info) - tick_len - 6
        y_pos_arrow = axon_plot_info[0]['surface_y_mean']

        # Draw arrow line (from bottom to top)
        ax.plot([leftmost_x, leftmost_x], [y_pos_arrow, y_pos_arrow],
               [z_min_arrow, z_max_arrow], color='black', linewidth=1.5, zorder=10)
        # Arrow head at top
        arrow_head_size = 2.0
        ax.plot([leftmost_x, leftmost_x], [y_pos_arrow, y_pos_arrow],
               [z_max_arrow - arrow_head_size, z_max_arrow],
               color='black', linewidth=2.5, zorder=10)
        # Add arrowhead lines
        ax.plot([leftmost_x - 0.5, leftmost_x, leftmost_x + 0.5],
               [y_pos_arrow, y_pos_arrow, y_pos_arrow],
               [z_max_arrow - arrow_head_size, z_max_arrow, z_max_arrow - arrow_head_size],
               color='black', linewidth=1.5, zorder=10)

        # Add rotated label "Arc length [μm]" alongside arrow (like y-axis label)
        # Use text2D for proper 2D rotation that doesn't depend on 3D view
        z_mid = (z_min_arrow + z_max_arrow) / 2
        ax.text2D(0.265, 0.5, 'Arc length [μm]', fontsize=14,
                 ha='center', va='center', color='black',
                 transform=ax.transAxes, rotation=90)

        # Add CV labels on top of each axon - same Z, but shift X for spacing
        cv_label_z = all_surface_z_max + 2  # Fixed Z for vertical alignment (slightly lower)
        n_axons = len(axon_plot_info)
        x_shifts = {0: -2.0, 1: 0.0, 2: 2.0}  # Shift left/middle/right
        for rank, info in enumerate(axon_plot_info):
            axon = rep_axons[info['orig_idx']]
            cv_val = axon['cv']
            x_pos = info['x_offset'] + x_shifts.get(rank, 0.0)
            y_pos = info['surface_y_mean']
            color = info['color']

            ax.text(x_pos, y_pos, cv_label_z,
                   f'CV = {cv_val:.2f}', fontsize=10, ha='center', va='bottom',
                   color=color, fontweight='bold', zorder=10)

    # Compute extent from rotated bounding box
    extent_um = rot_max - rot_min
    logging.info(f"  3D view bounding box: {extent_um[0]:.1f} x {extent_um[1]:.1f} x {extent_um[2]:.1f} μm")

    # Set tight axis limits with padding for labels
    pad = 1.0  # μm
    pad_top = 8.0  # Extra padding at top for CV labels
    ax.set_xlim(rot_min[0] - pad, rot_max[0] + pad)
    ax.set_ylim(rot_min[1] - pad, rot_max[1] + pad)
    ax.set_zlim(rot_min[2] - pad, rot_max[2] + pad_top)

    # Set box aspect based on actual data extent (preserves shape)
    extent_with_pad = extent_um.copy()
    extent_with_pad[2] += pad_top - pad  # Account for extra top padding
    ax.set_box_aspect(extent_with_pad + 2 * pad)

    # Set viewing angle: slight rotation to show depth variation
    ax.view_init(elev=5, azim=-85)
    # Bottom = start (arc length 0), Top = end
    ax.dist = 10  # Default

    # Hide axes, grid, and panes for clean look
    ax.set_axis_off()


def plot_cv_vs_radius(
    all_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict]],
    output_file: Path,
    mat_files: List[Path] = None
) -> None:
    """
    Create 2x3 panel plot:
    - (0,0): All 3 representative axons in one volume
    - (0,1): Straightened profiles
    - (0,2): CV histogram
    - (1,0): CV vs radius
    - (1,1): Std vs radius
    - (1,2): Slowdown factor vs axon size (bar plot)

    Args:
        all_data: List of (mean_radii, cv_values, std_values, slowdown, sample_name, profiles_dict) tuples
        output_file: Output PNG file path
        mat_files: List of mat files corresponding to each data tuple (for volume rendering)
    """
    # Create figure with GridSpec: subplot (a) spans full height, narrower column
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.6, 1, 1], height_ratios=[1, 1],
                          wspace=0.25, hspace=0.28)

    # Subplot (a): 3D rendering - spans both rows
    ax_vol = fig.add_subplot(gs[:, 0], projection='3d')
    # Position: extend beyond bounds, crop left whitespace
    ax_vol.set_position([-0.42, -0.5, 0.55, 2.0])
    # Subplots (b, c, e): standard grid positions
    ax_prof = fig.add_subplot(gs[0, 1])  # b
    ax_hist = fig.add_subplot(gs[0, 2])  # c
    ax_vel = fig.add_subplot(gs[1, 2])   # e

    # Subplot (d): CV vs radius
    ax_cv = fig.add_subplot(gs[1, 1])

    axes_list = [ax_vol, ax_prof, ax_hist, ax_cv, ax_vel]

    # Pool all data
    all_x = np.concatenate([r for r, _, _, _, _, _ in all_data])
    all_y = np.concatenate([cv for _, cv, _, _, _, _ in all_data])
    all_slowdown = np.concatenate([s for _, _, _, s, _, _ in all_data])

    # Get all profiles for representative selection
    all_profiles = [p for _, _, _, _, _, p in all_data]

    # Get settings
    hist_settings = settings.histogram
    font_settings = settings.fonts
    line_settings = settings.line
    main_color = settings.colors['category_a']  # Teal for main distribution

    # Representative example colors (purple, orange, green) - darker base for higher CV
    cv_colors = [settings.colors['example_3'],  # Purple for low CV
                 settings.colors['example_2'],  # Orange for mid CV
                 settings.colors['example_1']]  # Green for high CV
    cv_labels = ['Low CV', 'Mid CV', 'High CV']

    # Select representative axons using skeleton coords (memory efficient - no volume needed)
    rep_axons = select_representative_axons(all_profiles, min_length=50.0, use_spatial_selection=True)

    # Load cropped volume region for rendering (if mat files provided)
    vol = None
    vol_offset = None
    voxel_size = None
    if mat_files and rep_axons:
        import h5py
        from scipy import ndimage

        # Get file index and check if mat file exists
        file_idx = rep_axons[0]['file_idx']
        if file_idx < len(mat_files) and mat_files[file_idx].exists():
            mat_file = mat_files[file_idx]

            # Get voxel size from metadata
            from axonometry.io import load_json_metadata
            metadata = load_json_metadata(mat_file)
            voxel_size = 0.05  # Default
            if metadata and 'voxel_size' in metadata:
                vs = metadata['voxel_size']
                voxel_size = vs[0] if isinstance(vs, list) else vs

            labels = [ax['label'] for ax in rep_axons]
            max_label = max(labels)

            logger.info(f"Loading volume from {mat_file.name} for 3D rendering...")
            logger.info(f"  Looking for axon labels: {labels}")

            try:
                with h5py.File(str(mat_file), 'r') as f:
                    # Find volume key
                    volume_key = None
                    for key in f.keys():
                        if not key.startswith('#') and not key.startswith('_'):
                            volume_key = key
                            break

                    if volume_key:
                        full_volume = f[volume_key][:]
                        logger.info(f"  Full volume shape: {full_volume.shape} "
                                   f"({full_volume.nbytes / 1e6:.1f} MB)")

                        # Use find_objects to get bounding boxes for all labels up to max_label
                        # This is memory-efficient - doesn't create boolean masks
                        all_slices = ndimage.find_objects(full_volume, max_label)

                        # Compute combined bounding box for our selected labels
                        min_coords = [np.inf, np.inf, np.inf]
                        max_coords = [0, 0, 0]
                        for label in labels:
                            if label - 1 < len(all_slices) and all_slices[label - 1] is not None:
                                slc = all_slices[label - 1]
                                for i, s in enumerate(slc):
                                    min_coords[i] = min(min_coords[i], s.start)
                                    max_coords[i] = max(max_coords[i], s.stop)

                        # Add padding (50 voxels = 2.5 μm at 0.05 μm/voxel)
                        padding = 50
                        min_coords = [max(0, int(c) - padding) for c in min_coords]
                        max_coords = [min(full_volume.shape[i], int(c) + padding)
                                     for i, c in enumerate(max_coords)]

                        # Crop to bounding box and store offset for coordinate conversion
                        vol_offset = np.array(min_coords)
                        vol = full_volume[
                            min_coords[0]:max_coords[0],
                            min_coords[1]:max_coords[1],
                            min_coords[2]:max_coords[2]
                        ].copy()

                        # Free full volume immediately
                        del full_volume
                        del all_slices

                        logger.info(f"  Cropped to {vol.shape} ({vol.nbytes / 1e6:.1f} MB)")

            except Exception as e:
                logger.warning(f"Failed to load volume: {e}")
                vol = None
                vol_offset = None

    # === (a): All 3 axons in one volume ===
    if vol is not None and rep_axons:
        # Get labels for all rep axons
        axon_labels = [ax['label'] for ax in rep_axons]
        colors = cv_colors[:len(axon_labels)]

        render_multiple_axons_3d(vol, axon_labels, colors, voxel_size, ax_vol,
                                 rep_axons=rep_axons, arc_interval=10.0,
                                 volume_offset=vol_offset)
    else:
        ax_vol.text2D(0.5, 0.5, 'Provide --mat-dir\nfor 3D rendering',
                      ha='center', va='center', transform=ax_vol.transAxes, fontsize=12)

    # === (b): Straightened profiles ===
    # Per-axon trim offsets (same as 3D rendering)
    profile_trim_offsets = {2: 15.0}  # Index 2 (High CV/green) gets 15 μm extra trim
    base_trim = 2.0
    max_visible_arc = 0
    if rep_axons:
        for i, axon in enumerate(rep_axons):
            arc = axon['arc_lengths']
            radii = axon['radii']
            extra_trim = profile_trim_offsets.get(i, 0.0)
            trim_start_um = base_trim + extra_trim
            mask = arc >= trim_start_um
            visible_arc = arc[mask] - trim_start_um
            max_visible_arc = max(max_visible_arc, visible_arc.max())
            # Plot with x-axis starting from 0 (relative arc length)
            ax_prof.plot(visible_arc, radii[mask], color=cv_colors[i], linewidth=1.5,
                        label=f'CV = {axon["cv"]:.2f}')

    style_axis(ax_prof, xlabel='Arc length [μm]', ylabel='Axon radius [μm]')
    ax_prof.legend(loc='upper right', fontsize=font_settings['legend_size'], ncol=2)
    # Set x-ticks and xlim to match data (round up to nearest 10)
    x_max = int(np.ceil(max_visible_arc / 10) * 10) if max_visible_arc > 0 else 70
    ax_prof.set_xticks(range(0, x_max + 1, 10))
    ax_prof.set_xlim(0, x_max)
    # Extend y-axis to make room for legend
    ymin, ymax = ax_prof.get_ylim()
    ax_prof.set_ylim(ymin, ymax * 1.15)

    # === (c): CV histogram ===
    ax_hist.hist(all_y, bins=hist_settings['bins'], color=main_color,
                 edgecolor=hist_settings['edgecolor'], alpha=hist_settings['alpha'])
    ax_hist.axvline(np.mean(all_y), color=settings.colors['mean_line'], linestyle='--',
                    linewidth=line_settings['linewidth'], label=f'Mean = {np.mean(all_y):.3f}')
    ax_hist.axvline(np.median(all_y), color=settings.colors['median_line'], linestyle=':',
                    linewidth=line_settings['linewidth'], label=f'Median = {np.median(all_y):.3f}')
    style_axis(ax_hist, xlabel='CV', ylabel='Count')
    ax_hist.set_xlim(0, 0.8)  # Fixed x-axis for comparison
    ax_hist.legend(loc='upper right', fontsize=font_settings['legend_size'])
    # Use scientific notation for y-axis (×10^4 at top)
    ax_hist.ticklabel_format(axis='y', style='sci', scilimits=(4, 4), useMathText=True)
    ax_hist.yaxis.get_offset_text().set_fontsize(font_settings['tick_size'])

    # === (d): CV vs radius with marginal histogram on top ===
    x_max = np.percentile(all_x, 99.5)
    n_bins = 30
    bin_edges = np.linspace(0, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_medians = []
    bin_q25 = []
    bin_q75 = []

    for i in range(n_bins):
        mask = (all_x >= bin_edges[i]) & (all_x < bin_edges[i + 1])
        count = np.sum(mask)
        if count >= 100:
            bin_medians.append(np.median(all_y[mask]))
            bin_q25.append(np.percentile(all_y[mask], 25))
            bin_q75.append(np.percentile(all_y[mask], 75))
        else:
            bin_medians.append(np.nan)
            bin_q25.append(np.nan)
            bin_q75.append(np.nan)

    bin_medians = np.array(bin_medians)
    bin_q25 = np.array(bin_q25)
    bin_q75 = np.array(bin_q75)
    valid = ~np.isnan(bin_medians)

    # Get x-axis range from valid bins
    x_min_data = bin_centers[valid].min()
    x_max_data = bin_centers[valid].max()

    single_line_color = settings.colors['single_line']  # Dark gray for single lines
    ax_cv.plot(bin_centers[valid], bin_medians[valid], color=single_line_color, linestyle='-',
               linewidth=line_settings['linewidth'], marker='o',
               markersize=line_settings['marker_size'], label='Median')
    ax_cv.fill_between(bin_centers[valid], bin_q25[valid], bin_q75[valid],
                       color=single_line_color, alpha=line_settings['fill_alpha'], label='IQR (25-75%)')
    style_axis(ax_cv, xlabel='Along-axon mean radius [μm]', ylabel='CV')
    ax_cv.set_xlim(0.1, 0.6)  # Fixed x-axis for comparison
    ax_cv.set_ylim(0, 0.35)  # Fixed y-axis for comparison

    # === (1,2): Slowdown factor vs axon size (continuous median + IQR) ===
    # Bin axons by mean radius and compute slowdown stats per bin
    # Two slowdown metrics:
    #   1. Conduction velocity: v_eff/v(r_bar) where v(r)=r*sqrt(ln(1+d_m/r))
    #   2. Axial diffusion: 1/(1+4*CV²) - depends on area (r²), so stronger effect
    n_size_bins = 30  # More bins for smoother curves
    size_bin_edges = np.linspace(0, x_max, n_size_bins + 1)
    size_bin_centers = (size_bin_edges[:-1] + size_bin_edges[1:]) / 2

    slowdown_medians = []
    slowdown_q25 = []
    slowdown_q75 = []
    diffusion_slowdown_medians = []
    diffusion_slowdown_q25 = []
    diffusion_slowdown_q75 = []

    for i in range(n_size_bins):
        mask = (all_x >= size_bin_edges[i]) & (all_x < size_bin_edges[i + 1])
        count = np.sum(mask)
        if count >= 100:  # Same threshold as panel (d)
            slowdown_medians.append(np.median(all_slowdown[mask]))
            slowdown_q25.append(np.percentile(all_slowdown[mask], 25))
            slowdown_q75.append(np.percentile(all_slowdown[mask], 75))
            # Axial diffusion slowdown: 1/(1+4*CV²) per axon
            cv_bin = all_y[mask]
            diffusion_slow = 1.0 / (1.0 + 4.0 * cv_bin**2)
            diffusion_slowdown_medians.append(np.median(diffusion_slow))
            diffusion_slowdown_q25.append(np.percentile(diffusion_slow, 25))
            diffusion_slowdown_q75.append(np.percentile(diffusion_slow, 75))
        else:
            slowdown_medians.append(np.nan)
            slowdown_q25.append(np.nan)
            slowdown_q75.append(np.nan)
            diffusion_slowdown_medians.append(np.nan)
            diffusion_slowdown_q25.append(np.nan)
            diffusion_slowdown_q75.append(np.nan)

    slowdown_medians = np.array(slowdown_medians)
    slowdown_q25 = np.array(slowdown_q25)
    slowdown_q75 = np.array(slowdown_q75)
    diffusion_slowdown_medians = np.array(diffusion_slowdown_medians)
    diffusion_slowdown_q25 = np.array(diffusion_slowdown_q25)
    diffusion_slowdown_q75 = np.array(diffusion_slowdown_q75)
    valid_bins = ~np.isnan(slowdown_medians)

    # Convert efficiency to reduction percentage: (1 - efficiency) * 100
    cond_reduction_medians = (1 - slowdown_medians) * 100
    cond_reduction_q25 = (1 - slowdown_q75) * 100  # Note: q75 becomes lower bound
    cond_reduction_q75 = (1 - slowdown_q25) * 100  # Note: q25 becomes upper bound
    diff_reduction_medians = (1 - diffusion_slowdown_medians) * 100
    diff_reduction_q25 = (1 - diffusion_slowdown_q75) * 100
    diff_reduction_q75 = (1 - diffusion_slowdown_q25) * 100

    # Continuous line + IQR shaded area (like panel d)
    cond_color = '#C4A77D'  # Muted sand/tan - warm neutral
    diff_color = '#7BA3A8'  # Dusty teal - cool neutral

    # Conduction velocity reduction - line with IQR
    ax_vel.plot(size_bin_centers[valid_bins], cond_reduction_medians[valid_bins],
               color=cond_color, linestyle='-', linewidth=line_settings['linewidth'],
               marker='o', markersize=line_settings['marker_size'], label='Conduction velocity')
    ax_vel.fill_between(size_bin_centers[valid_bins], cond_reduction_q25[valid_bins],
                       cond_reduction_q75[valid_bins],
                       color=cond_color, alpha=line_settings['fill_alpha'])

    # Axial diffusion reduction - line with IQR
    ax_vel.plot(size_bin_centers[valid_bins], diff_reduction_medians[valid_bins],
               color=diff_color, linestyle='-', linewidth=line_settings['linewidth'],
               marker='s', markersize=line_settings['marker_size'], label='Diffusion (along-axon)')
    ax_vel.fill_between(size_bin_centers[valid_bins], diff_reduction_q25[valid_bins],
                       diff_reduction_q75[valid_bins],
                       color=diff_color, alpha=line_settings['fill_alpha'])

    ax_vel.legend(loc='upper right', fontsize=font_settings['legend_size'])
    style_axis(ax_vel, xlabel='Along-axon mean radius [μm]', ylabel='Reduction [%]')
    ax_vel.set_xlim(0.1, 0.6)  # Fixed x-axis for comparison
    ax_vel.set_ylim(0, 30)  # Fixed y-axis for comparison

    # Set aspect ratios for subplots (not the tall 3D volume)
    for ax in [ax_prof, ax_hist, ax_cv, ax_vel]:
        ax.set_aspect('auto')
        ax.set_box_aspect(1)  # Square subplot

    # Adjust layout for 2D subplots (start at ~12% from left)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.08, wspace=0.3, hspace=0.3)
    # Restore 3D axes position
    ax_vol.set_position([-0.42, -0.5, 0.55, 2.0])

    # Save figure (SVG + PNG)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')

    png_file = output_file.with_suffix('.png')
    plt.savefig(png_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file} and {png_file}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Plot coefficient of variation vs mean axon radius from 3D skeleton data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all HM axon profiles
  python plot_cv_vs_radius.py \\
      "data/processed/HM/*_axon_profiles.npz" \\
      fig/cv_vs_radius.png

  # Process specific files
  python plot_cv_vs_radius.py \\
      "data/processed/HM/sham_*_axon_profiles.npz" \\
      fig/sham_cv_vs_radius.png

  # Single file
  python plot_cv_vs_radius.py \\
      data/processed/HM/sham_25_ipsi_axon_profiles.npz \\
      fig/sham_25_ipsi_cv.png
        """
    )

    parser.add_argument('input', type=str,
                        help='Input: single NPZ file or glob pattern for 3D axon profiles')
    parser.add_argument('output', type=Path,
                        help='Output PNG file path')
    parser.add_argument('--mat-dir', type=Path, default=None,
                        help='Directory containing .mat files for 3D rendering (optional)')
    parser.add_argument('--min-length', type=float, default=20.0,
                        help='Minimum axon length in μm (default: 20.0)')

    args = parser.parse_args()

    # Find matching files
    files = find_axon_profile_files(args.input)

    if not files:
        logger.error(f"No files found matching pattern: {args.input}")
        return

    logger.info(f"Found {len(files)} axon profile file(s)")

    # Load all data
    all_data = []
    for f in files:
        mean_radii, cv, std_radii, slowdown, sample_name, profiles_dict = load_axon_cv_data(f, min_length=args.min_length)
        all_data.append((mean_radii, cv, std_radii, slowdown, sample_name, profiles_dict))

    # Find corresponding mat files if mat-dir is provided
    mat_files = None
    if args.mat_dir and args.mat_dir.exists():
        mat_files = []
        for f in files:
            # Convert NPZ filename to mat filename
            # e.g., sham_25_ipsi_axon_profiles.npz -> LM_25_ipsi_myelinated_axons.mat
            sample_name = f.stem.replace('_axon_profiles', '')
            parts = sample_name.split('_')
            # parts: ['sham', '25', 'ipsi'] or ['tbi', '24', 'contra']
            if len(parts) >= 3:
                rat_id = parts[1]
                hemisphere = parts[2]
                mat_name = f"LM_{rat_id}_{hemisphere}_myelinated_axons.mat"
                # Try to find it in mat_dir or subdirectories
                mat_path = args.mat_dir / mat_name
                if not mat_path.exists():
                    # Try condition subdirectory
                    condition = parts[0].capitalize()
                    mat_path = args.mat_dir / f"{condition}_{rat_id}_{hemisphere}" / mat_name
                mat_files.append(mat_path)
            else:
                mat_files.append(Path(""))
        logger.info(f"Found {len(mat_files)} mat files for 3D rendering")

    # Create plot
    plot_cv_vs_radius(all_data, args.output, mat_files=mat_files)

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)

    all_cv = np.concatenate([cv for _, cv, _, _, _, _ in all_data])
    all_radii = np.concatenate([r for r, _, _, _, _, _ in all_data])
    all_slowdown = np.concatenate([s for _, _, _, s, _, _ in all_data])

    # Compute conduction velocity using v(r) = r*sqrt(ln(1 + d_m/r)), d_m = 2*r_bar/3
    k_velocity, g_bar = 5.5, 0.6

    logger.info(f"Total axons: {len(all_cv)}")
    logger.info(f"Mean radius: {np.mean(all_radii):.3f} +/- {np.std(all_radii):.3f} um")
    logger.info(f"Mean CV: {np.mean(all_cv):.3f} +/- {np.std(all_cv):.3f}")
    logger.info(f"Median CV: {np.median(all_cv):.3f}")
    logger.info(f"Slowdown factor (v_eff / v(r_bar), Rushton model):")
    logger.info(f"  Mean: {np.mean(all_slowdown):.4f}")
    logger.info(f"  Range: {np.min(all_slowdown):.4f} - {np.max(all_slowdown):.4f}")
    logger.info(f"  Mean velocity reduction: {100*(1 - np.mean(all_slowdown)):.1f}%")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
