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


def load_axon_cv_data(npz_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Load axon data and compute CV for each axon.

    Args:
        npz_file: Path to 3D axon profiles NPZ file

    Returns:
        Tuple of (mean_radii, cv_values, std_values, slowdown_factors, sample_name, profiles_dict)
        slowdown_factor = harmonic_mean(r) / arithmetic_mean(r) for each axon
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

    # Compute slowdown factor = harmonic_mean / arithmetic_mean for each axon
    # Harmonic mean captures that conduction time is dominated by narrow regions
    slowdown = np.zeros(len(radii_profiles))
    for i, profile in enumerate(radii_profiles):
        profile = np.array(profile)
        valid = profile > 0
        if np.sum(valid) > 0:
            harmonic_mean = len(profile[valid]) / np.sum(1.0 / profile[valid])
            arith_mean = np.mean(profile[valid])
            slowdown[i] = harmonic_mean / arith_mean if arith_mean > 0 else np.nan
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

    # Filter out NaN values
    valid = np.isfinite(cv) & np.isfinite(mean_radii) & np.isfinite(std_radii) & np.isfinite(slowdown)
    mean_radii = mean_radii[valid]
    cv = cv[valid]
    std_radii = std_radii[valid]
    slowdown = slowdown[valid]

    logger.info(f"Loaded {len(mean_radii)} axons from {npz_file.name}")
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
                             padding_um: float = 2.0):
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
    """
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

    for label, color in zip(labels, colors):
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

        # Subsample for performance (keep more points for denser appearance)
        n_original = len(surface_coords)
        if len(surface_coords) > 20000:
            indices = np.random.choice(len(surface_coords),
                                       size=min(len(surface_coords) // 2, 20000),
                                       replace=False)
            surface_coords = surface_coords[indices]

        # Log extent in Z (after rotation this will be the vertical extent)
        z_extent = (surface_coords[:, 0].max() - surface_coords[:, 0].min()) * voxel_size
        logging.info(f"  Axon {label}: {n_original} surface pts -> {len(surface_coords)} rendered, Z-extent: {z_extent:.1f} μm")

        # Convert to physical coordinates (μm) - account for subvolume offset
        coords_um = (surface_coords + min_coords) * voxel_size

        # Use distance transform for smooth local radius estimation
        dist = ndimage.distance_transform_edt(mask)
        from scipy.ndimage import maximum_filter
        filter_size = 21  # Captures radius up to ~0.5 μm at 0.05 μm voxels
        dist_local_max = maximum_filter(dist, size=filter_size)
        dist_values = dist_local_max[surface_coords[:, 0], surface_coords[:, 1], surface_coords[:, 2]]

        # Convert to physical units (μm)
        dist_values_um = dist_values * voxel_size

        # Normalize to [0.15, 1.0] for intensity
        dist_max_um = dist.max() * voxel_size
        if dist_max_um > 0:
            intensities = 0.15 + 0.85 * (dist_values_um / dist_max_um)
            point_sizes = 1 + 7 * (dist_values_um / dist_max_um)
        else:
            intensities = np.ones(len(surface_coords)) * 0.5
            point_sizes = np.ones(len(surface_coords)) * 3

        # Create colors with varying intensity
        base_color = np.array(to_rgb(color))
        point_colors = np.outer(intensities, base_color)

        all_surface_points.append(coords_um)
        axon_data.append((coords_um, point_colors, point_sizes))

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

    # Check if we need to flip Z so that arc length 0 is at the bottom
    # Use the first axon's skeleton to determine orientation
    if rep_axons and len(rep_axons) > 0:
        first_skel = rep_axons[0].get('skeleton_coords')
        if first_skel is not None and len(first_skel) > 1:
            first_skel = np.array(first_skel)
            # Transform skeleton start and end points
            start_rot = (first_skel[0] - centroid) @ R.T
            end_rot = (first_skel[-1] - centroid) @ R.T
            # If start has higher Z than end, flip Z
            if start_rot[2] > end_rot[2]:
                R = np.diag([1, 1, -1]) @ R  # Flip Z axis

    # Apply rotation and plot each axon
    all_rotated = []
    for coords_um, point_colors, point_sizes in axon_data:
        # Center, rotate, then shift back
        rotated = (coords_um - centroid) @ R.T + centroid
        all_rotated.append(rotated)

        # Plot 3D scatter (x, y, z -> for vertical alignment, z is up)
        # Point size varies with local radius for visible thickness variation
        # rasterized=True ensures points are rendered as bitmap in SVG (not 60k vector elements)
        ax.scatter(rotated[:, 0], rotated[:, 1], rotated[:, 2],
                   c=point_colors, s=point_sizes, alpha=0.9, rasterized=True)

    # Compute extent from rotated points
    all_rotated = np.vstack(all_rotated)
    rot_min = all_rotated.min(axis=0)
    rot_max = all_rotated.max(axis=0)

    # TODO: Arc length markers along skeleton (disabled for now)
    # if rep_axons:
    #     for axon, color in zip(rep_axons, colors):
    #         skeleton_coords = axon.get('skeleton_coords')
    #         ... (add back arc length annotation code)

    # Compute extent from rotated bounding box
    extent_um = rot_max - rot_min
    logging.info(f"  3D view bounding box: {extent_um[0]:.1f} x {extent_um[1]:.1f} x {extent_um[2]:.1f} μm")

    # Set tight axis limits with small padding
    pad = 1.0  # μm
    ax.set_xlim(rot_min[0] - pad, rot_max[0] + pad)
    ax.set_ylim(rot_min[1] - pad, rot_max[1] + pad)
    ax.set_zlim(rot_min[2] - pad, rot_max[2] + pad)

    # Set box aspect based on actual data extent (preserves shape)
    ax.set_box_aspect(extent_um + 2 * pad)

    # Set viewing angle: slight rotation to show depth variation
    ax.view_init(elev=5, azim=-85)
    ax.invert_zaxis()  # Flip z-axis
    ax.dist = 10  # Default

    # Hide axes, grid, and panes for clean look
    ax.set_axis_off()


def plot_cv_vs_radius(
    all_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict]],
    output_file: Path,
    k_velocity: float = 5.5,
    g_ratio: float = 0.6,
    mat_files: List[Path] = None
) -> None:
    """
    Create 2x3 panel plot:
    - (0,0): All 3 representative axons in one volume
    - (0,1): Straightened profiles
    - (0,2): CV histogram
    - (1,0): CV vs radius
    - (1,1): Std vs radius
    - (1,2): Velocity with slowdown

    Args:
        all_data: List of (mean_radii, cv_values, std_values, slowdown, sample_name, profiles_dict) tuples
        output_file: Output PNG file path
        k_velocity: Conduction velocity constant (m/s per μm of fiber diameter), default 5.5
        g_ratio: Ratio of axon radius to fiber radius, default 0.6
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
    # Subplots (b-e): 2x2 grid in columns 1-2
    ax_prof = fig.add_subplot(gs[0, 1])  # b
    ax_hist = fig.add_subplot(gs[0, 2])  # c
    ax_cv = fig.add_subplot(gs[1, 1])    # d
    ax_vel = fig.add_subplot(gs[1, 2])   # e

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

    # Representative example colors (green, orange, purple)
    cv_colors = [settings.colors['example_1'],
                 settings.colors['example_2'],
                 settings.colors['example_3']]
    cv_labels = ['Low CV', 'Mid CV', 'High CV']

    # Select representative axons using skeleton coords (memory efficient - no volume needed)
    rep_axons = select_representative_axons(all_profiles, min_length=50.0, use_spatial_selection=True)

    # Load cropped volume region for rendering (if mat files provided)
    vol = None
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

                        # Crop to bounding box
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

    # === (a): All 3 axons in one volume ===
    if vol is not None and rep_axons:
        # Get labels for all rep axons
        axon_labels = [ax['label'] for ax in rep_axons]
        colors = cv_colors[:len(axon_labels)]

        render_multiple_axons_3d(vol, axon_labels, colors, voxel_size, ax_vol,
                                 rep_axons=rep_axons, arc_interval=10.0)
    else:
        ax_vol.text2D(0.5, 0.5, 'Provide --mat-dir\nfor 3D rendering',
                      ha='center', va='center', transform=ax_vol.transAxes, fontsize=12)

    # === (b): Straightened profiles ===
    if rep_axons:
        for i, axon in enumerate(rep_axons):
            ax_prof.plot(axon['arc_lengths'], axon['radii'], color=cv_colors[i], linewidth=1.5,
                        label=f'{cv_labels[i]} (CV={axon["cv"]:.2f})')

    style_axis(ax_prof, xlabel='Arc length [μm]', ylabel='Axon radius [μm]')
    ax_prof.legend(loc='upper right', fontsize=font_settings['legend_size'])
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
    ax_hist.legend(loc='upper right', fontsize=font_settings['legend_size'])

    # === (d): CV vs radius ===
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

    single_line_color = settings.colors['single_line']  # Dark gray for single lines
    ax_cv.plot(bin_centers[valid], bin_medians[valid], color=single_line_color, linestyle='-',
               linewidth=line_settings['linewidth'], marker='o',
               markersize=line_settings['marker_size'], label='Median')
    ax_cv.fill_between(bin_centers[valid], bin_q25[valid], bin_q75[valid],
                       color=single_line_color, alpha=line_settings['fill_alpha'], label='IQR (25-75%)')
    style_axis(ax_cv, xlabel=r'Along-axon $\bar{r}$ [μm]', ylabel='CV')
    ax_cv.legend(loc='upper right', fontsize=font_settings['legend_size'])
    ax_cv.set_xlim(0, x_max)

    # === (1,2): Velocity with slowdown ===
    fiber_diameter = 2 * all_x / g_ratio  # μm
    velocity = k_velocity * fiber_diameter  # m/s (ideal, uniform axon)
    velocity_slow = velocity * all_slowdown  # m/s (with slowdown from caliber variation)

    bins = np.linspace(0, np.percentile(velocity, 99.5), hist_settings['bins'] + 1)
    # Filled histogram for "Ideal" (baseline)
    ax_vel.hist(velocity, bins=bins, color=settings.colors['category_a'],
                edgecolor='white', linewidth=0.5,
                alpha=0.7, label=f'Ideal: {np.mean(velocity):.1f} m/s')
    # Step histogram for "With slowdown" (comparison)
    ax_vel.hist(velocity_slow, bins=bins, histtype='step',
                edgecolor=settings.colors['category_b_edge'], linewidth=2,
                label=f'With slowdown: {np.mean(velocity_slow):.1f} m/s')

    mean_slowdown = np.mean(all_slowdown)
    ax_vel.text(0.95, 0.75, f'Mean slowdown:\n{mean_slowdown:.3f}',
                transform=ax_vel.transAxes, ha='right', va='top',
                fontsize=font_settings['legend_size'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    style_axis(ax_vel, xlabel='Conduction velocity [m/s]', ylabel='Count')
    ax_vel.legend(loc='upper right', fontsize=font_settings['legend_size'])

    # Set 1:1 aspect ratio for square subplots (not the tall 3D volume)
    for ax in [ax_prof, ax_hist, ax_cv, ax_vel]:
        ax.set_aspect('auto')
        ax.set_box_aspect(1)  # Square subplot

    # Adjust layout for 2D subplots (start at ~12% from left)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.08, wspace=0.3, hspace=0.3)
    # Restore 3D axes position
    ax_vol.set_position([-0.42, -0.5, 0.55, 2.0])

    # Save figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plot to {output_file}")


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
        mean_radii, cv, std_radii, slowdown, sample_name, profiles_dict = load_axon_cv_data(f)
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

    # Compute conduction velocity (k=5.5, g=0.6)
    k_velocity, g_ratio = 5.5, 0.6
    fiber_diameter = 2 * all_radii / g_ratio
    velocity = k_velocity * fiber_diameter
    velocity_slow = velocity * all_slowdown

    logger.info(f"Total axons: {len(all_cv)}")
    logger.info(f"Mean radius: {np.mean(all_radii):.3f} +/- {np.std(all_radii):.3f} um")
    logger.info(f"Mean CV: {np.mean(all_cv):.3f} +/- {np.std(all_cv):.3f}")
    logger.info(f"Median CV: {np.median(all_cv):.3f}")
    logger.info(f"Slowdown factor (harmonic/arithmetic mean):")
    logger.info(f"  Mean: {np.mean(all_slowdown):.4f}")
    logger.info(f"  Range: {np.min(all_slowdown):.4f} - {np.max(all_slowdown):.4f}")
    logger.info(f"Conduction velocity (k={k_velocity}, g={g_ratio}):")
    logger.info(f"  Ideal mean: {np.mean(velocity):.1f} +/- {np.std(velocity):.1f} m/s")
    logger.info(f"  With slowdown: {np.mean(velocity_slow):.1f} +/- {np.std(velocity_slow):.1f} m/s")
    logger.info(f"  Velocity reduction: {100*(1 - np.mean(velocity_slow)/np.mean(velocity)):.1f}%")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
