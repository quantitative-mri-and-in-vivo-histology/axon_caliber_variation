"""
Axon radius profiling via skeleton extraction and cross-section sampling.

Two variants:
- axon_radius_profile(): uses verbatim DeepACSON code (axonometry.deepacson.original)
- axon_radius_profile_fast(): uses optimized version (axonometry.deepacson.fast)
"""

import contextlib
import io

import numpy as np

from .deepacson.original import (
    skeleton as deepacson_skeleton,
    sample_cross_section as deepacson_sample_cross_section,
    unit_tangent_vector as deepacson_unit_tangent_vector,
    get_line_length as deepacson_get_line_length,
)

from .deepacson.fast import (
    skeleton as fast_skeleton,
    sample_cross_section as fast_sample_cross_section,
    unit_tangent_vector as fast_unit_tangent_vector,
    get_line_length as fast_get_line_length,
)


def axon_radius_profile(binary_vol, g_radius, g_res=0.25, step_voxels=None,
                        verbose=False):
    """
    Extract radius profile using verbatim DeepACSON code.

    Args:
        binary_vol: 3D binary volume (1=axon, 0=background), float64
        g_radius: grid radius in voxels for cross-section sampling
        g_res: grid resolution in voxels (default 0.25, matching DeepACSON)
        step_voxels: if set, subsample skeleton at this spacing in voxels
        verbose: print skeleton segment lengths

    Returns:
        dict with 'radii_voxels', 'skeleton_points', 'n_segments',
        'length_voxels', or None if extraction failed
    """
    if verbose:
        skel_segments = deepacson_skeleton(binary_vol)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            skel_segments = deepacson_skeleton(binary_vol)

    if len(skel_segments) == 0:
        return None

    all_radii = []
    all_skel_points = []
    main_length = 0.0

    for skel_seg in skel_segments:
        if len(skel_seg) < 3:
            continue

        if step_voxels is not None:
            stride = max(1, round(step_voxels / 0.1))
            skel_seg = skel_seg[::stride]
            if len(skel_seg) < 3:
                continue

        tangent_vecs = deepacson_unit_tangent_vector(skel_seg)

        radii = []
        skel_points = []

        for pt, tangent in zip(skel_seg, tangent_vecs):
            area_pixels = deepacson_sample_cross_section(
                binary_vol, pt, tangent, g_radius, g_res
            )

            if area_pixels > 0:
                area_voxels = area_pixels * (g_res ** 2)
                radius_voxels = np.sqrt(area_voxels / np.pi)
                radii.append(radius_voxels)
                skel_points.append(pt.copy())

        if len(radii) < 2:
            continue

        seg_length = deepacson_get_line_length(skel_seg)
        if seg_length > main_length:
            main_length = seg_length

        all_radii.extend(radii)
        all_skel_points.extend(skel_points)

    if len(all_radii) < 2:
        return None

    return {
        'radii_voxels': np.array(all_radii),
        'skeleton_points': np.array(all_skel_points),
        'n_segments': len(skel_segments),
        'length_voxels': main_length,
    }


def axon_radius_profile_fast(binary_vol, g_radius, g_res=0.25, step_voxels=None,
                             verbose=False):
    """
    Extract radius profile using optimized DeepACSON code.

    Same algorithm as axon_radius_profile() but using deepacson_fast.

    Args:
        binary_vol: 3D binary volume (1=axon, 0=background), float64
        g_radius: grid radius in voxels for cross-section sampling
        g_res: grid resolution in voxels (default 0.25, matching DeepACSON)
        step_voxels: if set, subsample skeleton at this spacing in voxels
        verbose: print skeleton segment lengths

    Returns:
        dict with 'radii_voxels', 'skeleton_points', 'n_segments',
        'length_voxels', 'maxD', or None if extraction failed
    """
    skel_segments, maxD = fast_skeleton(binary_vol, verbose=verbose)

    if len(skel_segments) == 0:
        return None

    all_radii = []
    all_skel_points = []
    main_length = 0.0

    for skel_seg in skel_segments:
        if len(skel_seg) < 3:
            continue

        if step_voxels is not None:
            stride = max(1, round(step_voxels / 0.1))
            skel_seg = skel_seg[::stride]
            if len(skel_seg) < 3:
                continue

        tangent_vecs = fast_unit_tangent_vector(skel_seg)

        radii = []
        skel_points = []

        for pt, tangent in zip(skel_seg, tangent_vecs):
            area_pixels = fast_sample_cross_section(
                binary_vol, pt, tangent, g_radius, g_res
            )

            if area_pixels > 0:
                area_voxels = area_pixels * (g_res ** 2)
                radius_voxels = np.sqrt(area_voxels / np.pi)
                radii.append(radius_voxels)
                skel_points.append(pt.copy())

        if len(radii) < 2:
            continue

        seg_length = fast_get_line_length(skel_seg)
        if seg_length > main_length:
            main_length = seg_length

        all_radii.extend(radii)
        all_skel_points.extend(skel_points)

    if len(all_radii) < 2:
        return None

    return {
        'radii_voxels': np.array(all_radii),
        'skeleton_points': np.array(all_skel_points),
        'n_segments': len(skel_segments),
        'length_voxels': main_length,
        'maxD': float(maxD),
    }
