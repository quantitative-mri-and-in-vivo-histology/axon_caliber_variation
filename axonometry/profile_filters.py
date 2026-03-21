"""
Filtering utilities for 3D axon radius profile data.

Loads raw per-segment data from NPZ files and applies configurable
filters (endpoint trimming, minimum segment length). Analogous to
the 2D filtering in plot_2d_vs_3d_distribution_comparison.py.
"""

import numpy as np
from pathlib import Path
from typing import Union


# Default filter parameters
ENDPOINT_TRIM_POINTS = 20    # Points to trim from each segment end (matches gACSON)
MIN_SEGMENT_LENGTH_UM = 0.0  # Minimum segment length in μm


def load_and_filter_3d(npz_path: Union[str, Path],
                       endpoint_trim: int = ENDPOINT_TRIM_POINTS,
                       min_segment_length_um: float = MIN_SEGMENT_LENGTH_UM) -> dict:
    """
    Load 3D axon profile data and apply quality filters.

    Args:
        npz_path: Path to axon_profiles.npz file
        endpoint_trim: Number of points to trim from each segment end
                       (mitigates unreliable tangent vectors at endpoints)
        min_segment_length_um: Minimum segment length in μm to include

    Returns:
        dict with:
            labels: (N,) axon label IDs
            segment_radii_um: per-axon list of per-segment radius arrays (filtered)
            segment_lengths_um: per-axon list of per-segment lengths (filtered)
            all_radii_um: pooled radii from all filtered segments
            n_segments: (N,) number of segments per axon after filtering
            voxel_size_um: scalar
    """
    data = np.load(npz_path, allow_pickle=True)

    labels = data['labels']
    raw_seg_radii = data['segment_radii_um']
    raw_seg_lengths = data['segment_lengths_um']
    voxel_size = float(data['voxel_size_um'])

    filtered_seg_radii = []
    filtered_seg_lengths = []
    all_radii = []

    for i in range(len(labels)):
        segs_r = raw_seg_radii[i]
        segs_l = raw_seg_lengths[i]
        axon_segs_r = []
        axon_segs_l = []

        for j in range(len(segs_r)):
            r = np.asarray(segs_r[j])
            seg_len = float(segs_l[j])

            if seg_len < min_segment_length_um:
                continue

            # Endpoint trimming
            if endpoint_trim > 0 and len(r) > 2 * endpoint_trim:
                r = r[endpoint_trim:-endpoint_trim]

            if len(r) == 0:
                continue

            axon_segs_r.append(r)
            axon_segs_l.append(seg_len)
            all_radii.append(r)

        filtered_seg_radii.append(axon_segs_r)
        filtered_seg_lengths.append(axon_segs_l)

    all_radii = np.concatenate(all_radii).astype(np.float64) if all_radii else np.array([], dtype=np.float64)

    return {
        'labels': labels,
        'segment_radii_um': filtered_seg_radii,
        'segment_lengths_um': filtered_seg_lengths,
        'all_radii_um': all_radii,
        'n_segments': np.array([len(s) for s in filtered_seg_radii]),
        'voxel_size_um': voxel_size,
    }
