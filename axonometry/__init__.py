"""
Axonometry: Tools for axon morphometry analysis.

Includes utilities for:
- 3D skeletonization
- Cross-section sampling
- Radius profile extraction
- Effective radius computation
"""

from .io import (
    load_mat_volume,
    load_json_metadata,
    load_volume_with_metadata,
    parse_voxel_size,
    resample_to_isotropic,
    construct_output_path,
)

from .populations import (
    load_volume_downsampled,
    precompute_axon_voxels,
    compute_all_orientations,
    classify_by_dominant_axis,
    filter_sparse_axons,
    create_populations,
)

from .morphometry import (
    assign_myelin_to_axons,
    compute_fiber_metrics,
)

from .zarr_io import (
    write_ome_zarr_pyramid,
    downsample_nearest,
    downsample_mean,
)

from .plotting import (
    load_plot_settings,
    PlotSettings,
    get_plot_settings,
    add_panel_labels,
    style_axis,
)

__version__ = "0.1.0"

__all__ = [
    # I/O
    "load_mat_volume",
    "load_json_metadata",
    "load_volume_with_metadata",
    "parse_voxel_size",
    "resample_to_isotropic",
    "construct_output_path",
    # Populations
    "load_volume_downsampled",
    "precompute_axon_voxels",
    "compute_all_orientations",
    "classify_by_dominant_axis",
    "filter_sparse_axons",
    "create_populations",
    # Morphometry
    "assign_myelin_to_axons",
    "compute_fiber_metrics",
    # Zarr I/O
    "write_ome_zarr_pyramid",
    "downsample_nearest",
    "downsample_mean",
    # Plotting
    "load_plot_settings",
    "PlotSettings",
    "get_plot_settings",
    "add_panel_labels",
    "style_axis",
]
