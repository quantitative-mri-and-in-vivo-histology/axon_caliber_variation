"""DeepACSON CSD skeletonization and cross-section analysis.

Two implementations of the algorithm from Abdollahzadeh et al. (2019):
- ``original``: verbatim code from the DeepACSON repository
- ``fast``: optimized version with fused pointmin/Euler path tracing,
  in-place array ops, cached grids, and bool mask support

Both expose the same public API: ``skeleton``, ``sample_cross_section``,
``unit_tangent_vector``, ``get_line_length``.  The fast module additionally
provides ``find_g_radius`` for adaptive grid sizing.
"""
