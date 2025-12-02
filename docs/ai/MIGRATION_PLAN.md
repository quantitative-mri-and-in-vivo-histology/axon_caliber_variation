# Plan: Reorganize Repository into `axonometry` Library + Examples

## Overview

Reorganize the codebase incrementally:
- **`axonometry/`** - Reusable Python library for axon morphometry (caliber variation, g-ratio)
- **`examples/`** - Flat directory of standalone analysis scripts
- **`src/`** - Original code (will delete when migration complete)

Build incrementally by picking relevant analyses from `src/` one at a time, extracting reusable code into `axonometry` as needed.

---

## Completed

### Initial Setup ✓
- Created `axonometry/` package with `__init__.py`
- Created `examples/` directory
- Added `scikit-fmm` to `requirements.txt`

### Library Modules Created ✓

```
axonometry/
├── __init__.py          # Public API exports (v0.1.0)
├── skeleton.py          # Fast-marching skeletonization with Numba (~300 lines)
├── geometry.py          # Tangent vectors, rotations, arc length, resampling (~120 lines)
├── io.py                # MAT file loading, voxel size parsing, isotropic resampling (~90 lines)
└── sampling.py          # Cross-section sampling, Numba interpolation (~150 lines)
```

### Example Scripts Created ✓

```
examples/
└── compute_axon_profiles_3d.py   # Along-axon radius profiling (from compute_axon_radius_profiles_accel.py)
```

### Test Suite ✓

```
tests/
├── __init__.py
├── test_geometry.py     # 30 tests - tangents, rotations, arc length
├── test_skeleton.py     # 15 tests - path tracing, skeleton extraction
├── test_io.py           # 14 tests - MAT loading, voxel sizes, resampling
└── test_sampling.py     # 30 tests - interpolation, cross-sections, validation
```

**Total: 89 tests, all passing, 81% coverage**

### Project Configuration ✓

- `pyproject.toml` - Package metadata, pytest/coverage/black/ruff/mypy config
- `requirements-dev.txt` - Already had good dev dependencies

---

## Remaining Work

### Library Modules To Create

```
axonometry/
├── bundling.py          # Bundle identification, orientation clustering, filtering
├── formats.py           # OME-Zarr metadata, pyramid generation, label LUTs
└── analysis.py          # Effective radius computation (2D slice-wise)
```

### Example Scripts To Migrate

**Priority (core pipeline):**
1. `identify_bundles.py` ← from `preprocess_1_identify_bundles.py`
2. `extract_bundle_volumes.py` ← from `preprocess_1b_extract_bundle_volumes.py`
3. `extract_orthogonal_slices.py` ← from `preprocess_1c_extract_orthogonal_slices.py`
4. `compute_fiber_profiles_2d.py` ← from `analyze_effective_radius.py` (slice-wise)

**Optional (if needed):**
- `visualize_neuroglancer.py`
- Various plotting scripts

### Key Components Still To Extract

**From preprocess_1_identify_bundles.py → bundling.py:**
- `compute_axon_orientation_from_coords()` - PCA-based orientation
- `cluster_by_orientation()` - Hierarchical clustering
- `filter_sparse_axons()` - KNN-based spatial filtering

**From preprocess_1b/1c → formats.py (consolidate duplicates):**
- `create_label_lut()`, `filter_with_lut()`
- `downsample_segmentation_mode_fast()`
- `create_ome_ngff_metadata()`
- `compute_valid_slice_range()`, `filter_axons_by_coverage()`

**From analyze_effective_radius.py → analysis.py:**
- `extract_radii_per_slice()` - Core 2D radius extraction
- `compute_effective_radius()` - r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)

---

## Commands

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=axonometry --cov-report=term-missing

# Run example script
python examples/compute_axon_profiles_3d.py input.mat output.npz --voxel-size 0.05
```
