# CLAUDE.md

## Project Overview

This repository contains Python-based analysis and figure generation code for the manuscript on axon caliber variability and the effective axon radius, based on publicly available high-resolution axon segmentation data (Abdollazadeh et al., Nat. Commun., 2025; originally from a 2021 dataset).

The dataset consists of 10 labeled 3D volumes of myelinated axons from 5 rats:
- 2 rats with TBI (traumatic brain injury)
- 3 rats with sham operation
- Each rat has both ipsilateral and contralateral hemisphere volumes

Each volume contains multiple white matter tract populations identified by orientation-based clustering.

The focus is on characterizing and visualizing:
- Axon radius distributions in white matter tracts
- Effective MRI-visible radius (r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)) along tract length
- Per-slice radius profiles and global pooled statistics
- Comparisons between sham and TBI groups

## Repository Structure

### Directory Layout

```
data/
├── raw/                    # Raw .mat files organized by condition and hemisphere
│   ├── Sham_25_ipsi/
│   │   └── HM_25_ipsi_myelinated_axons.mat
│   ├── Sham_25_contra/
│   │   └── HM_25_contra_myelinated_axons.mat
│   ├── TBI_2_ipsi/
│   │   └── HM_2_ipsi_myelinated_axons.mat
│   └── ...
│
├── processed/              # Processed NPZ files organized by microscopy type
│   ├── HM/                 # High magnification data
│   │   ├── sham_25_ipsi_axon_profiles.npz
│   │   ├── sham_25_ipsi_slice_profiles.npz
│   │   ├── sham_25_contra_axon_profiles.npz
│   │   ├── tbi_2_ipsi_axon_profiles.npz
│   │   └── ...
│   └── LM/                 # Low magnification data
│       ├── sham_25_ipsi_axon_profiles.npz
│       └── ...
│
fig/                        # Generated figures and plots
│
src/                        # Python source code for preprocessing, analysis, and visualization
│
examples/                   # Example scripts for batch processing and visualization
```

**Naming Convention**: When using `--organize-by-microscopy` flag:
- Output files are organized into `HM/` and `LM/` subdirectories based on microscopy type
- Microscopy prefix (HM_/LM_) is removed from filenames (encoded in directory structure)
- "_myelinated_axons" suffix is removed from filenames (redundant)
- Condition prefix (Sham/TBI) is extracted from parent directory and converted to lowercase
- Files are named as `{condition}_{ratid}_{hemisphere}_{suffix}.npz` (all lowercase)
  - `suffix` is either `axon_profiles` (3D skeleton-based analysis) or `slice_profiles` (2D slice-based analysis)

**Example transformation**:
- Input: `data/raw/Sham_25_ipsi/HM_25_ipsi_myelinated_axons.mat`
- Output: `data/processed/HM/sham_25_ipsi_axon_profiles.npz`

**Rationale**: This structure enables easy comparison between different imaging modalities (HM/LM) for the same biological sample, while preserving critical experimental metadata (condition, subject ID, hemisphere) in filenames for analysis and filtering.

## Main Components

### Preprocessing Pipeline (3 steps)

**Step 1: Bundle Identification** ([preprocess_1_identify_bundles.py](src/preprocess_1_identify_bundles.py))
- Discover fiber bundles using orientation-based clustering (PCA + hierarchical clustering)
- No assumptions about specific tract identities (CC/CG)
- Filter bundles by size (≥500 axons) and remove spatially isolated axons using KNN distance threshold
- Save bundle metadata (axon labels, orientations, lengths) to JSON

**Step 2: Bundle Volume Extraction** ([preprocess_1b_extract_bundle_volumes.py](src/preprocess_1b_extract_bundle_volumes.py))
- Extract individual bundle volumes from bundle metadata
- Create axis-aligned OME-Zarr volumes with 5 pyramid levels
- Optimized for Neuroglancer visualization
- Two-phase processing: write level 0 (streaming), then generate pyramids

**Step 3: Orthogonal Slice Extraction** ([preprocess_1c_extract_orthogonal_slices.py](src/preprocess_1c_extract_orthogonal_slices.py))
- Extract true orthogonal cross-sections perpendicular to fiber direction
- Uses oblique sampling via `scipy.ndimage.map_coordinates`
- Parallel extraction with ThreadPoolExecutor
- Outputs OME-Zarr with multi-resolution pyramids
- Filters slices to include only bundle axons

### Analysis Pipeline

**Effective Radius Analysis** ([analyze_effective_radius.py](src/analyze_effective_radius.py))
- Supports both Zarr and HDF5 input formats
- Extract circular-equivalent radii per slice using regionprops
- Compute radius histograms (bin edges: 0-20 μm with 0.02 μm step)
- Calculate effective MRI-visible radius: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)
- Per-slice profiles showing radius variation along tract
- Global pooled effective radius from all slices combined
- Parallel processing with multiprocessing Pool
- Memory-efficient: reads slices directly from file
- Axon count filtering (min_axon_fraction) for symmetric slice selection

### Visualization

**Neuroglancer Viewer** ([visualize_neuroglancer.py](src/visualize_neuroglancer.py))
- Visualize bundle volumes in Neuroglancer
- Supports both HDF5 and OME-Zarr formats
- Multi-resolution HTTP serving for Zarr with CORS support
- Optional grayscale image overlay
- Auto-detection of segment IDs from stored metadata

**Effective Radius Plots**
- Per-slice effective radius vs position along tract
- Global effective radius as reference line (dashed)
- Axon count per slice subplot
- Publication-ready plots (200 DPI)

## Workflow

### 1. Obtain raw data

Download the raw axon segmentation data from:
[FAIR Data Dataset - f8ccc23a-1f1a-4c98-86b7-b63652a809c3](https://etsin.fairdata.fi/dataset/f8ccc23a-1f1a-4c98-86b7-b63652a809c3)

Place the 10 .mat files in `data/raw/LM/`:
- LM_2_ipsi_myelinated_axons.mat (Sham)
- LM_2_contra_myelinated_axons.mat (Sham)
- LM_24_ipsi_myelinated_axons.mat (Sham)
- LM_24_contra_myelinated_axons.mat (Sham)
- LM_25_ipsi_myelinated_axons.mat (TBI)
- LM_25_contra_myelinated_axons.mat (TBI)
- LM_28_ipsi_myelinated_axons.mat (Sham)
- LM_28_contra_myelinated_axons.mat (Sham)
- LM_49_ipsi_myelinated_axons.mat (TBI)
- LM_49_contra_myelinated_axons.mat (TBI)

### 2. Identify fiber bundles

```bash
python src/preprocess_1_identify_bundles.py \
    data/raw/LM/LM_25_ipsi_myelinated_axons.mat \
    data/processed/LM_25_ipsi/bundles.json \
    --downsample 4 \
    --min-axons 500 \
    --min-length 50.0 \
    --orientation-threshold 0.7 \
    --k-neighbors 10 \
    --max-neighbor-distance 30.0
```

Output: `data/processed/LM_25_ipsi/bundles.json`

### 3. Extract bundle volumes (axis-aligned)

```bash
python src/preprocess_1b_extract_bundle_volumes.py \
    data/raw/LM/LM_25_ipsi_myelinated_axons.mat \
    data/processed/LM_25_ipsi/bundles.json \
    data/processed/LM_25_ipsi/aligned \
    --voxel-size 0.05 \
    --n-levels 5
```

Output: `data/processed/LM_25_ipsi/aligned/bundle_01_aligned.zarr`, etc.

### 4. Extract orthogonal slices

```bash
python src/preprocess_1c_extract_orthogonal_slices.py \
    data/raw/LM/LM_25_ipsi_myelinated_axons.mat \
    data/processed/LM_25_ipsi/bundles.json \
    data/processed/LM_25_ipsi/orthogonal \
    --voxel-size 0.05 \
    --padding 0.5 \
    --n-levels 5 \
    --n-workers 32
```

Output: `data/processed/LM_25_ipsi/orthogonal/bundle_01_orthogonal.zarr`, etc.

### 5. Analyze effective radius

```bash
python src/analyze_effective_radius.py \
    data/processed/LM_25_ipsi/orthogonal/bundle_01_orthogonal.zarr \
    fig/LM_25_ipsi \
    --voxel-size 0.05 \
    --n-jobs -1 \
    --min-axon-fraction 0.75
```

Output: `fig/LM_25_ipsi/Bundle_01_effective_radius_profile.png`

### 6. Visualize in Neuroglancer

```bash
python src/visualize_neuroglancer.py \
    data/processed/LM_25_ipsi/orthogonal/bundle_01_orthogonal.zarr \
    --port 9999
```

Opens Neuroglancer viewer with multi-resolution support.

## Technical Details

### Voxel Size
- Original data: 0.05 μm/voxel (isotropic)
- Bundle identification uses 4× downsampling (0.2 μm/voxel) for speed
- Output volumes are at full resolution (0.05 μm/voxel)

### Data Formats
- **Input**: MATLAB .mat files (HDF5-based, v7.3)
- **Bundle metadata**: JSON with axon labels, orientations, lengths
- **Processed volumes**: OME-Zarr v3 format
  - 5 pyramid levels (2× downsampling per level)
  - ZSTD compression (level 3)
  - uint16 labels (supports up to 65,535 axons)
  - Segment IDs stored in `labels/segment_ids`
- **Output**: PNG figures (200 DPI)

### Memory Efficiency
- Zarr streaming: slice-by-slice processing avoids loading full volumes
- Two-phase extraction: write level 0, free original volume, then generate pyramids
- Parallel slice extraction with ThreadPoolExecutor
- LUT-based label filtering for fast bundle isolation
- Histogram aggregation: fixed memory footprint regardless of data size

### Performance
- Parallel processing: uses all CPU cores by default (`--n-jobs -1`)
- Batched map_coordinates calls for oblique sampling
- regionprops: efficient batch extraction of morphological properties
- Progress tracking: tqdm progress bars for all long operations

### Clustering Parameters
- **orientation_threshold**: Distance threshold for hierarchical clustering (0-1.4 range)
  - 0.3 = very strict (~17° tolerance)
  - 0.7 = moderate (~45° tolerance, recommended)
  - 1.0 = permissive (~60° tolerance)
- **k-neighbors / max-neighbor-distance**: KNN-based sparse axon filtering
