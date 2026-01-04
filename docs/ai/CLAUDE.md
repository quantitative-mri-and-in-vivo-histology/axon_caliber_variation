# CLAUDE.md

## Project Overview

This repository contains Python-based analysis and figure generation code for the manuscript on axon caliber variability and the effective axon radius, based on publicly available high-resolution axon segmentation data.

### Datasets

**1. Rat Brain 3D Segmentation Data** (Abdollazadeh et al., Nat. Commun., 2025)
- 10 labeled 3D volumes of myelinated axons from 5 rats
- 2 rats with TBI (traumatic brain injury), 3 rats with sham operation
- Each rat has both ipsilateral and contralateral hemisphere volumes
- Multiple white matter tract populations identified by orientation-based clustering
- Available in two magnifications:
  - **LM (Low Magnification)**: Primary dataset for all manuscript figures and analyses
  - **HM (High Magnification)**: Small test set for algorithm validation only; data too limited to report
- Used for: along-bundle stability analysis, effective radius profiles

**2. Human Corpus Callosum 2D Light Microscopy Data** (`data/raw_LM/`)
- Large-scale 2D axon radius distributions from human corpus callosum
- Pre-processed histogram data (bin edges, bin centers, counts)
- Multiple radius measures: circular equivalent, major axis, minor axis
- Multiple subjects (sub-ev01, sub-ev02) with ROI metadata
- Used for: parametric distribution fitting

### Analysis Focus

- Axon radius distributions in white matter tracts
- Effective MRI-visible radius (r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)) along tract length
- Per-slice radius profiles and global pooled statistics
- Comparisons between sham and TBI groups
- **Parametric distribution fitting** for radius distributions (following Sepehrband et al., 2016)

## Repository Structure

### Directory Layout

```
config/
├── plot_settings.yaml      # Global plot styling settings (colors, fonts, etc.)
│
data/
├── raw/                    # Raw .mat files organized by condition and hemisphere (rat 3D data)
│   ├── Sham_25_ipsi/
│   │   └── HM_25_ipsi_myelinated_axons.mat
│   ├── Sham_25_contra/
│   │   └── HM_25_contra_myelinated_axons.mat
│   ├── TBI_2_ipsi/
│   │   └── HM_2_ipsi_myelinated_axons.mat
│   └── ...
│
├── raw_LM/                 # Human corpus callosum 2D light microscopy data
│   ├── desc-binCenters_radii.tsv      # Histogram bin centers
│   ├── desc-binEdges_radii.tsv        # Histogram bin edges
│   ├── desc-countsCircularEq_radii.tsv    # Counts (circular equivalent radius)
│   ├── desc-countsMajorAxis_radii.tsv     # Counts (major axis radius)
│   ├── desc-countsMinorAxis_radii.tsv     # Counts (minor axis radius)
│   ├── roiinfo.tsv                    # ROI metadata
│   ├── desc-polygons_roicoords.tsv    # ROI polygon coordinates
│   ├── sub-ev01/                      # Subject 1 data
│   └── sub-ev02/                      # Subject 2 data
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

**Parametric Distribution Fitting** (human corpus callosum data)

Fit parametric distributions to axon radius histograms following the methodology of Sepehrband et al. (2016, NeuroImage). This analysis applies to the 2D light microscopy data in `data/raw_LM/`.

*Candidate distributions* (in order of expected performance):
- **Generalized Extreme Value (GEV)** - best performer in Sepehrband et al.
- Gamma - commonly used baseline
- Log-normal
- Inverse Gaussian
- Log-logistic
- Birnbaum-Saunders

*Light microscopy resolution limit*:
- Small axons with radius r < 0.3 μm cannot be reliably resolved
- Fits must account for this **left-truncation** (or left-censoring)
- Options:
  1. Fit truncated distributions with lower bound at r_min = 0.3 μm
  2. Fit to histogram data starting from first reliable bin
  3. Model the truncation explicitly in the likelihood function
- Report both raw fitted parameters and truncation-corrected estimates

*Fitting approach*:
- Input: histogram counts from TSV files (bin edges + counts)
- Method: maximum likelihood estimation (MLE) on binned data
- Goodness-of-fit: AIC, BIC, Kolmogorov-Smirnov test
- Output: fitted parameters, confidence intervals, diagnostic plots

*Reference*: Sepehrband F, et al. (2016). "Towards higher sensitivity and stability of axon diameter estimation with diffusion-weighted MRI." NMR Biomed. 29(3):293-308. PMID: 27303273

**Combined Distribution Fitting** ([fit_combined_distributions.py](examples/fit_combined_distributions.py))

Compares distribution fits and radius estimation bias across Human CC and Rat WM datasets:

*Outputs*:
1. **Summary figure** (2×4 layout): histogram + AIC + r_arith bias + r_eff bias for each dataset
2. **Scatter figures** (2×7 layout): subsampled vs whole-section values for Raw + 6 distributions

*Key features*:
- 6 Sepehrband distributions: GEV, Inverse Gaussian, Birnbaum-Saunders, Log Normal, Log Logistic, Gamma
- Fixed colors per distribution (consistent across all panels)
- Per-ROI/volume fitting with summed AIC aggregation
- Subsampling analysis: 50 subsamples × 1000 axons each, showing mean ± std error bars
- Optional EM correction for human data (`--em-correction` flag)

*Usage*:
```bash
python examples/fit_combined_distributions.py \
    --human-data data/raw_LM \
    --rat-data data/processed/LM \
    --output fig/combined_distribution_fits.png

# With EM correction for human data:
python examples/fit_combined_distributions.py \
    --human-data data/raw_LM \
    --rat-data data/processed/LM \
    --output fig/combined_distribution_fits_em.png \
    --em-correction data/raw_EM
```

*Key finding*: Best AIC (GEV) ≠ best r_eff predictor. Log Normal has near-zero r_eff bias for Human CC despite worse AIC.

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

### Plot Settings

All plot scripts in `examples/` must use centralized plot settings from `config/plot_settings.yaml` via the `axonometry` module. This ensures consistent styling across all figures.

**IMPORTANT: Always respect the color scheme guidelines defined in `config/plot_settings.yaml`.**

**Figure Color Scheme Guidelines:**

*Global Distinctions (consistent across all figures):*
- **Species** (when both shown):
  - Human: Blue (`#4A90D9`)
  - Rat: Red (`#D94A4A`)
- **Condition** (rat data only) - both clearly in red family:
  - Sham: Light brick red (`#E07070`)
  - TBI: Dark maroon (`#802020`)
- **Tract** (rat data only):
  - CC (corpus callosum): Circles (●)
  - CG (cingulum): Triangles (▲)
- **Representative examples** (when showing 3 selected ROIs/axons):
  - Example 1: Green (`#6B9E6B`)
  - Example 2: Orange (`#D9864A`)
  - Example 3: Purple (`#8B6BAE`)

*Local Binary Comparisons (overlapping histograms):*
Use filled vs step histogram:
- **Category A (baseline)**: Filled histogram, light gray (`#B0B0B0`, alpha=0.7)
- **Category B (comparison)**: Step histogram (unfilled), dark edge (`#303030`, linewidth=2)

For violin/box plots (high contrast):
- **Category A**: Light gray (`#D0D0D0`)
- **Category B**: Dark gray (`#505050`)

*Single lines/curves (no comparison):*
- Use dark gray (`#505050`) for single median/mean lines with shaded IQR

Examples: Ideal vs With slowdown, Intra-ROI vs Inter-ROI, CV vs radius (single line).

*Encoding Priority:*
1. Color for biological distinctions (species, condition)
2. Shape for anatomical categories (tract)
3. Grayscale for methodological comparisons

**Figure conventions:**
- **No suptitles**: Do not add figure-level titles (`fig.suptitle()`)
- **No subplot titles**: Do not add individual panel titles (`ax.set_title()`)
- **Panel labels**: Add lowercase letter labels (a, b, c, ...) to each panel
- **Units in square brackets**: Use square brackets for units in axis labels (e.g., `Radius [μm]`, not `Radius (μm)`)
- **Radius notation**: Use `$\bar{r}$` for arithmetic mean radius, `$r_{\mathrm{MRI}}$` for effective MRI-visible radius

**Usage in plot scripts:**
```python
from axonometry import get_plot_settings, add_panel_labels

settings = get_plot_settings()

# Species colors
human_color = settings.colors['human']  # '#4A90D9' (blue)
rat_color = settings.colors['rat']      # '#D94A4A' (red)

# Condition colors (rat data) - both in red family
sham_color = settings.colors['sham']    # '#E07070' (light brick red)
tbi_color = settings.colors['tbi']      # '#802020' (dark maroon)

# Binary comparison colors (filled vs step histogram)
filled_gray = settings.colors['category_a']      # '#B0B0B0' (light gray, use alpha=0.7)
step_edge = settings.colors['category_b_edge']   # '#303030' (dark edge, linewidth=2)
# For violin/box plots:
violin_light = settings.colors['category_a_violin']  # '#D0D0D0'
violin_dark = settings.colors['category_b_violin']   # '#505050'

# Representative example colors
colors = [settings.colors['example_1'],  # Green
          settings.colors['example_2'],  # Orange
          settings.colors['example_3']]  # Purple

# Markers for populations (tract)
marker = settings.get_marker('CC')   # 'o' (circle)
marker = settings.get_marker('CG')   # '^' (triangle)

# Apply consistent styling to axis (fonts, grid, box frame)
from axonometry import style_axis
style_axis(ax)                          # Apply all settings
style_axis(ax, xlabel='X', ylabel='Y')  # With labels

# Or manually:
ax.set_xlabel('X', fontsize=settings.fonts['label_size'])  # 14
ax.tick_params(labelsize=settings.fonts['tick_size'])      # 12
ax.legend(fontsize=settings.fonts['legend_size'])          # 11

# Error bar settings
ax.errorbar(..., capsize=settings.error_bars['capsize'],
            capthick=settings.error_bars['capthick'],
            elinewidth=settings.error_bars['linewidth'],
            alpha=settings.error_bars['alpha'])

# Histogram settings
ax.hist(..., bins=settings.histogram['bins'],
        alpha=settings.histogram['alpha'],
        edgecolor=settings.histogram['edgecolor'])

# Line plot settings
ax.plot(..., linewidth=settings.line['linewidth'],
        markersize=settings.line['marker_size'])
ax.fill_between(..., alpha=settings.line['fill_alpha'])

# Panel labels (a, b, c, ...) - add after tight_layout()
fig, axes = plt.subplots(2, 3)
# ... create plots ...
plt.tight_layout()
add_panel_labels(axes)  # Adds 'a' through 'f' with consistent styling

# Figure DPI
plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
```

**Settings file structure** (`config/plot_settings.yaml`):
- `colors`: Species (human, rat), condition (sham, tbi), examples (example_1/2/3), binary comparisons (category_a/b)
- `markers`: Population markers (cc, cg, default)
- `figure`: DPI, default figure size, suptitle/subplot_titles flags
- `panel_labels`: Enabled, fontsize, fontweight, position, alignment
- `fonts`: label_size (14), tick_size (12), title_size (16), legend_size (11)
- `grid`: enabled (false), alpha
- `frame`: box (true), linewidth - controls plot border
- `error_bars`: Capsize, thickness, linewidth, alpha
- `histogram`: Bin count, alpha, edge color
- `line`: Linewidth, marker size, fill alpha
