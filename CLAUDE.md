# CLAUDE.md

## Project Overview

This repository contains Python-based analysis and figure generation code for the manuscript on axon caliber variation and its impact on histological sampling of axon radius distributions, conduction velocity, and scalar statistics as mean radius and MRI-visible radius, based on publicly available high-resolution axon segmentation data.

### Datasets

**1. Rat Brain 3D Segmentation Data** (Sierra, Abdollahzadeh et al., 2021)
- 10 labeled 3D volumes of myelinated axons from 5 rats
- 2 rats with TBI (traumatic brain injury), 3 rats with sham operation
- Each rat has both ipsilateral and contralateral hemisphere volumes
- Multiple white matter tract populations identified by orientation-based clustering
- Available in two magnifications:
  - **LM (Low Magnification)**: Primary dataset for all manuscript figures and analyses
  - **HM (High Magnification)**: Small test set for algorithm validation only; data too limited to report
- Used for: along-bundle stability analysis, effective radius profiles

**2. Human Corpus Callosum 2D Light Microscopy Data** (Mordhorst et al., 2025; `data/raw/human/lm/`)
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
├── source/rat/             # Original .mat files organized by condition and hemisphere
│   ├── Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat
│   ├── Sham_25_contra/LM_25_contra_myelinated_axons.mat
│   └── ...
│
├── raw/
│   ├── human/
│   │   ├── lm/             # Human corpus callosum 2D light microscopy data
│   │   │   ├── desc-binCenters_radii.tsv
│   │   │   ├── desc-binEdges_radii.tsv
│   │   │   ├── desc-countsCircularEq_radii.tsv
│   │   │   ├── desc-countsMajorAxis_radii.tsv
│   │   │   ├── desc-countsMinorAxis_radii.tsv
│   │   │   ├── roiinfo.tsv
│   │   │   ├── sub-ev01/
│   │   │   └── sub-ev02/
│   │   ├── em/             # Internal EM data
│   │   └── em_ruthig/      # Ruthig et al. EM data (Zenodo)
│   └── rat/
│       ├── lm/             # Prepared OME-Zarr volumes (LM)
│       └── hm/             # Prepared OME-Zarr volumes (HM)
│
├── processed/rat/
│   ├── lm/                 # Low magnification processed profiles
│   │   ├── sham_25_contra_cc_myelin_axon_profiles.npz
│   │   ├── sham_25_contra_cc_myelin_slice_profiles.npz
│   │   └── ...
│   └── hm/                 # High magnification processed profiles
│
fig/                        # Generated figures and plots
│
scripts/
│   ├── preparation/        # Data preparation (ROI identification, volume extraction)
│   ├── processing/         # Data processing (slice profiles, axon profiles)
│   ├── figures/            # Manuscript figure generation
│   ├── visualization/      # Neuroglancer and interactive visualization
│   ├── exploratory/        # Exploratory analyses
│   └── benchmarks/         # Performance benchmarks
```

**Naming Convention**: When using `--organize-by-microscopy` flag:
- Output files are organized into `hm/` and `lm/` subdirectories based on microscopy type
- Microscopy prefix (HM_/LM_) is removed from filenames (encoded in directory structure)
- "_myelinated_axons" suffix is removed from filenames (redundant)
- Condition prefix (Sham/TBI) is extracted from parent directory and converted to lowercase
- Files are named as `{condition}_{ratid}_{hemisphere}_{suffix}.npz` (all lowercase)
  - `suffix` is either `axon_profiles` (3D skeleton-based analysis) or `slice_profiles` (2D slice-based analysis)

**Example transformation**:
- Input: `data/source/rat/Sham_25_ipsi/LM_25_ipsi_myelinated_axons.mat`
- Prepared: `data/raw/rat/lm/sham_25_ipsi_cc_myelin.zarr`
- Output: `data/processed/rat/lm/sham_25_ipsi_cc_myelin_axon_profiles.npz`

**Rationale**: This structure preserves critical experimental metadata (condition, subject ID, hemisphere, tract) in filenames for analysis and filtering.

## Main Components

### Preparation Pipeline (`scripts/preparation/`)

The preparation scripts convert raw heterogeneous data into a **unified canonical format** so that all downstream processing scripts can work with the same assumptions.

**Canonical output format:**
- OME-Zarr volumes with multi-resolution pyramids
- **Isotropic voxels** (resampled if source is anisotropic, e.g. HM 0.015×0.015×0.05 → isotropic)
- **Z axis = along-axon (fiber) direction** — axes are permuted so the dominant/fiber axis maps to axis 0 (Z)
- XY planes = cross-sections perpendicular to fibers

This means **processing scripts can always assume `slice_axis=0`** to get cross-sections, regardless of the original fiber orientation in the raw data.

**Step 1: Identify ROIs**

ROI bounding boxes for CC and CG populations were identified via PCA-based orientation classification and manually refined. The resulting `*_population_rois.json` files (with ROI bounds, dominant axes, and separation metadata) are stored under `data/`.

**Step 2: Extract volumes**

- `extract_rois.py`: Reads `*_population_rois.json`, crops each population's ROI, permutes the **separation axis → Z**, and writes OME-Zarr. LM data is already isotropic (0.05 μm). Outputs per-population: `*_cc_myelin.zarr`, `*_cg_myelin.zarr` (+ grayscale companions).

**Axis permutation logic** (same in both):
```
dominant/separation axis 0 → (0,1,2)  — already Z, no change
dominant/separation axis 1 → (1,0,2)  — swap Y↔Z
dominant/separation axis 2 → (2,0,1)  — move X to Z
```

### Analysis Pipeline

**Parametric Distribution Fitting** (human corpus callosum data)

Fit parametric distributions to axon radius histograms following the methodology of Sepehrband et al. (2016, NeuroImage). This analysis applies to the 2D light microscopy data in `data/raw/human/lm/`.

*Candidate distributions* (Sepehrband et al., 2016):
- **Generalized Extreme Value (GEV)** - best performer in Sepehrband et al.
- Inverse Gaussian
- Birnbaum-Saunders
- Log-normal
- Log-logistic
- Gamma

*Fitting approach* (implemented in `axonometry/distribution_fitting.py`):
- Input: histogram counts from TSV files (bin edges + counts)
- Method: binned MLE (multinomial likelihood over CDF-derived bin probabilities)
- Initialization: method-of-moments with perturbed restarts
- Goodness-of-fit: AIC, Wasserstein distance
- Radius estimation: numerical integration of fitted distribution for r_arith and r_eff

*Reference*: Sepehrband F, et al. (2016). "Towards higher sensitivity and stability of axon diameter estimation with diffusion-weighted MRI." NMR Biomed. 29(3):293-308. PMID: 27303273

**Combined Distribution Fitting** ([plot_fig6_parametric_fits.py](scripts/figures/plot_fig6_parametric_fits.py))

Compares distribution fits and radius estimation bias across Human CC and Rat WM datasets:

*Usage*:
```bash
python scripts/figures/plot_fig6_parametric_fits.py \
    --human-data data/raw/human/lm \
    --rat-data data/processed/rat/lm \
    --output fig/main/combined_distribution_fits.svg
```

*Key features*:
- 6 Sepehrband distributions with fixed colors per distribution
- Per-ROI/volume fitting with summed AIC aggregation
- Win rate, Wasserstein distance, r_arith and r_eff bias comparison

*Key finding*: Best AIC (GEV) ≠ best r_eff predictor. Log Normal has near-zero r_eff bias for Human CC despite worse AIC.


## Workflow

### 1. Obtain raw data

Download the raw axon segmentation data from:
[FAIR Data Dataset - f8ccc23a-1f1a-4c98-86b7-b63652a809c3](https://etsin.fairdata.fi/dataset/f8ccc23a-1f1a-4c98-86b7-b63652a809c3)

Place the 10 .mat files in `data/source/rat/`, organized by condition and hemisphere:
- Sham_2_ipsi/LM_2_ipsi_myelinated_axons.mat
- Sham_2_contra/LM_2_contra_myelinated_axons.mat
- Sham_24_ipsi/LM_24_ipsi_myelinated_axons.mat
- Sham_24_contra/LM_24_contra_myelinated_axons.mat
- Sham_28_ipsi/LM_28_ipsi_myelinated_axons.mat
- Sham_28_contra/LM_28_contra_myelinated_axons.mat
- TBI_25_ipsi/LM_25_ipsi_myelinated_axons.mat
- TBI_25_contra/LM_25_contra_myelinated_axons.mat
- TBI_49_ipsi/LM_49_ipsi_myelinated_axons.mat
- TBI_49_contra/LM_49_contra_myelinated_axons.mat


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
- **Output**: SVG figures (200 DPI raster fallback)

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

All plot scripts in `scripts/` must use centralized plot settings from `config/plot_settings.yaml` via the `axonometry` module. This ensures consistent styling across all figures.

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

For colored binary comparisons (scatter plots, error bars):
- **Binary A**: Red (`#D94A4A`)
- **Binary B**: Blue (`#4A90D9`)

*Single lines/curves (no comparison):*
- Use dark gray (`#505050`) for single median/mean lines with shaded IQR

Examples: Ideal vs With slowdown, Intra-ROI vs Inter-ROI, CV vs radius (single line).

*Encoding Priority:*
Red (`#D94A4A`) and blue (`#4A90D9`) are the primary contrast pair: species where species are
compared, binary methodological comparison otherwise. **No single figure may contain both.**
1. Color for the strongest distinction in the figure (species, condition, or binary comparison)
2. Shape for anatomical categories (tract)
3. Grayscale for methodological comparisons where a binary color pair is not needed

**Figure conventions:**
- **No suptitles**: Do not add figure-level titles (`fig.suptitle()`)
- **No subplot titles**: Do not add individual panel titles (`ax.set_title()`) — these are added manually during figure assembly
- **No panel labels**: Do not add subplot panel labels (a, b, c, ...) — these are added manually during figure assembly
- **Units in square brackets**: Use square brackets for units in axis labels (e.g., `Radius [μm]`, not `Radius (μm)`)
- **Radius notation**: Use `$\bar{r}$` for arithmetic mean radius, `$r_{\mathrm{MRI}}$` for effective MRI-visible radius

**Usage in plot scripts:**
```python
from axonometry import get_plot_settings

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
# For colored binary comparisons (scatter plots, error bars):
binary_a = settings.colors['binary_a']           # '#D94A4A' (red)
binary_b = settings.colors['binary_b']           # '#4A90D9' (blue)

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

# Figure DPI
plt.savefig(output_file, dpi=settings.figure['dpi'], bbox_inches='tight')
```

**Settings file structure** (`config/plot_settings.yaml`):
- `colors`: Species (human, rat), condition (sham, tbi), examples (example_1/2/3), binary comparisons (category_a/b)
- `markers`: Population markers (cc, cg, default)
- `figure`: DPI, default figure size, suptitle/subplot_titles flags
- `fonts`: label_size (14), tick_size (12), title_size (16), legend_size (11)
- `grid`: enabled (false), alpha
- `frame`: box (true), linewidth - controls plot border
- `error_bars`: Capsize, thickness, linewidth, alpha
- `histogram`: Bin count, alpha, edge color
- `line`: Linewidth, marker size, fill alpha
