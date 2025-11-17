CLAUDE.md
📦 Project Overview

This repository contains Python-based analysis and figure generation code for the manuscript on axon caliber variability and the effective axon radius, based on publicly available high-resolution axon segmentation data (Abdollazadeh et al., Nat. Commun., 2025; originally from a 2021 dataset).

The dataset consists of 10 labeled 3D volumes of myelinated axons from 5 rats:
- 2 rats with TBI (traumatic brain injury)
- 3 rats with sham operation
- Each rat has both ipsilateral and contralateral hemisphere volumes

Each volume contains two white matter tract populations:
- **Cingulum (CG)**: Aligned primarily with one volume axis
- **Corpus Callosum (CC)**: Aligned primarily with a different volume axis

The focus is on characterizing and visualizing:
- Axon radius distributions in white matter tracts
- Effective MRI-visible radius (r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)) along tract length
- Per-slice radius profiles and global pooled statistics
- Comparisons between sham and TBI groups, and between CC and CG populations

🗂 Repository Structure

**data/raw/LM/** – Expected location for raw .mat files from the FAIR dataset (10 volumes: LM_2_ipsi, LM_2_contra, LM_24_ipsi, LM_24_contra, LM_25_ipsi, LM_25_contra, LM_28_ipsi, LM_28_contra, LM_49_ipsi, LM_49_contra)

**data/processed/** – Cleaned and separated tract populations (CG and CC) ready for analysis

**fig/** – Output folder for generated figures

**src/** – Python source code for preprocessing, analysis, and visualization

📈 Main Components

**Preprocessing Pipeline** ([filter_axons_simple.py](src/filter_axons_simple.py))
- Extract and separate cingulum (CG) and corpus callosum (CC) populations from raw volumes
- Filter axons based on:
  - Minimum length threshold (remove short fragments)
  - Slice persistence (axons must appear in sufficient fraction of slices)
  - Orientation filtering (remove axons with excessive bending that creates skewed 2D projections)
- Identify top two populations by alignment axis (CC = most axons, CG = second most)
- Create axis-aligned volumes (permute axes so axons run along z-axis)
- Save separate HDF5 volumes for CC and CG populations
- Memory-efficient: processes slice-by-slice to handle large volumes

**Analysis Pipeline** ([analyze_effective_radius.py](src/analyze_effective_radius.py))
- Extract circular-equivalent radii per slice using regionprops
- Compute radius histograms (bin edges: 0-20 μm with 0.02 μm step)
- Calculate effective MRI-visible radius: r_eff = (⟨r⁶⟩/⟨r²⟩)^(1/4)
- Per-slice profiles showing radius variation along tract
- Global pooled effective radius from all slices combined
- Parallel processing support for faster computation
- Memory-efficient histogram storage

**Visualization**
- Effective radius profiles: per-slice values vs position along tract
- Global effective radius as reference line (dashed)
- Axon count per slice
- Publication-ready plots with proper labeling and units

▶️ Workflow

1. **Obtain raw data**
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

2. **Preprocess and extract tract populations**
   Run [filter_axons_simple.py](src/filter_axons_simple.py) to:
   - Load raw .mat file (downsampled for filtering)
   - Apply quality filters (alignment, straightness, extent)
   - Identify CC and CG populations by alignment axis
   - Create axis-aligned HDF5 volumes at full resolution

   Example:
   ```bash
   python src/filter_axons_simple.py \
       data/raw/LM/LM_25_ipsi_myelinated_axons.mat \
       data/processed/LM_25_ipsi \
       --downsample 4 \
       --min-alignment 0.7 \
       --min-extent 50 \
       --min-straightness 0.8
   ```

   Output: `data/processed/LM_25_ipsi/cc_aligned.h5` and `cg_aligned.h5`

3. **Analyze axon radii**
   Run [analyze_effective_radius.py](src/analyze_effective_radius.py) to:
   - Extract circular-equivalent radii per slice
   - Compute effective MRI-visible radius
   - Generate profile plots

   Example:
   ```bash
   python src/analyze_effective_radius.py \
       data/processed/LM_25_ipsi/cc_aligned.h5 \
       fig/LM_25_ipsi \
       --voxel-size 0.05 \
       --n-jobs -1
   ```

   Output: `fig/LM_25_ipsi/CC_effective_radius_profile.png`

4. **Compare populations and groups**
   Process all 10 volumes (both CC and CG for each) and compare:
   - TBI vs Sham
   - Ipsilateral vs Contralateral
   - CC vs CG populations

## Technical Details

**Voxel Size**
- Original data: 0.05 μm/voxel (isotropic)
- Filtering uses 4× downsampling (0.2 μm/voxel) for speed
- Output volumes are at full resolution (0.05 μm/voxel)

**Memory Efficiency**
- HDF5 streaming: slice-by-slice processing avoids loading full volumes
- Chunked storage: (100, 512, 512) chunks optimized for z-axis access
- Light compression: gzip level 1 for speed/size balance
- Histogram aggregation: fixed memory footprint regardless of data size

**Performance**
- Parallel processing: uses all CPU cores by default (`--n-jobs -1`)
- regionprops: efficient batch extraction of morphological properties
- Progress tracking: tqdm progress bars for all long operations

**Data Format**
- Input: MATLAB .mat files with labeled volumes
- Processing: HDF5 with chunked, compressed storage
- Output: PNG figures with 200 DPI resolution
