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
- Single-axon caliber fluctuations
- Ensemble-effective radius behavior along white matter tracts
- Comparisons between sham and TBI model groups
- Summary statistics and spectral features derived from gamma-decomposed axon morphologies

🗂 Repository Structure

**data/raw/LM/** – Expected location for raw .mat files from the FAIR dataset (10 volumes: LM_2_ipsi, LM_2_contra, LM_24_ipsi, LM_24_contra, LM_25_ipsi, LM_25_contra, LM_28_ipsi, LM_28_contra, LM_49_ipsi, LM_49_contra)

**data/processed/** – Cleaned and separated tract populations (CG and CC) ready for analysis

**fig/** – Output folder for generated figures

**src/** – Python source code for preprocessing, analysis, and visualization

📈 Main Components

**Preprocessing Pipeline**
- Extract and separate cingulum (CG) and corpus callosum (CC) populations from raw volumes
- Filter axons based on:
  - Minimum length threshold (remove short fragments)
  - Slice persistence (axons must appear in sufficient fraction of slices)
  - Orientation filtering (remove axons with excessive bending that creates skewed 2D projections)
- Generate per-slice radius histograms for both circular-equivalent and minor-axis radii

**Analysis Pipeline**
- Gamma decomposition for position-based caliber analysis
- FFT-based power spectral summaries
- Effective radius (r_eff) calculations along tract length
- Statistical comparisons between groups (TBI vs sham, ipsi vs contra)

**Visualization**
- Slice-wise and tract-level effective radius plots
- Error bar summaries and distributions
- 2D/3D comparison plots
- Spectral feature visualizations

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
   Run preprocessing scripts to separate CG and CC populations, apply quality filters, and generate cleaned intermediate datasets

3. **Analyze caliber variability**
   Compute morphological features, effective radii, and statistical summaries

4. **Generate figures**
   Create publication-ready visualizations
