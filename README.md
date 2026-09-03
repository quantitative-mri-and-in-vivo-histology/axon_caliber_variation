# 3D Histology Validates 2D Histology for Axon Radius Distributions and Conduction Velocities

Laurin Mordhorst, Nikolaus Weiskopf, Markus Morawski, Siawoosh Mohammadi

**Preprint:** [https://doi.org/10.64898/2026.03.25.714137](https://doi.org/10.64898/2026.03.25.714137)

This repository contains the analysis code and figure generation scripts for the above manuscript. It is intended for reproducing the published results.

## Data

### Rat brain 3D segmentation data

Download the 10 labeled volumes of myelinated axons from:

> Sierra, A., Abdollahzadeh, A., Belevich, I., Jokitalo, E., & Tohka, J. (2021). *Segmentation of brain ultrastructures in 3D electron microscopy.* [doi:10.23729/BAD417CA-553F-4FA6-AE0A-22EDDD29A230](https://etsin.fairdata.fi/dataset/f8ccc23a-1f1a-4c98-86b7-b63652a809c3)

Place the `.mat` files in `data/source/rat/`, organized by condition and hemisphere:

```
data/source/rat/
├── Sham_2_ipsi/LM_2_ipsi_myelinated_axons.mat
├── Sham_2_contra/LM_2_contra_myelinated_axons.mat
├── Sham_24_ipsi/LM_24_ipsi_myelinated_axons.mat
├── Sham_24_contra/LM_24_contra_myelinated_axons.mat
├── Sham_28_ipsi/LM_28_ipsi_myelinated_axons.mat
├── Sham_28_contra/LM_28_contra_myelinated_axons.mat
├── TBI_25_ipsi/LM_25_ipsi_myelinated_axons.mat
├── TBI_25_contra/LM_25_contra_myelinated_axons.mat
├── TBI_49_ipsi/LM_49_ipsi_myelinated_axons.mat
└── TBI_49_contra/LM_49_contra_myelinated_axons.mat
```

### Human corpus callosum 2D light microscopy data

Download the pre-processed histogram data from:

> Mordhorst, L., et al. (2025). *Ex-vivo dataset for validation of MRI-based axon radius mapping.* Zenodo. [doi:10.5281/zenodo.17431227](https://zenodo.org/records/17431227)

Place the files in `data/raw/human/lm/`:

```
data/raw/human/lm/
├── desc-binCenters_radii.tsv
├── desc-binEdges_radii.tsv
├── desc-countsCircularEq_radii.tsv
├── desc-countsMajorAxis_radii.tsv
├── desc-countsMinorAxis_radii.tsv
├── desc-polygons_roicoords.tsv
├── roiinfo.tsv
├── sub-ev01/
│   ├── sub-ev01.png
│   └── sub-ev01_label-cc_mask.nii.gz
└── sub-ev02/
    ├── sub-ev02.png
    └── sub-ev02_label-cc_mask.nii.gz
```

## Setup

Requires Python >= 3.10.

```bash
pip install -e ".[full]"
```

A `.devcontainer` configuration is provided for VS Code / GitHub Codespaces with all dependencies pre-installed.

## Reproducing figures

### Processing pipeline

Figures depend on preprocessed data. Run the pipeline stages in order:

1. **Preparation** (`scripts/preparation/`)
   - `extract_rois.py` — crops each population's ROI from the raw `.mat` volumes using pre-computed `*_population_rois.json` bounding boxes, permutes axes so Z = fiber direction, and writes OME-Zarr volumes for both the labeled segmentation and companion grayscale data (if available as `.h5` files alongside the `.mat` sources)

2. **Processing** (`scripts/processing/`)
   - `compute_2d_slice_profiles.py` — extracts per-instance regionprops (area, eccentricity, radii, ...) for every cross-sectional slice along the fiber axis (~1 hour for the whole dataset on a dual 36-core Intel Xeon Gold 6254 / 750 GB RAM server)
   - `compute_3d_axon_profiles.py` — traces each axon along its 3D skeleton, measuring radius variation along the bundle (~24 hours for the whole dataset on the same server)
   - `extract_representative_3d_axons.py` — selects representative axons (low, mid, high CoV) for visualization in Fig 2

3. **Figures** — generate manuscript figures from the processed data (`scripts/figures/`)

### Figure scripts

Figure generation is split into two stages so the plotted numbers are available as
source data:

1. **`gen_data_fig*.py`** compute the numerical results and write them as CSV
   source-data files under `data/figures/`.
2. **`plot_fig*.py`** read those CSVs and render the figure SVGs (under `fig/`).

Run the pair, e.g. for Figure 3:

```bash
python scripts/figures/gen_data_fig3_2d_vs_3d.py   # -> data/figures/fig_3_*.csv
python scripts/figures/plot_fig3_2d_vs_3d.py       # -> fig/main/fig_3.svg
```

Main figures (each has a `gen_data_*` + `plot_*` pair):

- **Fig 2** — `…_fig2ab_radius_profiles.py` (panels a–b) and `…_fig2ce_variation_stats.py` (panels c–e)
- **Fig 3** — `…_fig3_2d_vs_3d.py`
- **Fig 4** — `…_fig4_slice_vs_random.py`
- **Fig 5** — `…_fig5_sample_size.py`
- **Fig 6** — `…_fig6_parametric_fits.py`

Supplementary figures:

- **S1** — `…_figS1_radius_std_vs_mean.py` (radius standard deviation vs mean radius)
- **S2** — `…_figS2_myelin_thickness_sensitivity.py` (myelin-thickness model sensitivity of the conduction-velocity reduction)
- **S3** — circular-radius variant of Fig 3:
  ```bash
  python scripts/figures/gen_data_fig3_2d_vs_3d.py --radius-type circular --prefix fig_s3
  python scripts/figures/plot_fig3_2d_vs_3d.py --prefix fig_s3 --output fig/supplementary/fig_s3.svg
  ```
- **S4** — `plot_figS4_dists_per_panel.py` (per-panel distribution fits; reuses the Fig 6 fitting)

Source data for all figures lives in `data/figures/` (the same CSVs provided as
Supplementary Data). Use `--help` for per-script options.

## License

This project is licensed under the MIT License (see LICENSE file).
