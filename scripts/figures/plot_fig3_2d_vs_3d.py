"""
Compare 2D slice-based vs 3D skeleton-based radius distributions.

Uses the new canonical data format (per-instance flat tables from
compute_2d_slice_profiles.py and compute_3d_axon_profiles.py).

Creates a 2×2 panel figure:
(a) PDF: 3D reference line + 2D IQR envelope, with r̄ and r_MRI markers
(b) Wasserstein distances: within-ROI (sampling error) vs between-ROI (biological)
(c) Arithmetic mean radius scatter plot (2D vs 3D) for all samples
(d) Effective radius scatter plot (2D vs 3D) for all samples

Usage:
  python plot_fig3_2d_vs_3d.py \\
      --data-dir data/processed/rat/LM \\
      --output fig/main/distribution_2d_vs_3d_comparison.svg
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats

from axonometry import compute_r_eff, get_plot_settings, style_axis
from axonometry.io import load_2d_profiles, load_3d_profiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()

# ── Constants ──────────────────────────────────────────────────────────────
MIN_AXON_COUNT = 30       # Min axons per slice for valid statistics
BIN_WIDTH_UM = 0.02       # Histogram bin width in μm
MAX_RADIUS_UM = 5.0       # Max radius for histograms


# ── Data loading ───────────────────────────────────────────────────────────


def compute_per_slice_stats(data: Dict) -> Dict:
    """Compute per-slice r̄ and r_eff from filtered 2D data."""
    radii = data['radii']
    slices = data['slice_index']
    n_slices = data['n_slices']

    r_arith_per_slice = []
    r_eff_per_slice = []
    counts_per_slice = []

    for z in range(n_slices):
        r_z = radii[slices == z]
        n = len(r_z)
        if n < MIN_AXON_COUNT:
            continue
        counts_per_slice.append(n)
        r_arith_per_slice.append(np.mean(r_z))
        r_eff_per_slice.append(compute_r_eff(r_z))

    return {
        'r_arith': np.array(r_arith_per_slice),
        'r_eff': np.array(r_eff_per_slice),
        'counts': np.array(counts_per_slice),
        'n_valid': len(counts_per_slice),
    }


def compute_per_slice_pdfs(data: Dict) -> np.ndarray:
    """Compute per-slice PDFs (n_valid_slices × n_bins)."""
    bin_centers, bin_edges, bin_width = make_bins()
    radii = data['radii']
    slices = data['slice_index']
    n_slices = data['n_slices']

    pdfs = []
    for z in range(n_slices):
        r_z = radii[slices == z]
        if len(r_z) < MIN_AXON_COUNT:
            continue
        hist, _ = np.histogram(r_z, bins=bin_edges)
        total = hist.sum()
        if total > 0:
            pdfs.append(hist / (total * bin_width))

    return np.array(pdfs) if pdfs else np.empty((0, len(bin_centers)))


def load_3d_radii(npz_path: Path) -> np.ndarray:
    """Load pooled 3D radii with endpoint trimming."""
    return load_3d_profiles(npz_path)['all_radii_um']



# ── File matching ──────────────────────────────────────────────────────────

def find_matching_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Find matching 2D slice / 3D axon profile pairs.

    Returns list of (slice_file, axon_file, sample_name) tuples.
    """
    pairs = []
    for sf in sorted(data_dir.glob("*_myelin_slice_profiles.npz")):
        stem = sf.stem.replace('_slice_profiles', '')  # e.g. sham_25_ipsi_cc_myelin
        af = data_dir / f"{stem}_axon_profiles.npz"

        # Extract sample name (e.g. sham_25_ipsi_CC)
        parts = stem.replace('_myelin', '').rsplit('_', 1)  # ['sham_25_ipsi', 'cc']
        base_name = parts[0] if len(parts) == 2 else stem
        pop = parts[1].upper() if len(parts) == 2 else ''
        sample_name = f"{base_name}_{pop}" if pop else base_name

        if not af.exists():
            logger.warning(f"  No 3D file for {sf.name}, skipping")
            continue

        pairs.append((sf, af, sample_name))

    logger.info(f"Found {len(pairs)} matching 2D/3D pairs")
    return pairs


# ── Plotting ───────────────────────────────────────────────────────────────

def make_bins() -> Tuple[np.ndarray, np.ndarray, float]:
    """Standard bin centers, edges, and width for histogram comparison."""
    bin_centers = np.arange(BIN_WIDTH_UM / 2, MAX_RADIUS_UM, BIN_WIDTH_UM)
    bin_width = bin_centers[1] - bin_centers[0]
    bin_edges = np.concatenate([[bin_centers[0] - bin_width / 2],
                                 bin_centers + bin_width / 2])
    return bin_centers, bin_edges, bin_width



class HandlerLineInPatch(HandlerBase):
    """Legend handler drawing a median line centered inside a shaded band.

    Expects orig_handle to be a (patch, line) tuple; renders a full-height
    rectangle from the patch and a horizontal line (from the line) through it.
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        patch, line = orig_handle
        rect = Rectangle((xdescent, ydescent), width, height,
                         facecolor=patch.get_facecolor(), edgecolor='none',
                         transform=trans)
        ln = Line2D([xdescent, xdescent + width],
                    [ydescent + height / 2.0, ydescent + height / 2.0],
                    color=line.get_color(), linewidth=line.get_linewidth(),
                    linestyle=line.get_linestyle(), transform=trans)
        return [rect, ln]


def plot_pdf_panel(ax, slice_file: Path, axon_file: Path,
                   radius_type: str, x_max: float,
                   cache_2d: Optional[Dict] = None,
                   cache_3d: Optional[Dict] = None) -> None:
    """
    Panel (a): 3D and 2D pooled PDFs + r̄/r_MRI vertical markers.
    """
    bin_centers, bin_edges, bin_width = make_bins()

    # 2D: per-slice PDFs → median + IQR envelope
    data_2d = cache_2d[slice_file] if cache_2d else load_2d_profiles(slice_file, radius_type)
    pdfs_2d = compute_per_slice_pdfs(data_2d)
    stats_2d = compute_per_slice_stats(data_2d)

    if len(pdfs_2d) == 0:
        ax.text(0.5, 0.5, 'No valid 2D data', ha='center', va='center',
                transform=ax.transAxes)
        return

    pdf_2d_median = np.median(pdfs_2d, axis=0)
    pdf_2d_lo = np.percentile(pdfs_2d, 2.5, axis=0)
    pdf_2d_hi = np.percentile(pdfs_2d, 97.5, axis=0)

    # 3D: pooled PDF
    radii_3d = cache_3d[axon_file] if cache_3d else load_3d_radii(axon_file)
    hist_3d, _ = np.histogram(radii_3d, bins=bin_edges)
    pdf_3d = hist_3d / (hist_3d.sum() * bin_width) if hist_3d.sum() > 0 else hist_3d * 0.0

    # Crop to x_max
    mask = bin_centers <= x_max
    x = bin_centers[mask]

    # Colors: one per dimensionality (matching binary_a/b from variability stats)
    color_2d = settings.colors['binary_a']   # Sand/tan
    color_3d = settings.colors['binary_b']   # Dusty teal
    vline_lw = settings.line['linewidth']

    # 3D: solid PDF line
    ax.plot(x, pdf_3d[mask], color=color_3d, linewidth=settings.line['linewidth'],
            linestyle='-')
    # 2D: median line + central 95% shaded envelope across slices
    ax.fill_between(x, pdf_2d_lo[mask], pdf_2d_hi[mask], alpha=0.3, color=color_2d)
    ax.plot(x, pdf_2d_median[mask], color=color_2d, linewidth=settings.line['linewidth'],
            linestyle='-')

    # 3D summary statistics (pooled)
    r_arith_3d = np.mean(radii_3d)
    r_eff_3d = compute_r_eff(radii_3d)

    # 2D summary statistics (per-slice median + IQR)
    r_arith_2d_med = np.median(stats_2d['r_arith'])
    r_arith_2d_lo = np.percentile(stats_2d['r_arith'], 25)
    r_arith_2d_hi = np.percentile(stats_2d['r_arith'], 75)
    valid_reff = stats_2d['r_eff'][~np.isnan(stats_2d['r_eff'])]
    r_eff_2d_med = np.median(valid_reff) if len(valid_reff) else np.nan
    r_eff_2d_lo = np.percentile(valid_reff, 25) if len(valid_reff) else np.nan
    r_eff_2d_hi = np.percentile(valid_reff, 75) if len(valid_reff) else np.nan

    # r̄ markers (dotted): 3D line + 2D median line
    ax.axvline(r_arith_3d, color=color_3d, linewidth=vline_lw, linestyle=':', alpha=0.9)
    ax.axvline(r_arith_2d_med, color=color_2d, linewidth=vline_lw, linestyle=':', alpha=0.9)

    # r_MRI markers (dashed): 3D line + 2D median line
    ax.axvline(r_eff_3d, color=color_3d, linewidth=vline_lw, linestyle='--', alpha=0.9)
    if not np.isnan(r_eff_2d_med):
        ax.axvline(r_eff_2d_med, color=color_2d, linewidth=vline_lw, linestyle='--', alpha=0.9)

    # Legend
    handles, labels = [], []

    # 3D entries
    handles.append(Line2D([0], [0], color=color_3d, linewidth=1.5, linestyle='-'))
    labels.append('3D PDF')
    handles.append(Line2D([0], [0], color=color_3d, linewidth=vline_lw, linestyle=':'))
    labels.append(r'3D $\bar{r}$')
    handles.append(Line2D([0], [0], color=color_3d, linewidth=vline_lw, linestyle='--'))
    labels.append(r'3D $r_{\mathrm{MRI}}$')

    # 2D entries
    handles.append((Patch(facecolor=color_2d, alpha=0.3),
                    Line2D([0], [0], color=color_2d, linewidth=1.5, linestyle='-')))
    labels.append('2D median PDF\n(+ central 95%)')
    handles.append(Line2D([0], [0], color=color_2d, linewidth=vline_lw, linestyle=':'))
    labels.append(r'2D $\bar{r}$ median')
    handles.append(Line2D([0], [0], color=color_2d, linewidth=vline_lw, linestyle='--'))
    labels.append(r'2D $r_{\mathrm{MRI}}$ median')

    style_axis(ax, xlabel='Axon radius [μm]', ylabel='Probability density [μm⁻¹]')
    ax.legend(handles, labels, loc='upper right',
              fontsize=settings.fonts['legend_size'] - 1,
              framealpha=0.9,
              handler_map={tuple: HandlerLineInPatch()})
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_wasserstein_panel(ax, pairs: List[Tuple[Path, Path, str]],
                           radius_type: str,
                           cache_2d: Optional[Dict] = None,
                           cache_3d: Optional[Dict] = None) -> None:
    """
    Panel (b): Within-ROI vs between-ROI Wasserstein distances.
    """
    bin_centers, bin_edges, bin_width = make_bins()

    within_distances = []
    roi_cdfs_3d = []

    for sf, af, sn in pairs:
        # 3D CDF
        r3d = cache_3d[af] if cache_3d else load_3d_radii(af)
        if len(r3d) == 0:
            continue
        hist_3d, _ = np.histogram(r3d, bins=bin_edges)
        cdf_3d = np.cumsum(hist_3d) / hist_3d.sum()
        roi_cdfs_3d.append(cdf_3d)

        # Per-slice 2D CDFs → Wasserstein vs 3D
        data_2d = cache_2d[sf] if cache_2d else load_2d_profiles(sf, radius_type)
        for z in range(data_2d['n_slices']):
            r_z = data_2d['radii'][data_2d['slice_index'] == z]
            if len(r_z) < MIN_AXON_COUNT:
                continue
            hist_z, _ = np.histogram(r_z, bins=bin_edges)
            total = hist_z.sum()
            if total == 0:
                continue
            cdf_2d = np.cumsum(hist_z) / total
            w = np.sum(np.abs(cdf_2d - cdf_3d)) * bin_width
            within_distances.append(w)

    # Between-ROI distances (pairwise 3D)
    between_distances = []
    for i in range(len(roi_cdfs_3d)):
        for j in range(i + 1, len(roi_cdfs_3d)):
            w = np.sum(np.abs(roi_cdfs_3d[i] - roi_cdfs_3d[j])) * bin_width
            between_distances.append(w)

    within_distances = np.array(within_distances)
    between_distances = np.array(between_distances)

    logger.info(f"  Within: {len(within_distances)} slices, "
                f"Between: {len(between_distances)} ROI pairs")

    # Violin plot
    parts = ax.violinplot([within_distances, between_distances], positions=[1, 2],
                           showmeans=False, showmedians=True, widths=0.7)

    colors = [settings.colors['category_a_violin'], settings.colors['category_b_violin']]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.8)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    for key in ('cbars', 'cmins', 'cmaxes'):
        parts[key].set_color('gray')

    # Jittered points
    rng = np.random.default_rng(42)
    n_show = min(500, len(within_distances))
    idx = rng.choice(len(within_distances), n_show, replace=False) if len(within_distances) > n_show else np.arange(len(within_distances))
    jitter = rng.uniform(-0.15, 0.15, len(idx))
    ax.scatter(1 + jitter, within_distances[idx], alpha=0.4, s=5, color='#808080', zorder=0)

    jitter = rng.uniform(-0.15, 0.15, len(between_distances))
    ax.scatter(2 + jitter, between_distances, alpha=0.5, s=10, color='#404040', zorder=0)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Sampling error\n(intra-ROI 2D ↔ 3D)',
                         'Anat. variability\n(inter-ROI 3D)'],
                        fontsize=settings.fonts['tick_size'] - 1)
    style_axis(ax, ylabel='Wasserstein distance [μm]')
    ax.set_ylim(bottom=0)
    ax.set_box_aspect(1)


def plot_scatter_panel(ax, all_metrics: List[Tuple[Dict, Dict, str]],
                       metric: str, mc_results: Optional[Dict] = None) -> None:
    """
    Panel (c)/(d): 2D vs 3D scatter for r̄ or r_eff.
    """
    x_vals, y_vals, yerr_lo, yerr_hi = [], [], [], []

    for m2d, m3d, _ in all_metrics:
        x_vals.append(m3d[metric])
        y_vals.append(m2d[f'{metric}_median'])
        yerr_lo.append(m2d[f'{metric}_median'] - m2d[f'{metric}_lo'])
        yerr_hi.append(m2d[f'{metric}_hi'] - m2d[f'{metric}_median'])

    color = settings.colors['single_line']
    err = settings.error_bars

    ax.errorbar(x_vals, y_vals, yerr=[yerr_lo, yerr_hi],
                fmt='o', color=color, ecolor=to_rgba(color, 0.7),
                markersize=8,
                capsize=err['capsize'], capthick=err['capthick'],
                elinewidth=err['linewidth'],
                markerfacecolor=to_rgba(color, 0.3), markeredgecolor=color,
                markeredgewidth=1.5)

    if len(x_vals) < 2:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
        return

    all_vals = x_vals + y_vals
    lo = min(all_vals) * 0.95
    hi = max(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5,
            linewidth=settings.line['linewidth'], zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)

    if metric == 'r_arith':
        xlabel = r'$\bar{r}$ (3D) [μm]'
        ylabel = r'$\bar{r}$ (2D) [μm]'
        tick_step = 0.05
        tick_start = np.ceil(lo / tick_step) * tick_step
        tick_end = np.floor(hi / tick_step) * tick_step
        ticks = np.arange(tick_start, tick_end + tick_step / 2, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    else:
        xlabel = r'$r_{\mathrm{MRI}}$ (3D) [μm]'
        ylabel = r'$r_{\mathrm{MRI}}$ (2D) [μm]'
        # Match x tick locations to y (limits are equal)
        ax.set_xticks(ax.get_yticks())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    style_axis(ax, xlabel=xlabel, ylabel=ylabel)

    # Normalized mean bias: mean((2D - 3D) / 3D) * 100%
    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    nmb = np.mean((y_arr - x_arr) / x_arr) * 100
    bias_str = f'Bias = {nmb:+.1f}%'

    # Correlation annotation
    if mc_results and metric in mc_results:
        r_mean = mc_results[metric]['r_mean']
        r_std = mc_results[metric]['r_std']
        p_val = mc_results[metric].get('p_value', None)
        if p_val is not None:
            p_str = 'p < 0.001' if p_val < 0.001 else f'p = {p_val:.3f}'
            text = f'{bias_str}\n$R$ = {r_mean:.2f} ± {r_std:.2f}\n{p_str}'
        else:
            text = f'{bias_str}\n$R$ = {r_mean:.2f} ± {r_std:.2f}'
        ax.text(0.95, 0.05, text, transform=ax.transAxes,
                fontsize=settings.fonts['legend_size'], ha='right', va='bottom')
    else:
        r, p = stats.pearsonr(x_vals, y_vals)
        p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        ax.text(0.95, 0.05, f'{bias_str}\n$R$ = {r:.2f}, {p_str}', transform=ax.transAxes,
                fontsize=settings.fonts['legend_size'], ha='right', va='bottom')


# ── Monte Carlo ────────────────────────────────────────────────────────────

def run_monte_carlo(pairs: List[Tuple[Path, Path, str]],
                    radius_type: str, n_iterations: int = 10000,
                    seed: int = 42,
                    cache_2d: Optional[Dict] = None,
                    cache_3d: Optional[Dict] = None) -> Dict:
    """
    Monte Carlo correlation: randomly pick one slice per ROI, compute R.
    """
    rng = np.random.default_rng(seed)

    roi_data = []   # per-slice stats for each ROI
    x_3d_arith = []
    x_3d_reff = []

    for sf, af, sn in pairs:
        data_2d = cache_2d[sf] if cache_2d else load_2d_profiles(sf, radius_type)
        per_slice = compute_per_slice_stats(data_2d)

        if per_slice['n_valid'] < 1:
            logger.warning(f"  Skipping {sn} for MC - no valid slices")
            continue

        r3d = cache_3d[af] if cache_3d else load_3d_radii(af)
        if len(r3d) == 0:
            continue

        roi_data.append(per_slice)
        x_3d_arith.append(np.mean(r3d))
        x_3d_reff.append(compute_r_eff(r3d))

    x_3d_arith = np.array(x_3d_arith)
    x_3d_reff = np.array(x_3d_reff)
    n_rois = len(roi_data)

    logger.info(f"Monte Carlo: {n_rois} ROIs, {n_iterations} iterations")

    r_arith_iter = np.empty(n_iterations)
    r_reff_iter = np.empty(n_iterations)

    for it in range(n_iterations):
        y_arith = np.array([roi['r_arith'][rng.integers(roi['n_valid'])] for roi in roi_data])
        y_reff = np.array([roi['r_eff'][rng.integers(roi['n_valid'])] for roi in roi_data])
        r_arith_iter[it], _ = stats.pearsonr(x_3d_arith, y_arith)
        r_reff_iter[it], _ = stats.pearsonr(x_3d_reff, y_reff)

    # Permutation p-value
    rng_perm = np.random.default_rng(seed + 100)
    r_arith_obs = np.mean(r_arith_iter)
    r_reff_obs = np.mean(r_reff_iter)

    n_perm = n_iterations
    p_arith_count = 0
    p_reff_count = 0

    for _ in range(n_perm):
        perm = rng_perm.permutation(n_rois)
        y_arith = np.array([roi['r_arith'][rng_perm.integers(roi['n_valid'])] for roi in roi_data])
        y_reff = np.array([roi['r_eff'][rng_perm.integers(roi['n_valid'])] for roi in roi_data])
        r_a, _ = stats.pearsonr(x_3d_arith[perm], y_arith)
        r_e, _ = stats.pearsonr(x_3d_reff[perm], y_reff)
        if r_a >= r_arith_obs:
            p_arith_count += 1
        if r_e >= r_reff_obs:
            p_reff_count += 1

    return {
        'r_arith': {
            'r_mean': r_arith_obs,
            'r_std': np.std(r_arith_iter),
            'p_value': p_arith_count / n_perm,
        },
        'r_eff': {
            'r_mean': r_reff_obs,
            'r_std': np.std(r_reff_iter),
            'p_value': p_reff_count / n_perm,
        },
        'n_iterations': n_iterations,
        'n_rois': n_rois,
        'seed': seed,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare 2D vs 3D radius distributions (new canonical data format)')

    parser.add_argument('--data-dir', type=Path, default=Path('data/processed/rat/lm'),
                        help='Directory containing slice and axon profiles')
    parser.add_argument('--output', type=Path,
                        default=Path('fig/main/2d_vs_3d_distribution_comparison.svg'),
                        help='Output figure path')
    parser.add_argument('--radius-type', type=str, default='minor',
                        choices=['circular', 'minor'], help='Radius type to use')
    parser.add_argument('--x-max', type=float, default=1.0,
                        help='Max x-axis for PDF panel')
    parser.add_argument('--n-iterations', type=int, default=10000,
                        help='Monte Carlo iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--representative', type=str, default=None,
                        help='Stem of representative sample for panel (a), '
                             'e.g. "sham_25_ipsi_cg_myelin" (default: auto-select)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("2D vs 3D Distribution Comparison (v2 — canonical data)")
    logger.info("=" * 80)

    # Find all matching pairs
    all_pairs = find_matching_pairs(args.data_dir)
    if not all_pairs:
        logger.error("No matching 2D/3D pairs found!")
        return

    # Pre-load all 2D and 3D data (avoid redundant I/O and filtering)
    cache_2d = {}
    cache_3d = {}
    for sf, af, sn in all_pairs:
        cache_2d[sf] = load_2d_profiles(sf, args.radius_type)
        cache_3d[af] = load_3d_radii(af)

    # Pick representative sample for panel (a)
    if args.representative:
        rep_sf = args.data_dir / f"{args.representative}_slice_profiles.npz"
        rep_af = args.data_dir / f"{args.representative}_axon_profiles.npz"
        if not rep_sf.exists() or not rep_af.exists():
            logger.error(f"Representative sample files not found: {args.representative}")
            return
        # Ensure representative files are in the cache
        if rep_sf not in cache_2d:
            cache_2d[rep_sf] = load_2d_profiles(rep_sf, args.radius_type)
        if rep_af not in cache_3d:
            cache_3d[rep_af] = load_3d_radii(rep_af)
    else:
        # Auto: pick sample with highest mean axon count per slice (using filtered data)
        best_sf, best_af, best_name = None, None, None
        best_mean_count = 0
        for sf, af, sn in all_pairs:
            n_radii = len(cache_2d[sf]['radii'])
            n_slices = cache_2d[sf]['n_slices']
            mean_count = n_radii / max(n_slices, 1)
            if mean_count > best_mean_count:
                best_mean_count = mean_count
                best_sf, best_af, best_name = sf, af, sn
        rep_sf, rep_af = best_sf, best_af
        logger.info(f"Auto-selected representative: {best_name} "
                    f"(mean {best_mean_count:.0f} filtered instances/slice)")

    # Compute 2D/3D metrics for all pairs
    all_metrics = []
    for sf, af, sn in all_pairs:
        per_slice = compute_per_slice_stats(cache_2d[sf])
        if per_slice['n_valid'] < 1:
            continue

        r3d = cache_3d[af]
        if len(r3d) == 0:
            continue

        valid_reff = per_slice['r_eff'][~np.isnan(per_slice['r_eff'])]

        m2d = {
            'r_arith_median': np.median(per_slice['r_arith']),
            'r_arith_lo': np.percentile(per_slice['r_arith'], 25),
            'r_arith_hi': np.percentile(per_slice['r_arith'], 75),
            'r_eff_median': np.median(valid_reff) if len(valid_reff) else np.nan,
            'r_eff_lo': np.percentile(valid_reff, 25) if len(valid_reff) else np.nan,
            'r_eff_hi': np.percentile(valid_reff, 75) if len(valid_reff) else np.nan,
        }
        m3d = {
            'r_arith': np.mean(r3d),
            'r_eff': compute_r_eff(r3d),
        }
        all_metrics.append((m2d, m3d, sn))

    logger.info(f"Metrics computed for {len(all_metrics)} pairs")

    # Monte Carlo
    logger.info(f"\nRunning Monte Carlo ({args.n_iterations} iterations)...")
    mc_results = run_monte_carlo(all_pairs, args.radius_type,
                                  args.n_iterations, args.seed,
                                  cache_2d=cache_2d, cache_3d=cache_3d)
    logger.info(f"  r̄:    R = {mc_results['r_arith']['r_mean']:.3f} ± "
                f"{mc_results['r_arith']['r_std']:.3f}")
    logger.info(f"  r_MRI: R = {mc_results['r_eff']['r_mean']:.3f} ± "
                f"{mc_results['r_eff']['r_std']:.3f}")

    # ── Figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    logger.info("\nPanel (a): PDF stability...")
    plot_pdf_panel(axes[0, 0], rep_sf, rep_af, args.radius_type, args.x_max,
                   cache_2d=cache_2d, cache_3d=cache_3d)

    logger.info("Panel (b): Wasserstein distances...")
    plot_wasserstein_panel(axes[0, 1], all_pairs, args.radius_type,
                           cache_2d=cache_2d, cache_3d=cache_3d)

    logger.info("Panel (c): r̄ scatter...")
    plot_scatter_panel(axes[1, 0], all_metrics, 'r_arith', mc_results)

    logger.info("Panel (d): r_eff scatter...")
    plot_scatter_panel(axes[1, 1], all_metrics, 'r_eff', mc_results)

    plt.tight_layout(w_pad=2.5, h_pad=2.5)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"\nSaved to {args.output}")

    # Save metadata
    per_roi = []
    for m2d, m3d, sn in all_metrics:
        r_arith_ratio = m2d['r_arith_median'] / m3d['r_arith'] if m3d['r_arith'] > 0 else float('nan')
        r_eff_ratio = m2d['r_eff_median'] / m3d['r_eff'] if m3d['r_eff'] > 0 else float('nan')
        per_roi.append({
            'name': sn,
            '2d_r_arith_median': round(float(m2d['r_arith_median']), 4),
            '3d_r_arith': round(float(m3d['r_arith']), 4),
            'r_arith_ratio_2d_over_3d': round(float(r_arith_ratio), 4),
            '2d_r_eff_median': round(float(m2d['r_eff_median']), 4),
            '3d_r_eff': round(float(m3d['r_eff']), 4),
            'r_eff_ratio_2d_over_3d': round(float(r_eff_ratio), 4),
        })
    # Sort by largest r_eff difference
    per_roi.sort(key=lambda x: abs(1 - x['r_eff_ratio_2d_over_3d']), reverse=True)

    meta = {
        'n_pairs': len(all_metrics),
        'radius_type': args.radius_type,
        'filters': {
            'min_axon_count_per_slice': MIN_AXON_COUNT,
        },
        'per_roi': per_roi,
        'monte_carlo': {
            'n_iterations': mc_results['n_iterations'],
            'n_rois': mc_results['n_rois'],
            'seed': mc_results['seed'],
            'r_arith': mc_results['r_arith'],
            'r_eff': mc_results['r_eff'],
        },
    }
    json_path = args.output.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {json_path}")


if __name__ == '__main__':
    main()
