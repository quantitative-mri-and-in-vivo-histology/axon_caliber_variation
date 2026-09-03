"""
Plot Figure S1 (radius standard deviation vs mean radius) from source data.

Reads data/figures/fig_s1_std_vs_radius.csv (written by
gen_data_figS1_radius_std_vs_mean.py) and draws the binned median + IQR curve.

Usage:
    python scripts/figures/plot_figS1_radius_std_vs_mean.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from axonometry import get_plot_settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

settings = get_plot_settings()


def main():
    parser = argparse.ArgumentParser(description='Plot Fig S1 from source data')
    parser.add_argument('--data-dir', type=Path, default=Path('data/figures'))
    parser.add_argument('--output', type=Path, default=Path('fig/supplementary/fig_s1.svg'))
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / 'fig_s1_std_vs_radius.csv')

    width_in = 62 / 25.4  # 62 mm, square
    fig, ax = plt.subplots(figsize=(width_in, width_in))
    label_size, tick_size, linewidth, marker_size = 8, 7, 1.2, 3
    color = settings.colors['single_line']

    ax.plot(df['mean_radius_um'], df['std_median'], color=color, linestyle='-',
            linewidth=linewidth, marker='o', markersize=marker_size)
    ax.fill_between(df['mean_radius_um'], df['std_q25'], df['std_q75'], color=color, alpha=0.2)

    ax.set_xlabel('Mean (along-axon radius) [μm]', fontsize=label_size)
    ax.set_ylabel('Std (along-axon radius) [μm]', fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.set_box_aspect(1)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=settings.figure['dpi'], bbox_inches='tight')
    plt.close()
    logger.info(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
