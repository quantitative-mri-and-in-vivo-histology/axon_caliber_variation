"""
Generate source data for Figure 2b (radius profiles of the representative axons).

Reads the precomputed representative-axon NPZ and writes the radius-vs-arc-length
profiles of the three axons as a CSV:

  fig_2b_radius_profiles.csv   columns: axon_index (0=low,1=mid,2=high CoV),
                               cov, arc_length_um, radius_um

Figure 2a is a 3D volume rendering (no underlying chart data), so it has no
source-data CSV. Plotting is done by plot_fig2ab_radius_profiles.py.

Usage:
    python scripts/figures/gen_data_fig2ab_radius_profiles.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate Fig 2b source data")
    parser.add_argument("--input", type=Path,
                        default=Path("data/processed/rat/lm/representative_axons.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return

    data = np.load(args.input, allow_pickle=True)
    arc_lengths = data["arc_lengths"]
    radii = data["radii"]
    cv = data["cv"]

    rows = []
    for i in range(len(cv)):
        al = np.asarray(arc_lengths[i], dtype=np.float64)
        r = np.asarray(radii[i], dtype=np.float64)
        for a, rr in zip(al, r):
            rows.append({"axon_index": i, "cov": round(float(cv[i]), 4),
                         "arc_length_um": float(a), "radius_um": float(rr)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "fig_2b_radius_profiles.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"Wrote {out} ({len(cv)} axons, {len(rows)} rows)")


if __name__ == "__main__":
    main()
