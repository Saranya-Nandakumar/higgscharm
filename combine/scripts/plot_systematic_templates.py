#!/usr/bin/env python3
"""Plot Up/Down shape-systematic templates vs nominal, per process, for a
combine histograms ROOT file (as written by create_datacards_ctag2d.py).

One page per shape systematic; one row per process that carries it (parsed
from the datacard's own shape rows, so this never has to know the systematic
list by hand). Top panel: nominal/Up/Down overlaid. Bottom panel: Up/Nom and
Down/Nom ratio per bin -- this is what actually shows a flat/dead variation
(e.g. the CMS_ctag2d Down-variation anomaly flagged in README_HcZZ.md).

Usage:
    python3 plot_systematic_templates.py --card datacard_mva_with_zx.txt \\
        --dir /path/to/outputs_dir [--out systematic_templates.pdf]
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def parse_shape_rows(card_path):
    """Return {syst_name: [process_names_with_flag_1]} and shapes-line info."""
    processes = []
    shape_rows = {}
    hist_pattern = None
    with open(card_path) as f:
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith("shapes"):
            parts = line.split()
            # shapes * * <file> <nom_pattern> <syst_pattern>
            hist_pattern = (parts[3], parts[4], parts[5])
        if line.strip().startswith("process") and not processes:
            parts = line.split()[1:]
            if parts and not parts[0].isdigit():
                processes = parts
    for line in lines:
        parts = line.split()
        if len(parts) < 2 + len(processes):
            continue
        if parts[1] in ("shape", "shape1", "shapeN2"):
            name = parts[0]
            flags = parts[2:2 + len(processes)]
            active = [p for p, fl in zip(processes, flags) if fl == "1"]
            shape_rows[name] = active
    return shape_rows, hist_pattern, processes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True, help="datacard filename (relative to --dir)")
    ap.add_argument("--dir", required=True, help="directory containing the datacard + ROOT shapes")
    ap.add_argument("--out", default="systematic_templates.pdf", help="output PDF filename (relative to --dir)")
    args = ap.parse_args()

    workdir = Path(args.dir).resolve()
    card_path = workdir / args.card
    if not card_path.exists():
        sys.exit(f"datacard not found: {card_path}")

    shape_rows, hist_pattern, processes = parse_shape_rows(card_path)
    if not shape_rows:
        sys.exit("No shape-systematic rows found in this datacard.")

    root_file = workdir / hist_pattern[0]
    if not root_file.exists():
        sys.exit(f"histograms file not found: {root_file}")

    import uproot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    f = uproot.open(root_file)

    def get_hist(name):
        if name not in f:
            return None
        h = f[name]
        vals = h.values()
        edges = h.axis().edges()
        return vals, edges

    out_pdf = workdir / args.out
    n_syst = len(shape_rows)
    print(f"Plotting {n_syst} shape systematic(s) from {root_file.name} -> {out_pdf.name}")

    with PdfPages(out_pdf) as pdf:
        for syst, procs in shape_rows.items():
            if not procs:
                continue
            nrows = len(procs)
            fig, axes = plt.subplots(
                nrows, 2, figsize=(11, 2.6 * nrows), squeeze=False,
                gridspec_kw={"width_ratios": [2.2, 1], "hspace": 0.5, "wspace": 0.3},
            )
            fig.suptitle(f"Shape systematic: {syst}", fontsize=13, y=1.0)

            for row, proc in enumerate(procs):
                nom = get_hist(f"h_{proc}")
                up = get_hist(f"h_{proc}_{syst}Up")
                down = get_hist(f"h_{proc}_{syst}Down")
                ax_shape, ax_ratio = axes[row]

                if nom is None or up is None or down is None:
                    ax_shape.text(0.5, 0.5, f"{proc}: missing histogram(s)", ha="center", va="center")
                    ax_shape.set_axis_off()
                    ax_ratio.set_axis_off()
                    continue

                nom_v, edges = nom
                up_v, _ = up
                down_v, _ = down
                centers = 0.5 * (edges[:-1] + edges[1:])
                width = np.diff(edges)

                ax_shape.step(edges[:-1], nom_v, where="post", color="k", lw=1.4, label="nominal")
                ax_shape.step(edges[:-1], up_v, where="post", color="#d62728", lw=1.1, label="Up")
                ax_shape.step(edges[:-1], down_v, where="post", color="#1f77b4", lw=1.1, label="Down")
                ax_shape.set_title(f"{proc}  (nom rate={nom_v.sum():.4g})", fontsize=9)
                ax_shape.tick_params(labelsize=8)
                if row == 0:
                    ax_shape.legend(fontsize=7, loc="upper left")

                # ratio panel -- this is what exposes a flat/dead Down variation
                with np.errstate(divide="ignore", invalid="ignore"):
                    r_up = np.where(nom_v != 0, up_v / nom_v, np.nan)
                    r_down = np.where(nom_v != 0, down_v / nom_v, np.nan)
                ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
                ax_ratio.plot(centers, r_up, "o-", color="#d62728", ms=3, lw=1, label="Up/Nom")
                ax_ratio.plot(centers, r_down, "o-", color="#1f77b4", ms=3, lw=1, label="Down/Nom")
                ax_ratio.set_ylim(0.7, 1.3)
                ax_ratio.tick_params(labelsize=8)
                if row == 0:
                    ax_ratio.legend(fontsize=7, loc="upper left")
                # flag near-flat Down variation (the CMS_ctag2d-style anomaly)
                spread_down = np.nanmax(r_down) - np.nanmin(r_down) if np.any(~np.isnan(r_down)) else 0
                spread_up = np.nanmax(r_up) - np.nanmin(r_up) if np.any(~np.isnan(r_up)) else 0
                if spread_up > 0.03 and spread_down < 0.01:
                    ax_ratio.text(0.02, 0.05, "Down ~flat vs Up!", color="#1f77b4",
                                  transform=ax_ratio.transAxes, fontsize=7, fontweight="bold")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
