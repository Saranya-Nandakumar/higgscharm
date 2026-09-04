#!/usr/bin/env python3
"""Two figures for the ctag2d negative-bin / surgical-rebin slides."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# validated categorical palette (dataviz skill, references/palette.md)
BLUE   = "#2a78d6"
ORANGE = "#eb6834"
AQUA   = "#1baf7a"
YELLOW = "#eda100"
GREEN  = "#008300"
RED    = "#e34948"
INK    = "#0b0b0b"
MUTED  = "#52514e"
GRID   = "#dedcd6"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "svg.fonttype": "none",
})

data = json.load(open("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/plots/hist_data.json"))

# ---------------------------------------------------------------------------
# Figure 1: Signal template, original 20-bin vs surgical 15-bin, negative
# bins highlighted in red.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, key, title, neg_nominal, neg_syst_only in [
    (axes[0], "orig20_Signal", "Original — 20 quantile bins", {9}, {3}),
    (axes[1], "surgv2_Signal", "Surgical — 15 bins (2–4, 7–10 merged)", set(), set()),
]:
    vals = np.array(data[key]["vals"])
    n = len(vals)
    colors = [RED if i in neg_nominal else (ORANGE if i in neg_syst_only else BLUE) for i in range(n)]
    x = np.arange(n)
    ax.bar(x, np.abs(vals), color=colors, width=0.78, zorder=3)
    for i in neg_nominal:
        ax.annotate("neg.\n(nominal)", (i, np.abs(vals[i])), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7,
                     color=RED, fontweight="bold")
    for i in neg_syst_only:
        ax.annotate("neg.\n(2 systs)", (i, np.abs(vals[i])), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=7,
                     color=ORANGE, fontweight="bold")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel("bin index", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("|scaled Signal yield| per bin", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=7)
    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.suptitle("Signal template (h\\_Signal, nominal): where the negative bins were",
             fontsize=12, color=INK, y=1.02)
fig.text(0.5, -0.04,
         "Magnitude plotted (log scale). Red = negative in the nominal template too "
         "(bin 9, [0.00427,0.00508): 26 raw MC events, net $-3.30$). "
         "Orange = nominal positive but negative under 2 of its 4 usable systematics "
         "(bin 3, [0.00141,0.00169): 15 raw MC events, net $+0.62$ nominal, $-3.33$/$-1.02$ under CMS\\_ctag2d/CMS\\_pileup Up).",
         ha="center", fontsize=8.0, color=MUTED)
fig.tight_layout()
fig.savefig("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/plots/fig1_negbin_before_after.pdf",
            bbox_inches="tight")
print("saved fig1")

# ---------------------------------------------------------------------------
# Figure 2: r vs binning scheme, negative-bin count annotated
# ---------------------------------------------------------------------------
schemes = ["Original\n(20 bins)", "Uniform\n(10 bins)", "Uniform\n(8 bins)", "Surgical\n(15 bins)"]
r_vals = [309.5, 349.5, 374.0, 309.5]
neg_counts = [9, 3, 1, 0]
colors_bar = [ORANGE, ORANGE, ORANGE, GREEN]

fig2, ax = plt.subplots(figsize=(7.2, 4.4))
x = np.arange(len(schemes))
bars = ax.bar(x, r_vals, color=colors_bar, width=0.58, zorder=3)

for xi, (r, n) in zip(x, zip(r_vals, neg_counts)):
    ax.annotate(f"r = {r:.1f}", (xi, r), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=10, color=INK, fontweight="bold")
    badge_color = GREEN if n == 0 else RED
    ax.annotate(f"{n} neg. bin{'s' if n != 1 else ''}", (xi, r), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=8.5,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=badge_color, edgecolor="none"))

ax.set_xticks(x)
ax.set_xticklabels(schemes, fontsize=9.5)
ax.set_ylabel("Median expected limit, r (no Z+X)", fontsize=10)
ax.set_ylim(0, 430)
ax.grid(axis="x", visible=False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.set_title("Rebinning schemes: sensitivity vs. negative-bin count", fontsize=12, color=INK, pad=10)
fig2.text(0.5, -0.02,
          "Surgical rebin: same r as the original 20-bin scheme, but zero negative bins — uniform coarsening trades away real sensitivity instead.",
          ha="center", fontsize=8.5, color=MUTED)
fig2.tight_layout()
fig2.savefig("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/plots/fig2_r_vs_binning.pdf",
             bbox_inches="tight")
print("saved fig2")
