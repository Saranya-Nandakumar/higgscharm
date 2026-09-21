#!/usr/bin/env python3
"""
Rebuild docs/hczz_ctag2d_jesjer_results.pdf with fully current numbers (2026-09-21):
Other_Higgs's lhe_pdf/scalevar_muR/scalevar_muF systematics coverage completed
(previously excluded entirely; now only its private HPlusBottom sub-sample is
masked), JES/JER re-verified on top of that fix, official ctag2d SFs (L0 swap
correction still reverted, unchanged since 2026-09-15) -- 23 nuisances no-ZX /
25 with-ZX (same nuisance names/count as before; Other_Higgs's fix extends
per-process coverage of names already in the list, it doesn't add new ones).
The existing PDF was stale on the limits, the with-ZX Z+X yield (the fake-rate
production itself changed separately, from 133.24 to 31.11, since the last
rebuild of this doc), and the impact ranking (CMS_ctag2d no longer dominates by
~4x; scalevar_muR is a strong new #2 now that Other_Higgs carries it).

Builds table/text pages with matplotlib, merges in the real
combineTool.py -M Impacts plot PDFs (not redrawn) via pypdf.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
from pypdf import PdfReader, PdfWriter

NOZX_DIR = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_100150_ctag2d_rawsf_nozx_jecshifts_2026_09_21"
WITHZX_DIR = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_100150_ctag2d_rawsf_withzx_jecshifts_2026_09_21"
OUT_TABLES = "/tmp/snandaku/jesjer_results_tables.pdf"
OUT_FINAL = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/docs/hczz_ctag2d_jesjer_results.pdf"

LNN_INFO_NOZX = {
    "lumi_Run3": ("1.014", "Signal, ggZZ, qqZZ, Other_Higgs"),
    "QCDscale_gg": ("1.039", "Signal, ggZZ"),
    "pdf_gg": ("1.032", "Signal, ggZZ"),
    "kfactor_ggZZ": ("1.10", "ggZZ"),
    "BR_HZZ4l": ("1.02", "Signal, Other_Higgs"),
    "QCDscale_qqZZ": ("1.04", "qqZZ"),
}
LNN_INFO_WITHZX = dict(LNN_INFO_NOZX)
LNN_INFO_WITHZX["lumi_Run3"] = ("1.014", "Signal, ggZZ, qqZZ, Other_Higgs, ZX")
LNN_INFO_WITHZX["ZX_norm"] = ("1.30", "ZX")


def load_ranking(json_path, lnn_info):
    with open(json_path) as f:
        d = json.load(f)
    params = d["params"]
    ranked = sorted(params, key=lambda p: -abs(p.get("impact_r", 0)))
    rows = []
    for p in ranked:
        name = p["name"]
        impact = p["impact_r"]
        if name in lnn_info:
            size, procs = lnn_info[name]
            typ = "lnN"
            val = f"{size} ({procs})"
        elif p["type"] == "Unconstrained":
            fit = p["fit"]
            typ = "rateParam"
            val = f"{fit[1]:.2f} (+{fit[2]-fit[1]:.2f}/-{fit[1]-fit[0]:.2f}), free-floating"
        else:
            fit = p["fit"]
            typ = "shape"
            sign = "+" if fit[1] >= 0 else "-"
            val = f"pull {sign}0.00 ({fit[0]:+.3f}/{fit[2]:+.3f})"
        rows.append((name, typ, val, impact))
    return rows


def draw_table_page(pdf, rows_data, col_widths, header, title, subtitle, fig_h=11):
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
    ax.text(0.0, 0.99, subtitle, transform=ax.transAxes, fontsize=8.5,
            va="top", wrap=True)

    n_rows = len(rows_data) + 1
    tbl = Table(ax, bbox=[0, 0.0, 1, 0.85])
    for j, (h, w) in enumerate(zip(header, col_widths)):
        c = tbl.add_cell(0, j, w, 1.0 / n_rows, text=h, loc="center", facecolor="#2c3e50")
        c.get_text().set_color("white")
        c.get_text().set_fontweight("bold")
        c.get_text().set_fontsize(8)
    for i, row in enumerate(rows_data, start=1):
        bg = "#eef2f5" if i % 2 == 0 else "white"
        if row[0] == "1" or (len(row) > 0 and str(row[0]).strip() == "1"):
            bg = "#dce6f1"
        for j, (val, w) in enumerate(zip(row, col_widths)):
            loc = "left" if j == 1 else "center"
            c = tbl.add_cell(i, j, w, 1.0 / n_rows, text=str(val), loc=loc, facecolor=bg)
            c.get_text().set_fontsize(7.5)
            if i == 1:
                c.get_text().set_fontweight("bold")
    tbl.auto_set_font_size(False)
    ax.add_table(tbl)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    nozx_rows = load_ranking(f"{NOZX_DIR}/impacts_jecshifts_2026_09_21.json", LNN_INFO_NOZX)
    withzx_rows = load_ranking(f"{WITHZX_DIR}/impacts_withzx_jecshifts_2026_09_21.json", LNN_INFO_WITHZX)

    with PdfPages(OUT_TABLES) as pdf:
        # ---------------- Page 1: no-ZX yields + limits ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.suptitle("HcZZ ctag2d -- Results (Other_Higgs systematics completed, JES/JER, 2026-09-21)",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "Production: combine_run3_100150_ctag2d_rawsf_nozx_jecshifts_2026_09_21\n"
                "hplusc_mva_4class_ctag2d_scored_v8 + hplusc_mva_4class_ctag2d_jecshifts_scored_v2,\n"
                "[100,150] GeV, no-ZX (MC-only), 20 quantile bins, 23 nuisances. CMS_ctag2d uses\n"
                "the OFFICIAL up_Total/down_Total, unmodified (L0 swap correction remains\n"
                "reverted since 2026-09-15). Other_Higgs's lhe_pdf/scalevar_muR/scalevar_muF\n"
                "coverage completed 2026-09-20 (previously excluded entirely; only its private\n"
                "HPlusBottom sub-sample, ~4% of yield, is now masked). Rebuilt and verified\n"
                "2026-09-21.",
                transform=ax.transAxes, fontsize=8.5, va="top")

        ax.text(0.0, 0.685, "Yields  [100,150] GeV", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        yields = [("Signal", "0.053130"), ("ggZZ", "6.677889"),
                  ("qqZZ", "36.318223"), ("Other_Higgs", "55.077609")]
        tbl1 = Table(ax, bbox=[0.0, 0.50, 0.55, 0.17])
        for j, h in enumerate(["Process", "Rate"]):
            c = tbl1.add_cell(0, j, 0.5, 0.2, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        for i, (p, r) in enumerate(yields, start=1):
            c = tbl1.add_cell(i, 0, 0.5, 0.2, text=p, loc="center", facecolor="white")
            c = tbl1.add_cell(i, 1, 0.5, 0.2, text=r, loc="center", facecolor="white")
        ax.add_table(tbl1)
        ax.text(0.0, 0.47, "S/$\\sqrt{B}$ = 0.0054", transform=ax.transAxes, fontsize=9)

        ax.text(0.0, 0.41, "Final limits (AsymptoticLimits, --run blind)",
                transform=ax.transAxes, fontsize=12, fontweight="bold")
        limits = [("2.5%", "178.6641", "33.63"), ("16%", "242.9138", "44.79"),
                  ("50% (median)", "346.5000", "62.75"), ("84%", "510.8585", "91.24"),
                  ("97.5%", "730.3098", "129.26")]
        tbl2 = Table(ax, bbox=[0.0, 0.21, 0.75, 0.19])
        for j, h in enumerate(["Quantile", "r", "kappa_c"]):
            c = tbl2.add_cell(0, j, 0.33, 1/6, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        for i, (q, r, k) in enumerate(limits, start=1):
            bg = "#dce6f1" if "median" in q else "white"
            for j, val in enumerate([q, r, k]):
                c = tbl2.add_cell(i, j, 0.33, 1/6, text=val, loc="center", facecolor=bg)
                if "median" in q:
                    c.get_text().set_fontweight("bold")
        ax.add_table(tbl2)

        ax.text(0.0, 0.16,
                "Stat-only (--freezeParameters allConstrainedNuisances): r=273.5/kappa_c=50.09 --\n"
                "bit-identical to the pre-Other_Higgs-fix value (a pure systematics-side change\n"
                "cannot move a stat-only fit). Full-syst median moved 339.5 (no JES/JER, post-fix)\n"
                "-> 346.5 (this page, +JES/JER) -- +2.06% from JES/JER alone.",
                transform=ax.transAxes, fontsize=8.5, va="top")

        ax.text(0.0, 0.06,
                "CMS_ctag2d is still #1 but no longer dominates by ~4x -- now only ~1.74x the\n"
                "#2 nuisance (scalevar_muR, a real Other_Higgs shape for the first time,\n"
                "impact 84.96 vs CMS_ctag2d's 147.68). CMS_scale_j ranks #5 (29.79).",
                transform=ax.transAxes, fontsize=8.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---------------- Page 2: no-ZX full systematics table ----------------
        rows2 = [(str(i), r[0], r[1], r[2], f"{r[3]:.2f}") for i, r in enumerate(nozx_rows, 1)]
        draw_table_page(
            pdf, rows2, [0.05, 0.22, 0.13, 0.45, 0.15],
            ["#", "Systematic", "Type", "Value (lnN size / pull)", "Impact"],
            "Systematics: type, assigned value, and impact",
            "Ranked by impact_r (combineTool.py -M Impacts, --expectSignal 346.5, official\n"
            "ctag2d SFs, 23 nuisances, Other_Higgs systematics completed). lnN rows show the\n"
            "assigned size per affected process; shape rows show the post-fit pull instead\n"
            "(all prefit, max |pull| ~1e-7, expected for a blind Asimov fit).",
        )

        # ---------------- Page 4: with-ZX yields + limits ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.suptitle("HcZZ -- with Z+X (current 4-era estimate) results, official SFs",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "Production: combine_run3_100150_ctag2d_rawsf_withzx_jecshifts_2026_09_21\n"
                "[100,150] GeV, ctag2d scheme, 25 nuisances (18 shape/lnN + 4 JES/JER +\n"
                "ZX_norm/ZX_rate), current 4-era Z+X estimate via --zx-dir. Official ctag2d\n"
                "SFs (L0 swap correction reverted), Other_Higgs systematics completed\n"
                "2026-09-20 (see no-ZX page). Note: the Z+X yield below (31.11) is smaller\n"
                "than a 2026-09-15 value (133.24) this doc previously carried -- the\n"
                "fake-rate production itself changed in between, unrelated to this update.",
                transform=ax.transAxes, fontsize=8.5, va="top")

        ax.text(0.0, 0.73, "Yields in [100,150] GeV", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        yields2 = [("Signal", "0.0531"), ("ggZZ", "6.6779"), ("qqZZ", "36.3182"),
                   ("Other_Higgs", "55.0776"), ("Z+X", "31.1075"), ("Total bkg", "129.1813")]
        tbl3 = Table(ax, bbox=[0.0, 0.50, 0.55, 0.22])
        for j, h in enumerate(["Process", "Yield"]):
            c = tbl3.add_cell(0, j, 0.5, 1/7, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        for i, (p, r) in enumerate(yields2, start=1):
            bg = "#dce6f1" if p == "Total bkg" else "white"
            c = tbl3.add_cell(i, 0, 0.5, 1/7, text=p, loc="center", facecolor=bg)
            c = tbl3.add_cell(i, 1, 0.5, 1/7, text=r, loc="center", facecolor=bg)
            if p == "Total bkg":
                c.get_text().set_fontweight("bold")
        ax.add_table(tbl3)
        ax.text(0.0, 0.48, "S/$\\sqrt{B}$ = 0.0047", transform=ax.transAxes, fontsize=9)

        ax.text(0.0, 0.42, "Final limits (combine -M AsymptoticLimits -m 120 --run blind)",
                transform=ax.transAxes, fontsize=12, fontweight="bold")
        limits2 = [("2.5%", "180.9844", "34.03"), ("16%", "246.0685", "45.33"),
                   ("50% (median)", "351.0000", "63.53"), ("84%", "519.5917", "92.75"),
                   ("97.5%", "745.3238", "131.86")]
        tbl4 = Table(ax, bbox=[0.0, 0.22, 0.75, 0.19])
        for j, h in enumerate(["Quantile", "r", "kappa_c"]):
            c = tbl4.add_cell(0, j, 0.33, 1/6, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        for i, (q, r, k) in enumerate(limits2, start=1):
            bg = "#dce6f1" if "median" in q else "white"
            for j, val in enumerate([q, r, k]):
                c = tbl4.add_cell(i, j, 0.33, 1/6, text=val, loc="center", facecolor=bg)
                if "median" in q:
                    c.get_text().set_fontweight("bold")
        ax.add_table(tbl4)

        ax.text(0.0, 0.15, "Comparison against the no-ZX headline",
                transform=ax.transAxes, fontsize=11, fontweight="bold")
        comp = [("no-Z+X headline (23 syst, official SFs)", "346.5", "62.75"),
                ("with-Z+X, current 4-era (25 syst, official SFs)", "351.0", "63.53")]
        tbl5 = Table(ax, bbox=[0.0, 0.02, 0.95, 0.12])
        for j, h in enumerate(["Configuration", "r (median)", "kappa_c"]):
            w = 0.6 if j == 0 else 0.2
            c = tbl5.add_cell(0, j, w, 0.5, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        for i, (cfg, r, k) in enumerate(comp, start=1):
            bg = "#dce6f1" if i == 2 else "white"
            c = tbl5.add_cell(i, 0, 0.6, 0.5, text=cfg, loc="left", facecolor=bg)
            c = tbl5.add_cell(i, 1, 0.2, 0.5, text=r, loc="center", facecolor=bg)
            c = tbl5.add_cell(i, 2, 0.2, 0.5, text=k, loc="center", facecolor=bg)
        ax.add_table(tbl5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---------------- Page 5: with-ZX full systematics table ----------------
        rows5 = [(str(i), r[0], r[1], r[2], f"{r[3]:.2f}") for i, r in enumerate(withzx_rows, 1)]
        draw_table_page(
            pdf, rows5, [0.05, 0.22, 0.13, 0.45, 0.15],
            ["#", "Systematic", "Type", "Value (lnN size / pull)", "Impact"],
            "Systematics with Z+X: type, assigned value, and impact",
            "Ranked by impact_r (combineTool.py -M Impacts, --expectSignal 351.0, official\n"
            "ctag2d SFs, current 4-era Z+X included, 25 nuisances incl. the free-floating\n"
            "ZX_rate). ZX_rate is now #3 (impact 84.88) -- CMS_ctag2d (169.10) and\n"
            "scalevar_muR (89.92) still rank #1/#2 ahead of it.",
            fig_h=12,
        )

        # ---------------- Page 7: stat-only comparison ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.suptitle("Stat-only limit: with vs. without Z+X (official SFs)",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "combine -M AsymptoticLimits --freezeParameters allConstrainedNuisances\n"
                "(all constrained nuisances frozen; ZX_rate still floats since it is\n"
                "unconstrained), same official-SF production, [100,150] GeV, both re-verified\n"
                "2026-09-21. No-ZX stat-only is bit-identical to every prior rebuild (a pure\n"
                "systematics-side fix cannot move it); with-ZX stat-only moved (283.5->289.5)\n"
                "purely because the Z+X estimate itself changed (133.24->31.11 yield),\n"
                "unrelated to this session's systematics work.",
                transform=ax.transAxes, fontsize=8.5, va="top")

        stat = [("2.5%", "141.0234", "27.08", "148.1426", "28.32", "+7.12", "+5.0%"),
                ("16%", "190.7021", "35.72", "202.2560", "37.73", "+11.55", "+6.1%"),
                ("50% (median)", "273.5000", "50.09", "289.5000", "52.87", "+16.00", "+5.9%"),
                ("84%", "396.6906", "71.45", "421.6284", "75.78", "+24.94", "+6.3%"),
                ("97.5%", "553.8622", "98.69", "590.0568", "104.96", "+36.19", "+6.5%")]
        tbl7 = Table(ax, bbox=[0.0, 0.57, 1.0, 0.28])
        headers7 = ["Quantile", "r (no ZX)", "kc (no ZX)", "r (with ZX)", "kc (with ZX)", "dr", "dr %"]
        widths7 = [0.20, 0.15, 0.13, 0.15, 0.13, 0.12, 0.12]
        for j, h in enumerate(headers7):
            c = tbl7.add_cell(0, j, widths7[j], 1/6, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
            c.get_text().set_fontsize(8)
        for i, row in enumerate(stat, start=1):
            bg = "#dce6f1" if "median" in row[0] else "white"
            for j, val in enumerate(row):
                c = tbl7.add_cell(i, j, widths7[j], 1/6, text=val, loc="center", facecolor=bg)
                c.get_text().set_fontsize(8)
                if "median" in row[0]:
                    c.get_text().set_fontweight("bold")
        ax.add_table(tbl7)

        ax.text(0.0, 0.50,
                "Key point: stat-only, Z+X loosens the median limit by ~5.9%\n"
                "(r=273.5$\\to$289.5) -- somewhat larger than the ~1.3% full-syst loosening\n"
                "(r=346.5$\\to$351.0, official SFs). The full-syst effect is small because the\n"
                "free-floating ZX_rate parameter absorbs most of Z+X's own normalization\n"
                "freedom in the fit, leaving little room for the systematics side to add on\n"
                "top of the smaller current Z+X yield's own statistical effect.",
                transform=ax.transAxes, fontsize=9, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote table pages: {OUT_TABLES}")

    # ---------------- Merge real impacts plots as pages 3 and 6 ----------------
    tables = PdfReader(OUT_TABLES)
    impacts_nozx = PdfReader(f"{NOZX_DIR}/impacts_jecshifts_2026_09_21.pdf")
    impacts_withzx = PdfReader(f"{WITHZX_DIR}/impacts_withzx_jecshifts_2026_09_21.pdf")

    writer = PdfWriter()
    writer.add_page(tables.pages[0])  # page 1: no-ZX yields/limits
    writer.add_page(tables.pages[1])  # page 2: no-ZX systematics table
    writer.add_page(impacts_nozx.pages[0])  # page 3: real no-ZX impacts plot
    writer.add_page(tables.pages[2])  # page 4: with-ZX yields/limits
    writer.add_page(tables.pages[3])  # page 5: with-ZX systematics table
    writer.add_page(impacts_withzx.pages[0])  # page 6: real with-ZX impacts plot
    writer.add_page(tables.pages[4])  # page 7: stat-only comparison

    with open(OUT_FINAL, "wb") as f:
        writer.write(f)
    print(f"Wrote final 7-page PDF: {OUT_FINAL}")


if __name__ == "__main__":
    main()
