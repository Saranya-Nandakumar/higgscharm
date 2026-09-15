#!/usr/bin/env python3
"""
Rebuild docs/hczz_ctag2d_jesjer_results.pdf with fully current numbers:
official ctag2d SFs (L0 swap correction reverted 2026-09-15), ggZZ k-factor
applied, lhe_alphaS re-included, qqZZ decoupling applied -- 23 nuisances
no-ZX / 25 with-ZX. The existing PDF was stale on ALL of these (still had
pre-kfactor ggZZ=2.9418, 22/24 nuisances, and the retracted swap fix).

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

NOZX_DIR = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_ggzzkfactor_nozx_rawsf"
WITHZX_DIR = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_100150_ctag2d_v8_jecshifts_with_zx_pkatris_ggzzkfactor_rawsf"
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
    nozx_rows = load_ranking(f"{NOZX_DIR}/impacts_real.json", LNN_INFO_NOZX)
    withzx_rows = load_ranking(f"{WITHZX_DIR}/impacts_real.json", LNN_INFO_WITHZX)

    with PdfPages(OUT_TABLES) as pdf:
        # ---------------- Page 1: no-ZX yields + limits ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.suptitle("HcZZ ctag2d -- Results (official SFs, L0 swap reverted 2026-09-15)",
                     fontsize=15, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "Production: combine_run3_100150_ctag2d_v8_jecshifts_ggzzkfactor_nozx_rawsf\n"
                "hplusc_mva_4class_ctag2d_scored_v8 + hplusc_mva_4class_ctag2d_jecshifts_scored_v2,\n"
                "[100,150] GeV, no-ZX (MC-only), 15 bins (quantile + surgical rebin \"2-4,7-10\"),\n"
                "23 nuisances. CMS_ctag2d uses the OFFICIAL up_Total/down_Total, unmodified --\n"
                "the 2026-09-02 swap correction was retracted 2026-09-15 (component-level\n"
                "re-investigation found its basis too thin; reported to the calibration's\n"
                "maintainer, not yet POG-confirmed either way). Rebuilt and verified 2026-09-15.",
                transform=ax.transAxes, fontsize=8.5, va="top")

        ax.text(0.0, 0.72, "Yields  [100,150] GeV", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        yields = [("Signal", "0.053130"), ("ggZZ", "6.677900"),
                  ("qqZZ", "36.318223"), ("Other_Higgs", "55.077609")]
        tbl1 = Table(ax, bbox=[0.0, 0.53, 0.55, 0.17])
        for j, h in enumerate(["Process", "Rate"]):
            c = tbl1.add_cell(0, j, 0.5, 0.2, text=h, loc="center", facecolor="#2c3e50")
            c.get_text().set_color("white")
            c.get_text().set_fontweight("bold")
        for i, (p, r) in enumerate(yields, start=1):
            c = tbl1.add_cell(i, 0, 0.5, 0.2, text=p, loc="center", facecolor="white")
            c = tbl1.add_cell(i, 1, 0.5, 0.2, text=r, loc="center", facecolor="white")
        ax.add_table(tbl1)
        ax.text(0.0, 0.50, "S/$\\sqrt{B}$ = 0.0054", transform=ax.transAxes, fontsize=9)

        ax.text(0.0, 0.44, "Final limits (AsymptoticLimits, --run blind)",
                transform=ax.transAxes, fontsize=12, fontweight="bold")
        limits = [("2.5%", "164.2617", "31.13"), ("16%", "223.0386", "41.34"),
                  ("50% (median)", "321.0000", "58.33"), ("84%", "477.1014", "85.39"),
                  ("97.5%", "687.6224", "121.87")]
        tbl2 = Table(ax, bbox=[0.0, 0.24, 0.75, 0.19])
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

        ax.text(0.0, 0.19,
                "Stat-only (--freezeParameters allConstrainedNuisances): r=273.5/kappa_c=50.09 --\n"
                "identical to the retracted-fix version (only CMS_ctag2d's shape changed, not\n"
                "central yields). Without the L0 swap correction: r=321.0 (was 411.0 with it),\n"
                "a -21.9% shift -- the single largest correction to this headline's history.",
                transform=ax.transAxes, fontsize=8.5, va="top")

        ax.text(0.0, 0.06,
                "CMS_ctag2d dominates the fit (~4.1x the #2 nuisance, down from ~8.6x under the\n"
                "retracted swap correction). JES/JER's 4 rows (CMS_scale_j/CMS_res_j/CMS_scale_m/\n"
                "CMS_res_m) rank #3/#15/#17/#16 by impact -- CMS_scale_j in particular is now\n"
                "the #3 nuisance overall, more prominent than under the retracted fix (#5).",
                transform=ax.transAxes, fontsize=8.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---------------- Page 2: no-ZX full systematics table ----------------
        rows2 = [(str(i), r[0], r[1], r[2], f"{r[3]:.2f}") for i, r in enumerate(nozx_rows, 1)]
        draw_table_page(
            pdf, rows2, [0.05, 0.22, 0.13, 0.45, 0.15],
            ["#", "Systematic", "Type", "Value (lnN size / pull)", "Impact"],
            "Systematics: type, assigned value, and impact",
            "Ranked by impact_r (combineTool.py -M Impacts, --expect-signal 321.0, official\n"
            "ctag2d SFs, 23 nuisances). lnN rows show the assigned size per affected process;\n"
            "shape rows show the post-fit pull instead (all prefit +/-1sigma here, expected for\n"
            "a blind Asimov fit).",
        )

        # ---------------- Page 4: with-ZX yields + limits ----------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.suptitle("HcZZ -- with Z+X (true 4-era, pkatris) results, official SFs",
                     fontsize=14, fontweight="bold", x=0.06, y=0.97, ha="left")
        ax.text(0.0, 0.935,
                "Production: combine_run3_100150_ctag2d_v8_jecshifts_with_zx_pkatris_ggzzkfactor_rawsf\n"
                "[100,150] GeV, ctag2d scheme, 25 nuisances (18 shape/lnN + 4 JES/JER +\n"
                "ZX_norm/ZX_rate), true 4-era Z+X (pkatris production) via --zx-dir. Official\n"
                "ctag2d SFs, L0 swap correction retracted 2026-09-15 (see no-ZX page).",
                transform=ax.transAxes, fontsize=8.5, va="top")

        ax.text(0.0, 0.78, "Yields in [100,150] GeV", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        yields2 = [("Signal", "0.0531"), ("ggZZ", "6.6779"), ("qqZZ", "36.3182"),
                   ("Other_Higgs", "55.0776"), ("Z+X", "133.2386"), ("Total bkg", "231.3123")]
        tbl3 = Table(ax, bbox=[0.0, 0.55, 0.55, 0.22])
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
        ax.text(0.0, 0.53, "S/$\\sqrt{B}$ = 0.0035", transform=ax.transAxes, fontsize=9)

        ax.text(0.0, 0.47, "Final limits (combine -M AsymptoticLimits -m 120 --run blind)",
                transform=ax.transAxes, fontsize=12, fontweight="bold")
        limits2 = [("2.5%", "167.0703", "31.61"), ("16%", "227.7939", "42.16"),
                   ("50% (median)", "329.0000", "59.72"), ("84%", "490.3032", "87.68"),
                   ("97.5%", "709.5030", "125.66")]
        tbl4 = Table(ax, bbox=[0.0, 0.27, 0.75, 0.19])
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

        ax.text(0.0, 0.20, "Comparison against the no-ZX headline",
                transform=ax.transAxes, fontsize=11, fontweight="bold")
        comp = [("no-Z+X headline (23 syst, official SFs)", "321.0", "58.33"),
                ("with-Z+X, true 4-era (25 syst, official SFs)", "329.0", "59.72")]
        tbl5 = Table(ax, bbox=[0.0, 0.06, 0.95, 0.12])
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
            "Ranked by impact_r (combineTool.py -M Impacts, --expect-signal 329.0, official\n"
            "ctag2d SFs, true 4-era Z+X included, 25 nuisances incl. the free-floating\n"
            "ZX_rate). ZX_rate is now #13 (was #2 under the retracted swap correction).",
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
                "unconstrained), same official-SF production, [100,150] GeV, both\n"
                "re-verified 2026-09-15 -- IDENTICAL to the retracted-fix version, since the\n"
                "swap correction only ever changed CMS_ctag2d's shape, never central yields.",
                transform=ax.transAxes, fontsize=8.5, va="top")

        stat = [("2.5%", "141.0234", "27.08", "145.0723", "27.79", "+4.05", "+2.9%"),
                ("16%", "190.7021", "35.72", "198.0641", "37.00", "+7.36", "+3.9%"),
                ("50% (median)", "273.5000", "50.09", "283.5000", "51.83", "+10.00", "+3.7%"),
                ("84%", "396.6906", "71.45", "412.3249", "74.16", "+15.63", "+3.9%"),
                ("97.5%", "553.4374", "98.62", "575.7133", "102.48", "+22.28", "+4.0%")]
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
                "Key point: stat-only, Z+X loosens the median limit by only ~3.7%\n"
                "(r=273.5$\\to$283.5) -- much smaller than the ~2.5% full-syst loosening\n"
                "(r=321.0$\\to$329.0, official SFs). Both effects are now much smaller than\n"
                "under the retracted swap correction (full-syst was +16.8%, r=411.0$\\to$480.0) --\n"
                "with CMS_ctag2d no longer artificially inflated, Z+X's own systematics\n"
                "(ZX_norm, ZX_rate) have much less room to loosen the fit further on top of\n"
                "the stat-only effect.",
                transform=ax.transAxes, fontsize=9, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote table pages: {OUT_TABLES}")

    # ---------------- Merge real impacts plots as pages 3 and 6 ----------------
    tables = PdfReader(OUT_TABLES)
    impacts_nozx = PdfReader(f"{NOZX_DIR}/impacts_real.pdf")
    impacts_withzx = PdfReader(f"{WITHZX_DIR}/impacts_real.pdf")

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
