# ctag2d negative-bin investigation + surgical rebinning fix (2026-08-16)

Consolidated home for everything produced while investigating negative bins in the
`hplusc_mva_4class_ctag2d` (2D pseudo-continuous c-tag SF) combine datacard, and the
surgical-rebinning fix that resolved them. Previously scattered across sibling
`Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_*` directories and a session
`/tmp` scratchpad — moved here in one place 2026-08-16.

**Does not touch**: the sibling `../combine_run3_90160_ctag2d/` directory (the original
20-bin production ctag2d datacard/histograms, with-ZX and hand-derived no-ZX) — left
exactly where it was, still the reference this investigation started from. Also does not
touch `b-hive_ttcc/combine/create_datacards_ctag2d.py` — the surgical-rebin logic below
is **not yet wired into the production script**, only exists as the standalone script
`scripts/07_build_surgical_v2_noZX.py`.

## tl;dr

- Original 20-bin ctag2d Signal template has 9 histograms (nominal + 8 systematic
  variants) with negative bins — never crashed combine (process totals stay positive),
  but real, traced to two sparse low-`mva_score_Signal` bins where the private
  `HPlusCharm`/`SomeSMSignal` sample's ~26% negative-weight-event fraction fluctuates
  hard on ~15-26 raw MC events.
- Uniform coarsening (10 or 8 bins) removes the negative bins but costs real
  sensitivity (r: 309.5 → 349.5 → 374.0).
- **Surgical fix**: merge only the two sparse bins (old bins 2-4 and 7-10) into two
  wide bins, 20 → 15 bins, leaving every other bin at full original resolution.
  **Result: r = 309.5, identical to the original 20-bin median, 0/46 histograms
  negative.** Not a trade-off — both merged regions were places Signal had almost no
  statistics regardless of sign.
- Full CMS `combineTool.py -M Impacts` ranking run on the final surgical no-ZX card:
  `CMS_ctag2d` ranks #3 of 11 nuisances (impact 6.03), behind `lhe_alphaS` (17.3) and
  `lumi_Run3` (8.2).

## Layout

```
combine_run3_90160_ctag2d_negbin_rebinning/
├── README.md                    <- this file
├── surgical_v2_noZX/            <- FINAL RESULT: datacard, histograms, workspace,
│                                    impacts JSON+PDF, all AsymptoticLimits ROOT output
│                                    (r=309.5, 0 negative bins, 15 bins, no Z+X)
├── rebinning_tests/             <- superseded intermediate tests, kept for the record
│   ├── 10bins/                  <- uniform 10-bin coarsening (r=349.5, 3 neg. bins)
│   ├── 8bins/                   <- uniform 8-bin coarsening (r=374.0, 1 neg. bin)
│   └── surgical_v1_noZX/        <- surgical merge of bins 7-10 ONLY (r not rerun here;
│                                    incomplete fix -- bin-3 issue found only after this)
├── scripts/                     <- every script that did the actual work, in the order
│                                    they were run (01-09), paths fixed to point here
└── plots/                       <- rendered figures (PDF + PNG) + the JSON they were
                                    built from
```

## Scripts (`scripts/`), in order

| # | Script | What it does |
|---|---|---|
| 01 | `check_negative_bins.py` | First scan: any TH1 in the original 20-bin with-ZX ctag2d histogram file with negative bin content, vs. the 1D-SF production reference for comparison. |
| 02 | `find_bin9_raw_events.py` | Loads real Signal-matching scored parquets (`HPlusCharm`/`SomeSMSignal`, all 4 eras), finds the 26 raw MC events landing in bin 9 `[0.004268,0.005078)`, prints each with its `weight_nominal`. |
| 03 | `rebin_merge_test_bin9.py` | Tests whether merging bin 9 with its neighbors (in the original 20-bin edges) flips the sum positive. It does, for every merge width tried. |
| 04 | `find_bin3_raw_events.py` | Same as 02 but for the second sparse bin (old bin 3, `[0.001406,0.001688)`), including the `CMS_ctag2d`/`CMS_pileup` Up systematic weight columns (this is the bin that's only negative under those two systematics, not nominal). |
| 05 | `scan_low_score_region.py` | General per-bin sum table (nominal + `CMS_ctag2d`/`CMS_pileup` Up) across the whole low-score region, used to pick the minimal merge window that fixes bin 3 (bins 2-4). |
| 06 | `build_surgical_v1_noZX.py` | First surgical build: merges only bins 7-10. Leaves bin 3's systematic-only negatives unfixed (found afterward). Writes to `rebinning_tests/surgical_v1_noZX/`. |
| 07 | `build_surgical_v2_noZX.py` | **The final build.** Merges bins 2-4 AND 7-10 (20 → 15 bins). Writes to `surgical_v2_noZX/`. Reuses `create_datacards_ctag2d.py`'s own `load_sumw`/`load_mc_scored_parquets`/`compute_quantile_bins`/`build_histograms` (imported as a module, not copy-pasted) so normalization/systematics wiring stays identical to production; only the bin edges and the no-ZX datacard-writing are custom. |
| 08 | `final_negative_bin_scan_all_variants.py` | Consolidated re-check across every variant (original with-ZX, original no-ZX, 10-bin, 8-bin, surgical v1, surgical v2) in one pass — the numbers quoted in the tl;dr above. |
| 09 | `make_plots.py` | Builds the two comparison figures in `plots/` (dataviz-skill categorical palette): Signal template before/after (negative bins highlighted), and r-vs-binning-scheme bar chart. |

**To rerun from scratch**: `01` → `02`/`04` (diagnosis, order doesn't matter between
them) → `07` (the only build script needed for the final result — `06` is superseded)
→ `08` (verify) → `09` (plots). All scripts use the LCG_105 environment:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
python3 scripts/07_build_surgical_v2_noZX.py
```
Combine itself (workspace/impacts/AsymptoticLimits, already run and saved in
`surgical_v2_noZX/`) needs the separate CMSSW combine environment, in a different shell:
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
cd /eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/surgical_v2_noZX
combine -M AsymptoticLimits -m 120 --run blind --rAbsAcc 0.00001 --rRelAcc 0.00001 \
  datacard_surgical_v2_noZX.txt -n HcZZ_ctag2d_surgical_v2_noZX
```

## Slides

The same plots (plus a systematics-list table and the impact ranking) were turned into
4 beamer slides appended to the project's main HcZZ combine deck:
`second-brain/slides/combine_limit_hczz.tex` / `.pdf` (pages 9-12, added 2026-08-16).
That deck stays in the `second-brain` notes repo per this project's existing convention
(every other combine-result slide deck lives there, not in `Higgscharmnew`) — only the
raw analysis materials (scripts, combine outputs, plot sources) are consolidated here.

## Open items (unchanged from before consolidation)

- Surgical-rebin logic is **not yet promoted into `create_datacards_ctag2d.py`** — a
  normal rerun of the production pipeline will not reproduce `surgical_v2_noZX/`.
- MC-only (no Z+X) throughout — Z+X was deliberately excluded from this whole
  investigation per explicit instruction, not because of a technical limitation.
- `rebinning_tests/10bins`, `8bins`, `surgical_v1_noZX` are superseded but kept for the
  comparison table in the tl;dr and the slides — safe to delete if disk space matters,
  nothing downstream depends on them.
