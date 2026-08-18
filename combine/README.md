# HcZZ ctag2d combine pipeline

Self-contained pipeline for the H+c → ZZ → 4ℓ combine analysis under the 2D
pseudo-continuous c-tag SF scheme (`ctag2d`) — datacard building, the Combine
software itself, and every result produced with it, all under this one directory.

**Folded into the `higgscharm` repo 2026-08-17** (this directory used to be its own
standalone git repo under `Higgscharmnew/combine/`; moved here so the analysis code
and the combine pipeline that consumes its output live in one place/one push). CMSSW
exclusion and output-data exclusion rules live in the repo's top-level `.gitignore`,
not a local one here.

> ✅ **CMSSW rebuilt properly 2026-08-18**: `setup_cmssw.sh --force` (standard
> `cmsrel`/`git clone`/`scram b`) fixed the post-move breakage that used to require a
> per-shell `sed` path-patch workaround (see git history for that era if needed).
> `combine` now resolves and runs cleanly with no manual patching.
>
> ✅ **Default mass window changed to `[100,150]` GeV, 2026-08-18** (analysis
> decision — tighter S/√B, 0.0054 vs 0.0043). `./run_pipeline.sh` with no args now
> builds `[100,150]` via `create_datacards_ctag2d_100150.py`; pass `--window 90160`
> for the older window. **Z+X (fake-rate background) is deliberately out of scope
> at this window for now** (decision 2026-08-18, same date) — only `[90,160]`'s
> `zx_background_mva_os_90160_featmajor` estimate exists; `--with-zx` at the new
> default exits with a message pointing at `--window 90160 --with-zx` instead of
> silently reusing the wrong window's Z+X data.

**Not in scope here**: the production 1D c-tag SF pipeline (`create_datacards.py`,
`Analysis/combine_run3_90160_v5/`) — that lives in `b-hive_ttcc/combine/` and
`Analysis/`, per the project's older, separate `README_HcZZ.md` convention. This
directory is specifically the `ctag2d` comparison/candidate-replacement pipeline.

**`outputs/` is real, on-disk, and colocated here, but not git-tracked** — the
repo's own `.gitignore` already excludes `outputs/`/`*.root` everywhere (a
deliberate, pre-existing convention in this collaborative repo), and that's
respected here rather than overridden. Every result in `outputs/` is reproducible
via `run_pipeline.sh` below.

## Layout

```
combine/
├── README.md              <- this file
├── run_pipeline.sh         <- ONE COMMAND: build datacard -> run combine -> print r/kappa_c
├── setup_cmssw.sh          <- bootstraps CMSSW_14_1_0_pre4 + Combine v10.5.0 from scratch
├── scripts/
│   └── create_datacards_ctag2d.py   <- the datacard builder (canonical copy, see its own
│                                        docstring). --merge-bin-ranges / --skip-zx added
│                                        2026-08-17 (the "surgical rebin" fix, see below).
├── CMSSW_14_1_0_pre4/       <- built Combine v10.5.0 (gitignored, rebuild via setup_cmssw.sh)
└── outputs/                 <- gitignored (see above), reproducible, colocated on disk
    ├── combine_run3_90160_ctag2d/                  <- original 20-bin, with-ZX (+ hand-
    │                                                    derived no-ZX), production reference
    └── combine_run3_90160_ctag2d_negbin_rebinning/  <- negative-bin investigation: root
                                                          cause, rebinning comparison, plots,
                                                          the diagnostic scripts that led to
                                                          --merge-bin-ranges/--skip-zx above.
                                                          Full detail in its own README.md.
```

## Quickstart

```bash
cd /eos/user/s/snandaku/Higgscharmnew/higgscharm/combine
./run_pipeline.sh                 # DEFAULT (since 2026-08-18): [100,150] GeV, surgical rebin,
                                   #   no Z+X -> r=311.5 full-syst / 279.0 stat-only, 0 neg. bins
./run_pipeline.sh --with-zx       # NOT AVAILABLE at [100,150] -- Z+X deliberately out of scope for now
./run_pipeline.sh --original      # [100,150], unmerged, no Z+X (Z+X out of scope at this window for now)
./run_pipeline.sh --window 90160                # older [90,160] window, surgical, no Z+X -> r=324.5
./run_pipeline.sh --window 90160 --with-zx      # [90,160], surgical, WITH Z+X -> r=329.0
./run_pipeline.sh --window 90160 --original     # [90,160], unmerged, WITH Z+X -> r=329.0 (may have neg. bins)
```

First run bootstraps `CMSSW_14_1_0_pre4/` automatically if it isn't present (takes
a while — standard `cmsrel` + `git clone` + `scram b`). Every subsequent run reuses it.

To pass through raw flags instead of a preset:
```bash
./run_pipeline.sh --n-bins 20 --merge-bin-ranges "2-4,7-10" --skip-zx --output /some/other/dir
```

**Footgun already found and worked around, 2026-08-17**: `create_datacards_ctag2d.py`'s
own `--zx-dir` default points at the SS (same-sign) ZX estimate, but every documented
reference number in this repo (r=317.0 original with-ZX, r=309.5 surgical no-ZX) was
built against the OS (opposite-sign, `_featmajor`) estimate — confirmed the hard way:
the SS default silently gives r=364.5 for otherwise-identical `--original` settings, no
error, no warning. `run_pipeline.sh`'s `--with-zx`/`--original` presets pin the correct
`--zx-dir` explicitly; **if you invoke `create_datacards_ctag2d.py` directly with a
with-ZX card and no `--zx-dir`, you will silently get the wrong ZX estimate.**

## What "surgical rebin" means (tl;dr — full story in `outputs/.../negbin_rebinning/README.md`)

The original 20-bin quantile scheme has 9 histograms (Signal nominal + 8 systematic
variants) with negative bin content — never crashed combine (process totals stay
positive) but traced to two sparse low-`mva_score_Signal` bins where Signal's
private LHE sample's ~26% negative-weight-event fraction fluctuates on ~15-26 raw
MC events. `--merge-bin-ranges "2-4,7-10"` merges just those two regions (20 → 15
bins), leaving every other bin at full resolution: **r = 309.5, identical to the
unmerged 20-bin median, but 0/46 histograms negative** — a strict improvement, not
a sensitivity trade-off (unlike uniform coarsening, which does cost real
sensitivity: 10 bins → r=349.5, 8 bins → r=374.0).

**Caveat**: the merge ranges are empirically tuned for the current parquets' data-driven
quantile edges. If the underlying scored parquets change enough to shift those edges,
re-run the diagnostic scripts in `outputs/combine_run3_90160_ctag2d_negbin_rebinning/scripts/`
(01-05) before trusting the same `"2-4,7-10"` string.

## Environments

Two separate shells — mixing them corrupts the python interpreter path (established
gotcha throughout this project):

```bash
# Datacard building (create_datacards_ctag2d.py)
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh

# Combine itself (v10.5.0) -- separate shell
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
```

`run_pipeline.sh` handles both automatically.

## Status / open items

- **START HERE next session**: rebuild CMSSW (`./setup_cmssw.sh --force`), re-validate
  all 3 `run_pipeline.sh` presets, THEN stage+commit+push (see below) -- see the
  blocked-banner above for full detail.
- **Nothing has been committed to git yet.** This whole `combine/` directory (code +
  README, not `outputs/`/`CMSSW_14_1_0_pre4/` per `.gitignore`) is new, untracked
  content sitting in the `higgscharm` repo's working tree (branch
  `HcZZ-zx-estimation`). Stage ONLY this directory
  (`git add combine/ .gitignore` from the repo root) before committing --
  **that repo also has 38 pre-existing modified/untracked files from unrelated
  in-progress work; do not `git add -A` or otherwise sweep those in.**
  Push target confirmed 2026-08-17: `origin` (`github.com:Saranya-Nandakumar/
  higgscharm.git`, her own fork) -- never the `ua` remote
  (`ua-cms/higgscharm.git`, the collaborator's upstream) without being told to.
- This pipeline's outputs are a candidate/comparison result, not yet the analysis's
  official production number — the 1D c-tag SF scheme (`b-hive_ttcc/combine/`) is
  still what `README_HcZZ.md` documents as current production.
- No-ZX vs with-ZX: Z+X (the data-driven fake-rate background) is real and used
  elsewhere in the analysis — `--skip-zx` here is a deliberate MC-only comparison
  mode, not a claim that Z+X should be dropped from the real analysis.
- `ZX_rate`/`ZX_norm` degeneracy noted 2026-08-16 (impact ranking anomaly, see the
  negbin_rebinning README) is not fully root-caused.
