# H→ZZ + charm (H+c) — analysis pipeline

End-to-end guide for the **`hplusc_mva_4class`** workflow: NanoAOD → parquet →
MVA → datacard → limit. This documents the H→ZZ-specific layer built on top of
the base `higgscharm` framework (filesets, workflow YAML structure, condor
submission) — same framework family as `Chirayu18/higgscharm` (`hww-analysis`
branch), which runs the parallel H+c→WW analysis and is the template this doc
follows.

**Repo (EOS)**: `/eos/user/s/snandaku/Higgscharmnew/higgscharm`
**Repo (AFS)**: `/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm`
**Combine builder**: `/eos/home-s/snandaku/b-hive_ttcc/combine/create_datacards.py`
**Env**: LCG_105 (python/data-loading), separate CMSSW shell for combine itself

> **Two checkouts, not synced by git.** EOS and AFS `Higgscharmnew/higgscharm`
> are independent working copies (different `origin` remotes: EOS points at
> `Saranya-Nandakumar/higgscharm`, AFS at `ua-cms/higgscharm`), and have
> drifted before — a fix landed in one and not the other at least twice
> (PDF-normalization fix, Z+c CR `--jobflavor`). **Check which copy has the
> state you need before trusting either one**, and mirror any pipeline-code
> change to both.

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
```

---

## Contents

- [Pipeline overview](#pipeline-overview)
- [0. Prerequisites](#0-prerequisites)
- [1. Process NanoAOD → parquet](#1-process-nanoaod--parquet)
- [2. Postprocess: merge + MVA inference (consolidated)](#2-postprocess-merge--mva-inference-consolidated)
- [3. MVA: training](#3-mva-training)
- [4. MVA: inference](#4-mva-inference)
- [5. Build combine inputs](#5-build-combine-inputs)
- [6. Run the limit](#6-run-the-limit)
- [7. Impact ranking](#7-impact-ranking)
- [Systematics](#systematics)
- [Known traps](#known-traps)
- [Open items / next techniques to port from hww-analysis](#open-items--next-techniques-to-port-from-hww-analysis)

---

## Pipeline overview

**Production** (`hplusc_mva_4class`, historical split path — §4):
```
NanoAOD (DAS/EOS)
   |  runner.py --workflow hplusc_mva_4class          condor, 4 eras
   v
parquet   outputs/hplusc_mva_4class/<era>/<dataset>_<jobid>/base/    per-dataset, per-partition
   |  analysis/postprocess/run_mva_postprocess.py       (NOT run_postprocess.py --postprocess)
   v
mva parquet   outputs/hplusc_mva_4class_scored_v4/<era>/<dataset>_<jobid>/   mva_score_{qqZZ,ggZZ,Signal,Other_Higgs}
   |  b-hive_ttcc/combine/create_datacards.py
   v
datacard  outputs/combine_run3_<window>_v5/{datacard,histograms}.*    1 channel x 4 processes (+ZX)
   |  combine (separate CMSSW shell)
   v
limit
```

**New workflow variants going forward** (consolidated path — §2):
```
NanoAOD (DAS/EOS)
   |  runner.py --workflow <variant>                  condor, 4 eras
   v
parquet   outputs/<variant>/<era>/<dataset>_<jobid>/base/    per-dataset, per-partition
   |  run_postprocess.py --postprocess --mva-inference   (merge -> parquets_<sample>/ -> score, one step)
   v
mva parquet   outputs/<variant>_mvascores/<era>/<sample>/    mva_score_{qqZZ,ggZZ,Signal,Other_Higgs}
   |  b-hive_ttcc/combine/create_datacards_<variant>.py   (copy create_datacards.py, point at new dirs)
   v
datacard -> combine -> limit
```

One structural difference remains from the `hww-analysis` pipeline this
mirrors either way: one channel (`mva_score_Signal` as a single discriminant)
instead of argmax-defined SR+CR channels — kept deliberately, see the note in
[Open items](#open-items--next-techniques-to-port-from-hww-analysis).

---

## 0. Prerequisites

**Grid proxy** — re-init before any condor campaign:
```bash
voms-proxy-init --voms cms --valid 192:00
```

**EOS space** — check before any large campaign:
```bash
eos quota /eos/user/s/snandaku
```
This account runs close to its 2 TB group quota routinely (hit 87.82% once
and caused a real condor failure wave — see the Z+c CR incident in
`Tasks/HcZZ-fake-rate.md`, 2026-08-12). **Never delete anything on EOS
without confirming first.**

**Combine environment** — must be a **separate shell** from the LCG_105 one
above; mixing them corrupts the python interpreter path and crashes combine.
**Now lives inside `Higgscharmnew` itself** (added 2026-08-14, "everything in
one place"), not the older `Analysis/combine/` copy:
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /eos/user/s/snandaku/Higgscharmnew/combine/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
```
Same release + combine tag as the pre-existing `/eos/user/s/snandaku/Analysis/
combine/CMSSW_14_1_0_pre4/` copy (`CMSSW_14_1_0_pre4`, `el9_amd64_gcc12`,
`HiggsAnalysis/CombinedLimit` @ `v10.5.0`/commit `af57d49e`) — built fresh
here rather than copied (CMSSW areas don't reliably relocate). **Fully
validated 2026-08-14**, not just a version-string check: ran
`combine -M AsymptoticLimits` on this new environment against the real
`combine_run3_100150_noZX/datacard_100150_fullsyst_noZX.txt` datacard and
got median r = **312.5000**, an exact bit-for-bit match to the documented
reference number. The older `Analysis/combine/` copy still exists and still
works — this doesn't replace it, just adds a copy inside the consolidated
pipeline location.

**Long jobs** run in `tmux`, not `nohup`.

---

## 1. Process NanoAOD → parquet

```bash
python3 runner.py --workflow hplusc_mva_4class --year 2022postEE \
        --submit --eos --output_format parquet
```

Emits **weight-based** shape systematics baked in as extra `weight_<name>Up/Down`
columns (pileup, PS ISR/FSR, LHE scale, LHE PDF/alphaS, c-tag SF) — unlike
`hww-analysis`, there are currently **no object-shift (JES/JER/lepton
scale+res) shifted parquet directories** produced by this workflow. That means
the datacard has no `CMS_scale_j`/`CMS_res_j`/etc. rows at all yet (see
[Open items](#open-items--next-techniques-to-port-from-hww-analysis)).

Also writes a `sumw/<partition>.json` sidecar per MC chunk — the
**true pre-selection** `Σ genWeight`, written unconditionally even for
zero-event chunks (see the sumw fix in
[Known traps](#known-traps)).

Years: `2022preEE`, `2022postEE`, `2023preBPix`, `2023postBPix`.

**Monitor / resubmit:**
```bash
watch condor_q
python3 jobs_status.py --workflow hplusc_mva_4class --year 2022postEE --eos
```

---

## 2. Postprocess: merge + MVA inference (consolidated)

**Updated 2026-08-14 (user decision): new workflow variants should use the
consolidated `run_postprocess.py --postprocess --mva-inference` path below,
not the split/skip approach production history left behind.** Kept both
described here since `hplusc_mva_4class_scored_v4` (the directory every
existing r/κc reference number in
`Notes/HcZZ-combine-results-reproducibility.md` was built from) was produced
the old way — don't reinterpret those numbers as having gone through this
path, and don't re-run production through this path without a deliberate
decision to rebuild the reference numbers too.

**Historical state, still true for `hplusc_mva_4class`**: `run_postprocess.py`
supports the same `--postprocess`/`--skipmerging` merge step `hww-analysis`
uses to combine per-dataset parquets into per-process files before MVA
inference, plus a `--mva-inference` flag that runs inference on that merged
output directly (`analysis/postprocess/mva_inference.py::MVAPostProcessor`,
the *same* class `run_mva_postprocess.py` uses — one inference implementation,
two CLI entry points, not a duplicate). **Production never used either flag
together**: `run_mva_postprocess.py` was pointed directly at the raw,
unmerged, per-dataset-per-job-partition parquet tree from step 1 and globs
every `*.parquet` file it finds recursively, skipping the merge step entirely.

That split-path choice was **not a correctness bug** — the two things
`hww-analysis`'s README warns merging protects against were already handled
another way (sumw sidecars, filename-based process matching — see below) —
but it also means the "merge → infer" flow had **never actually been run**
for this analysis before 2026-08-14.

**Now fully working, confirmed by a real run AND a real accuracy check, not just code
inspection**:
```bash
python3 run_postprocess.py -w hplusc_mva_4class -y 2023postBPix \
    --postprocess --output_format parquet --nocutflow \
    --mva-inference \
    --mva-config /eos/user/s/snandaku/Higgscharmnew/higgscharm/models/hc_zzto4l_mw_training_4class_nomass.yml \
    --mva-model  /eos/user/s/snandaku/Higgscharmnew/higgscharm/models/best_model.pt \
    --mva-mass-window
```
**Use `models/best_model.pt` + `models/hc_zzto4l_mw_training_4class_nomass.yml`
(bundled together, documented in `models/README.md`)** — do not substitute a
`b-hive_ttcc/output/TrainingTask/.../best_model.pt` path even if documented
elsewhere as "production" (e.g. `b-hive_ttcc/combine/README.md` documents
`train_loose_oldcwithmassbal` for a *different* consumer, Z+X inference).
Exits 0, scores every dataset in the era, writes a per-sample
`<sample>_mva_scores.coffea` histogram file for each. **`--nocutflow` is
required** for this workflow (see Known traps) — without it the run gets all
the way through scoring-prep and then crashes in the cutflow-summary print,
one step before `--mva-inference` even starts.

Getting here took fixing **ten** independent plumbing/config bugs found by
actually running it against the full production output tree for the first
time, PLUS a real config-composition bug found afterward when the resulting
scores looked wrong (see [Known traps](#known-traps) and second-brain
memories `hww_analysis_pipeline_review` and
`hczz_mva_postprocessor_root_cause` for the complete blow-by-blow).

**Validated with a full-population accuracy check, not a small sample**: all
~40 real datasets in 2023postBPix, 526,011 events, argmax accuracy = **71.04%**
— exceeds the model's own native b-hive `InferenceTask` ground-truth benchmark
(69.64% on its own held-out test set) for the identical checkpoint. Mean
P(Signal|true class) — qqZZ 0.012, ggZZ 0.006, Signal 0.617, Other_Higgs
0.265 — closely matches that same native ground truth (0.068/0.086/0.598/0.231).
One known, real (not a bug) weak point: qqZZ recall is only 28.2%, mostly
misclassified as ggZZ — a genuine model characteristic (these two continuum
processes are kinematically similar, documented previously in this project's
"MVA vs Tagger" work), not a pipeline defect; Signal recall (78.9%, the class
that matters for the physics) is strong.

**Production's existing results are unaffected.** Checked directly:
production's actual `hplusc_mva_4class_scored_v4` parquets (source of every
r=327.0-style reference number) have `mva_score_Signal` on a completely
different scale (~[-0.86, 0.41], Signal mean 0.406, sensible separation from
qqZZ's -0.858) than this tool's `[0,1]`-ish softmax-style score — meaning
production was **never** produced via `MVAPostProcessor`/this consolidated
path at all. Nothing fixed or changed today touched production's scoring.
Reads merged `parquets_<sample>/` dirs (produced by the `--postprocess` merge
step), writes scored output to `outputs/<workflow>_mvascores/<year>/<sample>/`
by default (override with `--mva-output`) — a **different directory
convention** than `hplusc_mva_4class_scored_v4`, so `create_datacards.py
--scored-dir`/`--sumw-dir` need pointing at the new location, not assumed to
match.

**Why this still matters even though sumw/per-process aggregation are already
handled**: `hww-analysis`'s README documents a real incident — "inference
must cover all shift directories, not just nominal... cost 500 [limit]
units" — caught only because counting merged-per-process directories is
trivial (`find ... -name mva -type d | wc -l`, one number to check).
Verifying completeness over the old *per-partition* tree (dozens of numbered
subfolders per dataset, e.g. `HPlusCharm_2022postEE_1` through `_32`) is much
easier to get silently wrong — this consolidation removes that risk for any
new workflow that adopts it, even though it hasn't bitten `hplusc_mva_4class`
in practice (no object-shift directories exist yet to miss, see §1).

---

## 3. MVA: training

**Model**: `MLP_HcZZ_MW_Deep_4class`, 4 classes `[qqZZ, ggZZ, Signal, Other_Higgs]`
**Config**: `hc_zzto4l_mw_training_4class_nomass`
**Training run**: `train_loose_oldcwithmassbal`

Trained with **b-hive** (`b-hive_ttcc/`), separate from the coffea repo — same
tool `hww-analysis` uses for its own training. See `b-hive_ttcc/README.md` for
the training invocation; not duplicated here since it hasn't changed as part
of this doc pass.

---

## 4. MVA: inference (legacy path — still what production uses)

This is what actually produced `hplusc_mva_4class_scored_v4` and every
existing reference number. Kept documented as-is; §2 above is the path for
new workflow variants going forward, not a replacement for this section's
history.

```bash
python3 analysis/postprocess/run_mva_postprocess.py \
    --input-dir  /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class/2022postEE \
    --output-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_scored_v4/2022postEE \
    --config /path/to/b-hive_ttcc/config/hc_zzto4l_mw_training_4class_nomass.yml \
    --model  /path/to/b-hive_ttcc/output/TrainingTask/.../best_model.pt \
    --mass-window-min 90 --mass-window-max 160 \
    --all-eras --workers 8
```

Writes `mva_score_{qqZZ,ggZZ,Signal,Other_Higgs}` into every parquet file it
finds under `--input-dir`, recursively (`Path(input_dir).glob('**/*.parquet')`)
— there is no per-shift-directory iteration to miss the way `hww-analysis`'s
`run_inference.py` has, **because this pipeline doesn't produce shift
directories yet** (§1). If/when object-shift systematics are added, this
script (or its call site) will need the same "verify the count" discipline
`hww-analysis`'s README enforces.

---

## 5. Build combine inputs

```bash
python3 create_datacards.py \
  --scored-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_scored_v4 \
  --sumw-dir   /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class \
  --zx-dir     /eos/user/s/snandaku/Analysis/zx_background_mva_os_90160_featmajor \
  --output     /eos/user/s/snandaku/Analysis/combine_run3_90160_v5
```

One global channel (`hczz`), discriminant = `mva_score_Signal`, 20 quantile
bins by default. Full reproduction commands (all windows/bin-counts tried)
live in `second-brain/Notes/HcZZ-combine-results-reproducibility.md` — that
note is the results/reproducibility index; this README is the pipeline
structure doc, kept deliberately non-duplicative of it.

> **Production switched to the 2D c-tag SF, 2026-08-14** — decision made
> after confirming applying both 1D and 2D SFs together would double-count
> the same correction (a 2D per-category SF already covers a jet's full
> tagger response, it isn't a "companion" to a matching 1D WP). Both
> `hplusc_mva_4class.yaml` (`ctagging_2d: true`, replacing `ctagging: [wp:
> loose]`) and `create_datacards.py` (`CMS_ctag2d` replacing
> `CMS_ctag_b`/`CMS_ctag_c`, `CMS_ctag_light` exclusion retired — the 2D
> scheme has no `eff/(1-eff)` formula to trigger that bug) now reflect this.
> The `atleast_one_cjet` **selection** cut is unchanged (still 1D `loose`
> WP) — matches hww-analysis's own pattern (WP selection + `ctagging_2d`
> only, no 1D SF). **The command above and every existing reference number
> (r=327.0/304.0, κc≈53-57) still reflect the OLD 1D-SF scoring** —
> `hplusc_mva_4class_scored_v4` was produced before this switch. A real
> reprocessing campaign (condor, currently blocked — see Known traps) is
> needed before a new, valid combine number exists under the 2D scheme.

---

## 6. Run the limit

```bash
# (combine env from §0, separate shell)
cd /eos/user/s/snandaku/Analysis/combine_run3_90160_v5
combine -M AsymptoticLimits -m 120 --run blind --rAbsAcc 0.00001 --rRelAcc 0.00001 \
  datacard_mva_with_zx.txt -n HcZZ_v5_check
```

`--run blind` suppresses the fake Asimov-derived "Observed" line; tight
`rAbsAcc`/`rRelAcc` kept as a standing habit (confirmed bit-identical vs.
default tolerance in this project, but the SMEFT sister-project's own note
flagged default tolerance as sometimes too loose to resolve small
differences between similar-sensitivity models).

Current production reference (no-ZX): median r = 327.0 [90,160] full-syst /
316.5 stat-only. Full table (all windows, bin counts, κc conversions):
`second-brain/Notes/HcZZ-combine-results-reproducibility.md`.

---

## 7. Impact ranking

`b-hive_ttcc/combine/impact_ranking.py` — added 2026-08-14, ported/adapted
from `hww-analysis`'s `scripts/combine/make_impact_plot.py`. Same method
(cheap approximation, not a real CombineHarvester `-M Impacts` profile fit):
freeze one nuisance at a time, rerun `AsymptoticLimits`, rank by
`|median_r_frozen - median_r_nominal|`. Adapted to auto-parse nuisance names
directly from a plain-text datacard (`<name> <shape|lnN> ...` rows) instead
of `hww-analysis`'s yaml-driven `combine:` block, since `create_datacards.py`
here writes flat datacards, not a `WorkflowConfigBuilder`-mediated one.

```bash
# (combine env from §0)
cd /eos/user/s/snandaku/Higgscharmnew/combine/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
python3 /eos/home-s/snandaku/b-hive_ttcc/combine/impact_ranking.py \
  --card datacard_mva_with_zx.txt \
  --dir  /eos/user/s/snandaku/Analysis/combine_run3_90160_v5 \
  --plot
```

**Fully validated end-to-end 2026-08-14**: `parse_nuisances()` correctly extracted
all 13 real nuisances from `combine_run3_100150_noZX/datacard_100150_fullsyst_noZX.txt`,
and a real 2-nuisance smoke run (`--nuisance CMS_pileup CMS_ctag_b`) against
that same real datacard, using the new in-repo combine environment (§0),
correctly reproduced the known nominal median r = 312.5000 and produced
sane per-nuisance deltas (`CMS_pileup`: −1.0/−0.32%, `CMS_ctag_b`: +0.0/0%).
`--plot` needs matplotlib+uproot (not in the combine CMSSW python by
default — run that step from an LCG env against the written
`impact_ranking.json` if needed) — not yet exercised. Output:
`<dir>/impact_ranking.json` (+ `.png` with `--plot`).

A nuisance whose limit **improves** when frozen usually signals degeneracy
with a free-floating rateParam (here: `ZX_rate`), not a broken nuisance —
same caveat `hww-analysis`'s README notes for their `rate_tt`.

---

## Systematics

Declared in `create_datacards.py`'s `SYSTEMATICS`/`USABLE_SYST` dicts
(weight-based, read as `weight_<name>Up/Down` — same mechanism
`hww-analysis` calls `combine.shape_systematics`).

| kind | examples | notes |
|---|---|---|
| weight shapes | `lhe_pdf`, `lhe_alphaS`, `scalevar_muR`, `scalevar_muF`, `ps_isr`, `ps_fsr`, `CMS_pileup`, `CMS_ctag_b`, `CMS_ctag_c` | per-process usable subset, not uniform — see `USABLE_SYST` and Known traps |
| object shifts | **none** | `hww-analysis` has `CMS_scale_j/e/m`, `CMS_res_j/e/m` from JES/JER/lepton-scale shift directories; this pipeline doesn't produce those directories yet |
| lnN | `lumi_Run3` (1.4%), `pdf_gg`/`pdf_qq`, `QCDscale_ggZZ`/`QCDscale_qqZZ`, `ZX_norm` (30%) | see `b-hive_ttcc/combine/README.md` |
| rateParam | `ZX_rate` | Z+X floats freely from data, same pattern as `hww-analysis`'s `tt` |
| MC stat | **not yet `autoMCStats`** | `hww-analysis` runs `autoMCStats 10` per channel; not currently in this datacard |

`CMS_ctag_light` is **excluded entirely**, not just per-process-gated — see
Known traps.

### `ctag2d` variant (active as of 2026-08-17)

> **Default mass window changed to `[100,150]` GeV, 2026-08-18** (tighter S/√B,
> 0.0054 vs 0.0043 at `[90,160]`). `combine/run_pipeline.sh` with no args now
> builds `[100,150]` by default (`--window 90160` for the older window). **Z+X
> (fake-rate background) is deliberately excluded from scope for now** (decision
> 2026-08-18) — the production default at this window is MC-only (r=311.5
> full-syst / 279.0 stat-only). `[90,160]`'s own with-ZX number (r=329.0, from
> the OS/featmajor production) remains available via `--window 90160 --with-zx`.

Scripts: `combine/scripts/create_datacards_ctag2d.py` ([90,160] GeV, canonical),
`create_datacards_ctag2d_100150.py` ([100,150] GeV copy, `MASS_WINDOW` hardcoded
at line 117 — no CLI flag for the window), `create_datacards_ctag2d_3ratio.py`
(joint nested-tree 3-ratio discriminant variant of the same pipeline, see below).
All three share the same `SYSTEMATICS`/`USABLE_SYST` dicts and sumw/xs-scaling
logic as `create_datacards_ctag2d.py` — edit all three together when changing
the systematics model.

**16 systematics total** (up from 11, 2026-08-17): 5 `lnN` + 11 shape.

| kind | name | size/processes | notes |
|---|---|---|---|
| lnN | `lumi_Run3` | 1.4%, all | |
| lnN | `pdf_gg` | 5%, ggZZ+Signal | placeholder |
| lnN | `QCDscale_ggZZ` | 10%, ggZZ | placeholder |
| lnN | `BR_HZZ4l` | 2%, Signal+Other_Higgs | **added 2026-08-17**, sourced from Felix Heyen thesis Appendix D; no per-event weight column exists, pure rate lnN |
| lnN | `QCDscale_qqZZ` | 4%, qqZZ | **added 2026-08-17**, same source; separate from `QCDscale_ggZZ`, not a duplicate |
| shape | `lhe_pdf`/`lhe_alphaS`/`scalevar_muR`/`scalevar_muF` | qqZZ only (+`lhe_alphaS` for Other_Higgs) | excluded for Signal/ggZZ/most of Other_Higgs — see per-process reasons below |
| shape | `ps_isr`/`ps_fsr`/`CMS_pileup`/`CMS_ctag2d` | all 4 processes | |
| shape | `CMS_eff_e_reco_20to75`/`above75`/`below20` | all 4 processes | **added 2026-08-17** — already computed and stored in the scored parquets (`weight_CMS_eff_e_reco_*_<year>Up/Down`), never wired into `USABLE_SYST` before. Verified sane (≤6%, symmetric) for every process incl. Signal/HPlusBottom via direct per-event ratio check — a detector-level correction, unaffected by the private-LHE-sample bugs below. |

**Why `lhe_pdf`/`lhe_alphaS`/`scalevar_muR`/`scalevar_muF` are missing for some
processes** (full reasoning already inline as comments above `USABLE_SYST` in
the script; summarized here for discoverability):
- **Signal**: `lhe_pdf` — root cause found (missing `1/sqrt(N-1)` NNPDF
  normalization, fixed in source for future reprocessing) but a downstream
  patch on existing parquets still leaves ~88% PDF uncertainty (vs qqZZ's
  4.3%) — excluded until real reprocessing lands. `lhe_alphaS` — not a bug,
  Signal's PDF replica set genuinely has only 101 members, no dedicated
  αs-variation member exists. `scalevar_muR`/`muF` — separate, still-undiagnosed
  bug: Up and Down both shift the *same* direction (should be symmetric ±),
  specific to this private sample's `LHEScaleWeight` branch.
- **ggZZ**: simplest case — no `LHEScaleWeight` branch exists at all in these
  MC samples; `lhe_pdf`/`lhe_alphaS` degenerate (nothing to vary).
- **Other_Higgs**: a *pooled* process — official centrally-produced samples
  (ggH/VBF/WH/ZH/ttH/bbH) are fine (properly antisymmetric), but the private
  `HPlusBottom` component (same private LHE chain as Signal) shows the
  identical two bugs — `lhe_pdf` blown up to +857%/−657%, `scalevar_muR`/`muF`
  same-direction Up/Down. HPlusBottom is only ~4% of the yield but dominates
  the shape variation — this is what caused an earlier Combine "Bogus norm"
  crash once a masking clip was removed. `lhe_alphaS` is kept: larger for
  HPlusBottom than the official samples, but not pathological (no sign flip,
  no same-direction Up/Down).

**Not added (checked, deliberately skipped as redundant)**: `weight_scalevar_muR_muF`
and `weight_lhe_pdf_alphaS` (combined variants) exist in the parquets too, but
adding them alongside the already-used separate `muR`/`muF`/`pdf`/`alphaS` rows
would double-count the same uncertainty.

**Not addable without upstream work**: no muon ID/isolation/reco weight columns
exist anywhere in the scored parquets (checked directly across Signal/ggZZ/
Other_Higgs schemas) — needs new production, not a datacard-script change.
JES/JER object-shift systematics also need new shift-directory production
(see Open items #4).

**3-ratio discriminant variant** (`create_datacards_ctag2d_3ratio.py`): joint
$(r_1,r_2,r_3) = (P_s/(P_s{+}P_{qqZZ}), P_s/(P_s{+}P_{ggZZ}), P_s/(P_s{+}P_{OH}))$,
nested/conditional quantile tree binning (ported from `b-hive_ttcc/combine/
create_datacards_3ratio.py`'s "nested" mode), same 16-systematic set. Beats
plain score on full-syst at both mass windows (1–3% tighter) but not stat-only
(flat-to-slightly-worse) — opposite of the pattern documented in
`second-brain/Notes/SM-3ratio-kappa-decomposition-note.md` for the old
ctag_light/with-ZX test (4–10% better, both tiers). **Root cause found**:
rerunning the identical code against the *old* (`hplusc_mva_4class_scored_v4`,
scored 2026-08-04) parquets reproduces that old reference bit-exactly —
proves the 3-ratio construction/binning code itself is correct. The
`ctag2d`-scored parquets (scored 2026-08-16) sit on the other side of a
2026-08-14 MVA-inference bugfix (era-aware jet-tag one-hot boundaries, feature-
composition fix — see the config-composition trap below) that shifted Signal/
Other_Higgs classifier scores ~47–49% (qqZZ/ggZZ unaffected — those bugs are
jet-tagging-specific). Mechanism: each ratio $r=P_s/(P_s+P_\text{bkg})$
saturates in $[0,1]$ as $P_s$ grows, so it structurally captures less of a
large classifier improvement than the unbounded raw score does (raw
Signal-vs-qqZZ separation grew 71% from that fix; each ratio axis only grew
26–30%) — with a fixed 20-bin budget, plain score's full-resolution single
axis exploits that improvement fully, the 3-ratio's $2{\times}2{\times}5$
split across 3 axes cannot. Verified directly from per-event scores, not
hypothesized.

**New issue found while adding the electron-reco systematics above (2026-08-17,
not yet fixed)**: `CMS_ctag2d`'s Down variation is suspiciously flat (≈1.00)
for *every* process, while Up shows a real 9–25% shift — not the expected
symmetric ± pattern for a shape systematic (qqZZ: Up=1.088/Down=1.0001; ggZZ:
1.089/1.002; Signal: 1.249/1.028; Other_Higgs: 1.132/0.999). Looks like
`weight_CMS_ctag2d_*Down` may not be applying a real down-shift. Matters
because `CMS_ctag2d` ranks #2–3 of the nuisances in the impact-ranking plots —
that ranking may be affected. Needs digging into the upstream
`analysis/corrections/ctag2d.py` correction code; out of scope for the
datacard-script work that found it.

---

## Known traps

| trap | detail |
|---|---|
| **No merge step (historical)** | §2 above describes the old split path — superseded 2026-08-14 for new workflows by the consolidated `--postprocess --mva-inference` flow, which had never been run successfully before that date. |
| **`merge_parquets()` required identical schemas, crashed on real data** | Fixed 2026-08-14: rewrote to `pa.concat_tables(tables, promote=True)` (schema union) instead of `dask.dataframe.read_parquet(glob).compute()` (strict schema). Real mismatch found in `EGamma0v1D`/`DYJetsToLL_50` (data/DY, not analysis processes) — some files predate the workflow yaml adding `nSV`/`jet_btagUParTAK4*`/`jet_h_dphi_inclusive`. Signal/qqZZ/ggZZ/Other_Higgs directories confirmed schema-clean. Also fixed: empty-partition crash (`pa.concat_tables([])`) for datasets with zero legitimate rows. |
| **Unregistered private datasets crash multiple postprocess functions** | `2023postBPixHB`-style private datasets (condor-run directly, never added to `<era>_nanov<n>.yaml`) crash `filesets/utils.py::get_process_sample_map`, `postprocess/postprocessor.py::fill_histograms_from_parquets`, and `postprocess/postprocessor.py::save_histograms_by_sample`'s `get_lumi_weight` call, all with a bare `KeyError`. All three fixed 2026-08-14 to skip/degrade gracefully instead of crashing (the last one skips the whole sample's histogram build — no safe default xsec exists). A fourth instance, in the final cutflow-summary print (`key_process_map[key]` for `SomeSMSignal`), has no code fix — **use `--nocutflow`** (see below) instead, since that whole block is optional reporting output, not something `--mva-inference` needs. The real underlying gap (these datasets were never registered at all) is still open. |
| **`--nocutflow` required for `hplusc_mva_4class` + `--mva-inference`** | Without it, `run_postprocess.py` crashes in the final cutflow-summary print (see above) one step before MVA inference ever starts. Always include it for this workflow. |
| **Use `models/best_model.pt` + `models/hc_zzto4l_mw_training_4class_nomass.yml`, not a `b-hive_ttcc/output/TrainingTask/.../best_model.pt` path** | These two files are the DOCUMENTED (`models/README.md`), checkpoint-bundled pair for `run_postprocess.py --mva-inference`. The `.pt` is byte-identical (MD5-confirmed) to `train_loose_oldcwithmassbal/best_model.pt`, dated 2026-03-23 — same model, not a different one — but `b-hive_ttcc/config/hc_zzto4l_mw_training_4class_nomass.yml` (same filename, different file) had drifted to 103 `global_features`. An earlier same-day fix there (commenting out `jet_ht`/`nSV`) hit the right COUNT (101) by coincidence but the WRONG COMPOSITION (kept `jet_h_dphi_inclusive`, dropped `jet_ht` — the correct config is the reverse). Matching total dimension is NOT sufficient evidence a feature list is correct — diff the actual `global_features` names against a known-good config. Both `.yml` files now correctly match (fixed 2026-08-14) — see `hczz_mva_postprocessor_root_cause` memory for the full diagnostic trail (culminating in a full-population 71.04% argmax accuracy check on real data, exceeding the model's own native ground-truth benchmark of 69.64%). |
| **Two real code bugs in `mva_inference.py`, fixed 2026-08-14, unrelated to the config issue above** | (1) `MVA_JET_TAG_BOUNDARIES` was a single fixed (wrong, "pre-official-calibration") boundary set for the `jet_is_*` one-hot features (33/101 inputs) — now era-aware (`JET_TAG_BOUNDARIES_BY_ERA`, same real per-era boundaries validated for `ctag2d.py`), threaded via a new `MVAPostProcessor(era=...)` kwarg; `run_mva_postprocess.py`/`run_zx_inference.py` reuse one processor across a per-era loop so each needed `processor.era = era` set inside the loop, not just at construction. (2) per-candidate-SPECIFIC columns (`z1_l2_pt`, `z2_pt`, etc. — each a length-1 array for one specific lepton/Z role) were read with `idx=j` (the candidate slot), only correct for `j=0`; slots 1-3 silently zeroed. Fixed via a new `_get_candidate_col()` helper that reads specific-name columns at a fixed `idx=0`. Neither bug was the dominant cause of the collapse symptom (re-tested with both fixed, score collapse barely changed) — the config composition bug above was. |
| **`ZZto4L` / `GluGluHtoZZto4L` substring collision** | `"ZZto4L" in "GluGluHtoZZto4L"` — naive dict-order matching silently misrouted every `GluGluHtoZZto4L` event into `qqZZ` for an entire session. Fixed 2026-08-04 by matching the longest key first in `get_process_from_path()`; the fix is now load-bearing, don't reorder `PROCESS_MAPPING` casually. |
| **sumw from parquet shard metadata** | Same failure mode `hww-analysis` warns about (undercounts low-efficiency samples). Fixed here via `sumw/*.json` sidecars (§1) — `load_sumw()` raises rather than falling back if they're missing, don't defeat that. |
| **`CMS_ctag_light` sign flips (historical, retired 2026-08-14)** | `ctag.py`'s untagged-jet formula `(1-SF*eff)/(1-eff)` goes negative when `SF*eff>1` — hits the light-flavor mistag SF's up variation in some pT/η bins. Confirmed genuine sign flips in every process's existing 1D-scored parquets. **Clamped at the source, not truly fixed** — a numerical safety net (BTV-POG-standard mitigation) so the code doesn't crash, not a resolution of the underlying disagreement; the clamp is exactly the kind of thing this project's standing rule rejects as "fixed" (see the earlier ratio-clip incident). This is why `CMS_ctag_light` stayed excluded from the datacard even after clamping, right up until production switched its whole SF scheme to `ctagging_2d` (which has no `eff/(1-eff)` formula at all, so this failure mode is structurally impossible, not just clamped around). `hww-analysis`'s `ctag.py` has the identical unclamped formula, still live in their repo — irrelevant to their own results only because their production never calls it (`ctagging_2d` there too), same as ours going forward. |
| **Condor rejects `/eos` paths in submit files (CERN batch policy change since ~March)** | `condor/submit.sub` + `submit_condor.py`'s `CONDORDIR`/`LOGDIR` substitution point `executable`/`output`/`error`/`log` at literal `/eos/...` paths — confirmed working from this exact repo in March 2026 (found a successful old `.log`), now rejected outright: `ERROR: Failed to commit job submission into the queue... Standard batch schedds cannot use /eos paths directly`. **Workaround confirmed 2026-08-14: submit from the AFS checkout instead** (`/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm/`, same `runner.py -w <workflow> -y <era> --submit --eos --output_format parquet`) — condor accepts submit files whose paths originate from AFS even though `--eos` still points job *output* at `/eos`; no code change needed, just run from a different filesystem. Not a permanent code fix (submit.sub itself still hardcodes `/eos` if invoked from an EOS cwd) — still open whether to patch `submit.sub` properly (xrootd URLs / EOS-submit schedds) or keep "submit from AFS" as the accepted permanent workflow. |
| **AFS and EOS checkouts' fileset registries have diverged — EOS is missing signal samples entirely** | Confirmed 2026-08-14 right before the ctag2d condor campaign: running `runner.py`'s fileset discovery from the EOS checkout reports `hb not availabe` / `smsignal not availabe` for every era (would silently produce a datacard with zero signal events), while the identical discovery from AFS finds them correctly (`SomeSMSignal`, `HPlusCharm_<era>` all found via DAS phys03). Root cause not yet dug into — EOS's fileset registry JSON is just stale relative to AFS's. Deliberately not fixed before the campaign (user: "I know the issue... afs is fine. we can fix it after job submission as well") — still open. |
| **`lhe_pdf` formula: Hessian vs. MC-replica** | Current formula (`sqrt(Σ(w_k-w0)²)`, no `1/sqrt(N-1)`) is correct **only** for Hessian-eigenvector PDF sets. `hww-analysis`'s `lhepdf.py` uses the identical formula, justified there as correct because their samples are official/central production (Hessian-reduced NNPDF3.1). Our private `HPlusCharm`/`HPlusBottom` samples may carry a different (MC-replica) PDF format — unconfirmed, blocked on the private NanoAODs being gone from all storage sites. |
| **Signal/HPlusBottom negative-weight fraction is ~25%, not incidental** | Measured directly from `hplusc_mva_4class_scored_v4` (2026-08-14): `HPlusCharm`/`HPlusBottom` run **24–26% negative `weight_nominal`** across every era, roughly **8–17× higher** than every other process (`ZZto4L` 0.18%, `GluGlu*` ggZZ ~0.003–0.01%, Higgs backgrounds `WH`/`ZH`/`TTH` 1.4–1.7%). This is the same two processes with the already-diagnosed `lhe_pdf`/`scalevar_muR/muF` blowups — plausibly the same root cause (private amc@nlo-based generation), not yet connected. See [Open items](#open-items--next-techniques-to-port-from-hww-analysis). |

---

## Open items / next techniques to port from `hww-analysis`

Priority order set 2026-08-14; update as items land.

1. **`ctag2d` pseudo-continuous SF migration — done, active in production
   as of 2026-08-17** (16-systematic datacard pipeline, both mass windows,
   see the `ctag2d` variant subsection under Systematics above).
   `analysis/corrections/ctag2d.py`
   ported from `Chirayu18/higgscharm`'s `hww-analysis` branch (same
   `L0=0, C0..C4=40..44, B0..B4=50..54` WP-code mapping, `(syst, flavor, wp,
   abseta, pt)` evaluate order) — this resolved the CERN-SSO-gated-doc
   blocker in `hczz_pnet_ctag_sf_migration` memory outright, no guessing
   needed. **Directly verified against our real files** (2026-08-14,
   `flavTaggingSF_2022postEE.json.gz`): raw schema matches exactly, all WP
   codes evaluate without error, `up_Total`/`down_Total` give distinct sane
   values. A synthetic multi-jet smoke test of the full `CTag2DCorrector`
   class (0/1/2-jet events, product-across-jets combination) also passed.
   The file additionally exposes a full per-source decorrelation structure
   (`up/down_Stat_flav{L,C,B}_<cat>_pt_<bin>`, `up/down_JES`, `up/down_PUWeight`,
   etc.) beyond the single `Total` nuisance currently wired — worth revisiting
   once the coarse version is validated in a real datacard.

   **Scope decision made during porting**: `hww-analysis` applies this to one
   "candidate c-jet"; this port instead multiplies a per-jet SF across every
   jet in `events.selected_jets` (`ak.prod`), matching the scope of the 1D
   `CTagCorrector` it's meant to replace. Flagged in the module docstring for
   review.

   **Update 2026-08-14 (later): comparison workflow + datacard variant
   built.** `analysis/workflows/hplusc_mva_4class_ctag2d.yaml` (both
   checkouts) — byte-identical to production except `event_weights` swaps
   `ctagging: [wp: loose]` for `ctagging_2d: true`. The `atleast_one_cjet`
   **selection cut** (`working_points.jet_ctagging(events, 'loose', year)`)
   is deliberately untouched — that's a separate decision from the SF
   scheme (see "selection vs. SF" note below) and changing it would
   invalidate the MVA training + every reference number, not just add a
   comparison point. `b-hive_ttcc/combine/create_datacards_ctag2d.py` (copy
   of `create_datacards.py`, `CMS_ctag_b`/`CMS_ctag_c` swapped for a single
   `CMS_ctag2d` nuisance, default paths pointed at the ctag2d workflow's own
   output tree so it can never collide with production).

   **Offline yield-impact check, real data, no reprocessing needed**: since
   `CTag2DCorrector` only needs `jet_pt/eta/btagPNetCvL/btagPNetCvB/
   hadronFlavour` — already stored in existing production parquets as the
   *full* selected-jet collection, not just c-tagged jets — ran it directly
   against those stored columns and compared to the existing
   `weight_CMS_ctag_b × weight_CMS_ctag_c × weight_CMS_ctag_light` product,
   without touching NanoAOD/condor at all. Single-file check (10 events,
   `GluGlutoContinto2Zto4Tau_1`/2023postBPix): new/old yield ratio 0.942.
   Full-statistics version (all 4 eras, all 4 processes) run 2026-08-14 —
   see the "Offline ctag2d yield comparison" note below for the result.
   This offline check is a rough sanity read only — it reuses the OLD
   1D-scheme's selection (whatever jets already passed `loose` c-tag in the
   stored parquets), not a true reprocessing under the ctag2d workflow, so
   treat it as directional, not a substitute for a real campaign.

   **BUG FOUND AND FIXED 2026-08-14 (later): category boundaries were wrong.**
   The edges ported from `hww-analysis` (`[0.0,0.250,0.452,0.808,1.0]` /
   `[0.0,0.006,0.017,0.055,0.761,0.944,0.985,0.995,1.0]`) turned out to be the
   **2024 UParT** boundaries, not 2022/2023 PNet — caught when the user
   pasted the real `etsai.web.cern.ch` doc page (every entry there is
   explicitly "Tagger: UParT v2, NanoAODv15", files under `phys_top/Run3Vcb/`,
   not our `phys_higgs/cmshgg/` PNet files) and the actual per-era
   `HPC_ctag_WPs` JSON, which matched exactly what was already correctly
   implemented (independently, for a different purpose — MVA training
   features) in `b-hive_ttcc/utils/coffea_processors/lz4_hczz_processor.py::
   JET_TAG_BOUNDARIES_BY_ERA`. `ctag2d.py` now has real per-era boundaries
   (`BOUNDARIES_BY_ERA`, 4 PNet eras) copied from that confirmed-correct
   source; `_category_np()` takes an `era` argument. Re-validated: clean
   tiling of the full plane (0 unassigned points, 500×500 grid, all 4 eras),
   all 11 WP codes still evaluate. **The coordinate transform itself was
   never wrong** — only the edge values.

   **Corrected full-statistics offline comparison** (all 4 eras × all 4
   processes, re-run 2026-08-14 after the fix):

   | Process | Old yield (1D SF) | New yield (2D SF) | Ratio | Buggy-version ratio |
   |---|---|---|---|---|
   | Signal | 17,382.1 | 17,840.0 | **+2.63%** | +2.67% |
   | qqZZ | 1,082,525.6 | 1,083,540.3 | **+0.09%** | −3.40% |
   | ggZZ | 841,401.4 | 851,363.0 | **+1.18%** | −3.67% |
   | Other_Higgs | 3,271,437.1 | 3,278,789.6 | **+0.22%** | −2.33% |

   The corrected picture is much more modest than the buggy version
   suggested: Signal is still up ~2.6%, but backgrounds are close to flat
   (qqZZ +0.09%, Other_Higgs +0.22%) or even slightly up (ggZZ +1.18%) — not
   down 2-4% each like the wrong boundaries implied. **Do not cite the
   buggy-version ratios for anything** — they're left in the table only to
   show how much a wrong category mapping can distort a yield check that
   otherwise runs and completes without any error.

   **Reciprocal finding**: `hww-analysis`'s own `ctag2d.py` uses this same
   single fixed (2024 UParT) edge set across all 4 of *their* PNet eras too
   — their production analysis likely carries the identical bug. Worth
   flagging to that repo's maintainer.

   **Selection vs. SF, clarified**: the 2D file's `wp` categories
   (`L0`/`C0`-`C4`/`B0`-`B4`) are not a selectable "loose"-equivalent working
   point the way the 1D tagger has — they're the category grid the SF is
   binned in. Using them to redefine the **selection** cut (not just the SF)
   is a separate, bigger decision (different signal acceptance, needs its
   own purity/efficiency study, invalidates current training) — not
   something to bundle into this SF-only migration.

   **Wired but inert in production**: `correction_manager.py` (both EOS +
   AFS) has a `ctagging_2d` branch, gated behind `event_weights.ctagging_2d:
   true` — production `hplusc_mva_4class.yaml` still has only `ctagging:
   [wp: loose]`, untouched. Turning both on in the same workflow would
   double-apply c-tag SFs (both correctors touch the same jets) — this is
   exactly why the comparison lives in a separate `_ctag2d` workflow file
   rather than a flag flip on the production one.
   Also directly retires the `CMS_ctag_light` exclusion above (2D SF
   doesn't use the broken eff/(1-eff) untagged formula) — but only once
   parquets are reprocessed with it enabled.
2. **negrw ML reweighting — evaluate, likely needed.** `hww-analysis` built
   this for V+jets; we have no V+jets background, but the 25% negative-weight
   finding above means the *same class of problem* (NLO-generator
   negative-weight variance blowing up shape systematics) shows up in
   `Signal`/`HPlusBottom` instead. Worth trying before assuming it's
   unnecessary — this is the leading unresolved-bug candidate for the
   excluded `lhe_pdf`/`scalevar_muR/muF` rows in `USABLE_SYST`.
3. **`autoMCStats`** — cheap, no architecture change, not yet in the datacard.
   Update 2026-08 (separate investigation, `ctag2d` pipeline): found
   numerically degenerate with the current binning (several bins have MC-stat
   *errors* 10–90x the bin *content*) — both this note's "cheap" framing and
   that finding are true, not a contradiction; the degeneracy is real but
   specific to today's binning, not evidence `autoMCStats` is a bad idea in
   general.
4. **Object-shift (JES/JER/lepton scale+res) systematics** — bigger lift,
   needs new shift-directory production in `runner.py` plus inference
   re-coverage (§4); not started. **Update 2026-08-17**: electron reco is
   actually *not* blocked — `weight_CMS_eff_e_reco_{20to75,above75,below20}_
   <year>Up/Down` already exist in the `ctag2d` scored parquets (presumably
   from an earlier, unrelated production step) and are now wired into
   `create_datacards_ctag2d*.py`'s `USABLE_SYST` (16 systematics total, see
   the `ctag2d` variant subsection under Systematics above). Muon
   ID/isolation/reco remains genuinely blocked — no such weight columns
   exist anywhere in the scored parquets, checked directly. JES/JER still
   not started, still needs the shift-directory production described here.
5. **Per-event HF-composition-style weight** — `hww-analysis`'s
   `higgs_hf.py` replaces a mis-scoped flat lnN on a pooled background group
   with a per-event GEN-jet-flavour weight. Same shape of problem as our
   `Other_Higgs`/`HPlusBottom` pooling (see the `lhe_pdf`/`scalevar`
   exclusion above) — worth revisiting now that `ctag2d` has landed (item 1
   is done, this was gated on it).
6. **`BR_HZZ4l`/`QCDscale_qqZZ` lnN rows** — **done 2026-08-17**, see the
   `ctag2d` variant subsection under Systematics above.

**Deliberately not adopted**: the multi-channel argmax MVA architecture
(SR + per-background CRs, floating a background via `rateParam` the way
`hww-analysis` floats `tt`). Decision 2026-08-14: keep the single
`mva_score_Signal` discriminant — the current 4-class MVA and all reference
r/κc numbers stay valid; this is a documentation and systematics-parity
effort, not a redesign.
