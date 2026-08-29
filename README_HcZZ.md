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
| weight shapes | `lhe_pdf`, `lhe_alphaS` (excluded, sign bug fixed at source 2026-08-21 but non-retrofittable onto existing parquets — see update below), `scalevar_muR`, `scalevar_muF`, `ps_isr`, `ps_fsr`, `CMS_pileup`, `CMS_ctag2d`, `CMS_eff_e_reco_{below20,20to75,above75}` | per-process usable subset, not uniform — see `USABLE_SYST` and Known traps |
| **muon efficiency SF (ID only)** | **wired 2026-08-21** (`id: loose`), real single-file smoke test passed end-to-end — see the dated update below. `iso`/`trigger` deliberately left off (see rationale in the workflow yaml). Doesn't change any existing r number — needs a full reprocessing to take effect (condor-blocked). |
| **electron ID SF** | not applied (`id: false`) | documented, deliberate — smaller likely magnitude than the muon gap |
| object shifts | `CMS_scale_j`/`CMS_res_j` **in progress**, `CMS_scale_m`/`CMS_res_m` **in progress** (both 2026-08-19, `jec_shifts` workflow — see "Object-shift systematics" update below); production/consumption smoke-tested on 1 file only, never run at scale (condor-blocked) | `hww-analysis` has live `CMS_scale_j/e/m`, `CMS_res_j/e/m` from JES/JER/lepton-scale shift directories |
| lnN | `lumi_Run3` (1.4%), `pdf_gg`, `QCDscale_ggZZ`/`QCDscale_qqZZ`, `BR_HZZ4l` (2%), `ZX_norm` (30%) | values cross-validated against HIG-24-013 2026-08-21 (lumi 1.4% exact match); `pdf_qq` deliberately NOT added — qqZZ's real `lhe_pdf` shape already covers it, a separate lnN would double-count |
| rateParam | `ZX_rate` | Z+X floats freely from data, same pattern as `hww-analysis`'s `tt` |
| MC stat | `autoMCStats <threshold>`, **off by default since 2026-08-19** — confirmed non-convergent at threshold 10/50/100 (11.5h, zero quantiles) | `--automcstats-threshold` CLI flag opts back in for testing; real open gap, not a not-yet-tried item |

`CMS_ctag_light` is **excluded entirely**, not just per-process-gated — see
Known traps.

### `ctag2d` variant (active as of 2026-08-17)

> **Default mass window changed to `[100,150]` GeV, 2026-08-18** (tighter S/√B,
> 0.0054 vs 0.0043 at `[90,160]`). `combine/run_pipeline.sh` with no args now
> builds `[100,150]` by default (`--window 90160` for the older window). **Z+X
> (fake-rate background) is deliberately excluded from scope for now** (decision
> 2026-08-18) — the production default at this window is MC-only. `[90,160]`'s
> own with-ZX number (r=329.0, from the OS/featmajor production) remains
> available via `--window 90160 --with-zx`. **`lhe_alphaS` removed 2026-08-19
> pending a bug fix (see the dated update below).** **Muon ID efficiency SF
> (`CMS_eff_m_id`) added 2026-08-27 — current number: r=301.0 median /
> kappa_c=54.86, 16 systematics** (see the 2026-08-27 dated update below for
> the full CLs band and how it was produced — first result built from a
> genuinely complete, muon-SF-corrected `ctag2d` reprocessing). Superseded:
> r=309.5/15 systematics (2026-08-19, pre-muon-SF); r=311.5/16 systematics
> (2026-08-17, buggy `lhe_alphaS` still included).

Scripts: `combine/scripts/create_datacards_ctag2d.py` ([90,160] GeV, canonical),
`create_datacards_ctag2d_100150.py` ([100,150] GeV copy, `MASS_WINDOW` hardcoded
at line 117 — no CLI flag for the window), `create_datacards_ctag2d_3ratio.py`
(joint nested-tree 3-ratio discriminant variant of the same pipeline, see below).
All three share the same `SYSTEMATICS`/`USABLE_SYST` dicts and sumw/xs-scaling
logic as `create_datacards_ctag2d.py` — edit all three together when changing
the systematics model.

**16 systematics total** (was 15 as of 2026-08-19; `CMS_eff_m_id` added
2026-08-27; `lhe_alphaS` still removed everywhere pending a bug fix — see
the dated update below): 5 `lnN` + 11 shape.

| kind | name | size/processes | notes |
|---|---|---|---|
| lnN | `lumi_Run3` | 1.4%, all | |
| lnN | `pdf_gg` | 5%, ggZZ+Signal | placeholder |
| lnN | `QCDscale_ggZZ` | 10%, ggZZ | placeholder |
| lnN | `BR_HZZ4l` | 2%, Signal+Other_Higgs | **added 2026-08-17**, sourced from Felix Heyen thesis Appendix D; no per-event weight column exists, pure rate lnN |
| lnN | `QCDscale_qqZZ` | 4%, qqZZ | **added 2026-08-17**, same source; separate from `QCDscale_ggZZ`, not a duplicate |
| shape | `lhe_pdf`/`scalevar_muR`/`scalevar_muF` | qqZZ only | excluded for Signal/ggZZ/most of Other_Higgs — see per-process reasons below |
| shape | ~~`lhe_alphaS`~~ | **removed everywhere, 2026-08-19** | not a per-process exclusion — a genuine bug in the weight-computation code itself, see the dated update below |
| shape | `ps_isr`/`ps_fsr`/`CMS_pileup`/`CMS_ctag2d` | all 4 processes | |
| shape | `CMS_eff_e_reco_20to75`/`above75`/`below20` | all 4 processes | **added 2026-08-17** — already computed and stored in the scored parquets (`weight_CMS_eff_e_reco_*_<year>Up/Down`), never wired into `USABLE_SYST` before. Verified sane (≤6%, symmetric) for every process incl. Signal/HPlusBottom via direct per-event ratio check — a detector-level correction, unaffected by the private-LHE-sample bugs below. |
| shape | `CMS_eff_m_id` | all 4 processes | **added 2026-08-27** — muon ID efficiency SF (loose WP), wired into the workflow yaml 2026-08-21 but only reached a real scored-parquet tree and a combine run on 2026-08-27, once the ctag2d reprocessing campaign reached usable completeness. Detector-level correction like `CMS_eff_e_reco` above, not an LHE weight — usable for every process. |

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
  crash once a masking clip was removed. **`lhe_alphaS`'s "not pathological,
  no same-direction Up/Down" note here was itself wrong — corrected
  2026-08-19, see the dated update below: same-direction Up/Down for
  `lhe_alphaS` is universal (a code bug affecting every process with
  LHEPdfWeight members), not a property that happened to distinguish
  HPlusBottom from the official samples.**

**Not added (checked, deliberately skipped as redundant)**: `weight_scalevar_muR_muF`
and `weight_lhe_pdf_alphaS` (combined variants) exist in the parquets too, but
adding them alongside the already-used separate `muR`/`muF`/`pdf`/`alphaS` rows
would double-count the same uncertainty.

**Muon ID SF: resolved 2026-08-27** (was "not addable without upstream work" as
of this writing) — `weight_CMS_eff_m_id_<year>Up/Down` now exists in a real
scored production tree (`hplusc_mva_4class_ctag2d_scored_v2`) and is wired
into the datacard as `CMS_eff_m_id`, see the table above and the 2026-08-27
dated update below. Muon isolation/reco SF and electron ID/isolation SF are
still not applied (deliberate scope choices, see the 2026-08-21 dated update
for the muon isolation rationale). JES/JER object-shift systematics still
need new shift-directory production (see Open items #4).

### Update 2026-08-19: `jec_shifts` bug fixed, muon scale/resolution shift production added, `higgsHFWeight` smoke-tested, `autoMCStats` threshold made configurable

Picked back up after a laptop crash killed the prior shell mid-session — **nothing
was lost**: all edits from that session (`higgsHFWeight` wiring, `higgs_hf.py`,
`muon_ss.py`) were already on disk in the EOS checkout, just uncommitted. Confirmed
directly (`git status`) before doing anything else.

**Bug found and fixed: `object_corrector_manager`'s `"jec_shifts"` branch used to
`return apply_jerc_shifts(...)` immediately**, silently skipping `muon_ss`/
`electon_ss` entirely for `hplusc_mva_4class_ctag2d_jecshifts.yaml` even though the
workflow yaml lists both under `corrections.objects` — every event in that
workflow's output was missing the muon scale/smearing correction and (for
2022–2023) the electron scale/smearing correction, silently. Caught while wiring
in muon shift production (needed a place to hook `apply_muon_ss_shifts` into).
**Fixed**: `muon_ss`/`electon_ss` nominal corrections now run *before* JES/JER
shift production, not skipped. Order doesn't matter physically —
`apply_jerc_shifts` reads `events[met_field_key]` fresh at call time (confirmed by
reading `get_corrected_jets_with_shifts`), and independent MET deltas (Type-1 JEC,
muon SS) commute under addition — so every JES/JER variant automatically inherits
the nominal muon/electron correction too, with no extra plumbing.
Every workflow run through this path **before** 2026-08-19 is missing muon SS
(and, for 2022/2023, electron SS) — re-run before trusting any yield/shape from
`hplusc_mva_4class_ctag2d_jecshifts.yaml` dated earlier.

**Muon scale/resolution shift production added** (`CMS_scale_m_<year>`,
`CMS_res_m_<year>`, MC only) — `analysis/corrections/muon_ss.py::
apply_muon_ss_shifts()`, mirroring `jerc.py::apply_jerc_shifts`'s
`(collections, shift_name)` list contract. Reuses the file's existing
`pt_scale_var`/`pt_resol_var` functions (already implemented, previously unused)
for the Up/Down math. MET propagation reuses `met.py::update_met`'s own tested
px/py-delta formula via a throwaway array rather than re-deriving that algebra —
deliberate, to avoid adding a second, independently-drifting copy of a formula
after this project's history of exactly that kind of bug (ratio-clip/sign-bug
precedents elsewhere in `analysis/corrections/`).

`object_corrector_manager` now assembles JES/JER and muon-scale/res variants into
one combined list, **explicitly carrying `Jet`/`MET`/`Muon` in every entry even
where unchanged from nominal** — required because `base.py`'s shift loop mutates
`events` cumulatively across iterations rather than resetting to nominal each
time; an entry that omitted a key would silently inherit whatever the previous
loop iteration left behind. `base.py` extended to apply a `"Muon"` override the
same way it already applies `"Jet"`/`"MET"`.

**Smoke-tested for real** (`apptainer exec` into the production coffea image,
one real `GluGluHtoZZto4L`/2023postBPix file, `hplusc_mva_4class_ctag2d_jecshifts`
workflow — needed `--bind /eos --bind /afs --bind /run/user/<uid> --bind /cvmfs`
plus `X509_USER_PROXY` pointed at a real voms proxy; none of those binds are
automatic in this environment, unlike a real lxplus node). Ran clean, exit 0, all
9 shift directories written (`base` nominal + `CMS_scale_j/res_j_2023{Up,Down}` +
`CMS_scale_m/res_m_2023{Up,Down}`). Verified from the actual parquet output, not
just "no crash":
- JES/JER variants: event count changes sensibly (183–201 vs nominal 193 — jet-
  dependent selection responding to shifted jet kinematics), `jet_pt` shifts,
  `lepton_pt` stays close to nominal.
- Muon variants: `jet_pt` is **bit-identical** to nominal across all 4 muon
  variants (44.575, confirms `Jet` correctly stays pinned to nominal in every
  muon-shift entry — the "no stale state across loop iterations" design works).
  `lepton_pt` shows the expected small (sub-percent) shift. Event count unchanged
  (193 in all 4 — muon momentum shifts here are small enough to essentially never
  cross a selection threshold, unlike JES/JER).
- `weight_higgs_plus_c` / `weight_nominal` ratio is exactly **1.5 (Up) / 0.5
  (Down)** for the 13/1728 events carrying a genuine c-flavour GEN jet, exactly
  1.0 for every other event and for the nominal weight itself — matches the
  documented ±50% design exactly, correctly scoped.

**Not yet done**: `weight_higgs_plus_c` is computed in the parquet but **not yet
wired into `create_datacards_ctag2d*.py`'s `SYSTEMATICS`/`USABLE_SYST` dicts** —
next concrete step once a real (non-smoke-test) `jecshifts` production run exists.
JES/JER and muon-scale/res shift *production* is code-complete and validated;
building a datacard that actually USES these as shape systematics (reading from
the new per-shift subdirectories) is separate, not-yet-started work.

**AFS synced 2026-08-19**: per user instruction ("we need the exact copy in afs
as well because we will be submitting jobs from afs" — EOS runs postprocess/
inference/combine, AFS is what condor submission actually uses), copied the
full correction chain + `base.py` + both `ctag2d` workflow yamls from EOS to
AFS: `correction_manager.py`, `ctag.py`, `ctag2d.py`, `higgs_hf.py`, `jerc.py`,
`jec_params_correctionlib.yaml`, `lhepdf.py`, `muon_ss.py`, `utils.py`,
`processors/base.py`, `hplusc_mva_4class_ctag2d.yaml`,
`hplusc_mva_4class_ctag2d_jecshifts.yaml`. Verified `py_compile`-clean on AFS
after copying. **Deliberately did NOT touch** other outstanding EOS/AFS drift
(filesets, postprocess, CR workflows, production `hplusc_mva_4class.yaml`) —
out of scope for this sync, still diverged, still needs its own resolution per
the standing two-checkout warning at the top of this file.

**`autoMCStats` threshold made a CLI flag** (`--automcstats-threshold`, default
10, unchanged behavior) in both `create_datacards_ctag2d.py` and `_100150.py` —
to test whether raising it past the current sparse-bin degeneracy (several bins
have MC-stat errors 10–90× the bin content, noted 2026-08-18) stabilizes the fit.
Rationale: `autoMCStats <threshold>` compares each bin's *effective* MC event
count (`n_eff = (Σw)²/Σw²`, not raw entry count) against the threshold — above it,
Combine uses the cheap merged Barlow-Beeston-**lite** Gaussian nuisance per bin;
below it, the exact per-process Poisson/gamma treatment. Default 10 is standard
combine practice, but this analysis's Signal/HPlusBottom carry ~25% negative-
weight MC events (documented private-LHE-generator issue), which makes `n_eff` a
noisier estimate of real statistical power than in a normal positively-weighted
sample — bins that nominally clear `n_eff≥10` may still sit on very few raw
events, right where the lite Gaussian approximation is weakest. Raising the
cutoff routes more of those marginal bins to the exact treatment instead.
**Result not yet in hand** — a threshold=10 baseline reproduction (same
`--merge-bin-ranges "2-4,7-10" --skip-zx` production recipe) was still running
past 19+ minutes at last check (started 2026-08-19 ~16:38, PID 3003391 in that
shell — PID won't survive a machine reboot, but the log path below will), well
beyond this pipeline's normal <2-minute `AsymptoticLimits` runtime — itself
informative (the sparse-bin degeneracy is a real fit-performance cost, not just
a theoretical concern) but not yet a comparable number for the 50/100 tests.
Session ended before it finished (by choice, not a crash) — **resume here**:

```bash
# 1. Check whether the threshold=10 baseline finished:
tail -60 /tmp/snandaku/automcstats_thr10.log
# (built into $HOME/.../combine/outputs/automcstats_test_thr10/ -- if it's not
#  there or the log looks incomplete, it may have been killed by a shell/session
#  restart; just relaunch it, same command as below with 10.)

# 2. Once 10 has a result (or after relaunching it), run 50 and 100 the same way,
#    each into its own --output dir so they don't collide:
cd /eos/user/s/snandaku/Higgscharmnew/higgscharm/combine
./run_pipeline.sh --merge-bin-ranges "2-4,7-10" --skip-zx \
  --automcstats-threshold 50 --output "$PWD/outputs/automcstats_test_thr50" \
  > /tmp/snandaku/automcstats_thr50.log 2>&1 &

./run_pipeline.sh --merge-bin-ranges "2-4,7-10" --skip-zx \
  --automcstats-threshold 100 --output "$PWD/outputs/automcstats_test_thr100" \
  > /tmp/snandaku/automcstats_thr100.log 2>&1 &

# 3. Compare median r (and total runtime / whether it converged cleanly at all)
#    across 10/50/100 -- runtime and convergence behavior matter here as much
#    as the r value itself, since the whole point is testing fit STABILITY,
#    not just getting a number.
```

**Result, 2026-08-19: definitive non-convergence at all three thresholds, sweep
abandoned.** Relaunched all three (10/50/100) fresh — the prior session's `/tmp`
logs didn't survive the laptop crash. Let them run **11.5 hours** (Aug 18 23:04 →
Aug 19 10:34), confirmed genuinely CPU-bound the whole time (`ps` CPU-time tracked
wall-clock 1:1 at ~99.6% CPU, not sleeping/deadlocked) via two checks roughly 25
minutes apart plus the final one. **None of the three produced even a first
`Expected N%: r < ...` quantile line** — not "slow", genuinely non-converging,
and critically, **raising the threshold did not help**: threshold=100 (which
should route almost every bin to the exact per-process treatment instead of the
fragile Barlow-Beeston-lite Gaussian) was exactly as stuck as threshold=10. This
rules out the sweep's own hypothesis (marginal bins near a low threshold being
the culprit) — the degeneracy isn't about *which* bins get the lite treatment,
it runs deeper, plausibly in how `autoMCStats`'s Gaussian nuisances interact with
this analysis's ~25% negative-weight Signal/HPlusBottom MC regardless of the
threshold. Killed all three (`kill -TERM`, confirmed clean) rather than let them
burn CPU indefinitely. **Conclusion: raising the threshold alone does not fix
it — that specific idea is closed, not the underlying problem.** `autoMCStats`
stays out of the production datacard for now, but **this is a real, unresolved
gap, not an accepted permanent state**: per-bin MC statistical uncertainty on
the templates currently has no nuisance covering it at all. A CMS analysis note
needs some accounting for this; "omit it because it doesn't converge" is a
documented workaround, not a resolution. **Candidate next steps, not yet
started, roughly in order of expected effort-to-payoff**:
1. Identify exactly which bins have the 10–90× error/content ratio and merge
   just those further (the pipeline already merges some ranges for the
   negative-bin issue) — if `n_eff` clears the threshold everywhere, plain
   `autoMCStats` may converge normally without needing any threshold games.
2. Attack the root cause directly: Signal/HPlusBottom's ~25% negative-weight
   MC fraction (same likely root cause as the already-diagnosed `lhe_pdf`/
   `scalevar_muR/muF` blowups) makes `n_eff` an unreliable proxy for real
   statistical power — fixing how negative weights are summed/handled could
   resolve several problems at once, not just this one.
3. Manual per-bin `shapeN`/gamma nuisances for just the worst offending
   bins/processes instead of the automatic whole-channel treatment.
4. Debug the fit itself before assuming it's structural — untried:
   `--cminDefaultMinimizerStrategy 0`, `--X-rtd MINIMIZER_analytic`, looser
   `--rAbsAcc`/`--rRelAcc`, or plain `MultiDimFit` instead of the full
   `AsymptoticLimits` CLs scan, to isolate whether the CLs scan itself is the
   bottleneck versus the underlying likelihood being genuinely broken. The
   11.5-hour test only ruled out "threshold alone fixes it."

### Update 2026-08-19: `lhe_alphaS` sign bug found — same-direction Up/Down was
never real physics, removed from every datacard pending a fix

While investigating why `lhe_alphaS` ranked an unexpectedly high #3 in the
16-systematic Impacts (impact_r=17.6 — see the rebuild in the previous
section), checked the actual templates directly rather than trusting the fit:
both `qqZZ` (±1.5%) and `Other_Higgs` (±2.7%) move in the **same direction**
together — Up always raises both, Down always lowers both, smoothly and
without spikes across all 20 bins. That's the tell: a real, independent physics
effect on two different production processes shouldn't be forced into lockstep
like that.

**Root cause, found in `analysis/corrections/lhepdf.py:60`**:
```python
delta_alpha = 0.5 * np.abs(w_as_high - w_as_low)
w_up_alpha = 1 + delta_alpha
w_down_alpha = 1 - delta_alpha
```
The `np.abs()` destroys the *sign* of the difference between the two αs
members (`w_as_high` = LHEPdfWeight member 102, αs=0.120; `w_as_low` = member
101, αs=0.116) before symmetrizing. Since `delta_alpha` is forced non-negative,
`w_up_alpha >= 1` and `w_down_alpha <= 1` for **every event of every process,
unconditionally** — "Up" is mathematically guaranteed to raise the yield and
"Down" to lower it, regardless of which direction the real αs variation
pushes that event's weight. This is why `qqZZ` and `Other_Higgs` were seen
moving together: not a shared physics effect, an artifact of the formula. Any
other process with a 103-member `LHEPdfWeight` (Signal/ggZZ don't have one —
separately degenerate, unaffected) would show the identical manufactured
pattern.

Contrast with `lhe_pdf` immediately above it in the same function (lines
43-55): that one is *supposed* to be a symmetric envelope —
`sqrt(Σ(w_k-w0)²)` is the correct PDF4LHC treatment for ~100 Hessian-eigenvector
replica members, where "symmetric uncertainty, direction not physically
meaningful" is the right convention. αs has only **2** members, not 100 — the
physically correct treatment is a genuinely *signed* shift (higher αs → member
102's actual weight, lower αs → member 101's), not an absolute-value envelope.
The two-members case wrongly reused the many-eigenvectors convention.

**This also retroactively corrects a wrong conclusion recorded earlier in this
file** (the `USABLE_SYST` per-process reasoning above): `lhe_alphaS` was
described as "kept for Other_Higgs: larger for HPlusBottom than the official
samples, but not pathological (no sign flip, no same-direction Up/Down)" — that
comparison was never meaningful, since same-direction Up/Down turns out to be
universal, not something distinguishing HPlusBottom from the official samples.

**Not yet fixed, deliberately** — excluded from `USABLE_SYST` instead
(`create_datacards_ctag2d.py`, `create_datacards_ctag2d_100150.py`,
`create_datacards_ctag2d_3ratio.py`, all three kept in sync) pending
verification that a signed fix (drop the `np.abs()`) genuinely changes
direction for some events rather than just relabeling the same magnitude —
that check wasn't done before documenting, to avoid publishing an unverified
"fixed" number.

**Updated production limit, 15 systematics, `lhe_alphaS` excluded**: median
r=309.5 full-syst, no-ZX, `[100,150]` (was r=311.5 with the buggy systematic
included — a small, expected shift since removing one shape systematic
slightly changes the fit, not a sign the old number was wildly wrong, just
built on an incorrect nuisance). Converged cleanly in <2 minutes (confirms this
whole investigation was never about `autoMCStats`/convergence — that's a
separate, still-open issue, see above).

**Impact ranking rebuilt, 15 systematics** (same real `combineTool.py -M
Impacts` profile-likelihood method as before, not the cheap approximation):

| Rank | Nuisance | impact on r |
|---|---|---|
| 1 | `CMS_ctag2d` | 71.2 |
| 2 | `CMS_eff_e_reco_below20` | 28.2 |
| 3 | `BR_HZZ4l` | 12.0 |
| 4 | `lumi_Run3` | 9.6 |
| 5 | `CMS_eff_e_reco_20to75` | 7.3 |
| 6 | `CMS_pileup` | 6.5 |
| 7 | `ps_fsr` | 5.4 |
| 8 | `QCDscale_qqZZ` | 4.0 |
| 9 | `ps_isr` | 3.0 |
| 10 | `scalevar_muF` | 2.7 |
| 11 | `scalevar_muR` | 1.3 |
| 12 | `QCDscale_ggZZ` | 0.7 |
| 13 | `lhe_pdf` | 0.4 |
| 14 | `pdf_gg` | 0.3 |
| 15 | `CMS_eff_e_reco_above75` | 0.2 |

`CMS_ctag2d` remains the dominant nuisance by a wide margin — unaffected by
this fix, reinforcing that resolving its Down-variation asymmetry (kept as a
documented caveat, see the Known traps entry) is the highest-value remaining
systematics work. Output:
`combine_run3_100150_ctag2d_pipeline_run/impacts_100150_15syst_noalphaS.{json,pdf}`.

**Also caught and fixed while rebuilding, unrelated regression**: the
`autoMCStats` code from the section above turned out to default to **on**
(`hczz autoMCStats 10` silently written into every datacard unless a value is
explicitly passed) — directly contradicting the standing decision to keep it
out, and confirmed by watching a freshly-rebuilt production datacard hang
again during this same session. Fixed: `--automcstats-threshold` now defaults
to `None` (no line written at all) in both `create_datacards_ctag2d.py` and
`create_datacards_ctag2d_100150.py`; pass an explicit value to opt back in for
testing.

**Next step, not started**: verify the signed fix (`delta_alpha = 0.5 *
(w_as_high - w_as_low)`, dropping `np.abs()`) actually flips direction for a
real subset of events before re-including `lhe_alphaS` and rebuilding again.

---

### Update 2026-08-21: `lhe_alphaS` signed fix applied (non-retrofittable),
muon efficiency SF wired, one real regression caught and fixed by an actual
smoke test

**Full reproducibility reference** (exact commands, exact diffs, exact expected output for
everything below): `second-brain/Notes/HcZZ-systematics-audit-2026-08-21.md`. The reusable
smoke-test script itself is `smoketest_muon_sf.py` at this repo's root (EOS+AFS, identical) —
run it directly to re-verify the muon-SF wiring end to end.

**`lhe_alphaS` signed fix applied**: dropped the `np.abs()` per the "next
step" above. Checked whether this could be retrofitted onto already-scored
parquets (real `2023postBPix` ctag2d-scored file schemas) — only the final
post-abs() `weight_lhe_alphaSUp/Down` columns survive, no raw
`LHEPdfWeight[:,101]`/`[:,102]` components to recompute the true sign from.
**Confirmed non-retrofittable**, same situation as the already-documented
`lhe_pdf` normalization bug. `lhe_alphaS` stays excluded from `USABLE_SYST`
until a real reprocessing exists; rebuilt the production `[100,150]` no-ZX
card with the fix present (still excluded) — r=309.5000, bit-identical to
the pre-fix baseline, as expected (the fix only benefits a future
reprocessing).

**Real regression caught by an actual smoke test, not a config check**:
applying this fix initially left `w_as_low`/`w_as_high` undefined —
`NameError: name 'w_as_high' is not defined` — a genuine bug that would have
crashed every future reprocessing (this function runs for every 103-member
`LHEPdfWeight` event, i.e. essentially all MC). Never caught by the earlier
audit pass because that pass only rebuilt the datacard from already-scored
*parquets*, which never call this NanoAOD-level correction function at all —
only a real single-file `run_uproot_job` smoke test (see below) exercises
this code path. Fixed by restoring the two extraction lines
(`w_as_low = pdfweights[:, 101]`, `w_as_high = pdfweights[:, 102]`) before
the signed-shift formula, in both EOS and AFS `lhepdf.py`.

**Muon efficiency SF (ID only) wired into both ZZ→4l workflow yamls**
(`hplusc_mva_4class.yaml`, `hplusc_mva_4class_ctag2d.yaml`, EOS+AFS),
closing the gap found in the 2026-08-21 systematics-completeness audit
(second-brain memory `hczz_systematics_completeness_audit`) — no ZZ→4l
workflow previously had a `muon:` key at all.
```yaml
muon:
  - id: loose
  - iso: false
  - trigger: false
```
- **`id: loose`**, not `tight`: this analysis's actual muon selection
  (`zzto4l.yaml` → `is_loose` = isGlobal|isTracker + kinematic/IP cuts,
  `is_tight` = `is_loose` + `is_relaxed` (SIP3D<4) + `isPFcand`) is the
  standard CMS H→ZZ→4l lepton recipe, which is much closer to POG **loose**
  muon ID (PF muon + isGlobal|isTracker) than POG **tight** ID (which
  additionally requires normalized χ², station-match, and tracker-layer
  cuts this selection never checks). Using `tight` here would apply an SF
  for a selection efficiency tighter than what's actually cut on —
  overcorrecting, not fixing a gap. `correctionlib` safely no-ops (SF=1)
  for muons below the SF binning's pT floor (10–15 GeV depending on nano
  version; see `MuonWeights.get_id_weights`/`unflat_sf`), so this
  analysis's `pt>5` GeV muons are only corrected above that floor — no
  crash risk, no silent extrapolation.
- **`iso: false`, deliberately not wired**: this analysis's isolation is a
  bespoke FSR-photon-corrected combined relative isolation
  (`select_zzto4l_leptons` in `analysis/selections/object_selections.py`),
  not a standard PF-relIso working point at a matching cone/threshold — the
  generic `NUM_*RelIso_DEN_*ID` correctionlib SFs don't correspond to this
  cut. Applying one anyway would risk a **new** mismatch bias, not fix one;
  left open rather than silently misapplied.
- **`trigger: false`**: matches the electron block's own choice and the
  `hww.yaml` sibling; a real multi-path (SingleMu/DiMu/TriMu/SingleEle/
  DiEle/MuEle) trigger SF needs combinatorial OR-of-paths efficiency
  modeling beyond a single flag flip — out of scope for this pass.

**Validated with a real end-to-end smoke test, not just YAML/config
inspection**: ran the actual `WorkflowConfigBuilder` → confirmed
`corrections_config["event_weights"]["muon"]` parses to
`{"id": "loose", "iso": False, "trigger": False}` exactly as
`correction_manager.py`'s muon branch expects (list-of-single-key-dicts →
merged dict, same mechanism the working `electron:` block already uses).
Then ran a **real single-file `processor.run_uproot_job`** (one
`GluGluHtoZZto4L`/2023postBPix file, `hplusc_mva_4class_ctag2d` workflow,
interactively in LCG_105 — no apptainer needed this time since coffea
imports directly there) end to end: exit clean, 191 events selected,
`weight_CMS_eff_m_id_2023`/`Up`/`Down` written to the output parquet with
sane values (mean SF 0.999, spread ≤0.5% per event, exactly matching this
framework's `weight_<name>Up/Down` convention that `create_datacards_
ctag2d*.py` already knows how to read).

**Two incidental bugs found and fixed while setting up this smoke test**
(both pre-existing, unrelated to the muon-SF work itself, only surfaced
because this was the first real interactive `run_uproot_job` test run
outside apptainer in a while):
1. The `lhe_alphaS` `NameError` above.
2. `analysis/data/nnlo_ps/` was missing an `__init__.py` (unlike its sibling
   `analysis/data/jec/__init__.py`), making it an implicit namespace
   package — Python 3.9's `importlib.resources.open_text` resolved it
   inconsistently in this interactive environment, constructing a wrong
   path (`/eos/home-s/.../NNLOPS_reweight.json`, missing the
   `analysis/data/nnlo_ps/` subdirectory entirely) and crashing
   `add_nnlops_weight`. Added the matching empty `__init__.py` (EOS+AFS) —
   turns it into a regular package, same fix pattern as the existing `jec/`
   sibling. Apparently never bitten real condor production (different
   environment setup there), but a real latent fragility worth having
   fixed regardless.

**At the time of this update, this did not yet change any existing combine
number** — `weight_CMS_eff_m_id_*` only existed in that session's throwaway
smoke-test parquet, not in any production scored-parquet tree. See the
2026-08-27 dated update immediately below for the full reprocessing +
datacard + combine run that finally used it for real.

### Update 2026-08-27: muon SF reaches a real combine number for the first
time — new production reference r=301.0, kappa_c=54.86 (16 systematics)

**Full reproducibility reference**: `second-brain/Tasks/HcZZ-fake-rate.md`
"Update 2026-08-27 ~19:30 CEST"; memory
`hczz_systematics_completeness_audit` §16-18 for the full chain of findings
that unblocked this (proxy-auth fix, `jobs_status.py` completeness-tool
fix, real-vs-apparent job completeness).

**Context**: the muon SF gap (§2 of the 2026-08-21 audit) was wired into the
workflow yamls the same day, but stayed blocked on reaching a real scored
production tree — the ctag2d reprocessing campaign that would carry
`weight_CMS_eff_m_id_*` into production appeared stuck at 65-90% completeness
for several days (2026-08-25 through 2026-08-27), traced through a site-wide
XRootD outage, then an AFS-proxy auth bug, before a same-session forensic
dig found the deeper cause: `jobs_status.py`'s completeness check couldn't
tell "job ran fine, zero events survived a tight selection" (expected
physics, especially for background MC and even some data streams) from
"job never ran" — real completeness was already 97-100% per era. Fixed
`jobs_status.py` (AFS checkout) to fall back to checking the always-written
flat `{dataset}_{N}.coffea` file before declaring a job missing.

**Rebuilt the MVA-scored parquets from scratch** (`analysis/postprocess/
run_mva_postprocess.py`, LCG_105 env, the documented `models/best_model.pt`
+ `hc_zzto4l_mw_training_4class_nomass.yml` bundled pair) into a new
directory, `hplusc_mva_4class_ctag2d_scored_v2` (old `_scored` dir from
2026-08-16 left untouched) — all 4 eras, 5035 parquet files, **0 errors**,
2,883,041 total events. Confirmed `weight_CMS_eff_m_id_2022/2023Up/Down`
present for the first time ever in a scored production tree.

**Added `CMS_eff_m_id`** (era-dependent, same convention as `CMS_pileup`/
`CMS_ctag2d`) to `SYSTEMATICS`/`USABLE_SYST` in all three ctag2d datacard
scripts — usable for every process (detector-level SF, not an LHE weight,
none of the private-sample exclusions apply). 16 systematics total now.

**Ran the standard production preset** (`run_pipeline.sh --merge-bin-ranges
"2-4,7-10" --skip-zx`, i.e. `[100,150]`, surgical rebin, no-ZX) against the
new scored dir:

**Result: r < 301.0 (median, 50% CL), kappa_c = 54.86.** Full CLs band:

| CL | r | kappa_c |
|---|---|---|
| 2.5% | 154.0273 | 29.35 |
| 16.0% | 209.1421 | 38.92 |
| 50.0% | 301.0000 | 54.86 |
| 84.0% | 441.3765 | 79.20 |
| 97.5% | 624.6505 | 110.96 |

Tightens modestly from the pre-muon-SF reference (r=309.5, 2026-08-19, 15
systematics) — expected direction/magnitude for one real systematic plus a
much more complete underlying production. Negative-bin scan: 15/74
histograms with a single ~1e-6-magnitude negative bin at index 0 (2 new
entries are `CMS_eff_m_idUp/Down`, same tiny benign magnitude as the rest,
same pattern documented in the 2026-08-18 negative-bins update). A ROOT
`TNetXNGFile::Open [3001]` error appeared in the combine log right after
CMSSW startup but didn't block or crash the fit — looks like a benign ROOT
plugin-manager probe, not investigated further.

**Re-ran impact ranking** (§7 method, freeze-and-compare) against this new
datacard: `CMS_eff_m_id` has **zero measurable impact** on the median
(`delta_r = 0.00`, exactly) — dominated by `CMS_eff_e_reco_below20`
(-3.82%) and `CMS_ctag2d` (-3.49%), same top-2 as the pre-muon-SF ranking.
Consistent with its small per-event magnitude (mean SF 0.999, <=0.5% spread,
per the 2026-08-21 smoke test) — correctly wired and included, just
genuinely small at this method's resolution, not a sign anything is wrong.
Output: `combine/outputs/combine_run3_100150_ctag2d_muonSF/impact_ranking.
{json,png}`.

**Then resubmitted the remaining genuinely-missing jobs** (9 ctag2d across
3 eras + 6 `ztoee` + ~103 `zplusl_ss`, using the fixed `jobs_status.py` to
correctly identify only the real gaps) — draining as of this update; a
still-more-complete number is expected once that lands and the pipeline is
rerun again.

**Nothing old was overwritten** — new scored dir
(`hplusc_mva_4class_ctag2d_scored_v2`) and new output dir
(`combine/outputs/combine_run3_100150_ctag2d_muonSF/`), both additive.

---

### Update 2026-08-28: `lhescale.py` Up/Down swap found and fixed; Signal/
HPlusBottom `lhe_pdf`/`scalevar_muR/muF` root cause confirmed from raw
NanoAOD (not a code bug); ggZZ exclusion confirmed structural

**Full reproducibility reference**: `second-brain/Notes/HcZZ-systematics-
audit-2026-08-28.md`; narrative in `second-brain/Tasks/HcZZ-fake-rate.md`
"Update 2026-08-28 (later still): lhe_pdf/scalevar_muR/muF audit".

**Audited every up/down systematic in `analysis/corrections/*.py`** for the
same class of bug as the historical `lhe_pdf`/`lhe_alphaS` issues. Every file
except `lhescale.py` delegates Up/Down to a correctionlib-named systematic
string (`up`/`down`, `systup`/`systdown`, `sfup`/`sfdown`, `up_Total`/
`down_Total`) — no custom-index risk. `jerc.py`'s JES/JER shift production
(`apply_jerc_shifts`) checked and confirmed correct (`fac=+1/-1` around
`jesunc` for JES; JER delegates to correctionlib's named systematic).
`partonshower.py`'s `PSWeight` indices checked against the standard CMS
NanoAOD producer convention — correct.

**Real bug found in `lhescale.py`**: confirmed `coffea.analysis_tools.
Weights.add()`'s signature is `(name, weight, weightUp, weightDown)`. Per
the file's own docstring index table (index[1]/[7]=muR down/up,
index[3]/[5]=muF down/up, index[0]/[8]=both down/up), the code passed
`weightUp`/`weightDown` in the **opposite** order for all three of
`scalevar_muR`/`scalevar_muF`/`scalevar_muR_muF` — Up and Down were
physically swapped, for every process, since this file was written.
`scalevar_muR_muF` also had a missing `/nom` ratio normalization (the only
one of the three without it). **Fixed at the source**, both EOS+AFS
checkouts, `ast.parse`-checked. Affects qqZZ/Other_Higgs's current
`USABLE_SYST` entries in every existing scored parquet — needs a
reprocessing to take effect, same non-retrofittable class as the
`lhe_alphaS` fix above.

**Signal/HPlusBottom root cause, confirmed from real NanoAOD (not the
2026-08-03 parquet-level retrofit math)**: pulled real events directly via
`uproot` from the regenerated `HPlusCharm_2023postBPix`/
`HPlusBottom_2023postBPix` production (`hplusc_htozz1`/`hplusb_htozz1`
paths). Confirmed index[4] (central ratio point) is exactly `1.000000`,
std=0, for every event — rules out the "branch normalization is broken"
hypothesis open since 2026-08-03. But every *other* grid index sits at
mean/median ≈1.5–2.5× (not the expected ≈0.8–1.2×): Signal
muR-up=1.96×/muR-down=2.24×, HPlusBottom muR-up=1.71×/muR-down=2.03× —
median tracks mean closely, not an outlier artifact. `delta_pdf`
recomputed from the raw 101/103-member `LHEPdfWeight` arrays with the
current fixed formula: Signal mean=1.08/median=1.27, HPlusBottom
mean=0.85/median=1.00 — same ~90-130% range as the earlier ~88% figure,
now independently confirmed from raw LHE weights.

**Explanation**: both datasets are tagged `MuRFScaleDynX0p50` — central
scale = half of some dynamical reference. Every other 9-point-grid entry
multiplies renscfact/facscfact onto that already-halved central scale, so
the "central" weight sits at an outlier point in its own envelope rather
than the middle — a genuine generator-setup artifact, not a downstream
formula bug. Wiring a ~100-200% lnN/shape for Signal would dominate every
other nuisance and misrepresent this as real theory uncertainty; keeping
the exclusion. Real fix (if pursued) is regenerating with a standard
central-scale definition — new item in Open items below, not a datacard
change.

**ggZZ exclusion confirmed structural**: pulled real
`GluGluToContinto2Zto2E2Mu` events — `LHEScaleWeight`/`LHEPdfWeight`
branches exist in the schema but are zero-length for every single event.
Not present-but-degenerate — genuinely empty; this MCFM-based sample was
never produced with these vectors filled.

---

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
that ranking may be affected. **Root-caused 2026-08-18**: the Up/Down weight-combination
logic itself is standard/correct — the real source is placeholder calibration values
(`central=1.000, up_Total=3.000, down_Total=0.300`) in statistically-empty (flavor, WP)
corners of the official `flavTaggingSF_<year>.json.gz`, amplified by `CTag2DCorrector`
taking `ak.prod` across every selected jet rather than one. **Decision (2026-08-18,
user sign-off): keep the all-jets `ak.prod` scope** — it matches the 1D `CTagCorrector`
this replaces, keeping pre/post-migration yields directly comparable, so the asymmetry
stays a documented caveat rather than triggering a scope change. Restricting to one jet
would likely smooth it out but was explicitly rejected as a bigger, separate
validation task that changes what the corrector represents.

---

### Update 2026-08-29: fresh systematics correctness re-audit — no new bugs,
`lhescale.py` fix confirmed correctly implemented in code

**Full reproducibility reference**: `second-brain/Notes/HcZZ-systematics-
audit-2026-08-29.md`; narrative in `second-brain/Tasks/HcZZ-fake-rate.md`
"Update 2026-08-29".

Re-read every `analysis/corrections/*.py` Up/Down call from the current
checkout (not a recall of the 2026-08-28 audit) and cross-checked against
`combine/scripts/create_datacards_ctag2d*.py`'s `SYSTEMATICS`/`USABLE_SYST`.
Confirmed `lhescale.py`'s fix is correctly implemented (index[7]/[5]/[8] as
Up, index[1]/[3]/[0] as Down, `/nom` present on all three variables) and
diff-identical between the EOS and AFS checkouts. `lhepdf.py`, `ctag.py`,
`ctag2d.py`, `pileup.py`, `partonshower.py`, `muon.py`, `electron.py` all
re-verified correct. `electron_ss.py`/`muon_ss.py`/`met.py`/`jetvetomaps.py`
confirmed to carry no weight systematics at all (object-level corrections
only, correctly out of scope). `USABLE_SYST` confirmed identical across all
three production datacard scripts (`create_datacards_ctag2d.py`,
`..._100150.py`, `..._3ratio.py`). No new bugs found.

**No code changed this session** — verification only. `lhescale.py`'s fix was
pushed to `Saranya-Nandakumar/higgscharm` (`HcZZ-zx-estimation`, `075478d`)
earlier the same day; this audit confirms that push is correct as-is.

---

## Known traps

| trap | detail |
|---|---|
| **No merge step (historical)** | §2 above describes the old split path — superseded 2026-08-14 for new workflows by the consolidated `--postprocess --mva-inference` flow, which had never been run successfully before that date. |
| **Local `apptainer exec` smoke tests need explicit binds, not automatic** | Found 2026-08-19 running a local single-file smoke test outside condor: `/eos`, `/afs`, `/cvmfs`, and the Kerberos ticket cache dir (`/run/user/<uid>`) are **not** auto-mounted into the container in this environment the way they are on a real condor worker / lxplus node. Symptoms in order as each missing bind was found: `/eos` missing → `FileNotFoundError` on `submit.py` itself; ticket cache missing → EOS `Permission denied` even with `/eos` bound (klist inside the container shows no cache until `/run/user/<uid>` is also bound); `/cvmfs` missing → correctionlib files under `/cvmfs/cms-griddata.cern.ch/...` not found even though the image itself lives on `/cvmfs` (loading the image doesn't imply the whole `/cvmfs` tree is bound). Full working invocation: `apptainer exec --bind /eos --bind /afs --bind /run/user/<uid> --bind /cvmfs <image> bash -c 'export HOME=<scratch>; export PYTHONNOUSERSITE=1; export X509_USER_PROXY=<path to a valid voms proxy>; ...'` — also needs a genuinely valid (non-expired) X509 proxy for xrootd reads (Kerberos alone isn't enough for CMS xrootd doors), separate from the EOS-filesystem Kerberos requirement above. |
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
| **`lhe_pdf` formula: Hessian vs. MC-replica (RESOLVED 2026-08-03, root cause of residual size confirmed 2026-08-28)** | Fixed 2026-08-03: formula now divides by `N-1=99` (`sqrt(Σ(w_k-w0)²/99)`), correct for the NNPDF Monte-Carlo-replica prescription our private samples use. The private NanoAODs were regenerated (`hplusc_htozz1`/`hplusb_htozz1`, reachable again) and read directly 2026-08-28 — the formula is not the remaining problem. The residual ~90-130% `delta_pdf` for Signal/HPlusBottom is real and traced to the samples' `MuRFScaleDynX0p50` central-scale choice (half a dynamical reference scale), confirmed from raw `LHEScaleWeight`/`LHEPdfWeight` branches, not a formula bug — see the 2026-08-28 Systematics update above. |
| **Signal/HPlusBottom negative-weight fraction is ~25%, not incidental** | Measured directly from `hplusc_mva_4class_scored_v4` (2026-08-14): `HPlusCharm`/`HPlusBottom` run **24–26% negative `weight_nominal`** across every era, roughly **8–17× higher** than every other process (`ZZto4L` 0.18%, `GluGlu*` ggZZ ~0.003–0.01%, Higgs backgrounds `WH`/`ZH`/`TTH` 1.4–1.7%). This is the same two processes with the already-diagnosed `lhe_pdf`/`scalevar_muR/muF` blowups — plausibly the same root cause (private amc@nlo-based generation), not yet connected. See [Open items](#open-items--next-techniques-to-port-from-hww-analysis). |

---

## Open items / next techniques to port from `hww-analysis`

**Priority order re-ranked 2026-08-21** after a systematics-completeness audit vs Felix
Heyen's thesis / HIG-24-013 / the HWW deck (full writeup: second-brain memory
`hczz_systematics_completeness_audit`, `Tasks/HcZZ-fake-rate.md` "Update 2026-08-21"). New
#0: **muon efficiency SF (ID/ISO/trigger) is entirely unapplied for ZZ→4l** — not a missing
uncertainty like the items below, the *central value* itself is missing (no `muon:` key in
any ZZ→4l workflow yaml, confirmed via `grep -rn "^\s*muon:" analysis/workflows/*.yaml`;
`MuonWeights.add_id_weights()`/`add_iso_weights()` in `analysis/corrections/muon.py` already
work, just never invoked here). Felix's thesis quotes the equivalent `muonSF` nuisance at
~4.5–9% yield effect — since 3 of 4 ZZ→4l final states carry muons, this is a plausible
few-percent normalization bias on every HcZZ number on record, not just an uncertainty gap.
Needs a `muon:` key added to the workflow yaml plus a full reprocessing — blocked behind the
same condor `/eos`-path issue as items 3–4 below. Electron ID SF has the same class of gap
(`id: false`, documented/deliberate, smaller likely magnitude). Also newly flagged: a
dedicated FS (flavour-scheme, 4FS/5FS envelope) uncertainty for Signal/HPlusBottom — Felix's
thesis has one (~59%/42% yield effect, his "clearly leading" systematic), HcZZ has none; a
candidate *addition*, not a fix for the already-excluded `lhe_pdf`/`scalevar` rows (different
problem). `pdf_qq` was checked and resolved as **not** a gap — qqZZ's real `lhe_pdf` shape
systematic already covers it.

Original priority order set 2026-08-14; update as items land.

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
   `CTagCorrector` it's meant to replace. **Reviewed and confirmed 2026-08-18**
   after root-causing the CMS_ctag2d Down-variation asymmetry to this scope (see
   Known traps above) — user sign-off to keep all-jets, not a bug.

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
3. **`autoMCStats`** — **added to the datacard 2026-08-18** (threshold 10).
   Update 2026-08 (separate investigation, `ctag2d` pipeline): found
   numerically degenerate with the current binning (several bins have MC-stat
   *errors* 10–90x the bin *content*). **Update 2026-08-19**: threshold made a
   CLI flag (`--automcstats-threshold`) to test raising it past this
   degeneracy — see the dated update in the Systematics section above for the
   full rationale and in-progress result.
4. **Object-shift (JES/JER/lepton scale+res) systematics** — bigger lift,
   needs new shift-directory production in `runner.py` plus inference
   re-coverage (§4). **Update 2026-08-17**: electron reco is actually *not*
   blocked — `weight_CMS_eff_e_reco_{20to75,above75,below20}_<year>Up/Down`
   already exist in the `ctag2d` scored parquets (presumably from an earlier,
   unrelated production step) and are now wired into `create_datacards_
   ctag2d*.py`'s `USABLE_SYST` (16 systematics total, see the `ctag2d`
   variant subsection under Systematics above). **Update 2026-08-19**:
   JES/JER (`CMS_scale_j`/`CMS_res_j`) and muon scale/resolution
   (`CMS_scale_m`/`CMS_res_m`) shift *production* is now code-complete and
   smoke-tested (`jec_shifts` workflow, `hplusc_mva_4class_ctag2d_
   jecshifts.yaml`) — see the dated update in the Systematics section above.
   Building a datacard that actually reads these shift directories as shape
   systematics is separate, not-yet-started work. Muon ID/isolation/reco
   (efficiency SF, different from scale/resolution) remains genuinely
   blocked — no such weight columns exist anywhere in the scored parquets,
   checked directly, and no `muon:` block exists in the `ctag2d` workflow
   yamls' `event_weights` at all.
5. **Per-event HF-composition-style weight** — `hww-analysis`'s
   `higgs_hf.py` replaces a mis-scoped flat lnN on a pooled background group
   with a per-event GEN-jet-flavour weight. Same shape of problem as our
   `Other_Higgs`/`HPlusBottom` pooling (see the `lhe_pdf`/`scalevar`
   exclusion above). **Update 2026-08-19**: ported (`analysis/corrections/
   higgs_hf.py`, `higgsHFWeight: true` in both `ctag2d` workflow yamls),
   wired into `weight_manager()`, and smoke-tested — see the dated update in
   the Systematics section above. **Not yet wired into `create_datacards_
   ctag2d*.py`'s `SYSTEMATICS`/`USABLE_SYST` dicts** — the column exists in
   the parquet, the datacard doesn't use it yet.
6. **`BR_HZZ4l`/`QCDscale_qqZZ` lnN rows** — **done 2026-08-17**, see the
   `ctag2d` variant subsection under Systematics above.

**Deliberately not adopted**: the multi-channel argmax MVA architecture
(SR + per-background CRs, floating a background via `rateParam` the way
`hww-analysis` floats `tt`). Decision 2026-08-14: keep the single
`mva_score_Signal` discriminant — the current 4-class MVA and all reference
r/κc numbers stay valid; this is a documentation and systematics-parity
effort, not a redesign.
