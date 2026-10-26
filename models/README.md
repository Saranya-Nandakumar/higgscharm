# MVA Models

This directory stores trained MVA model checkpoints used for post-processing inference in the H+c → ZZ → 4ℓ analysis.

See [`INFERENCE.md`](INFERENCE.md) for the internals of how `MVAPostProcessor` scores events, and how the reducible (Z+X) background gets run through the same model.

## Model format

Models are saved as PyTorch `.pt` checkpoint files produced by the [b-hive](https://github.com/deoache/b-hive) training framework. Each checkpoint encodes the model architecture parameters (`input_dim`, `hidden_dim`, `num_layers`, `num_classes`) so no separate architecture config is needed at inference time.

## Available models

The 2022/2023 analysis (2022preEE, 2022postEE, 2023preBPix, 2023postBPix) uses one model:

| File | Value |
|------|-------|
| Model | `models/best_model.pt` (md5 `fd19031f…`), `MLP_HcZZ_MW_Deep_4class`, 4 classes `[qqZZ, ggZZ, Signal, Other_Higgs]`, 101 inputs |
| Config | `models/hc_zzto4l_mw_training_4class_nomass.yml` (`config_4class.yml` is an identical copy) |
| Training (b-hive) | config `hc_zzto4l_mw_training_4class_nomass`, dataset `hcZZ_big4classjetyesmasslooseoldcbal`, training `train_loose_oldcwithmassbal`, 40 epochs, learning rate 5e-4 |

Always use this model with this config: a config with the right feature *count* but a different
feature *list* scores silently wrong. The era passed to the scorer selects the jet-category table
(`JET_TAG_BOUNDARIES_BY_ERA`, PNet for 2022/2023).

## Running MVA inference

There are two entry points. Both use `MVAPostProcessor` (`analysis/postprocess/mva_inference.py`)
and give the same per-event scores; they differ in which files they score and where the output goes.

### 1. Production (used for all datacards): `run_mva_postprocess.py`

A separate step after the condor jobs, on the raw per-job parquets. It needs no `--postprocess`
and does no merging. It mirrors the parquet tree (`<era>/<dataset>_<N>/base/*.parquet` and the
object-shift directories `base/<CMS_scale_j_…Up/Down, …>/*.parquet`) into `--output-dir` and adds the
score columns to every file, so the shifted templates are scored too. The `sumw/*.json` sidecars are
not copied; the datacard builder reads them from the unscored workflow directory (`--sumw-dir`).

```bash
source /eos/user/s/snandaku/miniforge3/etc/profile.d/conda.sh && conda activate b_hive_new
cd /eos/user/s/snandaku/Higgscharmnew/higgscharm
O=/eos/user/s/snandaku/higgscharm/outputs

# run once per era, or --all-eras
python3 -u analysis/postprocess/run_mva_postprocess.py \
  --input-dir  $O/<workflow> \
  --output-dir $O/<workflow>_scored_vN \
  --config models/hc_zzto4l_mw_training_4class_nomass.yml \
  --model  models/best_model.pt \
  --era <era> --no-mass-window --no-histograms --workers 8

```

`<workflow>` is the analysis production (the `hplusc_mva_4class_ctag2d*` family, see
`README_HcZZ.md` "Current state"). For long runs, submit the command as a condor job from AFS
(example: `/afs/cern.ch/user/s/snandaku/condor_hczz_2026_10_22/score.sub`).

| Flag | Description |
|------|-------------|
| `--input-dir`, `-i` | Workflow output directory (`<era>/<dataset>_<N>/...` below it). Required. |
| `--output-dir`, `-o` | Where the scored copy goes. Default: the input directory, i.e. files are updated **in place**; always give a new `_scored_vN` directory. |
| `--config`, `-c` / `--model`, `-m` | b-hive config YAML and `.pt` checkpoint (above). Required for the `score` stage. |
| `--era`, `-e` / `--all-eras` | Score one era, or every era found below `--input-dir`. |
| `--no-mass-window` | Score all events. Used in production: the m4l window (105–160 GeV) is applied in the datacard builder. Without it, only 90 ≤ m4l < 160 (`--mass-window-min/max`) is scored. |
| `--no-histograms` | Skip the score histograms (the cards are built from the parquets). |
| `--datasets` | Only score partition directories that start with these dataset names. |
| `--workers`, `-w` | Parallel workers (default 8). |
| `--stages` | `score` (default), `bookkeep`, `plot`; see below. |

### Bookkeeping and plots (2022/2023)

`analysis/postprocess/bookkeeping.py` gives the HcZZ productions what `run_postprocess.py --postprocess
--plot` gives the other workflows: cutflows, yield tables, normalised histograms and data/MC plots. It
works on the per-job outputs (no merging) and normalises exactly as the datacard builder
(`weight_nominal × xs × lumi / Σw` per dataset and era, `dc.event_scale`). It can run as a stage of
the scoring command, or standalone over several productions.

**Full 2022/2023 bookkeeping** (inputs of the nominal card; backgrounds, H+c and data come from
three productions, each restricted to its processes with `--source RAW_DIR SCORED_DIR KINDS`):

```bash
source /eos/user/s/snandaku/miniforge3/etc/profile.d/conda.sh && conda activate b_hive_new
cd /eos/user/s/snandaku/Higgscharmnew/higgscharm
O=/eos/user/s/snandaku/higgscharm/outputs
python3 -u analysis/postprocess/bookkeeping.py \
  --source $O/hplusc_mva_4class_ctag2d_jecshifts_bkg    $O/hplusc_mva_4class_ctag2d_jecshifts_bkg_scored_v1 background \
  --source $O/hplusc_mva_4class_ctag2d_jecshifts_fschad $O/hplusc_mva_4class_ctag2d_fschad_scored_v1        signal \
  --source $O/hplusc_mva_4class_ctag2d                  $O/hplusc_mva_4class_ctag2d_scored_v8               data \
  --eras 2022preEE 2022postEE 2023preBPix 2023postBPix \
  --sumw-table /eos/user/s/snandaku/Analysis/v4v3_card_2026_10_23/sumw_table.json \
  --output-dir <dir> --workers 16
```

**As a stage of the scoring command** (one production: `--input-dir` = raw, `--output-dir` = scored;
`--config/--model` are only needed for `score`; `score` alone is the default and unchanged):

```bash
python3 -u analysis/postprocess/run_mva_postprocess.py \
  --input-dir  $O/hplusc_mva_4class_ctag2d_jecshifts_bkg \
  --output-dir $O/hplusc_mva_4class_ctag2d_jecshifts_bkg_scored_v1 \
  --all-eras --stages bookkeep,plot --workers 16 \
  --sumw-table /eos/user/s/snandaku/Analysis/v4v3_card_2026_10_23/sumw_table.json --bookkeep-dir <dir>
```

Output, per era, per year (2022, 2023) and for all eras (`<tag>` = era, year or `all`):

| File | Content |
|------|---------|
| `cutflow_<tag>.csv` | Weighted cutflow per process: generated (pre jet veto) → jet veto map → trigger → lumimask → good vertex → ZZ candidate → ≥1 c-jet (per-job `.coffea` metadata), then selected / m4l window / card input (scored parquets) |
| `cutflow_eff_<tag>.csv` | The same relative to the generated row (data: relative to the input events) |
| `yields_<tag>.csv`, `.tex` | Selected, m4l window and card input (window and `mva_score_Signal ≥ 0`), stat. uncertainty, raw events |
| `sumw_check_<tag>.csv` | Σw of the `.coffea` files vs the Σw used for the normalisation; ≠ 1 means missing/duplicated jobs or a Σw table over a different set of chunks |
| `job_check_<tag>.csv` | Per process: events written according to the `.coffea` jobs (`raw_final_nevents`) vs rows in the scored parquets. If they differ, `.coffea` jobs are missing and that process's cutflow rows are left empty (NaN); its yields from the parquets are unaffected |
| `histograms_<tag>.coffea` | `{variable: hist.Hist(region, process, variable)}`, regions `selected` / `window`, with under/overflow; m4l, MVA scores, Z masses, jet and c-jet variables |
| `plots/` | Stacked MC with MC-stat band, H+c ×100, data with ratio panel |
| `bookkeeping_<tag>.json` | All numbers and the inputs used |

| Flag | Description |
|------|-------------|
| `--source RAW SCORED KINDS` | Repeatable; `KINDS` = `background`, `signal`, `data`, `all` or process names (comma-separated) |
| `--raw-dir`, `--scored-dir` | Single production instead of `--source` |
| `--sumw-table` | Pre-veto Σw table of the card (`nom` + `sig` entries); default: sidecars with `--sumw-key` (default `sumw_preveto`) |
| `--mass-window LO HI` | Window of the yield table and the `window` region (default 105 160, as the card) |
| `--no-split-higgs` | One pooled Other_Higgs instead of ggH/VBF/WminusH/WplusH/ZH/ttH/bbH/HPlusB |
| `--exclude` | Regexes of sample directories to skip |
| `--unblind`, `--blind-m4l LO HI` | Data blinding (default: data only outside the mass window in the tables and the `window` region; m4l plots hide 115–135 GeV) |
| `--stages`, `--signal-scale` | `bookkeep`, `plot` (standalone default: both); H+c overlay scale (default 100) |

Validated (2026-10-26): the background yields equal the nominal card's loader (r < 429, V4/V3, optbin10)
for every process to < 1e-4; the qqZZ cutflow equals the old `run_postprocess.py --postprocess` cutflow of
`hplusc_mva_4class`; scoring output is bit-identical to the previous script.
Not included: Z+X (data-driven, its own chain), the stitched 4FS+3FS signal and its flat muon ID SF (the
Signal column is the plain 4FS sample at 97.62 fb × BR) and the shifted templates. Data come from the
`_ctag2d` production (JEC V2/V1); 90 of its 261 data jobs have no `.coffea` (August production), so
the data cutflow is left empty and only the data yields (from the parquets) are given. Run time about
1 min for the four eras.

Known effect found by `sumw_check`: the card's Σw table sums only chunks that have a parquet file, and a
chunk with no selected event writes none. For the ggZZ→τ samples this lowers Σw (up to ×1/1.88 for 4τ in
2022preEE) and raises their yield; the effect on the card is +0.0056 events, 0.10% of ggZZ.

### 2. Optional, not used for the cards: `run_postprocess.py --mva-inference`

`--mva-inference` is a flag on `run_postprocess.py`, not a separate script. It scores only the
merged `parquets_<sample>/` directories that `--postprocess` writes (it stops with an error if they
do not exist), so run it together with `--postprocess` or after it. It writes scored parquets and
per-sample score/feature histograms to `--mva-output` (default `outputs/<workflow>_mvascores/<year>`).
The merged files have no shift directories, so this output cannot be used to build datacards; it is
meant for quick plots and checks.

```bash
python3 run_postprocess.py \
  --workflow <workflow> --year <era> --output_format parquet \
  --postprocess --mva-inference \
  --mva-config models/hc_zzto4l_mw_training_4class_nomass.yml \
  --mva-model  models/best_model.pt \
  --mva-output /eos/user/<u>/<username>/higgscharm/outputs/<workflow>_mvascores/<era>
```

| Flag | Description |
|------|-------------|
| `--mva-inference` | Score the merged parquets after (or together with) `--postprocess` |
| `--mva-config` / `--mva-model` | Config and checkpoint (required with `--mva-inference`) |
| `--mva-output` | Output directory (default `outputs/<workflow>_mvascores/<year>`) |
| `--mva-mass-window` | Score only events inside the config's `mass_window` (100–150 GeV if the config has none) |

## Outputs

Both entry points add these columns to every scored parquet:
- `mva_score_<class>` — softmax score per class (`mva_score_qqZZ`, `mva_score_ggZZ`, `mva_score_Signal`, `mva_score_Other_Higgs`)
- `mva_signal_score` — the Signal score
- `mva_class_prediction` — predicted class index (argmax)

Location:
- `run_mva_postprocess.py`: `<output-dir>/<era>/<dataset>_<N>/base/[<shift>/]*.parquet`, same layout as the input
  (plus score histograms unless `--no-histograms`).
- `run_postprocess.py --mva-inference`: `<mva-output>/<sample>/base/*.parquet`, plus
  `<mva-output>/<sample>_mva_scores.coffea` with one `hist.Hist` per score column (axes `[score, process, variation]`)
  and per-feature histograms.

## Config

The b-hive config YAML defines the feature list used at training time. The same config must be used at inference to ensure the feature vector matches the model's `input_dim`. Configs are stored in the b-hive repository under `config/`.

`MVAPostProcessor` (`analysis/postprocess/mva_inference.py`) reads these keys from the config:

| Key | Purpose |
|-----|---------|
| `class_names` (or `labels` / `classes`) | Output class order, e.g. `['qqZZ', 'ggZZ', 'Signal', 'Other_Higgs']`. Falls back to `class_0, class_1, ...` with a warning if missing. |
| `mass_window` | `{min, max}` in GeV, used for `--mva-mass-window` and the `in_mass_window` global feature. Defaults to 100/150 if absent. |
| `global_features` | Flat per-event features (e.g. `n_jet`, `m4l`, `n_ctagged_jets`). |
| `cpf_candidates`, `n_cpf_candidates` | Per-jet ("c-jet") features and how many leading jets to use (default 3). |
| `npf_candidates`, `n_npf_candidates` | Per-lepton features and lepton count (default 4). |
| `vtx_candidates`/`vtx_features`, `n_vtx_candidates` | Per-vertex features and count (default 2). |

`input_dim`, `hidden_dim`, `num_layers`, and `num_classes` are **not** read from the config — they're auto-detected from the checkpoint's `state_dict` tensor shapes at load time, so the config only needs to describe *which* features to build, not the resulting dimensions.

**Feature ordering is candidate-major, not feature-major** (candidate outer loop, feature inner loop), concatenated as `cpf_candidates` → `npf_candidates` → `vtx_features`, matching `b-hive_ttcc/tasks/dataset.py`'s training-time layout. This was verified empirically 2026-08-04: candidate-major gives AUC=0.957 (matches the model's true performance), feature-major gives AUC=0.559 (near-random) — an earlier "fix" that switched to feature-major was wrong and was reverted. Don't reorder the loops in `prepare_features()` without re-validating AUC against a known-good checkpoint.

**No ONNX path**: `onnxruntime` is imported opportunistically (`HAS_ONNX` flag) but never actually used — there's no `.onnx` model file and no `InferenceSession` call anywhere in this module. Inference is PyTorch-only (`torch.load` + `load_state_dict` into the natively-defined `MLP_HcZZ_MW_Deep`/`ResidualBlock` classes in this file).
