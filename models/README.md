# MVA Models

This directory stores trained MVA model checkpoints used for post-processing inference in the H+c → ZZ → 4ℓ analysis.

See [`INFERENCE.md`](INFERENCE.md) for the internals of how `MVAPostProcessor` scores events, and how the reducible (Z+X) background gets run through the same model.

## Model format

Models are saved as PyTorch `.pt` checkpoint files produced by the [b-hive](https://github.com/deoache/b-hive) training framework. Each checkpoint encodes the model architecture parameters (`input_dim`, `hidden_dim`, `num_layers`, `num_classes`) so no separate architecture config is needed at inference time.

## Available models

| File | Architecture | Classes | Description |
|------|-------------|---------|-------------|
| `best_model.pt` | MLP_HcZZ_MW_Deep | 4 | 4-class classifier: qqZZ, ggZZ, Signal, Other_Higgs |

## Running MVA inference

After running the standard postprocessing step (`--postprocess`), run MVA inference with:

```bash
python3 run_postprocess.py \
  --workflow hplusc_mva_4class \
  --year <year> \
  --output_format parquet \
  --mva-inference \
  --mva-config /path/to/config.yml \
  --mva-model /path/to/best_model.pt \
  --mva-output /eos/user/<u>/<username>/higgscharm/outputs/hplusc_mvascores/<year>
```

To also apply a mass window cut (100 < m4l < 150 GeV) before inference, add `--mva-mass-window`:

```bash
python3 run_postprocess.py \
  --workflow hplusc_mva_4class \
  --year <year> \
  --output_format parquet \
  --mva-inference \
  --mva-mass-window \
  --mva-config /path/to/config.yml \
  --mva-model /path/to/best_model.pt \
  --mva-output /eos/user/<u>/<username>/higgscharm/outputs/hplusc_mvascores/<year>
```

You can also combine `--postprocess` and `--mva-inference` in a single command to run both steps together:

```bash
python3 run_postprocess.py \
  --workflow hplusc_mva_4class \
  --year <year> \
  --postprocess \
  --output_format parquet \
  --mva-inference \
  --mva-mass-window \
  --mva-config /path/to/config.yml \
  --mva-model /path/to/best_model.pt
```

## CLI flags

| Flag | Description |
|------|-------------|
| `--mva-inference` | Enable MVA inference step |
| `--mva-config` | Path to b-hive config YAML (required with `--mva-inference`) |
| `--mva-model` | Path to trained `.pt` model file (required with `--mva-inference`) |
| `--mva-output` | Output directory for scored parquets and histograms (default: `outputs/<workflow>_mvascores/<year>`) |
| `--mva-mass-window` | Apply 100 < m4l < 150 GeV mass window before inference |

## Outputs

For each sample, the inference step produces:

- **Scored parquets** saved under `<mva-output>/<sample>/base/` — original parquet columns plus:
  - `mva_score_<class>` — softmax score for each class (e.g. `mva_score_Signal`, `mva_score_qqZZ`, ...)
  - `mva_signal_score` — signal class score
  - `mva_class_prediction` — predicted class index (argmax)
- **Score histograms** saved as `<mva-output>/<sample>_mva_scores.coffea` — one `hist.Hist` per score column with axes `[score, process, variation]`

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
