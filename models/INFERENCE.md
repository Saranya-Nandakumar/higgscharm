# Running Inference with the `.pt` Model

This note explains the mechanics of scoring events with `best_model.pt`
(see `README.md` for the CLI-level quick start) and how the reducible
(Z+X) background gets scored by the same model even though it isn't one
of the model's trained classes.

## What the model is

`best_model.pt` is a PyTorch checkpoint (`state_dict`, not a scripted/
traced model) trained by [b-hive](https://github.com/deoache/b-hive) for
the architecture `MLP_HcZZ_MW_Deep`, defined natively in
`analysis/postprocess/mva_inference.py` (not imported from b-hive) so
inference has no b-hive runtime dependency. It's a 4-class classifier —
`qqZZ`, `ggZZ`, `Signal` (H+c), `Other_Higgs` — trained only on prompt
MC. There is no ONNX export; `onnxruntime` is imported opportunistically
in this file but never called.

## How `MVAPostProcessor` scores events

1. **Load** (`_load_model`): `torch.load(model_path, map_location='cpu',
   weights_only=False)` reads the checkpoint, then `input_dim`,
   `hidden_dim`, `num_layers`, and `num_classes` are read back out of the
   `state_dict` tensor shapes (e.g. `state_dict['input_proj.0.weight'].shape[1]`)
   rather than from the config — the config only has to describe *which*
   features to build, since the model's own weights fix the dimensions.
   `class_names` comes from the config (`class_names`/`labels`/`classes`),
   falling back to `class_0, class_1, ...` if absent.
2. **Feature prep** (`prepare_features`): builds a flat feature vector per
   event from the config's `global_features` / `cpf_candidates` /
   `npf_candidates` / `vtx_features` lists, using column-name aliases
   (e.g. `zz_mass_inclusive` or `m4l`) so it works against slightly
   different parquet schemas. Ordering is **candidate-major** (loop over
   candidates outer, features inner), concatenated
   `cpf_candidates → npf_candidates → vtx_features`, matching b-hive's own
   training-time layout in `b-hive_ttcc/tasks/dataset.py`. This was
   confirmed empirically (2026-08-04): candidate-major → AUC 0.957,
   feature-major → AUC 0.559 (near-random). Getting this ordering wrong
   silently produces a working-but-useless model — validate against a
   known AUC after touching `prepare_features()`.
3. **Predict** (`predict`): a single forward pass through the MLP +
   softmax, returning per-class scores, the `Signal`-class score, and the
   argmax class prediction.
4. **Write out** (`process_parquets`): the CLI entry point
   (`run_postprocess.py --mva-inference`) drives all three steps above
   over every sample's parquet files, optionally applying the mass-window
   cut first, and writes `mva_score_<class>`, `mva_signal_score`,
   `mva_class_prediction` columns plus per-class score histograms. This
   is the path documented in `README.md`.

## Including the reducible (Z+X) background

Z+X (fake-lepton) background is **not** a trained class of the model —
it can't be, since it isn't a clean MC process. Instead it's estimated
data-driven from loose-lepton control regions and then run through the
*same* trained model to land on the same score axis as the 4 MC classes,
so it can be binned into the same histogram/datacard.

The chain (3 separate scripts, in order):

1. **`scripts/estimate_zx_background.py`** — computes the fake-rate (OS
   method) yield from the 3P1F/2P2F control regions:
   ```
   N_bkg_SR = (1 - N_ZZ_3P1F/N_3P1F) * sum_3P1F[f/(1-f)]
              - sum_2P2F[f1*f2 / ((1-f1)*(1-f2))]
   ```
   Writes `zx_3p1f_<era>.parquet` / `zx_2p2f_<era>.parquet` with a
   per-event `zx_weight` column (the fake-rate-derived weight) — no MVA
   involvement yet.

2. **`scripts/run_zx_inference.py`** — scores those control-region
   parquets with the **same** `best_model.pt` + config used for the MC
   samples. It doesn't call `process_parquets()` directly; instead it
   remaps CR-specific column names (`m4l_3p1f` → `m4l`, `z1_mass_3p1f` →
   `z1_mass`, etc., separate maps for 3P1F vs 2P2F) onto the MVA's
   expected feature names via `prepare_df_for_inference()`, then calls
   `processor.prepare_features()` + `processor.predict()` directly on the
   remapped frame. `zx_weight` (and `weight_nominal`, `zx_sign`) pass
   through untouched as part of `PASSTHROUGH`. Output:
   `<output>/<era>/zx_{3p1f,2p2f}_<era>_mva.parquet`, now carrying
   `mva_score_<class>` / `mva_signal_score` / `mva_class_prediction`
   alongside `zx_weight`.

3. **`combine/scripts/create_datacards_ctag2d_100150.py --zx-dir <dir>`**
   — `load_zx_parquets()` reads `mva_score_Signal` and `zx_weight` back
   out of those `_mva.parquet` files (already in the mass window) and
   histograms them with the same `MVA_BINS` binning as the 4 MC classes
   into a 5th process, `ZX`. It gets a 30% lnN uncertainty (`ZX_norm`)
   and floats freely in the fit via a `rateParam` (`ZX_rate`), since it's
   data-driven rather than MC-normalized. Pass `--skip-zx` to build a
   Z+X-free datacard instead (drops the process/row entirely).

So "including reducible background in inference" doesn't mean adding a
5th output class to the model — it means running the existing 4-class
model over the fake-rate control-region parquets via
`run_zx_inference.py`, using their pre-computed `zx_weight` (not the
model's classification) as the event weight, and letting
`create_datacards_ctag2d_100150.py` merge the two streams at histogram
time.
