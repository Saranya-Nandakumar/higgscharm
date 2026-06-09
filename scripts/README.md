# Z+X Reducible Background Estimation — Scripts

This directory contains the full pipeline for estimating the Z+X (reducible) background
in the **H+c → ZZ → 4ℓ** analysis using the fake rate (matrix) method.

## Overview

The Z+X background arises from DY+jets, WZ, and tt̄ events where one or both Z2 leptons
are non-prompt or misidentified ("fake"). The estimation proceeds in five stages:

```
[0] Run condor workflows       hplusc_mva_4class_CR  (3P1F/2P2F parquets)
                               zplusl_ss              (fake rate measurement coffea)
         ↓
[1] Merge CR parquets          merge_cr_parquets.py
         ↓
[2] Measure fake rates         compute_fake_rates.py
         ↓
[3] Estimate Z+X yield         estimate_zx_background.py
         ↓
[4] Run MVA inference          run_zx_inference.py
         ↓
[5] Validate & cross-check     validate_zx_mc_shape.py
                               compute_an_ss_yield.py
```

---

## Prerequisites

### Condor workflows to run first

Submit both workflows via the standard higgscharm condor submission before running
any script here:

```bash
# 3P1F / 2P2F control region parquets (input to Stage 1 and Stage 3)
for era in 2022preEE 2022postEE 2023preBPix 2023postBPix; do
  python submit_condor.py -w hplusc_mva_4class_CR -y $era --output_format parquet --eos --submit
done

# Z+1 lepton CR coffea histograms (input to Stage 2)
for era in 2022preEE 2022postEE 2023preBPix 2023postBPix; do
  python submit_condor.py -w zplusl_ss -y $era --submit
done
```

Check job completion with `jobs_status.py` before proceeding.

### Environment

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh
# or your local conda env with: coffea, numpy, pandas, pyarrow, torch, mplhep
```

---

## Stage 1 — Merge CR Parquets

**Script:** `merge_cr_parquets.py`

The `hplusc_mva_4class_CR` workflow produces thousands of per-job parquet files
(one per condor partition). This script merges them into one file per era and category
before running the fake rate estimation.

```bash
# Merge all eras at once
python scripts/merge_cr_parquets.py --year all

# Or a single era
python scripts/merge_cr_parquets.py --year 2022postEE
```

**Output:**
```
/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR_merged/<era>/
    CR_3P1F.parquet   # all 3P1F events merged
    CR_2P2F.parquet   # all 2P2F events merged
```

| Era | CR_3P1F events | CR_2P2F events |
|-----|----------------|----------------|
| 2022preEE | 12,660 | 25,319 |
| 2022postEE | 36,723 | 76,864 |
| 2023preBPix | 23,637 | 45,917 |
| 2023postBPix | 13,285 | 24,908 |
| **Total** | **86,305** | **173,008** |

---

## Stage 2 — Measure Lepton Fake Rates

**Script:** `compute_fake_rates.py`

Measures $f(p_T, |\eta|, \text{flavour})$ from the Z+1 lepton control region using
tag-and-probe on the `zplusl_ss` (or `zplusl_os`) coffea output.
Produces **uncorrected** (raw data ratio) and **WZ-corrected** fake rates,
where the WZ real-lepton contamination is subtracted from numerator and denominator.

> **Important:** The fake rate is a **data-driven measurement**. The script reads
> DATA coffea files to compute `f = N_tight / N_loose`, then subtracts WZ MC
> (and only WZ, matching AN-18-340) to correct for real prompt leptons from WZ → 3ℓν.
> The `--input` must point to the **base higgscharm outputs directory**, not to any
> workflow subdirectory — the script internally appends `/<workflow>/<era>/`.

```bash
for era in 2022preEE 2022postEE 2023preBPix 2023postBPix; do
  python scripts/compute_fake_rates.py \
      --input    /eos/user/s/snandaku/higgscharm/outputs \
      --workflow zplusl_ss \
      --year     $era \
      --output   /eos/user/s/snandaku/higgscharm/outputs/fake_rates/$era
done
```

Use `--workflow zplusl_os` for the OS variant (higher fake rates, closer to CJLST/UAntwerp).

**Output:** `fake_rates/<era>/fake_rates.pkl` — dictionary with keys:
```
fake_rates['electron']['fake_rate']        # shape (n_pt_bins, n_eta_bins)
fake_rates['electron']['fake_rate_err']
fake_rates['muon']['fake_rate']
fake_rates['muon']['fake_rate_err']
```
The `safe_corrected` variant is used by default: falls back to uncorrected in bins where
WZ MC statistics are insufficient for a stable subtraction.

### Visualise fake rates

```bash
# 1D plots: fake rate vs pT, split by barrel/endcap (all 4 eras in one figure)
python scripts/plot_fake_rates_1d.py \
    --input  /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --output docs/plots/fake_rates

# Print tabular summary
python scripts/print_fake_rate_table.py \
    --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates
```

---

## Stage 3 — Estimate Z+X Yield (OS method)

**Script:** `estimate_zx_background.py`

Applies the per-event fake rates from Stage 2 to the merged CR parquets from Stage 1
to compute the net Z+X yield using the standard OS formula:

$$N_{Z+X}^{SR} = \left(1 - \frac{N_{ZZ}^{MC}}{N_{3P1F}}\right) \sum_{3P1F} \frac{f}{1-f} - \sum_{2P2F} \frac{f_1 f_2}{(1-f_1)(1-f_2)}$$

- **3P1F**: Z1(tight,tight) + Z2(tight, **failing**) — positive contribution
- **2P2F**: Z1(tight,tight) + Z2(**failing**, **failing**) — double-count subtraction
- ZZ correction factor: removes genuine ZZ events leaking into 3P1F

```bash
for era in 2022preEE 2022postEE 2023preBPix 2023postBPix; do
  python scripts/estimate_zx_background.py \
      --cr-input   /eos/user/s/snandaku/higgscharm/outputs \
      --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates/$era/fake_rates.pkl \
      --year       $era \
      --output     /eos/user/s/snandaku/Analysis/zx_background/$era
done
```

**Output:** per-era parquet files of weighted Z+X events:
```
/eos/user/s/snandaku/Analysis/zx_background/<era>/
    zx_3p1f_<era>.parquet   # 3P1F events with zx_weight (positive)
    zx_2p2f_<era>.parquet   # 2P2F events with zx_weight (negative, subtraction term)
```
Columns include: `zx_weight`, `m4l`, `final_state`, lepton kinematics (`l3_pt`, `l3_eta`, ...).

### Cross-check: AN-18-340 Eq. 11 (SS method)

An independent cross-check using same-sign 2P2F events:

```bash
python scripts/compute_an_ss_yield.py \
    --cr-dir     /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --output     docs/zx_an_ss_yield.txt
```

---

## Stage 4 — Run MVA Inference on Z+X Events

**Script:** `run_zx_inference.py`

Applies the trained 4-class MLP to the weighted Z+X parquet files to produce
MVA score distributions for each class (qqZZ, ggZZ, Signal, Other Higgs).

```bash
python scripts/run_zx_inference.py \
    --input  /eos/user/s/snandaku/Analysis/zx_background \
    --config /path/to/hc_zzto4l_mw_training_4class_nomass.yml \
    --model  models/best_model.pt \
    --output /eos/user/s/snandaku/Analysis/zx_background_mva
```

The `models/best_model.pt` in this repository is the 4-class MLP
(trained on `hc_zzto4l_mw_training_4class_nomass` config).

**Output:**
```
/eos/user/s/snandaku/Analysis/zx_background_mva/<era>/
    zx_3p1f_<era>_mva.parquet   # with mva_score_qqZZ, mva_score_ggZZ,
    zx_2p2f_<era>_mva.parquet   #      mva_score_Signal, mva_score_Other_Higgs
```

**Key results (Run 3, already computed):**
- ~96% of Z+X events classified as gg→ZZ
- Median signal score ≈ 0.002
- No events with signal score > 0.5 — Z+X does not contaminate the signal region

### Print yield summary

```bash
python scripts/print_zx_summary_table.py \
    --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --zx-input   /eos/user/s/snandaku/Analysis/zx_background \
    --latex       # optional: print as LaTeX table
```

---

## Stage 5 — Validation

### 5a. Data-driven vs MC-driven MVA shape

```bash
python scripts/validate_zx_mc_shape.py \
    --zx-input       /eos/user/s/snandaku/Analysis/zx_background_mva \
    --cr-input       /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --fake-rates-dir /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --config         /path/to/hc_zzto4l_mw_training_4class_nomass.yml \
    --model          models/best_model.pt \
    --output         docs/plots/zx_validation
```

### 5b. Feature distributions in CR vs SR

```bash
python scripts/validate_zx_reducible_features.py \
    --cr-input /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --output   docs/plots/zx_cr_features

python scripts/validate_zx_sideband_features.py \
    --cr-input /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --output   docs/plots/zx_sideband_features
```

---

## Control Region Definitions

| Category | Z1 | Z2 | Sign | Contribution |
|---|---|---|---|---|
| 3P1F | tight, tight | tight, **failing** | OS | +f/(1−f) |
| 2P2F | tight, tight | **failing**, **failing** | OS | −f₁f₂/[(1−f₁)(1−f₂)] |
| CR_2P2F_SS | tight, tight | **failing**, **failing** | SS | used in AN Eq. 11 cross-check |

---

## Fake Rate Definitions

| Version | Formula | Notes |
|---|---|---|
| Uncorrected | $N_\text{tight}^\text{data} / N_\text{loose}^\text{data}$ | Includes real WZ leptons |
| WZ-corrected | $(N_\text{tight}^\text{data} - N_\text{tight}^{WZ}) / (N_\text{loose}^\text{data} - N_\text{loose}^{WZ})$ | Preferred |
| `safe_corrected` | Falls back to uncorrected where WZ subtraction is unstable | Used by default |

Fake rates are binned in 7 $p_T$ bins × 5 $|\eta|$ bins, separately for electrons and muons.
SS fake rates (zplusl_ss) are ~2–3× lower than OS (zplusl_os / CJLST/UAntwerp) by construction.

---

## Script Reference

| Script | Stage | Purpose |
|---|---|---|
| `merge_cr_parquets.py` | 1 | Merge per-job parquets into one file per era/category |
| `compute_fake_rates.py` | 2 | Measure f(pT, η, flavour) from zplusl_ss/os CR |
| `plot_fake_rates_1d.py` | 2 | Plot fake rate vs pT (barrel/endcap) |
| `plot_zx_cr_distributions.py` | 2 | 2D fake rate maps in (pT, η) |
| `print_fake_rate_table.py` | 2 | Tabular fake rate summary |
| `estimate_zx_background.py` | 3 | Compute Z+X yield via OS formula |
| `compute_an_ss_yield.py` | 3 | Cross-check yield via AN Eq. 11 (SS) |
| `run_zx_inference.py` | 4 | Score Z+X events with trained MVA |
| `print_zx_summary_table.py` | 4 | Print yield table per era × final state |
| `validate_zx_mc_shape.py` | 5 | Compare data-driven vs MC-driven MVA shape |
| `validate_zx_reducible_features.py` | 5 | Feature distributions in CR |
| `validate_zx_sideband_features.py` | 5 | Feature distributions in sideband |
| `compute_roc.py` | 5 | ROC curves for Z+X vs signal separation |
| `compute_feature_importance.py` | 5 | MVA feature importance ranking |
| `compare_inference_sensitivity.py` | 5 | Sensitivity comparison across configurations |
