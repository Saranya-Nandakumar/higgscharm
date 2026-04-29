# Z+X Reducible Background Estimation — Scripts

This directory contains the full pipeline for estimating the Z+X (reducible) background
in the **H+c → ZZ → 4ℓ** analysis using the fake rate (matrix) method.

## Overview

The Z+X background arises from DY+jets, WZ, and tt̄ events where one or both Z2 leptons
are non-prompt or misidentified ("fake"). The estimation proceeds in four stages:

```
[1] Measure fake rates        compute_fake_rates.py
         ↓
[2] Estimate Z+X yield        estimate_zx_background.py
         ↓
[3] Run MVA inference         run_zx_inference.py
         ↓
[4] Validate & cross-check    validate_zx_mc_shape.py
                              compute_an_ss_yield.py
```

---

## Prerequisites

### Workflow outputs needed

| Workflow | What it produces | Used by |
|---|---|---|
| `zplusl_ss` (or `zplusl_os`) | Z+1 lepton CR coffea files | `compute_fake_rates.py` |
| `hplusc_mva_4class_CR` | 3P1F / 2P2F / CR_2P2F_SS parquet files | `estimate_zx_background.py`, `compute_an_ss_yield.py` |

Run these workflows first via the standard higgscharm condor submission before running any script here.

### Environment

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh
# or your local conda env with: coffea, numpy, pandas, torch, mplhep
```

---

## Stage 1 — Measure Lepton Fake Rates

**Script:** `compute_fake_rates.py`

Measures $f(p_T, |\eta|, \text{flavour})$ from the Z+1 lepton control region using tag-and-probe.
Produces both **uncorrected** (raw data ratio) and **WZ-corrected** fake rates,
where the WZ real-lepton contamination is subtracted from numerator and denominator.

```bash
python scripts/compute_fake_rates.py \
    --input  /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR/<era> \
    --year   2022postEE \
    --output /eos/user/s/snandaku/higgscharm/outputs/fake_rates/2022postEE
```

Run for each era: `2022preEE`, `2022postEE`, `2023preBPix`, `2023postBPix`.

**Output:** `fake_rates/<era>/fake_rates.npz` — numpy archive with keys:
```
electron__uncorrected__fake_rate
electron__uncorrected__fake_rate_err
electron__uncorrected__n_total
electron__safe_corrected__fake_rate   ← used in yield estimation
...
```

### Visualise fake rates

```bash
# 1D plots: fake rate vs pT, split by barrel/endcap (all 4 eras in one figure)
python scripts/plot_fake_rates_1d.py \
    --input  /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --output docs/plots/fake_rates

# 2D maps: fake rate in (pT, |eta|) plane
python scripts/plot_zx_cr_distributions.py \
    --input  /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --output docs/plots/fake_rate_maps

# Print tabular summary
python scripts/print_fake_rate_table.py \
    --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates
```

---

## Stage 2 — Estimate Z+X Yield (OS method)

**Script:** `estimate_zx_background.py`

Applies fake rates to 3P1F and 2P2F control region events to compute the net Z+X yield
in the signal region using the standard OS formula:

$$N_{Z+X}^{SR} = \left(1 - \frac{N_{ZZ}^{MC}}{N_{3P1F}}\right) \sum_{3P1F} \frac{f}{1-f} - \sum_{2P2F} \frac{f_1 f_2}{(1-f_1)(1-f_2)}$$

- **3P1F**: Z1(tight,tight) + Z2(tight, **failing**) — positive contribution
- **2P2F**: Z1(tight,tight) + Z2(**failing**, **failing**) — double-count subtraction
- ZZ correction factor: removes genuine ZZ events that leak into 3P1F

```bash
python scripts/estimate_zx_background.py \
    --cr-input   /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates/2022postEE/fake_rates.npz \
    --year       2022postEE \
    --output     /eos/user/s/snandaku/Analysis/zx_background/2022postEE
```

Run for each era.

**Output:** per-era parquet files of weighted Z+X events with columns:
`weight_zx`, `m4l`, `final_state`, lepton kinematics.

### Cross-check: AN-18-340 Eq. 11 (SS method)

An independent cross-check using same-sign 2P2F events from `CR_2P2F_SS`
(Z1(tight,tight) + Z2(same-sign, both loose-failing)):

$$N_{Z+X} = \left(\frac{N_{OS}}{N_{SS}}\right)^{MC} \times \sum_{\text{DATA, SS}} f_1 \times f_2$$

- **(OS/SS)^MC**: lumi-weighted ratio of reducible MC (DY+WZ+tt̄) in OS vs SS 2P2F CRs.
  Corrects for any charge-asymmetric fake topologies. Typically 1.01–1.19 across eras.
- **Σ f₁×f₂**: sum over SS DATA events in the signal m4l window [100, 150] GeV,
  each event weighted by the product of the two Z2 fake rates.
- The SS region is free of genuine ZZ contamination by construction (real Z cannot decay to same-sign).

```bash
python scripts/compute_an_ss_yield.py \
    --cr-dir     /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --output     docs/zx_an_ss_yield.txt
```

**Results — AN Eq. 11, signal region m4ℓ ∈ [100, 150] GeV:**

| Era | N_SS^DATA | (OS/SS)^MC | Σ f₁f₂ | Z2→ee | Z2→μμ | Total |
|---|---|---|---|---|---|---|
| 2022 preEE | 1552 | 1.194 | 3.67 | 1.00 ± 0.04 | 3.38 ± 0.29 | **4.38 ± 0.29** |
| 2022 postEE | 4099 | 1.013 | 11.52 | 4.75 ± 0.09 | 6.91 ± 0.36 | **11.67 ± 0.37** |
| 2023 preBPix | 3417 | 1.082 | 8.46 | 4.38 ± 0.09 | 4.77 ± 0.30 | **9.15 ± 0.32** |
| 2023 postBPix | 1934 | 1.081 | 5.15 | 2.08 ± 0.06 | 3.49 ± 0.28 | **5.57 ± 0.28** |
| **Total** | **10002** | — | **28.80** | **12.22 ± 0.15** | **18.55 ± 0.61** | **30.77 ± 0.64** |

- Uncertainty: CR statistical only, σ = √(Σ(f₁f₂)ᵢ²)
- Z2→μμ contributes ~60% of total yield (higher muon fake rate)
- 2022 postEE is the largest era (26.7 fb⁻¹)

**Comparison: Eq. 10 (OS SS-based) vs AN Eq. 11:**

| Era | Eq. 10 SS | AN Eq. 11 |
|---|---|---|
| 2022 preEE | 7.0 ± 1.3 | 4.4 ± 0.3 |
| 2022 postEE | 18.6 ± 2.0 | 11.7 ± 0.4 |
| 2023 preBPix | 15.0 ± 1.9 | 9.2 ± 0.3 |
| 2023 postBPix | 7.5 ± 1.5 | 5.6 ± 0.3 |
| **Total** | **48.1 ± 3.4** | **30.8 ± 0.6** |

AN Eq. 11 is lower because it uses only the SS 2P2F events (no 3P1F term),
while Eq. 10 includes the 3P1F contribution which adds the larger positive term.

### OS vs SS cross-check by final state

Computed for 2022 postEE without mass window (m4ℓ > 70 GeV) using `estimate_zx_background.py`
for OS and `compute_an_ss_yield.py` for SS. Combination = (OS + SS) / 2.

| | 4μ | 4e | 2e2μ | 2μ2e |
|---|---|---|---|---|
| OS | 17.58 ± 2.07 | 8.38 ± 0.66 | 15.85 ± 1.86 | 7.38 ± 0.64 |
| SS | 13.46 ± 2.50 | 10.43 ± 1.02 | 14.40 ± 2.47 | 8.30 ± 1.01 |
| Combination | 15.52 ± 1.62 | 9.41 ± 0.61 | 15.13 ± 1.55 | 7.84 ± 0.60 |
| OS/SS | 1.306 ± 0.287 | 0.803 ± 0.101 | 1.101 ± 0.229 | 0.889 ± 0.133 |

OS/SS close to 1 across all final states confirms consistency of the two methods.
Deviations at the 20–30% level are within statistical uncertainties and consistent
with HIG-24-013 findings.

---

## Stage 3 — Run MVA Inference on Z+X Events

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

**Key results:**
- ~96% of Z+X events classified as gg→ZZ
- Median signal score ≈ 0.002
- No events with signal score > 0.5 (no signal-region contamination)

### Print yield summary

```bash
python scripts/print_zx_summary_table.py \
    --fake-rates /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --zx-input   /eos/user/s/snandaku/Analysis/zx_background \
    --latex       # optional: print as LaTeX table
```

---

## Stage 4 — Validation

### 4a. Data-driven vs MC-driven MVA shape

Applies the same fake-rate weighting to MC reducible samples (DY, WZ, tt̄) in the CR
and compares the MVA score shape to the data-driven estimate.
Agreement confirms the method correctly extrapolates from CR to SR kinematics.

```bash
python scripts/validate_zx_mc_shape.py \
    --zx-input    /eos/user/s/snandaku/Analysis/zx_background_mva \
    --cr-input    /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --fake-rates-dir /eos/user/s/snandaku/higgscharm/outputs/fake_rates \
    --config      /path/to/hc_zzto4l_mw_training_4class_nomass.yml \
    --model       models/best_model.pt \
    --output      docs/plots/zx_validation
```

### 4b. Feature distributions in CR vs SR

```bash
python scripts/validate_zx_reducible_features.py \
    --cr-input /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --output   docs/plots/zx_cr_features

python scripts/validate_zx_sideband_features.py \
    --cr-input /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR \
    --output   docs/plots/zx_sideband_features
```

### 4c. Sensitivity and ROC

```bash
python scripts/compute_roc.py \
    --zx-input /eos/user/s/snandaku/Analysis/zx_background_mva \
    --output   docs/plots/roc

python scripts/compare_inference_sensitivity.py \
    --zx-input /eos/user/s/snandaku/Analysis/zx_background_mva \
    --output   docs/plots/sensitivity
```

---

## Control Region Definitions

| Category | Z1 | Z2 | Sign | Contribution |
|---|---|---|---|---|
| 3P1F | tight, tight | tight, **failing** | OS | +f/(1−f) |
| 2P2F | tight, tight | **failing**, **failing** | OS | −f₁f₂/[(1−f₁)(1−f₂)] |
| CR_2P2F_SS | tight, tight | **failing**, **failing** | SS | used in AN Eq. 11 |

---

## Fake Rate Definitions

| Version | Formula | Notes |
|---|---|---|
| Uncorrected | $N_\text{tight}^\text{data} / N_\text{loose}^\text{data}$ | Includes real WZ leptons |
| WZ-corrected | $(N_\text{tight}^\text{data} - N_\text{tight}^{WZ}) / (N_\text{loose}^\text{data} - N_\text{loose}^{WZ})$ | Used in estimation |
| `safe_corrected` | Falls back to uncorrected in bins where WZ subtraction is unstable | Stored in `.npz`, used by default |

Fake rates are binned in 7 $p_T$ bins × 5 $|\eta|$ bins, separately for electrons and muons.
Barrel: $|\eta| < 1.5$; Endcap: $1.5 < |\eta| < 2.5$.

---

~96% classified as gg→ZZ by the MVA.

---

## Script Reference

| Script | Purpose |
|---|---|
| `compute_fake_rates.py` | Measure f(pT, η, flavour) from Z+1ℓ CR |
| `plot_fake_rates_1d.py` | Plot fake rate vs pT (barrel/endcap) |
| `plot_zx_cr_distributions.py` | 2D fake rate maps in (pT, η) |
| `print_fake_rate_table.py` | Tabular fake rate summary |
| `estimate_zx_background.py` | Compute Z+X yield via OS formula |
| `compute_an_ss_yield.py` | Cross-check yield via AN Eq. 11 (SS) |
| `run_zx_inference.py` | Score Z+X events with trained MVA |
| `print_zx_summary_table.py` | Print yield table per era × final state |
| `validate_zx_mc_shape.py` | Compare data-driven vs MC-driven MVA shape |
| `validate_zx_reducible_features.py` | Feature distributions in CR |
| `validate_zx_sideband_features.py` | Feature distributions in sideband |
| `compute_roc.py` | ROC curves for Z+X vs signal separation |
| `compute_feature_importance.py` | MVA feature importance ranking |
| `compare_inference_sensitivity.py` | Sensitivity comparison across configurations |
