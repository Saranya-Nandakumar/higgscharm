"""
Post-process MVA Inference module for H+c -> ZZ -> 4l analysis.

Runs MVA inference on parquet files using b-hive model and config.
All feature definitions and class names are read from the config - nothing is hardcoded.

Config should contain:
    - class_names (or labels/classes): List of class names, e.g. ['qqZZ', 'ggZZ', 'Signal', 'Other_Higgs']
    - mass_window: {min: 100, max: 150}
    - global_features, cpf_candidates, vtx_features, etc. for feature definitions

Usage:
    from analysis.postprocess.mva_inference import MVAPostProcessor

    mva_processor = MVAPostProcessor(
        bhive_config_path='/eos/user/s/snandaku/b-hive_ttcc/config/hc_zzto4l_4class.yml',
        bhive_model_path='/eos/user/s/snandaku/b-hive_ttcc/output/.../best_model.pt',
    )
    mva_processor.process_parquets(input_dir)
"""

import os
import sys
import yaml
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Try to import ONNX
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# =============================================================================
# MLP Model Definition (matches b-hive MLP_HcZZ_MW_Deep)
# =============================================================================
if HAS_PYTORCH:
    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout=0.3):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * 2)
            self.fc2 = nn.Linear(dim * 2, dim)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.GELU()

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x + residual

    class MLP_HcZZ_MW_Deep(nn.Module):
        def __init__(self, input_dim, num_classes=5, hidden_dim=384, num_layers=12, dropout=0.4):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)])
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            return self.head(x)

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    class ImprovedResidualBlockV2(nn.Module):
        def __init__(self, dim, dropout=0.3, drop_path=0.0, expansion=2):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.act = nn.SiLU()
            self.fc2 = nn.Linear(dim * expansion, dim)
            self.dropout = nn.Dropout(dropout)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.se = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, dim),
                nn.Sigmoid(),
            )
            self.gamma = nn.Parameter(torch.ones(dim) * 0.1)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            se_weight = self.se(residual)
            x = x * se_weight
            return residual + self.drop_path(self.gamma * x)

    class MLP_HcZZ_MW_Deep_4class_Optimized(nn.Module):
        """Matches b-hive MLP_HcZZ_MW_Deep_4class_Optimized for standalone inference."""

        def __init__(self, global_dim, cpf_total, npf_total, vtx_total, lt_total,
                     num_classes=4, hidden_dim=192, num_layers=4, dropout=0.4, drop_path_rate=0.1):
            super().__init__()
            lepton_out_dim = hidden_dim // 2
            z_out_dim = hidden_dim // 4
            global_out_dim = hidden_dim // 2
            global_higgs_dim = global_dim + lt_total

            self.jet_encoder = nn.Sequential(
                nn.Linear(cpf_total, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            )
            self.lepton_encoder = nn.Sequential(
                nn.Linear(npf_total, lepton_out_dim), nn.LayerNorm(lepton_out_dim), nn.SiLU(), nn.Dropout(dropout),
            )
            self.z_encoder = nn.Sequential(
                nn.Linear(vtx_total, z_out_dim), nn.LayerNorm(z_out_dim), nn.SiLU(), nn.Dropout(dropout),
            )
            self.global_encoder = nn.Sequential(
                nn.Linear(global_higgs_dim, global_out_dim), nn.LayerNorm(global_out_dim), nn.SiLU(), nn.Dropout(dropout),
            )
            fusion_dim = hidden_dim + lepton_out_dim + z_out_dim + global_out_dim
            self.fusion_proj = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout * 0.5),
            )
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
            self.blocks = nn.ModuleList([ImprovedResidualBlockV2(hidden_dim, dropout=dropout, drop_path=dpr[i]) for i in range(num_layers)])
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, global_f, cpf_f, npf_f, vtx_f, lt_f):
            B = global_f.shape[0]
            jet_enc = self.jet_encoder(cpf_f.reshape(B, -1))
            lepton_enc = self.lepton_encoder(npf_f.reshape(B, -1))
            z_enc = self.z_encoder(vtx_f.reshape(B, -1))
            global_enc = self.global_encoder(torch.cat([global_f, lt_f.reshape(B, -1)], dim=-1))
            x = self.fusion_proj(torch.cat([jet_enc, lepton_enc, z_enc, global_enc], dim=-1))
            for block in self.blocks:
                x = block(x)
            return self.head(x)


# =============================================================================
# MVA Jet Tag Boundaries (for computing jet_is_* features)
# =============================================================================
# BUG FOUND AND FIXED 2026-08-14: this used to be a single fixed boundary set
# (kept below, renamed, as the documented fallback) that turned out to be the
# "BACKUP: hand-optimized boundaries (pre-official-calibration)" set from
# b-hive_ttcc/utils/coffea_processors/lz4_hczz_processor.py -- NOT the real
# per-era boundaries (JET_TAG_BOUNDARIES_BY_ERA in that same file) that the
# actual training-time LZ4 array construction uses. Confirmed via a real
# numeric check on 2023postBPix Signal jets: only 78.5% agreement between the
# two boundary sets -- consistent with (and a strong contributing cause of)
# MVAPostProcessor's scores collapsing to ~99% one class (mean P(Signal) on
# true Signal events = 0.0011, vs the model's own native b-hive ground-truth
# InferenceTask output of 0.5975 for the identical checkpoint). `jet_is_*`
# one-hot flags are 33 of this model's 101 total input features (11 flags x 3
# jet slots), so a wrong categorization here corrupts a third of the vector.
#
# Fixed: real per-era boundaries, copied from the confirmed-correct
# lz4_hczz_processor.py::JET_TAG_BOUNDARIES_BY_ERA (itself cross-checked
# 2026-08-14 against the real HiggsDNA metaconditions HPC_ctag_WPs JSON, the
# same exercise that fixed analysis/corrections/ctag2d.py's identical class of
# bug). _compute_jet_tags() now takes an `era` argument and uses these;
# MVA_JET_TAG_BOUNDARIES_LEGACY_FALLBACK is used only if era is unrecognized
# (matching lz4_hczz_processor.py's own fallback behavior -- not a "safe"
# default, just what the reference implementation does).
JET_TAG_BOUNDARIES_BY_ERA = {
    "2022preEE": {
        "L0": {"x": (0.0,   0.160), "y": (0.0,   1.0)},
        "C0": {"x": (0.160, 0.332), "y": (0.000, 1.000)},
        "C1": {"x": (0.332, 0.706), "y": (0.000, 1.000)},
        "C2": {"x": (0.706, 1.000), "y": (0.090, 0.261)},
        "C3": {"x": (0.706, 1.000), "y": (0.036, 0.090)},
        "C4": {"x": (0.706, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.706, 1.000), "y": (0.261, 0.799)},
        "B1": {"x": (0.706, 1.000), "y": (0.799, 0.933)},
        "B2": {"x": (0.706, 1.000), "y": (0.933, 0.978)},
        "B3": {"x": (0.706, 1.000), "y": (0.978, 0.993)},
        "B4": {"x": (0.706, 1.000), "y": (0.993, 1.000)},
    },
    "2022postEE": {
        "L0": {"x": (0.0,   0.160), "y": (0.0,   1.0)},
        "C0": {"x": (0.160, 0.332), "y": (0.000, 1.000)},
        "C1": {"x": (0.332, 0.706), "y": (0.000, 1.000)},
        "C2": {"x": (0.706, 1.000), "y": (0.090, 0.261)},
        "C3": {"x": (0.706, 1.000), "y": (0.036, 0.090)},
        "C4": {"x": (0.706, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.706, 1.000), "y": (0.261, 0.799)},
        "B1": {"x": (0.706, 1.000), "y": (0.799, 0.932)},
        "B2": {"x": (0.706, 1.000), "y": (0.932, 0.978)},
        "B3": {"x": (0.706, 1.000), "y": (0.978, 0.993)},
        "B4": {"x": (0.706, 1.000), "y": (0.993, 1.000)},
    },
    "2023preBPix": {
        "L0": {"x": (0.0,   0.144), "y": (0.0,   1.0)},
        "C0": {"x": (0.144, 0.292), "y": (0.000, 1.000)},
        "C1": {"x": (0.292, 0.646), "y": (0.000, 1.000)},
        "C2": {"x": (0.646, 1.000), "y": (0.078, 0.240)},
        "C3": {"x": (0.646, 1.000), "y": (0.036, 0.078)},
        "C4": {"x": (0.646, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.646, 1.000), "y": (0.240, 0.752)},
        "B1": {"x": (0.646, 1.000), "y": (0.752, 0.915)},
        "B2": {"x": (0.646, 1.000), "y": (0.915, 0.972)},
        "B3": {"x": (0.646, 1.000), "y": (0.972, 0.992)},
        "B4": {"x": (0.646, 1.000), "y": (0.992, 1.000)},
    },
    "2023postBPix": {
        "L0": {"x": (0.0,   0.144), "y": (0.0,   1.0)},
        "C0": {"x": (0.144, 0.288), "y": (0.000, 1.000)},
        "C1": {"x": (0.288, 0.640), "y": (0.000, 1.000)},
        "C2": {"x": (0.640, 1.000), "y": (0.082, 0.243)},
        "C3": {"x": (0.640, 1.000), "y": (0.036, 0.082)},
        "C4": {"x": (0.640, 1.000), "y": (0.000, 0.036)},
        "B0": {"x": (0.640, 1.000), "y": (0.243, 0.742)},
        "B1": {"x": (0.640, 1.000), "y": (0.742, 0.909)},
        "B2": {"x": (0.640, 1.000), "y": (0.909, 0.969)},
        "B3": {"x": (0.640, 1.000), "y": (0.969, 0.992)},
        "B4": {"x": (0.640, 1.000), "y": (0.992, 1.000)},
    },
}

MVA_JET_TAG_BOUNDARIES_LEGACY_FALLBACK = {
    "C0": {"x": (0.15, 0.4), "y": (0.0, 0.9)},
    "C1": {"x": (0.4, 0.7339), "y": (0.0, 0.9)},
    "C2": {"x": (0.7339, 1.0), "y": (0.1851, 0.2688)},
    "C3": {"x": (0.7339, 1.0), "y": (0.0382, 0.1851)},
    "C4": {"x": (0.7339, 1.0), "y": (0.0, 0.0382)},
    "B0": {"x": (0.7339, 1.0), "y": (0.2688, 0.4057)},
    "B1": {"x": (0.7339, 1.0), "y": (0.4057, 0.4068)},
    "B2": {"x": (0.7339, 1.0), "y": (0.4068, 0.6788)},
    "B3": {"x": (0.7339, 1.0), "y": (0.6788, 0.8406)},
    "B4": {"x": (0.7339, 1.0), "y": (0.8406, 1.0)},
    "L0": {"x": (0.0, 0.15), "y": (0.0, 0.9)},
}


class MVAPostProcessor:
    """Post-processor for MVA inference using b-hive config and model."""

    def __init__(
        self,
        bhive_config_path: str,
        bhive_model_path: str,
        apply_mass_window: bool = True,
        era: str = None,
    ):
        """
        Args:
            bhive_config_path: Path to b-hive config YAML
            bhive_model_path: Path to model file (.pt)
            apply_mass_window: Whether to apply mass window from config
            era: dataset year/era (2022preEE/2022postEE/2023preBPix/2023postBPix),
                selects the real per-era jet-category boundaries in
                JET_TAG_BOUNDARIES_BY_ERA for the jet_is_* features. If None or
                unrecognized, falls back to MVA_JET_TAG_BOUNDARIES_LEGACY_FALLBACK
                (WRONG boundaries -- see that constant's docstring; only kept as a
                fallback to match lz4_hczz_processor.py's own behavior, not because
                it's a safe default). Always pass a real era in production use.
        """
        self.bhive_config_path = bhive_config_path
        self.bhive_model_path = bhive_model_path
        self.apply_mass_window = apply_mass_window
        self.era = era
        if era not in JET_TAG_BOUNDARIES_BY_ERA:
            logging.warning(
                f"MVAPostProcessor: era='{era}' not in JET_TAG_BOUNDARIES_BY_ERA "
                f"{list(JET_TAG_BOUNDARIES_BY_ERA)} -- falling back to the legacy "
                f"(WRONG, pre-official-calibration) single fixed jet-tag boundary "
                f"set. jet_is_* features (33/101 of this model's inputs) will not "
                f"match training. Pass a real era to MVAPostProcessor(era=...)."
            )

        # Load config
        self.config = self._load_config()

        # Get mass window from config
        mw = self.config.get('mass_window', {})
        self.mass_window_min = mw.get('min', 100)
        self.mass_window_max = mw.get('max', 150)

        # Get class names from config (read dynamically, not hardcoded)
        # Try multiple possible keys in the config
        self.class_names = (
            self.config.get('class_names') or
            self.config.get('labels') or
            self.config.get('classes') or
            None  # Will be set from model if not in config
        )
        self._num_classes = None  # Will be set when model is loaded
        self._model_type = 'flat'  # 'flat' or 'optimized', set in _load_model()

        # Model loaded lazily
        self._model = None

    def _load_config(self) -> Dict:
        """Load b-hive config YAML."""
        if not os.path.exists(self.bhive_config_path):
            raise FileNotFoundError(f"Config not found: {self.bhive_config_path}")
        with open(self.bhive_config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self):
        """Load PyTorch model."""
        if self._model is not None:
            return

        if not HAS_PYTORCH:
            raise ImportError("PyTorch required")

        checkpoint = torch.load(self.bhive_model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Auto-detect architecture from state dict keys
        if 'jet_encoder.0.weight' in state_dict:
            # MLP_HcZZ_MW_Deep_4class_Optimized — separate encoders per feature group
            self._model_type = 'optimized'
            num_classes = state_dict['head.4.weight'].shape[0]
            hidden_dim = state_dict['jet_encoder.0.weight'].shape[0]
            num_layers = sum(1 for k in state_dict if k.startswith('blocks.') and k.endswith('.norm.weight'))

            # Derive feature group dims from config
            n_cpf = self.config.get('n_cpf_candidates', 3)
            n_npf = self.config.get('n_npf_candidates', 4)
            n_vtx = self.config.get('n_vtx_candidates', 2)
            n_lt  = self.config.get('n_lt_candidates', 1)
            n_global = len(self.config.get('global_features', []))
            cpf_dim = len(self.config.get('cpf_candidates', []))
            npf_dim = len(self.config.get('npf_candidates', []))
            vtx_dim = len(self.config.get('vtx_features', []))
            lt_dim  = len(self.config.get('lt_candidates', []))

            # Offsets into the flat feature vector produced by prepare_features()
            self._global_end = n_global
            self._cpf_end    = n_global + n_cpf * cpf_dim
            self._vtx_end    = n_global + n_cpf * cpf_dim + n_vtx * vtx_dim
            self._npf_end    = n_global + n_cpf * cpf_dim + n_vtx * vtx_dim + n_npf * npf_dim
            self._lt_end     = self._npf_end + n_lt * lt_dim

            logging.info(f"Detected Optimized model: hidden_dim={hidden_dim}, num_layers={num_layers}, num_classes={num_classes}")
            self._model = MLP_HcZZ_MW_Deep_4class_Optimized(
                global_dim=n_global,
                cpf_total=n_cpf * cpf_dim,
                npf_total=n_npf * npf_dim,
                vtx_total=n_vtx * vtx_dim,
                lt_total=n_lt * lt_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        else:
            # MLP_HcZZ_MW_Deep — flat input MLP
            self._model_type = 'flat'
            input_dim = state_dict['input_proj.0.weight'].shape[1]
            num_classes = state_dict['head.1.weight'].shape[0]
            hidden_dim = state_dict['input_proj.0.weight'].shape[0]
            num_layers = sum(1 for k in state_dict if k.startswith('blocks.') and k.endswith('.norm.weight'))
            logging.info(f"Detected flat MLP: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_classes={num_classes}")
            self._model = MLP_HcZZ_MW_Deep(input_dim, num_classes, hidden_dim, num_layers)

        self._num_classes = num_classes

        if self.class_names is None:
            self.class_names = [f'class_{i}' for i in range(num_classes)]
            logging.warning(f"No class_names in config, using defaults: {self.class_names}")
        elif len(self.class_names) != num_classes:
            self.class_names = (list(self.class_names) + [f'class_{i}' for i in range(num_classes)])[:num_classes]

        logging.info(f"Class names: {self.class_names}")
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def _compute_jet_tags(self, B, CvB, CvL):
        """Compute jet tag categories from PNet scores, using this instance's era
        (see __init__'s `era` docstring -- real per-era boundaries if era is
        recognized, the wrong legacy fallback otherwise)."""
        pBvsC = 1.0 - CvB
        pBplusC = B + (1.0 - B) * CvL
        pBplusC = np.clip(pBplusC, 0, 1)
        pBvsC = np.clip(pBvsC, 0, 1)

        boundaries = JET_TAG_BOUNDARIES_BY_ERA.get(self.era, MVA_JET_TAG_BOUNDARIES_LEGACY_FALLBACK)
        tags = {}
        for cat, bounds in boundaries.items():
            x_min, x_max = bounds["x"]
            y_min, y_max = bounds["y"]
            mask = (pBplusC >= x_min) & (pBplusC < x_max) & (pBvsC >= y_min) & (pBvsC < y_max)
            tags[f'jet_is_{cat}'] = mask.astype(np.float32)
        return tags

    def _get_col(self, df, names, default=0.0, idx=None):
        """Get column by trying multiple names. Handles array-valued columns.

        Args:
            df: DataFrame
            names: Column name(s) to try
            default: Default value if column not found
            idx: If column contains arrays, extract this index (None = flatten/take first)
        """
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name not in df.columns:
                continue

            col_data = df[name].values
            if len(col_data) == 0:
                continue

            sample = col_data[0]

            # Check if this is an array-valued column
            is_array_col = isinstance(sample, np.ndarray)

            if is_array_col:
                # Column contains arrays - extract appropriately
                extract_idx = idx if idx is not None else 0
                result = []
                for arr in col_data:
                    if isinstance(arr, np.ndarray):
                        if len(arr) > extract_idx:
                            val = arr[extract_idx]
                        else:
                            # Candidate slot beyond this event's actual candidate
                            # count (e.g. requesting jet index 1 for a 1-jet event)
                            # -- must zero-pad like training time, NOT fall back to
                            # arr[0] (which silently duplicated the leading
                            # candidate into missing slots and corrupted the CPF
                            # block for the ~70% of events with <3 jets; found
                            # 2026-08-04 via direct feature-vector diff against
                            # the true LZ4 training-data ground truth).
                            val = default
                    else:
                        val = arr if arr is not None else default
                    result.append(float(val) if val is not None else default)
                return np.nan_to_num(np.array(result, dtype=np.float32), nan=default)
            else:
                # Scalar values - use directly (ignore idx)
                result = df[name].fillna(default).values
                # Handle any remaining object types
                if result.dtype == object:
                    result = np.array([float(x) if x is not None else default for x in result], dtype=np.float32)
                return np.nan_to_num(result.astype(np.float32), nan=default)

        return np.full(len(df), default, dtype=np.float32)

    def _get_candidate_col(self, df, specific_name, generic_name, j, default=0.0):
        """Get one candidate slot's value, correctly distinguishing two DIFFERENT
        column shapes that both exist in these parquets:

          - a per-candidate-specific column (e.g. `z1_l2_pt`, `z2_pt`) that names
            ONE particular candidate directly -- stored as a length-1 array (not a
            true scalar), so it must always be read at index 0, regardless of
            which candidate SLOT j we're filling.
          - a generic multi-candidate column (e.g. `lepton_pt`, `z_pt`) that holds
            ALL candidates in one array, indexed by slot j.

        BUG FOUND AND FIXED 2026-08-14: `prepare_features()` used to call
        `_get_col(df, [specific_name, generic_name], idx=j)` -- a single idx
        applied to BOTH names. For the specific_name branch this is wrong: it
        looks up index j into a length-1 array, which is only in-bounds for j=0
        and silently returns `default` (0.0) for every other slot. Confirmed on
        real data: `z1_l1_pt`/`z1_l2_pt`/`z2_l1_pt`/`z2_l2_pt` are each
        independent length-1 arrays (one per named lepton role), 0 null/0 zero
        across 12541 real events -- genuinely populated, just misindexed. This
        zeroed leptons 2-4's pt/eta/phi and the entire second Z boson's
        (pt/eta/phi) on EVERY event (12 of 101 total input features), which is
        certainly sufficient to explain the near-total-collapse symptom this fix
        was written to resolve (true-Signal mean P(Signal) was 0.0011 pre-fix vs.
        the model's own native ground truth of 0.5975 for the same checkpoint).
        """
        if specific_name in df.columns:
            return self._get_col(df, [specific_name], default=default, idx=0)
        return self._get_col(df, [generic_name], default=default, idx=j)

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features based on config."""
        n = len(df)

        # Get feature lists from config
        global_feats = self.config.get('global_features', [])
        cpf_feats = self.config.get('cpf_candidates', [])
        vtx_feats = self.config.get('vtx_features', [])
        npf_feats = self.config.get('npf_candidates', [])
        lt_feats = self.config.get('lt_candidates', [])

        n_cpf = self.config.get('n_cpf_candidates', 3)
        n_vtx = self.config.get('n_vtx_candidates', 2)
        n_npf = self.config.get('n_npf_candidates', 4)
        n_lt = self.config.get('n_lt_candidates', 1)

        all_features = []

        # Column name mappings
        col_map = {
            'n_jet': ['jet_multiplicity', 'n_jet'],
            'Jet_HT': ['jet_ht', 'Jet_HT'],
            'n_lepton': ['n_lepton'],
            'm4l': ['zz_mass_inclusive', 'm4l'],
            'eta_4l': ['zz_eta_inclusive', 'eta_4l'],
            'pT_4l': ['zz_pt_inclusive', 'pT_4l'],
            'phi_4l': ['zz_phi_inclusive', 'phi_4l'],
            'deltaR_ZZ': ['deltaR_ZZ'],
            'z_pt': ['z1_pt', 'z2_pt'],
            'z_eta': ['z1_eta', 'z2_eta'],
            'z_phi': ['z1_phi', 'z2_phi'],
            'z_mass': ['z1_mass', 'z2_mass'],
        }

        # --- Global features ---
        for feat in global_feats:
            if feat == 'in_mass_window':
                m4l = self._get_col(df, ['zz_mass_inclusive', 'm4l'])
                all_features.append(((m4l >= self.mass_window_min) & (m4l < self.mass_window_max)).astype(np.float32))
            elif feat == 'n_ctagged_jets':
                # Count jets in C categories
                count = np.zeros(n, dtype=np.float32)
                for cat in ['C0', 'C1', 'C2', 'C3', 'C4']:
                    col = f'leadjet_is_{cat}'
                    if col in df.columns:
                        count += df[col].fillna(0).values
                all_features.append(count)
            elif feat == 'n_btagged_jets':
                count = np.zeros(n, dtype=np.float32)
                for cat in ['B0', 'B1', 'B2', 'B3', 'B4']:
                    col = f'leadjet_is_{cat}'
                    if col in df.columns:
                        count += df[col].fillna(0).values
                all_features.append(count)
            elif feat == 'n_lepton':
                all_features.append(np.full(n, 4, dtype=np.float32))
            else:
                names = col_map.get(feat, [feat])
                all_features.append(self._get_col(df, names))

        # --- CPF (c-Jets) features ---
        # Precompute jet tags per jet index (loop order below is feature-major, not jet-major)
        jet_tags_by_j = []
        for j in range(n_cpf):
            B = self._get_col(df, ['cjets_btagPNetB', 'jet_btagPNetB'], idx=j)
            CvB = self._get_col(df, ['cjets_btagPNetCvB', 'jet_btagPNetCvB'], idx=j)
            CvL = self._get_col(df, ['cjets_btagPNetCvL', 'jet_btagPNetCvL'], idx=j)
            jet_tags_by_j.append(self._compute_jet_tags(B, CvB, CvL))

        # NOTE: candidate-major (candidate outer, feat inner) -- verified 2026-08-04
        # empirically against the real model: base_model.py::create_feature_shapes()
        # sets input_dims[1] = (n_cpf_candidates, cpf_dim), i.e. reshape(batch, n_cpf,
        # cpf_dim), which requires the flat vector to already be candidate-major.
        # Direct A/B test (same model, same real events): candidate-major AUC=0.957
        # (matches the model's true ~0.95-0.97 capability) vs feature-major AUC=0.559
        # (near-random). The 2026-08-03 "feature-major fix" was wrong; reverted here.
        for j in range(n_cpf):
            for feat in cpf_feats:
                if feat.startswith('cjet_is_') or feat.startswith('jet_is_'):
                    cat = feat.replace('cjet_is_', '').replace('jet_is_', '')
                    if f'jet_is_{cat}' in jet_tags_by_j[j]:
                        all_features.append(jet_tags_by_j[j][f'jet_is_{cat}'])
                    else:
                        all_features.append(np.zeros(n, dtype=np.float32))
                elif feat in ['cjets_pt', 'jet_pt']:
                    all_features.append(self._get_col(df, ['cjets_pt', 'jet_pt'], idx=j))
                elif feat in ['cjets_eta', 'jet_eta']:
                    all_features.append(self._get_col(df, ['cjets_eta', 'jet_eta'], idx=j))
                elif feat in ['cjets_phi', 'jet_phi']:
                    all_features.append(self._get_col(df, ['cjets_phi', 'jet_phi'], idx=j))
                elif feat in ['cjets_mass', 'jet_mass']:
                    all_features.append(self._get_col(df, ['cjets_mass', 'jet_mass'], idx=j))
                else:
                    # Generic array column with index
                    all_features.append(self._get_col(df, [feat], idx=j))

        # --- NPF (Leptons) features ---
        # Leptons can be stored as z1_l1_*, z1_l2_*, z2_l1_*, z2_l2_* or as lepton_* arrays
        # NOTE: NPF must come BEFORE VTX to match b-hive_ttcc/tasks/dataset.py's training-time
        # concatenation order (cpf_candidates + npf_candidates + vtx_features). Candidate-major
        # (lepton outer, feat inner) -- see cpf block comment above for the empirical evidence.
        lepton_prefixes = ['z1_l1', 'z1_l2', 'z2_l1', 'z2_l2']
        for j in range(n_npf):
            prefix = lepton_prefixes[j] if j < len(lepton_prefixes) else f'lepton_{j}'
            for feat in npf_feats:
                if feat == 'lepton_pt':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_pt', 'lepton_pt', j))
                elif feat == 'lepton_eta':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_eta', 'lepton_eta', j))
                elif feat == 'lepton_phi':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_phi', 'lepton_phi', j))
                elif feat == 'lepton_mass':
                    # Try array column first, then use default 0
                    all_features.append(self._get_col(df, ['lepton_mass'], default=0.0, idx=j))
                elif feat == 'lepton_charge':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_charge', 'lepton_charge', j))
                elif feat == 'lepton_pfRelIso03_all':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_pfRelIso03_all', 'lepton_pfRelIso03_all', j))
                elif feat == 'lepton_sip3d':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_sip3d', 'lepton_sip3d', j))
                elif feat == 'lepton_is_tight':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_is_tight', 'lepton_is_tight', j))
                elif feat == 'lepton_mvaHZZIso':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_mvaHZZIso', 'lepton_mvaHZZIso', j))
                elif feat == 'lepton_isPFcand':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_isPFcand', 'lepton_isPFcand', j))
                elif feat == 'lepton_highPtId':
                    all_features.append(self._get_candidate_col(df, f'{prefix}_highPtId', 'lepton_highPtId', j))
                else:
                    all_features.append(self._get_candidate_col(df, f'{feat}_{j}', feat, j))

        # --- VTX (Z bosons) features ---
        # Candidate-major (Z-candidate outer, feat inner) -- see cpf block comment above.
        for j in range(n_vtx):
            for feat in vtx_feats:
                if feat == 'z_pt':
                    all_features.append(self._get_candidate_col(df, f'z{j+1}_pt', 'z_pt', j))
                elif feat == 'z_eta':
                    all_features.append(self._get_candidate_col(df, f'z{j+1}_eta', 'z_eta', j))
                elif feat == 'z_phi':
                    all_features.append(self._get_candidate_col(df, f'z{j+1}_phi', 'z_phi', j))
                elif feat == 'z_mass':
                    all_features.append(self._get_candidate_col(df, f'z{j+1}_mass', 'z_mass', j))
                else:
                    all_features.append(self._get_candidate_col(df, f'{feat}_{j}', feat, j))

        # --- LT (Higgs) features ---
        for feat in lt_feats:
            names = col_map.get(feat, [feat])
            all_features.append(self._get_col(df, names))

        return np.column_stack(all_features)

    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference."""
        self._load_model()

        with torch.no_grad():
            if self._model_type == 'optimized':
                # Split flat feature array into the 5 groups the Optimized model expects.
                # prepare_features() builds jet-major: [jet0_all_feats, jet1_all_feats, ...]
                # b-hive training used feature-major: swapaxes(1,2) → [feat0_all_jets, feat1_all_jets, ...]
                # So we must transpose each candidate block before passing to the model.
                n_cpf = self.config.get('n_cpf_candidates', 3)
                n_npf = self.config.get('n_npf_candidates', 4)
                n_vtx = self.config.get('n_vtx_candidates', 2)
                n_lt  = self.config.get('n_lt_candidates', 1)
                cpf_dim = len(self.config.get('cpf_candidates', []))
                npf_dim = len(self.config.get('npf_candidates', []))
                vtx_dim = len(self.config.get('vtx_features', []))
                lt_dim  = len(self.config.get('lt_candidates', []))

                g   = torch.from_numpy(features[:, :self._global_end]).float()
                # jet-major [B, n*d] → feature-major [B, n*d] via reshape+permute
                cp  = torch.from_numpy(features[:, self._global_end:self._cpf_end]).float()
                cp  = cp.reshape(-1, n_cpf, cpf_dim).permute(0, 2, 1).reshape(-1, n_cpf * cpf_dim)
                vt  = torch.from_numpy(features[:, self._cpf_end:self._vtx_end]).float()
                vt  = vt.reshape(-1, n_vtx, vtx_dim).permute(0, 2, 1).reshape(-1, n_vtx * vtx_dim)
                np_ = torch.from_numpy(features[:, self._vtx_end:self._npf_end]).float()
                np_ = np_.reshape(-1, n_npf, npf_dim).permute(0, 2, 1).reshape(-1, n_npf * npf_dim)
                lt  = torch.from_numpy(features[:, self._npf_end:self._lt_end]).float()
                # n_lt=1 → no swapaxes needed in b-hive either
                logits = self._model(g, cp, np_, vt, lt)
            else:
                x = torch.from_numpy(features).float()
                logits = self._model(x)
            scores = torch.softmax(logits, dim=1).numpy()

        # Find signal index (try 'Signal', then look for any class containing 'signal')
        signal_idx = None
        for i, name in enumerate(self.class_names):
            if name == 'Signal' or 'signal' in name.lower():
                signal_idx = i
                break
        if signal_idx is None:
            signal_idx = 0  # Default to first class if no signal class found
            logging.warning(f"No 'Signal' class found in {self.class_names}, using class 0 for signal_score")

        return {
            'scores': scores,
            'signal_score': scores[:, signal_idx],
            'class_prediction': np.argmax(scores, axis=1),
        }

    def process_parquet(self, input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Process single parquet file."""
        df = pd.read_parquet(input_file)
        n = len(df)

        if n == 0:
            return df

        # Load model first to get class names
        self._load_model()

        # Get m4l
        m4l = self._get_col(df, ['zz_mass_inclusive', 'm4l'])

        if self.apply_mass_window:
            mask = (m4l >= self.mass_window_min) & (m4l < self.mass_window_max)
            n_in = mask.sum()

            df['mva_signal_score'] = -1.0
            df['mva_class_prediction'] = -1
            for cls in self.class_names:
                df[f'mva_score_{cls}'] = -1.0

            if n_in > 0:
                features = self.prepare_features(df[mask])
                result = self.predict(features)

                df.loc[mask, 'mva_signal_score'] = result['signal_score']
                df.loc[mask, 'mva_class_prediction'] = result['class_prediction']
                for i, cls in enumerate(self.class_names):
                    df.loc[mask, f'mva_score_{cls}'] = result['scores'][:, i]

                logging.info(f"Processed {n_in}/{n} events in mass window")
        else:
            features = self.prepare_features(df)
            result = self.predict(features)
            df['mva_signal_score'] = result['signal_score']
            df['mva_class_prediction'] = result['class_prediction']
            for i, cls in enumerate(self.class_names):
                df[f'mva_score_{cls}'] = result['scores'][:, i]

        output_path = output_file or input_file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logging.info(f"Saved: {output_path}")

        return df

    def process_parquets(self, input_dir: str, output_dir: Optional[str] = None, pattern: str = "**/*.parquet"):
        """Process all parquets in directory."""
        files = glob.glob(os.path.join(input_dir, pattern), recursive=True)
        logging.info(f"Found {len(files)} parquet files")

        for f in files:
            out = os.path.join(output_dir, os.path.relpath(f, input_dir)) if output_dir else f
            try:
                self.process_parquet(f, out)
            except Exception as e:
                logging.error(f"Error processing {f}: {e}")


def run_mva_inference(
    workflow: str,
    year: str,
    output_dir: str,
    bhive_config_path: str,
    bhive_model_path: str,
    apply_mass_window: bool = True,
):
    """Convenience function to run MVA inference."""
    input_dir = os.path.join(output_dir, workflow, year)
    if not os.path.exists(input_dir):
        logging.error(f"Not found: {input_dir}")
        return

    processor = MVAPostProcessor(bhive_config_path, bhive_model_path, apply_mass_window)
    processor.process_parquets(input_dir)
    logging.info(f"MVA inference complete for {workflow}/{year}")
