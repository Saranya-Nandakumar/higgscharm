"""
Flexible PyTorch MVA Inference module for H+c -> ZZ -> 4l analysis.

This module can load ANY PyTorch model architecture as long as it uses the same
feature configuration. Supports:
  - TorchScript models (.pt saved with torch.jit.save)
  - Full model saves (.pt saved with torch.save(model, path))
  - State dict with externally provided model class

Usage:
    from analysis.utils.mva_inference_pytorch import FlexibleMVAInference

    # Load TorchScript or full model (recommended)
    mva = FlexibleMVAInference(model_path='/path/to/model.pt')

    # Compute scores
    scores = mva.predict(events, objects)
"""

import os
import sys
import numpy as np
import awkward as ak
import yaml

# Increase recursion limit for deeply nested awkward arrays
_original_recursion_limit = sys.getrecursionlimit()
if _original_recursion_limit < 50000:
    sys.setrecursionlimit(50000)

try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("Warning: PyTorch not installed. PyTorch inference disabled.")


class FlexibleMVAInference:
    """
    Flexible MVA inference that works with any PyTorch model architecture.

    The model must accept inputs in one of these formats:
      1. Tuple: (global_features, cpf_features, npf_features, vtx_features, lt_features)
      2. Dict: {'global': ..., 'cpf': ..., 'npf': ..., 'vtx': ..., 'lt': ...}
      3. Single tensor: concatenated features

    Feature configuration can be customized via config file or parameters.
    """

    # Default feature configurations (can be overridden)
    DEFAULT_GLOBAL_FEATURES = [
        'n_jet',
        'Jet_HT',
        'n_lepton',
        'm4l',
        'eta_4l',
        'deltaR_ZZ',
        'n_ctagged_jets',
        'n_btagged_jets',
        'in_mass_window',
    ]

    DEFAULT_CPF_FEATURES = [
        'jet_pt',
        'jet_eta',
        'jet_phi',
        'jet_mass',
        'jet_is_C0',
        'jet_is_C1',
        'jet_is_C2',
        'jet_is_C3',
        'jet_is_C4',
        'jet_is_B0',
        'jet_is_B1',
        'jet_is_B2',
        'jet_is_B3',
        'jet_is_B4',
        'jet_is_L0',
    ]

    DEFAULT_VTX_FEATURES = [
        'z_pt',
        'z_eta',
        'z_phi',
        'z_mass',
    ]

    DEFAULT_NPF_FEATURES = [
        'lepton_pt',
        'lepton_eta',
        'lepton_phi',
        'lepton_mass',
        'lepton_charge',
        'lepton_pfRelIso03_all',
        'lepton_sip3d',
        'lepton_is_tight',
        'lepton_mvaHZZIso',
        'lepton_isPFcand',
        'lepton_highPtId',
    ]

    DEFAULT_LT_FEATURES = [
        'pT_4l',
        'eta_4l',
        'phi_4l',
        'm4l',
    ]

    # Default class names
    DEFAULT_CLASS_NAMES_5 = ['qqgg_toZZ', 'ggH', 'Signal', 'HPlusB', 'Other']
    DEFAULT_CLASS_NAMES_6 = ['Signal', 'HPlusB', 'ggZZ', 'qqZZ', 'Other', 'ggH']

    # Jet tagging boundaries
    MVA_JET_TAG_BOUNDARIES = {
        "C4": {"x": (0.7339, 1.0000), "y": (0.0000, 0.0382)},
        "C3": {"x": (0.7339, 1.0000), "y": (0.0382, 0.1851)},
        "C2": {"x": (0.7339, 1.0000), "y": (0.1851, 0.2688)},
        "C1": {"x": (0.4, 0.7339), "y": (0.0000, 0.9000)},
        "C0": {"x": (0.15, 0.4), "y": (0.0000, 0.9000)},
        "L0": {"x": (0.0000, 0.15), "y": (0.0000, 0.9000)},
        "B0": {"x": (0.7339, 1.0000), "y": (0.2688, 0.4057)},
        "B1": {"x": (0.7339, 1.0000), "y": (0.4057, 0.4068)},
        "B2": {"x": (0.7339, 1.0000), "y": (0.4068, 0.6788)},
        "B3": {"x": (0.7339, 1.0000), "y": (0.6788, 0.8406)},
        "B4": {"x": (0.7339, 1.0000), "y": (0.8406, 1.0000)},
    }
    CATEGORY_ORDER = ["C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4", "L0"]

    def __init__(
        self,
        model_path,
        config_path=None,
        n_cpf=3,
        n_vtx=2,
        n_npf=4,
        n_lt=1,
        num_classes=5,
        class_names=None,
        input_format='tuple',
        device='cpu',
    ):
        """
        Initialize flexible MVA inference.

        Args:
            model_path: Path to PyTorch model file (.pt)
                - TorchScript model (torch.jit.save)
                - Full model (torch.save(model, path))
            config_path: Optional path to YAML config file with feature definitions
            n_cpf: Number of jet candidates (default 3)
            n_vtx: Number of Z boson candidates (default 2)
            n_npf: Number of lepton candidates (default 4)
            n_lt: Number of Higgs candidates (default 1)
            num_classes: Number of output classes (default 5)
            class_names: List of class names (default based on num_classes)
            input_format: How to pass inputs to model:
                - 'tuple': (global, cpf, npf, vtx, lt)
                - 'dict': {'global': ..., 'cpf': ..., ...}
                - 'concat': single concatenated tensor
            device: Device to run inference on ('cpu' or 'cuda')
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch is required for this module")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.n_cpf = n_cpf
        self.n_vtx = n_vtx
        self.n_npf = n_npf
        self.n_lt = n_lt
        self.num_classes = num_classes
        self.input_format = input_format
        self.device = torch.device(device)

        # Load config if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._use_default_config()

        # Set class names
        if class_names:
            self.class_names = class_names
        elif num_classes == 6:
            self.class_names = self.DEFAULT_CLASS_NAMES_6
        else:
            self.class_names = self.DEFAULT_CLASS_NAMES_5

        # Load model
        self._load_model()

    def _use_default_config(self):
        """Use default feature configuration."""
        self.global_features = self.DEFAULT_GLOBAL_FEATURES.copy()
        self.cpf_features = self.DEFAULT_CPF_FEATURES.copy()
        self.vtx_features = self.DEFAULT_VTX_FEATURES.copy()
        self.npf_features = self.DEFAULT_NPF_FEATURES.copy()
        self.lt_features = self.DEFAULT_LT_FEATURES.copy()

    def _load_config(self, config_path):
        """Load feature configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract feature lists from config
        # Supports both direct feature lists and nested structures
        if 'features' in config:
            features = config['features']
        else:
            features = config

        self.global_features = features.get('global_features', self.DEFAULT_GLOBAL_FEATURES)
        self.cpf_features = features.get('cpf_features', self.DEFAULT_CPF_FEATURES)
        self.vtx_features = features.get('vtx_features', self.DEFAULT_VTX_FEATURES)
        self.npf_features = features.get('npf_features', self.DEFAULT_NPF_FEATURES)
        self.lt_features = features.get('lt_features', self.DEFAULT_LT_FEATURES)

        # Also check for candidate counts in config
        if 'n_cpf' in config:
            self.n_cpf = config['n_cpf']
        if 'n_vtx' in config:
            self.n_vtx = config['n_vtx']
        if 'n_npf' in config:
            self.n_npf = config['n_npf']
        if 'n_lt' in config:
            self.n_lt = config['n_lt']

        print(f"Loaded config from {config_path}")
        print(f"  Global features: {len(self.global_features)}")
        print(f"  CPF features: {len(self.cpf_features)} x {self.n_cpf}")
        print(f"  VTX features: {len(self.vtx_features)} x {self.n_vtx}")
        print(f"  NPF features: {len(self.npf_features)} x {self.n_npf}")
        print(f"  LT features: {len(self.lt_features)} x {self.n_lt}")

    def _load_model(self):
        """Load PyTorch model (TorchScript or regular)."""
        try:
            # Try loading as TorchScript first
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.is_torchscript = True
            print(f"Loaded TorchScript model: {self.model_path}")
        except Exception:
            # Fall back to regular torch.load
            try:
                loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)

                # Check if it's a full model or just state_dict
                if isinstance(loaded, torch.nn.Module):
                    self.model = loaded
                    self.is_torchscript = False
                    print(f"Loaded full PyTorch model: {self.model_path}")
                elif isinstance(loaded, dict) and 'model' in loaded:
                    # Checkpoint format with full model
                    self.model = loaded['model']
                    self.is_torchscript = False
                    print(f"Loaded model from checkpoint: {self.model_path}")
                else:
                    raise ValueError(
                        "Model file contains only state_dict. "
                        "Please save the full model with torch.save(model, path) "
                        "or use TorchScript with torch.jit.save(model, path)"
                    )
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")

        self.model.to(self.device)
        self.model.eval()

        # Print model info
        total_input_dim = (
            len(self.global_features) +
            self.n_cpf * len(self.cpf_features) +
            self.n_vtx * len(self.vtx_features) +
            self.n_npf * len(self.npf_features) +
            self.n_lt * len(self.lt_features)
        )
        print(f"  Total input dimension: {total_input_dim}")
        print(f"  Output classes: {self.num_classes} {self.class_names}")
        print(f"  Input format: {self.input_format}")

    def compute_jet_tags(self, B, CvB, CvL):
        """
        Compute jet tagging category flags from PNet scores.

        Args:
            B: PNet B-tag score (n_events, n_jets)
            CvB: PNet CvB score (n_events, n_jets)
            CvL: PNet CvL score (n_events, n_jets)

        Returns:
            Dict of jet tag arrays
        """
        pBvsC = 1.0 - CvB
        pBplusC = B + (1.0 - B) * CvL
        pBplusC = np.clip(pBplusC, 0, 1)
        pBvsC = np.clip(pBvsC, 0, 1)

        tags = {}
        for cat_name in self.CATEGORY_ORDER:
            bounds = self.MVA_JET_TAG_BOUNDARIES[cat_name]
            x_min, x_max = bounds["x"]
            y_min, y_max = bounds["y"]

            mask = (
                (pBplusC >= x_min) & (pBplusC < x_max) &
                (pBvsC >= y_min) & (pBvsC < y_max)
            )
            tags[f'jet_is_{cat_name}'] = mask.astype(np.float32)

        return tags

    def _to_1d_numpy(self, val, n_events):
        """Convert awkward array or scalar to 1D numpy array."""
        try:
            if isinstance(val, (int, float)):
                return np.full(n_events, val, dtype=np.float32)

            if isinstance(val, np.ndarray):
                arr = val.copy()
            else:
                try:
                    arr = np.asarray(val)
                except (ValueError, TypeError, RecursionError):
                    try:
                        arr = np.array(ak.to_list(ak.fill_none(val, 0)), dtype=np.float32)
                    except (RecursionError, Exception):
                        return np.zeros(n_events, dtype=np.float32)

            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            while arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.squeeze(axis=-1)

            if arr.ndim > 1:
                arr = arr.reshape(-1)

            if len(arr) != n_events:
                if len(arr) > n_events:
                    arr = arr[:n_events]
                else:
                    arr = np.pad(arr, (0, n_events - len(arr)), constant_values=0)

            return arr.astype(np.float32)

        except (RecursionError, Exception):
            return np.zeros(n_events, dtype=np.float32)

    def _to_2d_numpy(self, val, n_events, n_items):
        """Convert awkward array to 2D numpy array."""
        try:
            if isinstance(val, np.ndarray):
                arr = val
            else:
                try:
                    filled = ak.fill_none(val, 0)
                    arr = ak.to_numpy(filled)
                except (ValueError, TypeError, RecursionError):
                    try:
                        filled = ak.fill_none(ak.pad_none(val, n_items, clip=True), 0)
                        arr = np.array(ak.to_list(filled), dtype=np.float32)
                    except (RecursionError, Exception):
                        return np.zeros((n_events, n_items), dtype=np.float32)

            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            if arr.ndim == 1:
                if len(arr) == n_events * n_items:
                    arr = arr.reshape(n_events, n_items)
                else:
                    arr = arr.reshape(n_events, -1)
            elif arr.ndim > 2:
                while arr.ndim > 2:
                    arr = arr.squeeze(axis=-1)

            if arr.shape != (n_events, n_items):
                if arr.size >= n_events * n_items:
                    arr = arr.flatten()[:n_events * n_items].reshape(n_events, n_items)
                else:
                    new_arr = np.zeros((n_events, n_items), dtype=np.float32)
                    flat = arr.flatten()
                    new_arr.flat[:len(flat)] = flat
                    arr = new_arr

            return arr.astype(np.float32)

        except (RecursionError, Exception):
            return np.zeros((n_events, n_items), dtype=np.float32)

    def prepare_features(self, events, objects):
        """
        Prepare input features from events and objects.

        Args:
            events: NanoAOD events
            objects: Selected objects dictionary

        Returns:
            Tuple of (global_features, cpf_features, npf_features, vtx_features, lt_features)
        """
        n_events = len(events)

        best_zz = objects.get('best_zzcandidate', None)
        jets = objects.get('jets', None)
        leptons = objects.get('leptons', None)

        # =================================================================
        # Compute ZZ kinematics
        # =================================================================
        if best_zz is not None:
            try:
                def safe_to_numpy(arr, default=0):
                    try:
                        filled = ak.fill_none(arr, default)
                        result = np.asarray(filled)
                        if result.ndim > 1:
                            result = result.flatten()
                        return result
                    except (RecursionError, Exception):
                        try:
                            filled = ak.fill_none(arr, default)
                            result = np.array(ak.to_list(filled), dtype=np.float32)
                            if result.ndim > 1:
                                result = result.flatten()
                            return result
                        except (RecursionError, Exception):
                            return np.zeros(n_events, dtype=np.float32)

                # Extract lepton kinematics
                z1_l1_pt = safe_to_numpy(best_zz.z1.l1.pt)
                z1_l1_eta = safe_to_numpy(best_zz.z1.l1.eta)
                z1_l1_phi = safe_to_numpy(best_zz.z1.l1.phi)
                z1_l1_mass = safe_to_numpy(best_zz.z1.l1.mass)

                z1_l2_pt = safe_to_numpy(best_zz.z1.l2.pt)
                z1_l2_eta = safe_to_numpy(best_zz.z1.l2.eta)
                z1_l2_phi = safe_to_numpy(best_zz.z1.l2.phi)
                z1_l2_mass = safe_to_numpy(best_zz.z1.l2.mass)

                z2_l1_pt = safe_to_numpy(best_zz.z2.l1.pt)
                z2_l1_eta = safe_to_numpy(best_zz.z2.l1.eta)
                z2_l1_phi = safe_to_numpy(best_zz.z2.l1.phi)
                z2_l1_mass = safe_to_numpy(best_zz.z2.l1.mass)

                z2_l2_pt = safe_to_numpy(best_zz.z2.l2.pt)
                z2_l2_eta = safe_to_numpy(best_zz.z2.l2.eta)
                z2_l2_phi = safe_to_numpy(best_zz.z2.l2.phi)
                z2_l2_mass = safe_to_numpy(best_zz.z2.l2.mass)

                def compute_4vec(pt, eta, phi, mass):
                    px = pt * np.cos(phi)
                    py = pt * np.sin(phi)
                    pz = pt * np.sinh(eta)
                    E = np.sqrt(mass**2 + pt**2 * np.cosh(eta)**2)
                    return px, py, pz, E

                z1_l1_px, z1_l1_py, z1_l1_pz, z1_l1_E = compute_4vec(z1_l1_pt, z1_l1_eta, z1_l1_phi, z1_l1_mass)
                z1_l2_px, z1_l2_py, z1_l2_pz, z1_l2_E = compute_4vec(z1_l2_pt, z1_l2_eta, z1_l2_phi, z1_l2_mass)
                z2_l1_px, z2_l1_py, z2_l1_pz, z2_l1_E = compute_4vec(z2_l1_pt, z2_l1_eta, z2_l1_phi, z2_l1_mass)
                z2_l2_px, z2_l2_py, z2_l2_pz, z2_l2_E = compute_4vec(z2_l2_pt, z2_l2_eta, z2_l2_phi, z2_l2_mass)

                z1_px = z1_l1_px + z1_l2_px
                z1_py = z1_l1_py + z1_l2_py
                z1_pz = z1_l1_pz + z1_l2_pz
                z1_E = z1_l1_E + z1_l2_E

                z2_px = z2_l1_px + z2_l2_px
                z2_py = z2_l1_py + z2_l2_py
                z2_pz = z2_l1_pz + z2_l2_pz
                z2_E = z2_l1_E + z2_l2_E

                z1_mass = np.sqrt(np.maximum(z1_E**2 - z1_px**2 - z1_py**2 - z1_pz**2, 0))
                z2_mass = np.sqrt(np.maximum(z2_E**2 - z2_px**2 - z2_py**2 - z2_pz**2, 0))
                z1_pt = np.sqrt(z1_px**2 + z1_py**2)
                z2_pt = np.sqrt(z2_px**2 + z2_py**2)
                z1_eta = np.arctanh(np.clip(z1_pz / np.sqrt(z1_px**2 + z1_py**2 + z1_pz**2 + 1e-10), -0.9999, 0.9999))
                z2_eta = np.arctanh(np.clip(z2_pz / np.sqrt(z2_px**2 + z2_py**2 + z2_pz**2 + 1e-10), -0.9999, 0.9999))
                z1_phi = np.arctan2(z1_py, z1_px)
                z2_phi = np.arctan2(z2_py, z2_px)

                zz_px = z1_px + z2_px
                zz_py = z1_py + z2_py
                zz_pz = z1_pz + z2_pz
                zz_E = z1_E + z2_E

                m4l_val = np.sqrt(np.maximum(zz_E**2 - zz_px**2 - zz_py**2 - zz_pz**2, 0))
                pt_4l_val = np.sqrt(zz_px**2 + zz_py**2)
                zz_p = np.sqrt(zz_px**2 + zz_py**2 + zz_pz**2 + 1e-10)
                eta_4l_val = np.arctanh(np.clip(zz_pz / zz_p, -0.9999, 0.9999))
                phi_4l_val = np.arctan2(zz_py, zz_px)

                deta = z1_eta - z2_eta
                dphi = z1_phi - z2_phi
                dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
                dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
                deltaR_ZZ = np.sqrt(deta**2 + dphi**2)

                # Clean NaN
                for arr in [m4l_val, pt_4l_val, eta_4l_val, phi_4l_val, deltaR_ZZ,
                           z1_pt, z2_pt, z1_eta, z2_eta, z1_phi, z2_phi, z1_mass, z2_mass]:
                    arr = np.nan_to_num(arr, nan=0.0)

            except Exception as e:
                print(f"Warning: Failed to extract ZZ kinematics: {e}")
                m4l_val = pt_4l_val = eta_4l_val = phi_4l_val = deltaR_ZZ = np.zeros(n_events, dtype=np.float32)
                z1_pt = z2_pt = z1_eta = z2_eta = z1_phi = z2_phi = z1_mass = z2_mass = np.zeros(n_events, dtype=np.float32)
        else:
            m4l_val = pt_4l_val = eta_4l_val = phi_4l_val = deltaR_ZZ = np.zeros(n_events, dtype=np.float32)
            z1_pt = z2_pt = z1_eta = z2_eta = z1_phi = z2_phi = z1_mass = z2_mass = np.zeros(n_events, dtype=np.float32)

        # =================================================================
        # GLOBAL FEATURES
        # =================================================================
        global_features = np.zeros((n_events, len(self.global_features)), dtype=np.float32)

        # Map feature names to values
        feature_values = {
            'n_jet': self._to_1d_numpy(ak.num(jets), n_events) if jets is not None else np.zeros(n_events),
            'Jet_HT': self._to_1d_numpy(ak.sum(jets.pt, axis=1), n_events) if jets is not None and ak.count(jets) > 0 else np.zeros(n_events),
            'n_lepton': self._to_1d_numpy(ak.num(leptons), n_events) if leptons is not None else np.zeros(n_events),
            'm4l': m4l_val,
            'eta_4l': eta_4l_val,
            'deltaR_ZZ': deltaR_ZZ,
            'in_mass_window': ((m4l_val >= 100) & (m4l_val < 150)).astype(np.float32),
            'pT_4l': pt_4l_val,
            'phi_4l': phi_4l_val,
        }

        # n_ctagged and n_btagged will be computed after jet tags
        n_ctagged = np.zeros(n_events, dtype=np.float32)
        n_btagged = np.zeros(n_events, dtype=np.float32)

        # =================================================================
        # CPF FEATURES (Jets)
        # =================================================================
        cpf_features = np.zeros((n_events, self.n_cpf, len(self.cpf_features)), dtype=np.float32)
        jet_tags = {}

        if jets is not None and ak.count(jets) > 0:
            jets_padded = ak.pad_none(jets, self.n_cpf, clip=True)

            # Get PNet scores for jet tagging
            B = self._to_2d_numpy(jets_padded.btagPNetB, n_events, self.n_cpf)
            CvB = self._to_2d_numpy(jets_padded.btagPNetCvB, n_events, self.n_cpf)
            CvL = self._to_2d_numpy(jets_padded.btagPNetCvL, n_events, self.n_cpf)

            jet_tags = self.compute_jet_tags(B, CvB, CvL)

            # Count tagged jets
            for cat_name in ['C0', 'C1', 'C2', 'C3', 'C4']:
                n_ctagged += jet_tags[f'jet_is_{cat_name}'].sum(axis=1)
            for cat_name in ['B0', 'B1', 'B2', 'B3', 'B4']:
                n_btagged += jet_tags[f'jet_is_{cat_name}'].sum(axis=1)

            # Fill CPF features based on config
            jet_feature_map = {
                'jet_pt': self._to_2d_numpy(jets_padded.pt, n_events, self.n_cpf),
                'jet_eta': self._to_2d_numpy(jets_padded.eta, n_events, self.n_cpf),
                'jet_phi': self._to_2d_numpy(jets_padded.phi, n_events, self.n_cpf),
                'jet_mass': self._to_2d_numpy(jets_padded.mass, n_events, self.n_cpf),
            }
            # Add jet tag features
            for cat_name in self.CATEGORY_ORDER:
                jet_feature_map[f'jet_is_{cat_name}'] = jet_tags[f'jet_is_{cat_name}']

            for i, feat_name in enumerate(self.cpf_features):
                if feat_name in jet_feature_map:
                    cpf_features[:, :, i] = jet_feature_map[feat_name]

        # Update feature values with computed counts
        feature_values['n_ctagged_jets'] = n_ctagged
        feature_values['n_btagged_jets'] = n_btagged

        # Fill global features based on config
        for i, feat_name in enumerate(self.global_features):
            if feat_name in feature_values:
                global_features[:, i] = feature_values[feat_name]

        # =================================================================
        # VTX FEATURES (Z bosons)
        # =================================================================
        vtx_features = np.zeros((n_events, self.n_vtx, len(self.vtx_features)), dtype=np.float32)

        z_feature_map = {
            'z_pt': [z1_pt, z2_pt],
            'z_eta': [z1_eta, z2_eta],
            'z_phi': [z1_phi, z2_phi],
            'z_mass': [z1_mass, z2_mass],
        }

        for i, feat_name in enumerate(self.vtx_features):
            if feat_name in z_feature_map:
                for j in range(min(self.n_vtx, 2)):
                    vtx_features[:, j, i] = z_feature_map[feat_name][j]

        # =================================================================
        # NPF FEATURES (Leptons)
        # =================================================================
        npf_features = np.zeros((n_events, self.n_npf, len(self.npf_features)), dtype=np.float32)

        if leptons is not None and ak.count(leptons) > 0:
            leptons_padded = ak.pad_none(leptons, self.n_npf, clip=True)

            lepton_feature_map = {
                'lepton_pt': self._to_2d_numpy(leptons_padded.pt, n_events, self.n_npf),
                'lepton_eta': self._to_2d_numpy(leptons_padded.eta, n_events, self.n_npf),
                'lepton_phi': self._to_2d_numpy(leptons_padded.phi, n_events, self.n_npf),
                'lepton_mass': self._to_2d_numpy(leptons_padded.mass, n_events, self.n_npf),
                'lepton_charge': self._to_2d_numpy(leptons_padded.charge, n_events, self.n_npf),
                'lepton_pfRelIso03_all': self._to_2d_numpy(leptons_padded.pfRelIso03_all, n_events, self.n_npf),
                'lepton_sip3d': self._to_2d_numpy(leptons_padded.sip3d, n_events, self.n_npf),
            }

            # Optional fields
            for field, attr in [
                ('lepton_is_tight', 'isTight'),
                ('lepton_mvaHZZIso', 'mvaHZZIso'),
                ('lepton_isPFcand', 'isPFcand'),
                ('lepton_highPtId', 'highPtId'),
            ]:
                try:
                    lepton_feature_map[field] = self._to_2d_numpy(
                        ak.fill_none(getattr(leptons_padded, attr), 0), n_events, self.n_npf
                    )
                except:
                    lepton_feature_map[field] = np.zeros((n_events, self.n_npf), dtype=np.float32)

            for i, feat_name in enumerate(self.npf_features):
                if feat_name in lepton_feature_map:
                    npf_features[:, :, i] = lepton_feature_map[feat_name]

        # =================================================================
        # LT FEATURES (Higgs/ZZ system)
        # =================================================================
        lt_features = np.zeros((n_events, self.n_lt, len(self.lt_features)), dtype=np.float32)

        lt_feature_map = {
            'pT_4l': pt_4l_val,
            'eta_4l': eta_4l_val,
            'phi_4l': phi_4l_val,
            'm4l': m4l_val,
        }

        for i, feat_name in enumerate(self.lt_features):
            if feat_name in lt_feature_map:
                lt_features[:, 0, i] = lt_feature_map[feat_name]

        return global_features, cpf_features, npf_features, vtx_features, lt_features

    def predict(self, events, objects):
        """
        Compute MVA scores for events.

        Args:
            events: NanoAOD events
            objects: Selected objects dictionary

        Returns:
            Dictionary with scores and predictions
        """
        global_features, cpf_features, npf_features, vtx_features, lt_features = self.prepare_features(events, objects)

        with torch.no_grad():
            global_t = torch.from_numpy(global_features).float().to(self.device)
            cpf_t = torch.from_numpy(cpf_features).float().to(self.device)
            npf_t = torch.from_numpy(npf_features).float().to(self.device)
            vtx_t = torch.from_numpy(vtx_features).float().to(self.device)
            lt_t = torch.from_numpy(lt_features).float().to(self.device)

            # Pass inputs based on format
            if self.input_format == 'tuple':
                inputs = (global_t, cpf_t, npf_t, vtx_t, lt_t)
            elif self.input_format == 'dict':
                inputs = {
                    'global': global_t,
                    'cpf': cpf_t,
                    'npf': npf_t,
                    'vtx': vtx_t,
                    'lt': lt_t,
                }
            elif self.input_format == 'concat':
                batch_size = global_t.shape[0]
                inputs = torch.cat([
                    global_t,
                    cpf_t.reshape(batch_size, -1),
                    npf_t.reshape(batch_size, -1),
                    vtx_t.reshape(batch_size, -1),
                    lt_t.reshape(batch_size, -1),
                ], dim=-1)
            else:
                raise ValueError(f"Unknown input_format: {self.input_format}")

            logits = self.model(inputs)
            scores = torch.softmax(logits, dim=1).cpu().numpy()

        signal_idx = self.class_names.index('Signal') if 'Signal' in self.class_names else 0
        signal_score = scores[:, signal_idx]
        class_prediction = np.argmax(scores, axis=1)

        return {
            'scores': scores,
            'signal_score': signal_score,
            'class_prediction': class_prediction,
            'class_names': self.class_names,
        }

    def predict_signal_score(self, events, objects):
        """Convenience method to get just the signal score."""
        return self.predict(events, objects)['signal_score']
