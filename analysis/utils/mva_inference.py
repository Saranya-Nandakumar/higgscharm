"""
MVA Inference module for H+c -> ZZ -> 4l analysis.

Loads trained ONNX or PyTorch model and computes MVA scores during event processing.

Usage in processor:
    from analysis.utils.mva_inference import MVAInference

    # Initialize with ONNX model
    mva = MVAInference(model_path='/path/to/model.onnx')

    # Or initialize with PyTorch model
    mva = MVAInference(model_path='/path/to/best_model.pt', use_pytorch=True)

    # Compute scores for events
    scores = mva.predict(events, objects)
"""

import os
import sys
import numpy as np
import awkward as ak

# Increase recursion limit to handle deeply nested awkward array structures
_original_recursion_limit = sys.getrecursionlimit()
if _original_recursion_limit < 50000:
    sys.setrecursionlimit(50000)

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: onnxruntime not installed. ONNX inference disabled.")

try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    # Attempt to install PyTorch automatically
    print("PyTorch not found. Attempting to install...")
    import subprocess
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
            "-q"  # quiet mode
        ])
        import torch
        import torch.nn as nn
        HAS_PYTORCH = True
        print("PyTorch installed successfully.")
    except Exception as e:
        HAS_PYTORCH = False
        print(f"Warning: Failed to install PyTorch: {e}. PyTorch inference disabled.")


# =============================================================================
# PyTorch Model Definitions (minimal versions for inference)
# Compatible with hc_zzto4l_part_training_cvb.yml config
# =============================================================================

if HAS_PYTORCH:
    import math
    import copy
    from functools import partial

    class ResidualBlock(nn.Module):
        """Residual block with pre-norm architecture"""

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

    # =========================================================================
    # ParticleTransformer2_HcZZ Components
    # =========================================================================

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return output.type_as(x) * self.weight

    class SwiGLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.SiLU()

        def forward(self, x):
            x1, x2 = x.chunk(2, dim=-1)
            return x1 * self.act(x2)

    class FFNBlock(nn.Module):
        def __init__(self, dim, mult=4, swiglu=True, dropout=0.1):
            super().__init__()
            self.hid_dim = dim * 4 * 2 if swiglu else dim * 4
            self.dense_1 = nn.Linear(dim, self.hid_dim)
            self.activation = SwiGLU() if swiglu else nn.SiLU()
            self.dense_2 = nn.Linear(dim * 4, dim)
            self.dropout = nn.Dropout(dropout)
            self.attn_norm = RMSNorm(dim * 4)

        def forward(self, x):
            x_proj = self.dense_1(x)
            x = self.attn_norm(self.activation(x_proj))
            x = self.dense_2(x)
            x = self.dropout(x)
            return x

    class InputConv(nn.Module):
        def __init__(self, in_chn, out_chn, dropout_rate=0.1):
            super().__init__()
            self.lin = nn.Linear(in_chn, out_chn)
            self.bn1 = RMSNorm(in_chn)
            self.act = nn.SiLU()
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, sc, skip=True):
            x2 = self.dropout(self.act(self.lin(self.bn1(x))))
            return sc + x2 if skip else x2

    class InputProcess(nn.Module):
        def __init__(self, cpf_dim, npf_dim, vtx_dim, lt_dim, embed_dim):
            super().__init__()
            self.cpf_conv1 = InputConv(cpf_dim, embed_dim)
            self.cpf_conv3 = InputConv(embed_dim, embed_dim)
            self.npf_conv1 = InputConv(npf_dim, embed_dim)
            self.npf_conv3 = InputConv(embed_dim, embed_dim)
            self.vtx_conv1 = InputConv(vtx_dim, embed_dim)
            self.vtx_conv3 = InputConv(embed_dim, embed_dim)
            self.lt_conv1 = InputConv(lt_dim, embed_dim)
            self.lt_conv3 = InputConv(embed_dim, embed_dim)

        def forward(self, cpf, npf, vtx, lt):
            cpf = self.cpf_conv1(cpf, cpf, skip=False)
            cpf = self.cpf_conv3(cpf, cpf, skip=True)
            npf = self.npf_conv1(npf, npf, skip=False)
            npf = self.npf_conv3(npf, npf, skip=True)
            vtx = self.vtx_conv1(vtx, vtx, skip=False)
            vtx = self.vtx_conv3(vtx, vtx, skip=True)
            lt = self.lt_conv1(lt, lt, skip=False)
            lt = self.lt_conv3(lt, lt, skip=True)
            return torch.cat((cpf, npf, vtx, lt), dim=1)

    def delta_phi(a, b):
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def delta_r2(eta1, phi1, eta2, phi2):
        return (eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2

    def to_pt2(x, eps=1e-8):
        pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
        return pt2.clamp(min=eps) if eps else pt2

    def to_m2(x, eps=1e-8):
        m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
        return m2.clamp(min=eps) if eps else m2

    def atan2(y, x):
        sx, sy = torch.sign(x), torch.sign(y)
        pi_part = (sy + sx * (sy**2 - 1)) * (sx - 1) * (-math.pi / 2)
        atan_part = torch.arctan(y / (x + (1 - sx**2))) * sx**2
        return atan_part + pi_part

    def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
        px, py, pz, energy = x[:, :4, :].split((1, 1, 1, 1), dim=1)
        pt = torch.sqrt(to_pt2(x, eps=eps))
        rapidity = 0.5 * torch.log((1 + (2 * pz) / (energy - pz).clamp(min=1e-20)).clamp(min=1e-21))
        phi = (atan2 if for_onnx else torch.atan2)(py, px)
        if not return_mass:
            return torch.cat((pt, rapidity, phi), dim=1)
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)

    def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
        pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
        ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
        ai = torch.ne(pti, 0.0).to(xi.dtype)
        aj = torch.ne(ptj, 0.0).to(xi.dtype)
        mask = ai * aj
        delta = delta_r2(rapi, phii, rapj, phij).sqrt()
        lndelta = torch.log(delta.clamp(min=eps) + 1)
        if num_outputs == 1:
            return lndelta
        ptmin = torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps) + 1)
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps) + 1)
        outputs = [lnkt, lnz, lndelta]
        if num_outputs > 3:
            xij = xi + xj
            lnm2 = torch.log(to_m2(xij, eps=eps) + 1)
            outputs.append(lnm2)
        return torch.cat(outputs, dim=1) * mask

    class PairEmbed(nn.Module):
        def __init__(self, input_dim, dims, normalize_input=True, activation="SiLU", eps=1e-8, for_onnx=False):
            super().__init__()
            self.for_onnx = for_onnx
            self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=4, eps=eps, for_onnx=for_onnx)
            module_list = []
            for dim in dims:
                module_list.extend([nn.BatchNorm1d(input_dim), nn.SiLU(), nn.Conv1d(input_dim, dim, 1)])
                input_dim = dim
            self.embed = nn.Sequential(*module_list)
            self.out_dim = dims[-1]

        def forward(self, x):
            batch_size, _, seq_len = x.size()
            i, j = torch.tril_indices(seq_len, seq_len, offset=-1, device=x.device)
            x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
            xi = x[:, :, i, j]
            xj = x[:, :, j, i]
            x = self.pairwise_lv_fts(xi, xj)
            elements = self.embed(x)
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=x.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
            return y

    class HF_TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1, swiglu=True):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.attn_norm = RMSNorm(d_model)
            self.ffn_norm = RMSNorm(d_model)
            self.ffn = FFNBlock(d_model, 4, swiglu=swiglu, dropout=dropout)

        def forward(self, src, mask, padding_mask):
            residual = src
            src = self.attn_norm(src)
            src = self.self_attn(src, src, src, attn_mask=mask)[0]
            src = residual + src
            residual = src
            src = self.ffn_norm(src)
            src = self.ffn(src)
            return residual + src

    class HF_TransformerEncoder(nn.Module):
        def __init__(self, encoder_layer, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        def forward(self, src, mask, padding_mask):
            output = src
            for mod in self.layers:
                output = mod(output, mask, padding_mask)
            return output

    class CLS_TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dropout=0.1, swiglu=True):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.attn_norm = RMSNorm(d_model)
            self.ffn_norm = RMSNorm(d_model)
            self.cls_norm = RMSNorm(d_model)
            self.ffn = FFNBlock(d_model, 4, swiglu=swiglu, dropout=dropout)

        def forward(self, cls_token, x, padding_mask):
            residual = cls_token
            src = torch.cat((cls_token, x), dim=1)
            padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            enc = self.cls_norm(cls_token)
            src_normed = self.attn_norm(src)
            src = self.self_attn(enc, src_normed, src_normed, key_padding_mask=padding_mask)[0]
            src = residual + src
            residual = src
            src = self.ffn_norm(src)
            src = self.ffn(src)
            return residual + src

    def build_E_p_from_mass(tensor):
        """Convert (pt, eta, phi, mass) to (px, py, pz, E)."""
        pt, eta, phi, mass = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], tensor[:, :, 3]
        out = torch.zeros(tensor.shape[0], tensor.shape[1], 4, device=tensor.device, dtype=tensor.dtype)
        out[:, :, 0] = pt * torch.cos(phi)
        out[:, :, 1] = pt * torch.sin(phi)
        out[:, :, 2] = pt * torch.sinh(eta)
        out[:, :, 3] = torch.sqrt(pt.square() * torch.cosh(eta).square() + mass.square())
        return out

    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(min=a, max=b)
            return tensor

    class ParticleTransformer2_HcZZ_Inference(nn.Module):
        """
        Inference-only version of ParticleTransformer2_HcZZ.

        Input dimensions (from checkpoint):
          - cpf_dim: 15 (jets)
          - npf_dim: 11 (leptons)
          - vtx_dim: 4 (Z bosons)
          - lt_dim: 4 (Higgs)
          - embed_dim: 32
          - num_classes: 5
        """

        def __init__(self, cpf_dim=15, npf_dim=11, vtx_dim=4, lt_dim=4,
                     num_classes=5, num_enc=3, num_head=4, embed_dim=32,
                     for_inference=True, swiglu=True):
            super().__init__()
            self.for_inference = for_inference
            self.num_enc_layers = num_enc
            self.dtype = torch.float32

            self.InputProcess = InputProcess(cpf_dim, npf_dim, vtx_dim, lt_dim, embed_dim)
            self.Linear = nn.Linear(embed_dim, num_classes)
            self.pair_embed = PairEmbed(4, [3*num_head]*2 + [num_head], for_onnx=False)
            self.cls_norm = RMSNorm(embed_dim)

            encoder_layer = HF_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu)
            self.Encoder = HF_TransformerEncoder(encoder_layer, num_layers=num_enc)

            self.CLS_EncoderLayer1 = CLS_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu)
            if num_enc > 3:
                self.CLS_EncoderLayer2 = CLS_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)

        def forward(self, inpt):
            global_features, cpf_features, npf_features, vtx_features, lt_features = inpt

            cpf, npf, vtx, lt = cpf_features, npf_features, vtx_features, lt_features
            cpf_4v = cpf_features[:, :, :4]
            npf_4v = npf_features[:, :, :4]
            vtx_4v = vtx_features[:, :, :4]
            lt_4v = lt_features[:, :, :4]

            # Padding mask based on pt (first feature)
            padding_mask = torch.cat((cpf_4v[:, :, :1], npf_4v[:, :, :1], vtx_4v[:, :, :1], lt_4v[:, :, :1]), dim=1)
            padding_mask = torch.eq(padding_mask[:, :, 0], 0.0)
            pad = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(cpf.dtype) * torch.finfo(self.dtype).min

            # Convert (pt, eta, phi, mass) -> (px, py, pz, E)
            cpf_4v = build_E_p_from_mass(cpf_4v)
            npf_4v = build_E_p_from_mass(npf_4v)
            vtx_4v = build_E_p_from_mass(vtx_4v)
            lt_4v = build_E_p_from_mass(lt_4v)

            enc = self.InputProcess(cpf, npf, vtx, lt)

            lorentz_vectors = torch.cat((cpf_4v, npf_4v, vtx_4v, lt_4v), dim=1)
            v = lorentz_vectors.transpose(1, 2)
            attn_mask = self.pair_embed(v)
            attn_mask = attn_mask + pad.unsqueeze(dim=1)

            enc = self.Encoder(enc, attn_mask.reshape(-1, v.size(-1), v.size(-1)), padding_mask)

            cls_tokens = self.cls_token.expand(enc.size(0), 1, -1)
            cls_tokens = self.CLS_EncoderLayer1(cls_tokens, enc, padding_mask)
            if self.num_enc_layers > 3:
                cls_tokens = self.CLS_EncoderLayer2(cls_tokens, enc, padding_mask)

            x = cls_tokens.squeeze(dim=1)
            output = self.Linear(self.cls_norm(x))

            if self.for_inference:
                output = torch.softmax(output, dim=1)

            return output

    class MLP_HcZZ_MW_Deep_Inference(nn.Module):
        """
        Inference-only version of MLP_HcZZ_MW_Deep for loading best_model.pt

        Compatible with hc_zzto4l_part_training_cvb.yml config:
          - global_features: 9 features
          - cpf_candidates (Jets): 3 jets x 16 features
          - vtx_features (Z bosons): 2 Z x 4 features
          - npf_candidates (Leptons): 4 leptons x 11 features
          - lt_candidates (Higgs): 1 x 4 features
        """

        def __init__(
            self,
            global_dim=9,
            cpf_dim=16,
            npf_dim=11,
            vtx_dim=4,
            lt_dim=4,
            n_cpf=3,
            n_vtx=2,
            n_npf=4,
            n_lt=1,
            num_classes=5,
            hidden_dim=384,
            num_layers=12,
            dropout=0.4,
        ):
            super().__init__()

            self.num_classes = num_classes

            # Total input dimension
            self.total_input_dim = (
                global_dim +
                (n_cpf * cpf_dim) +
                (n_vtx * vtx_dim) +
                (n_npf * npf_dim) +
                (n_lt * lt_dim)
            )

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(self.total_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )

            # Residual blocks
            self.blocks = nn.ModuleList()
            for _ in range(num_layers):
                self.blocks.append(ResidualBlock(hidden_dim, dropout))

            # Output head
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, inpt):
            """Forward pass - accepts tuple of (global, cpf, npf, vtx, lt)"""
            global_features, cpf_features, npf_features, vtx_features, lt_features = (
                inpt[0], inpt[1], inpt[2], inpt[3], inpt[4]
            )

            batch_size = global_features.shape[0]

            # Flatten all candidate features
            cpf_flat = cpf_features.reshape(batch_size, -1)
            npf_flat = npf_features.reshape(batch_size, -1)
            vtx_flat = vtx_features.reshape(batch_size, -1)
            lt_flat = lt_features.reshape(batch_size, -1)

            # Concatenate all features
            x = torch.cat([global_features, cpf_flat, npf_flat, vtx_flat, lt_flat], dim=-1)

            # Project to hidden dimension
            x = self.input_proj(x)

            # Pass through residual blocks
            for block in self.blocks:
                x = block(x)

            # Classification
            output = self.head(x)

            return output


class MVAInference:
    """
    MVA inference for H+c -> ZZ -> 4l analysis using ONNX or PyTorch model.

    Compatible with hc_zzto4l_part_training_cvb.yml config:
        - global_features: 9 event-level features
        - cpf_features: Per-jet features (n_jets x 16 features)
        - npf_features: Per-lepton features (n_leptons x 11 features)
        - vtx_features: Per-Z features (n_Z x 4 features)
        - lt_features: Higgs features (1 x 4 features)
    """

    # Feature names matching hc_zzto4l_part_training_cvb.yml (9 features)
    GLOBAL_FEATURES = [
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

    # CPF = Jets (15 features per jet)
    CPF_FEATURES = [
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

    # VTX = Z bosons (4 features per Z)
    VTX_FEATURES = [
        'z_pt',
        'z_eta',
        'z_phi',
        'z_mass',
    ]

    # NPF = Leptons (11 features per lepton)
    NPF_FEATURES = [
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

    # LT = Higgs/ZZ system (4 features)
    LT_FEATURES = [
        'pT_4l',
        'eta_4l',
        'phi_4l',
        'm4l',
    ]

    # =========================================================================
    # ParticleTransformer2_HcZZ Features (different order - 4v first)
    # =========================================================================

    # CPF for ParT: 4v + 11 tagging flags = 15 features
    CPF_FEATURES_PART = [
        'jet_pt', 'jet_eta', 'jet_phi', 'jet_mass',  # 4-vector (first 4)
        'jet_is_C0', 'jet_is_C1', 'jet_is_C2', 'jet_is_C3', 'jet_is_C4',
        'jet_is_B0', 'jet_is_B1', 'jet_is_B2', 'jet_is_B3', 'jet_is_B4',
        'jet_is_L0',
    ]

    # NPF for ParT: 4v + 7 features = 11 features
    NPF_FEATURES_PART = [
        'lepton_pt', 'lepton_eta', 'lepton_phi', 'lepton_mass',  # 4-vector (first 4)
        'lepton_charge', 'lepton_pfRelIso03_all', 'lepton_sip3d',
        'lepton_is_tight', 'lepton_mvaHZZIso', 'lepton_isPFcand', 'lepton_highPtId',
    ]

    # VTX for ParT: 4 features (just 4-vector)
    VTX_FEATURES_PART = ['z_pt', 'z_eta', 'z_phi', 'z_mass']

    # LT for ParT: 4 features (just 4-vector)
    LT_FEATURES_PART = ['pT_4l', 'eta_4l', 'phi_4l', 'm4l']

    # Class names for 5-class output
    CLASS_NAMES_5CLASS = ['qqgg_toZZ', 'ggH', 'Signal', 'HPlusB', 'Other']

    # Class names for 6-class output
    CLASS_NAMES_6CLASS = ['Signal', 'HPlusB', 'ggZZ', 'qqZZ', 'Other', 'ggH']

    # MVA jet tag boundaries (for computing jet_is_* and jet_cvb_score)
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

    # CVB scores from training config: log(N_c / N_b) for each category
    CVB_SCORES = {
        "C4": 4.30,   # log(2964/40)
        "C3": 2.67,   # log(7161/495)
        "C2": 1.64,   # log(1669/323)
        "C1": 1.79,   # log(6095/1018)
        "C0": 2.23,   # log(10125/1091)
        "L0": 2.70,   # log(3672/247)
        "B0": 1.11,   # log(1499/493)
        "B1": 1.73,   # log(17/3)
        "B2": 0.36,   # log(1657/1156)
        "B3": -0.74,  # log(168/351)
        "B4": -3.32,  # log(96/2642)
    }

    def __init__(self, model_path, n_cpf=3, n_vtx=2, n_npf=4, n_lt=1,
                 use_6class=False, use_pytorch=False, hidden_dim=64, num_layers=2,
                 model_type='mlp'):
        """
        Initialize MVA inference.

        Args:
            model_path: Path to ONNX (.onnx) or PyTorch (.pt) model file
            n_cpf: Number of jet candidates (default 3)
            n_vtx: Number of Z boson candidates (default 2)
            n_npf: Number of lepton candidates (default 4)
            n_lt: Number of Higgs candidates (default 1)
            use_6class: Whether model is 6-class (default False for 5-class)
            use_pytorch: If True, load as PyTorch model; if False, load as ONNX
            hidden_dim: Hidden dimension for PyTorch model (default 64)
            num_layers: Number of residual layers for PyTorch model (default 2)
            model_type: 'mlp' for MLP_HcZZ_MW_Deep, 'part' for ParticleTransformer2_HcZZ
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.n_cpf = n_cpf
        self.n_vtx = n_vtx
        self.n_npf = n_npf
        self.n_lt = n_lt
        self.use_6class = use_6class
        self.use_pytorch = use_pytorch
        self.model_type = model_type.lower()
        self.class_names = self.CLASS_NAMES_6CLASS if use_6class else self.CLASS_NAMES_5CLASS
        self.num_classes = len(self.class_names)

        if use_pytorch:
            if not HAS_PYTORCH:
                raise ImportError("PyTorch required for PyTorch model inference")
            self._load_pytorch_model(hidden_dim, num_layers)
        else:
            if not HAS_ONNX:
                raise ImportError("onnxruntime required for ONNX model inference")
            self._load_onnx_model()

    def _load_pytorch_model(self, hidden_dim, num_layers):
        """Load PyTorch model from best_model.pt"""
        if self.model_type == 'part':
            # ParticleTransformer2_HcZZ model
            # Feature dimensions from checkpoint: cpf=15, npf=11, vtx=4, lt=4, embed=32
            self.model = ParticleTransformer2_HcZZ_Inference(
                cpf_dim=15,  # 4 (pt,eta,phi,mass) + 11 (tagging flags)
                npf_dim=11,  # lepton features
                vtx_dim=4,   # Z boson features
                lt_dim=4,    # Higgs features
                num_classes=self.num_classes,
                num_enc=3,
                num_head=4,
                embed_dim=32,
                for_inference=True,
                swiglu=True,
            )
            print(f"ParticleTransformer2_HcZZ model created")
        else:
            # MLP_HcZZ_MW_Deep model (default)
            self.model = MLP_HcZZ_MW_Deep_Inference(
                global_dim=len(self.GLOBAL_FEATURES),
                cpf_dim=len(self.CPF_FEATURES),
                npf_dim=len(self.NPF_FEATURES),
                vtx_dim=len(self.VTX_FEATURES),
                lt_dim=len(self.LT_FEATURES),
                n_cpf=self.n_cpf,
                n_vtx=self.n_vtx,
                n_npf=self.n_npf,
                n_lt=self.n_lt,
                num_classes=self.num_classes,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

        # Load weights (handle both checkpoint format and direct state_dict)
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"PyTorch MVA model loaded: {self.model_path}")
        print(f"  Model type: {self.model_type}")
        if self.model_type == 'mlp':
            print(f"  Architecture: hidden_dim={hidden_dim}, num_layers={num_layers}")
            print(f"  Input dim: {self.model.total_input_dim}")
        else:
            print(f"  Architecture: ParticleTransformer2_HcZZ (embed_dim=32, num_enc=3)")
        print(f"  Classes: {self.class_names}")

    def _load_onnx_model(self):
        """Load ONNX model"""
        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"ONNX MVA model loaded: {self.model_path}")
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
        print(f"  Classes: {self.class_names}")

    def compute_jet_tags_and_cvb(self, B, CvB, CvL):
        """
        Compute jet tagging category flags and CVB score from PNet scores.

        Args:
            B: PNet B-tag score (n_events, n_jets)
            CvB: PNet CvB score (n_events, n_jets)
            CvL: PNet CvL score (n_events, n_jets)

        Returns:
            Tuple of (tags dict, cvb_scores array)
        """
        # Transform to boundary coordinates
        pBvsC = 1.0 - CvB
        pBplusC = B + (1.0 - B) * CvL
        pBplusC = np.clip(pBplusC, 0, 1)
        pBvsC = np.clip(pBvsC, 0, 1)

        tags = {}
        cvb_scores = np.zeros_like(B, dtype=np.float32)

        for cat_name in self.CATEGORY_ORDER:
            bounds = self.MVA_JET_TAG_BOUNDARIES[cat_name]
            x_min, x_max = bounds["x"]
            y_min, y_max = bounds["y"]

            mask = (
                (pBplusC >= x_min) & (pBplusC < x_max) &
                (pBvsC >= y_min) & (pBvsC < y_max)
            )
            tags[f'jet_is_{cat_name}'] = mask.astype(np.float32)

            # Assign CVB score for jets in this category
            cvb_scores = np.where(mask, self.CVB_SCORES[cat_name], cvb_scores)

        return tags, cvb_scores

    def _to_1d_numpy(self, val, n_events):
        """Convert awkward array or scalar to 1D numpy array."""
        try:
            if isinstance(val, (int, float)):
                return np.full(n_events, val, dtype=np.float32)

            if isinstance(val, np.ndarray):
                arr = val.copy()
            else:
                # Try direct conversion first
                try:
                    arr = np.asarray(val)
                except (ValueError, TypeError, RecursionError):
                    # Fallback: use ak.to_list() which is more robust
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
                # Try direct conversion first
                try:
                    filled = ak.fill_none(val, 0)
                    arr = ak.to_numpy(filled)
                except (ValueError, TypeError, RecursionError):
                    # Fallback: use ak.to_list() which is more robust
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
            objects: Selected objects dictionary containing:
                - 'best_zzcandidate': ZZ candidate with z1, z2, and leptons
                - 'jets': Selected jets
                - 'leptons': Selected leptons

        Returns:
            Tuple of (global_features, cpf_features, npf_features, vtx_features, lt_features)
        """
        n_events = len(events)

        # Get objects
        best_zz = objects.get('best_zzcandidate', None)
        jets = objects.get('jets', None)
        leptons = objects.get('leptons', None)

        # =================================================================
        # GLOBAL FEATURES (9 features)
        # =================================================================
        global_features = np.zeros((n_events, len(self.GLOBAL_FEATURES)), dtype=np.float32)

        # n_jet
        if jets is not None:
            global_features[:, 0] = self._to_1d_numpy(ak.num(jets), n_events)

        # Jet_HT
        if jets is not None and ak.count(jets) > 0:
            jet_ht = ak.sum(jets.pt, axis=1)
            global_features[:, 1] = self._to_1d_numpy(jet_ht, n_events)

        # n_lepton
        if leptons is not None:
            global_features[:, 2] = self._to_1d_numpy(ak.num(leptons), n_events)

        # Compute ZZ kinematics using numpy (to avoid vector4D recursion issues)
        if best_zz is not None:
            try:
                # Helper to safely convert awkward arrays to numpy
                # Handles RecursionError from deeply nested awkward arrays
                def safe_to_numpy(arr, default=0):
                    try:
                        filled = ak.fill_none(arr, default)
                        result = np.asarray(filled)
                        # Ensure 1D output - flatten if extra dimensions exist
                        if result.ndim > 1:
                            result = result.flatten()
                        return result
                    except (RecursionError, Exception):
                        try:
                            # Fallback 1: use ak.to_list which is more robust
                            filled = ak.fill_none(arr, default)
                            result = np.array(ak.to_list(filled), dtype=np.float32)
                            if result.ndim > 1:
                                result = result.flatten()
                            return result
                        except (RecursionError, Exception):
                            # Fallback 2: return zeros
                            return np.zeros(n_events, dtype=np.float32)

                # Extract lepton kinematics from Z1 and Z2
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

                # Compute 4-vectors for all 4 leptons
                z1_l1_px, z1_l1_py, z1_l1_pz, z1_l1_E = compute_4vec(z1_l1_pt, z1_l1_eta, z1_l1_phi, z1_l1_mass)
                z1_l2_px, z1_l2_py, z1_l2_pz, z1_l2_E = compute_4vec(z1_l2_pt, z1_l2_eta, z1_l2_phi, z1_l2_mass)
                z2_l1_px, z2_l1_py, z2_l1_pz, z2_l1_E = compute_4vec(z2_l1_pt, z2_l1_eta, z2_l1_phi, z2_l1_mass)
                z2_l2_px, z2_l2_py, z2_l2_pz, z2_l2_E = compute_4vec(z2_l2_pt, z2_l2_eta, z2_l2_phi, z2_l2_mass)

                # Compute Z1 and Z2 4-vectors
                z1_px = z1_l1_px + z1_l2_px
                z1_py = z1_l1_py + z1_l2_py
                z1_pz = z1_l1_pz + z1_l2_pz
                z1_E = z1_l1_E + z1_l2_E

                z2_px = z2_l1_px + z2_l2_px
                z2_py = z2_l1_py + z2_l2_py
                z2_pz = z2_l1_pz + z2_l2_pz
                z2_E = z2_l1_E + z2_l2_E

                # Z masses and pts
                z1_mass = np.sqrt(np.maximum(z1_E**2 - z1_px**2 - z1_py**2 - z1_pz**2, 0))
                z2_mass = np.sqrt(np.maximum(z2_E**2 - z2_px**2 - z2_py**2 - z2_pz**2, 0))
                z1_pt = np.sqrt(z1_px**2 + z1_py**2)
                z2_pt = np.sqrt(z2_px**2 + z2_py**2)
                z1_eta = np.arctanh(np.clip(z1_pz / np.sqrt(z1_px**2 + z1_py**2 + z1_pz**2 + 1e-10), -0.9999, 0.9999))
                z2_eta = np.arctanh(np.clip(z2_pz / np.sqrt(z2_px**2 + z2_py**2 + z2_pz**2 + 1e-10), -0.9999, 0.9999))
                z1_phi = np.arctan2(z1_py, z1_px)
                z2_phi = np.arctan2(z2_py, z2_px)

                # Compute ZZ 4-vector
                zz_px = z1_px + z2_px
                zz_py = z1_py + z2_py
                zz_pz = z1_pz + z2_pz
                zz_E = z1_E + z2_E

                # ZZ system observables
                m4l_val = np.sqrt(np.maximum(zz_E**2 - zz_px**2 - zz_py**2 - zz_pz**2, 0))
                pt_4l_val = np.sqrt(zz_px**2 + zz_py**2)
                zz_p = np.sqrt(zz_px**2 + zz_py**2 + zz_pz**2 + 1e-10)
                eta_4l_val = np.arctanh(np.clip(zz_pz / zz_p, -0.9999, 0.9999))
                phi_4l_val = np.arctan2(zz_py, zz_px)

                # deltaR_ZZ
                deta = z1_eta - z2_eta
                dphi = z1_phi - z2_phi
                dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
                dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
                deltaR_ZZ = np.sqrt(deta**2 + dphi**2)

                # Clean up NaN values
                m4l_val = np.nan_to_num(m4l_val, nan=0.0)
                pt_4l_val = np.nan_to_num(pt_4l_val, nan=0.0)
                eta_4l_val = np.nan_to_num(eta_4l_val, nan=0.0)
                phi_4l_val = np.nan_to_num(phi_4l_val, nan=0.0)
                deltaR_ZZ = np.nan_to_num(deltaR_ZZ, nan=0.0)
                z1_pt = np.nan_to_num(z1_pt, nan=0.0)
                z2_pt = np.nan_to_num(z2_pt, nan=0.0)
                z1_eta = np.nan_to_num(z1_eta, nan=0.0)
                z2_eta = np.nan_to_num(z2_eta, nan=0.0)
                z1_phi = np.nan_to_num(z1_phi, nan=0.0)
                z2_phi = np.nan_to_num(z2_phi, nan=0.0)
                z1_mass = np.nan_to_num(z1_mass, nan=0.0)
                z2_mass = np.nan_to_num(z2_mass, nan=0.0)

            except Exception as e:
                print(f"Warning: Failed to extract ZZ kinematics: {e}")
                m4l_val = np.zeros(n_events, dtype=np.float32)
                pt_4l_val = np.zeros(n_events, dtype=np.float32)
                eta_4l_val = np.zeros(n_events, dtype=np.float32)
                phi_4l_val = np.zeros(n_events, dtype=np.float32)
                deltaR_ZZ = np.zeros(n_events, dtype=np.float32)
                z1_pt = np.zeros(n_events, dtype=np.float32)
                z2_pt = np.zeros(n_events, dtype=np.float32)
                z1_eta = np.zeros(n_events, dtype=np.float32)
                z2_eta = np.zeros(n_events, dtype=np.float32)
                z1_phi = np.zeros(n_events, dtype=np.float32)
                z2_phi = np.zeros(n_events, dtype=np.float32)
                z1_mass = np.zeros(n_events, dtype=np.float32)
                z2_mass = np.zeros(n_events, dtype=np.float32)
        else:
            m4l_val = np.zeros(n_events, dtype=np.float32)
            pt_4l_val = np.zeros(n_events, dtype=np.float32)
            eta_4l_val = np.zeros(n_events, dtype=np.float32)
            phi_4l_val = np.zeros(n_events, dtype=np.float32)
            deltaR_ZZ = np.zeros(n_events, dtype=np.float32)
            z1_pt = np.zeros(n_events, dtype=np.float32)
            z2_pt = np.zeros(n_events, dtype=np.float32)
            z1_eta = np.zeros(n_events, dtype=np.float32)
            z2_eta = np.zeros(n_events, dtype=np.float32)
            z1_phi = np.zeros(n_events, dtype=np.float32)
            z2_phi = np.zeros(n_events, dtype=np.float32)
            z1_mass = np.zeros(n_events, dtype=np.float32)
            z2_mass = np.zeros(n_events, dtype=np.float32)

        # Fill global features: m4l, eta_4l, deltaR_ZZ
        global_features[:, 3] = m4l_val  # m4l
        global_features[:, 4] = eta_4l_val  # eta_4l
        global_features[:, 5] = deltaR_ZZ  # deltaR_ZZ

        # n_ctagged_jets, n_btagged_jets (count jets in C and B categories)
        n_ctagged = np.zeros(n_events, dtype=np.float32)
        n_btagged = np.zeros(n_events, dtype=np.float32)
        # Will be computed after jet tags are computed below

        # in_mass_window
        in_mass_window = ((m4l_val >= 100) & (m4l_val < 150)).astype(np.float32)
        global_features[:, 8] = in_mass_window

        # =================================================================
        # CPF FEATURES (Jets: 3 jets x 16 features)
        # =================================================================
        cpf_features = np.zeros((n_events, self.n_cpf, len(self.CPF_FEATURES)), dtype=np.float32)

        if jets is not None and ak.count(jets) > 0:
            jets_padded = ak.pad_none(jets, self.n_cpf, clip=True)

            # Kinematics
            cpf_features[:, :, 0] = self._to_2d_numpy(jets_padded.pt, n_events, self.n_cpf)
            cpf_features[:, :, 1] = self._to_2d_numpy(jets_padded.eta, n_events, self.n_cpf)
            cpf_features[:, :, 2] = self._to_2d_numpy(jets_padded.phi, n_events, self.n_cpf)
            cpf_features[:, :, 3] = self._to_2d_numpy(jets_padded.mass, n_events, self.n_cpf)

            # Get PNet scores for jet tag computation
            B = self._to_2d_numpy(jets_padded.btagPNetB, n_events, self.n_cpf)
            CvB = self._to_2d_numpy(jets_padded.btagPNetCvB, n_events, self.n_cpf)
            CvL = self._to_2d_numpy(jets_padded.btagPNetCvL, n_events, self.n_cpf)

            # Compute jet tags and CVB score
            jet_tags, cvb_scores = self.compute_jet_tags_and_cvb(B, CvB, CvL)

            # Fill jet tag features (indices 4-14)
            for i, cat_name in enumerate(self.CATEGORY_ORDER):
                cpf_features[:, :, 4 + i] = jet_tags[f'jet_is_{cat_name}']

            # Count c-tagged and b-tagged jets
            c_categories = ['C0', 'C1', 'C2', 'C3', 'C4']
            b_categories = ['B0', 'B1', 'B2', 'B3', 'B4']

            for cat_name in c_categories:
                n_ctagged += jet_tags[f'jet_is_{cat_name}'].sum(axis=1)
            for cat_name in b_categories:
                n_btagged += jet_tags[f'jet_is_{cat_name}'].sum(axis=1)

        global_features[:, 6] = n_ctagged  # n_ctagged_jets
        global_features[:, 7] = n_btagged  # n_btagged_jets

        # =================================================================
        # VTX FEATURES (Z bosons: 2 x 4 features)
        # =================================================================
        vtx_features = np.zeros((n_events, self.n_vtx, len(self.VTX_FEATURES)), dtype=np.float32)

        # Z1
        vtx_features[:, 0, 0] = z1_pt
        vtx_features[:, 0, 1] = z1_eta
        vtx_features[:, 0, 2] = z1_phi
        vtx_features[:, 0, 3] = z1_mass

        # Z2
        vtx_features[:, 1, 0] = z2_pt
        vtx_features[:, 1, 1] = z2_eta
        vtx_features[:, 1, 2] = z2_phi
        vtx_features[:, 1, 3] = z2_mass

        # =================================================================
        # NPF FEATURES (Leptons: 4 x 11 features)
        # =================================================================
        npf_features = np.zeros((n_events, self.n_npf, len(self.NPF_FEATURES)), dtype=np.float32)

        if leptons is not None and ak.count(leptons) > 0:
            leptons_padded = ak.pad_none(leptons, self.n_npf, clip=True)

            npf_features[:, :, 0] = self._to_2d_numpy(leptons_padded.pt, n_events, self.n_npf)
            npf_features[:, :, 1] = self._to_2d_numpy(leptons_padded.eta, n_events, self.n_npf)
            npf_features[:, :, 2] = self._to_2d_numpy(leptons_padded.phi, n_events, self.n_npf)
            npf_features[:, :, 3] = self._to_2d_numpy(leptons_padded.mass, n_events, self.n_npf)
            npf_features[:, :, 4] = self._to_2d_numpy(leptons_padded.charge, n_events, self.n_npf)
            npf_features[:, :, 5] = self._to_2d_numpy(leptons_padded.pfRelIso03_all, n_events, self.n_npf)
            npf_features[:, :, 6] = self._to_2d_numpy(leptons_padded.sip3d, n_events, self.n_npf)

            # Boolean features - convert to float
            try:
                npf_features[:, :, 7] = self._to_2d_numpy(
                    ak.fill_none(leptons_padded.isTight, 0), n_events, self.n_npf)
            except:
                pass  # Field may not exist

            try:
                npf_features[:, :, 8] = self._to_2d_numpy(
                    ak.fill_none(leptons_padded.mvaHZZIso, 0), n_events, self.n_npf)
            except:
                pass

            try:
                npf_features[:, :, 9] = self._to_2d_numpy(
                    ak.fill_none(leptons_padded.isPFcand, 0), n_events, self.n_npf)
            except:
                pass

            try:
                npf_features[:, :, 10] = self._to_2d_numpy(
                    ak.fill_none(leptons_padded.highPtId, 0), n_events, self.n_npf)
            except:
                pass

        # =================================================================
        # LT FEATURES (Higgs/ZZ: 1 x 4 features)
        # =================================================================
        lt_features = np.zeros((n_events, self.n_lt, len(self.LT_FEATURES)), dtype=np.float32)

        lt_features[:, 0, 0] = pt_4l_val   # pT_4l
        lt_features[:, 0, 1] = eta_4l_val  # eta_4l
        lt_features[:, 0, 2] = phi_4l_val  # phi_4l
        lt_features[:, 0, 3] = m4l_val     # m4l

        return global_features, cpf_features, npf_features, vtx_features, lt_features

    def prepare_features_part(self, events, objects):
        """
        Prepare features for ParticleTransformer2_HcZZ model.

        Key difference from MLP: 4-vector features (pt, eta, phi, mass) must be
        the FIRST 4 features of each candidate type.
        """
        n_events = len(events)

        best_zz = objects.get('best_zzcandidate', None)
        jets = objects.get('jets', None)
        leptons = objects.get('leptons', None)

        # Global features (not used by ParT but kept for compatibility)
        global_features = np.zeros((n_events, 2), dtype=np.float32)
        if jets is not None:
            global_features[:, 0] = self._to_1d_numpy(ak.num(jets), n_events)
            if ak.count(jets) > 0:
                global_features[:, 1] = self._to_1d_numpy(ak.sum(jets.pt, axis=1), n_events)

        # Compute ZZ kinematics
        if best_zz is not None:
            try:
                def safe_to_numpy(arr, default=0):
                    try:
                        filled = ak.fill_none(arr, default)
                        result = np.asarray(filled)
                        if result.ndim > 1:
                            result = result.flatten()
                        return result
                    except:
                        return np.zeros(n_events, dtype=np.float32)

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

                z1_px, z1_py = z1_l1_px + z1_l2_px, z1_l1_py + z1_l2_py
                z1_pz, z1_E = z1_l1_pz + z1_l2_pz, z1_l1_E + z1_l2_E
                z2_px, z2_py = z2_l1_px + z2_l2_px, z2_l1_py + z2_l2_py
                z2_pz, z2_E = z2_l1_pz + z2_l2_pz, z2_l1_E + z2_l2_E

                z1_pt = np.sqrt(z1_px**2 + z1_py**2)
                z2_pt = np.sqrt(z2_px**2 + z2_py**2)
                z1_mass = np.sqrt(np.maximum(z1_E**2 - z1_px**2 - z1_py**2 - z1_pz**2, 0))
                z2_mass = np.sqrt(np.maximum(z2_E**2 - z2_px**2 - z2_py**2 - z2_pz**2, 0))
                z1_eta = np.arctanh(np.clip(z1_pz / np.sqrt(z1_px**2 + z1_py**2 + z1_pz**2 + 1e-10), -0.9999, 0.9999))
                z2_eta = np.arctanh(np.clip(z2_pz / np.sqrt(z2_px**2 + z2_py**2 + z2_pz**2 + 1e-10), -0.9999, 0.9999))
                z1_phi, z2_phi = np.arctan2(z1_py, z1_px), np.arctan2(z2_py, z2_px)

                zz_px, zz_py = z1_px + z2_px, z1_py + z2_py
                zz_pz, zz_E = z1_pz + z2_pz, z1_E + z2_E
                m4l_val = np.sqrt(np.maximum(zz_E**2 - zz_px**2 - zz_py**2 - zz_pz**2, 0))
                pt_4l_val = np.sqrt(zz_px**2 + zz_py**2)
                eta_4l_val = np.arctanh(np.clip(zz_pz / np.sqrt(zz_px**2 + zz_py**2 + zz_pz**2 + 1e-10), -0.9999, 0.9999))
                phi_4l_val = np.arctan2(zz_py, zz_px)

                # Clean NaN
                for arr in [z1_pt, z1_eta, z1_phi, z1_mass, z2_pt, z2_eta, z2_phi, z2_mass,
                            m4l_val, pt_4l_val, eta_4l_val, phi_4l_val]:
                    arr[:] = np.nan_to_num(arr, nan=0.0)

            except Exception as e:
                z1_pt = z1_eta = z1_phi = z1_mass = np.zeros(n_events, dtype=np.float32)
                z2_pt = z2_eta = z2_phi = z2_mass = np.zeros(n_events, dtype=np.float32)
                m4l_val = pt_4l_val = eta_4l_val = phi_4l_val = np.zeros(n_events, dtype=np.float32)
        else:
            z1_pt = z1_eta = z1_phi = z1_mass = np.zeros(n_events, dtype=np.float32)
            z2_pt = z2_eta = z2_phi = z2_mass = np.zeros(n_events, dtype=np.float32)
            m4l_val = pt_4l_val = eta_4l_val = phi_4l_val = np.zeros(n_events, dtype=np.float32)

        # CPF Features (15): 4v + 11 tags
        cpf_features = np.zeros((n_events, self.n_cpf, 15), dtype=np.float32)
        if jets is not None and ak.count(jets) > 0:
            jets_padded = ak.pad_none(jets, self.n_cpf, clip=True)
            cpf_features[:, :, 0] = self._to_2d_numpy(jets_padded.pt, n_events, self.n_cpf)
            cpf_features[:, :, 1] = self._to_2d_numpy(jets_padded.eta, n_events, self.n_cpf)
            cpf_features[:, :, 2] = self._to_2d_numpy(jets_padded.phi, n_events, self.n_cpf)
            cpf_features[:, :, 3] = self._to_2d_numpy(jets_padded.mass, n_events, self.n_cpf)

            B = self._to_2d_numpy(jets_padded.btagPNetB, n_events, self.n_cpf)
            CvB = self._to_2d_numpy(jets_padded.btagPNetCvB, n_events, self.n_cpf)
            CvL = self._to_2d_numpy(jets_padded.btagPNetCvL, n_events, self.n_cpf)
            jet_tags, _ = self.compute_jet_tags_and_cvb(B, CvB, CvL)

            tag_order = ['C0', 'C1', 'C2', 'C3', 'C4', 'B0', 'B1', 'B2', 'B3', 'B4', 'L0']
            for i, cat in enumerate(tag_order):
                cpf_features[:, :, 4 + i] = jet_tags[f'jet_is_{cat}']

        # NPF Features (11): 4v + 7 features
        npf_features = np.zeros((n_events, self.n_npf, 11), dtype=np.float32)
        if leptons is not None and ak.count(leptons) > 0:
            leptons_padded = ak.pad_none(leptons, self.n_npf, clip=True)
            npf_features[:, :, 0] = self._to_2d_numpy(leptons_padded.pt, n_events, self.n_npf)
            npf_features[:, :, 1] = self._to_2d_numpy(leptons_padded.eta, n_events, self.n_npf)
            npf_features[:, :, 2] = self._to_2d_numpy(leptons_padded.phi, n_events, self.n_npf)
            npf_features[:, :, 3] = self._to_2d_numpy(leptons_padded.mass, n_events, self.n_npf)
            npf_features[:, :, 4] = self._to_2d_numpy(leptons_padded.charge, n_events, self.n_npf)
            npf_features[:, :, 5] = self._to_2d_numpy(leptons_padded.pfRelIso03_all, n_events, self.n_npf)
            npf_features[:, :, 6] = self._to_2d_numpy(leptons_padded.sip3d, n_events, self.n_npf)
            try:
                npf_features[:, :, 7] = self._to_2d_numpy(ak.fill_none(leptons_padded.isTight, 0), n_events, self.n_npf)
            except: pass
            try:
                npf_features[:, :, 8] = self._to_2d_numpy(ak.fill_none(leptons_padded.mvaHZZIso, 0), n_events, self.n_npf)
            except: pass
            try:
                npf_features[:, :, 9] = self._to_2d_numpy(ak.fill_none(leptons_padded.isPFcand, 0), n_events, self.n_npf)
            except: pass
            try:
                npf_features[:, :, 10] = self._to_2d_numpy(ak.fill_none(leptons_padded.highPtId, 0), n_events, self.n_npf)
            except: pass

        # VTX Features (4): Z boson 4-vectors
        vtx_features = np.zeros((n_events, self.n_vtx, 4), dtype=np.float32)
        vtx_features[:, 0, 0], vtx_features[:, 0, 1] = z1_pt, z1_eta
        vtx_features[:, 0, 2], vtx_features[:, 0, 3] = z1_phi, z1_mass
        vtx_features[:, 1, 0], vtx_features[:, 1, 1] = z2_pt, z2_eta
        vtx_features[:, 1, 2], vtx_features[:, 1, 3] = z2_phi, z2_mass

        # LT Features (4): Higgs 4-vector
        lt_features = np.zeros((n_events, self.n_lt, 4), dtype=np.float32)
        lt_features[:, 0, 0], lt_features[:, 0, 1] = pt_4l_val, eta_4l_val
        lt_features[:, 0, 2], lt_features[:, 0, 3] = phi_4l_val, m4l_val

        return global_features, cpf_features, npf_features, vtx_features, lt_features

    def predict(self, events, objects):
        """
        Compute MVA scores for events.

        Args:
            events: NanoAOD events
            objects: Selected objects dictionary

        Returns:
            Dictionary with:
                - 'scores': Full probability array (n_events, n_classes)
                - 'signal_score': Signal class probability (n_events,)
                - 'class_prediction': Predicted class index (n_events,)
                - 'class_names': List of class names
        """
        # Prepare features based on model type
        if self.model_type == 'part':
            global_features, cpf_features, npf_features, vtx_features, lt_features = self.prepare_features_part(events, objects)
        else:
            global_features, cpf_features, npf_features, vtx_features, lt_features = self.prepare_features(events, objects)

        if self.use_pytorch:
            return self._predict_pytorch(global_features, cpf_features, npf_features, vtx_features, lt_features)
        else:
            return self._predict_onnx(global_features, cpf_features, npf_features, vtx_features, lt_features)

    def _predict_pytorch(self, global_features, cpf_features, npf_features, vtx_features, lt_features):
        """PyTorch inference"""
        with torch.no_grad():
            # Convert to tensors
            global_t = torch.from_numpy(global_features).float()
            cpf_t = torch.from_numpy(cpf_features).float()
            npf_t = torch.from_numpy(npf_features).float()
            vtx_t = torch.from_numpy(vtx_features).float()
            lt_t = torch.from_numpy(lt_features).float()

            # Forward pass
            logits = self.model((global_t, cpf_t, npf_t, vtx_t, lt_t))
            scores = torch.softmax(logits, dim=1).numpy()

        # Get signal score
        signal_idx = self.class_names.index('Signal')
        signal_score = scores[:, signal_idx]
        class_prediction = np.argmax(scores, axis=1)

        return {
            'scores': scores,
            'signal_score': signal_score,
            'class_prediction': class_prediction,
            'class_names': self.class_names,
        }

    def _predict_onnx(self, global_features, cpf_features, npf_features, vtx_features, lt_features):
        """ONNX inference"""
        # Map features to model inputs by name
        feature_map = {
            'global_features': global_features,
            'cpf_features': cpf_features,
            'npf_features': npf_features,
            'vtx_features': vtx_features,
            'lt_features': lt_features,
        }

        inputs = {name: feature_map[name] for name in self.input_names if name in feature_map}
        outputs = self.session.run(self.output_names, inputs)

        scores = outputs[0]

        # Handle output shape
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.flatten()
            signal_score = scores
            class_prediction = (scores > 0.5).astype(np.int32)
        else:
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            signal_idx = self.class_names.index('Signal')
            signal_score = scores[:, signal_idx] if scores.shape[1] > signal_idx else scores[:, 0]
            class_prediction = np.argmax(scores, axis=1)

        return {
            'scores': scores,
            'signal_score': signal_score,
            'class_prediction': class_prediction,
            'class_names': self.class_names,
        }

    def predict_signal_score(self, events, objects):
        """
        Convenience method to get just the signal score.

        Returns:
            numpy array of signal probabilities (n_events,)
        """
        result = self.predict(events, objects)
        return result['signal_score']
