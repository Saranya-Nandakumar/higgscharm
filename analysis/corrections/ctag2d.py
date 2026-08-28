"""PNet 2D pseudo-continuous HF-tagging scale factor (ParticleNetAK4_pseudocontinuous).

Preliminary "2D_HF_Tagging" SF (correction_files["ctagging_2d"]), distinct from the 1D
fixed-WP CTagCorrector (BTV particleNet_wc/tnp/light, ctag.py) this is meant to replace.

WP-code mapping (0=L0, 40-44=C0-C4, 50-54=B0-B4) and the (syst, flavor, wp, abseta, pt)
evaluate order were ported from `Chirayu18/higgscharm` (`hww-analysis` branch,
analysis/corrections/ctag2d.py) -- that repo runs the identical SF files
(flavTaggingSF_<campaign>.json.gz, correction name ParticleNetAK4_pseudocontinuous) in
production. This resolved the WP-code documentation blocker recorded in second-brain memory
hczz_pnet_ctag_sf_migration (the etsai.web.cern.ch reference doc is CERN-SSO-gated).

**Category boundaries, CORRECTED 2026-08-14** (originally shipped 2026-08-14 with a bug,
fixed same day): the FIRST version of this file used the hww-analysis reference's edges
verbatim -- confirmed after the fact (user pasted the real etsai.web.cern.ch page content) to
be the 2024 UParTAK4 boundaries, NOT the 2022/2023 PNet boundaries this file's SF_FILES
actually need. Real per-era PNet boundaries (source: HiggsDNA metaconditions Era2022_v1.json
"HPC_ctag_WPs" block, independently confirmed 2026-08-14 by the user pasting that exact JSON)
were already correctly implemented in this repo at
`b-hive_ttcc/utils/coffea_processors/lz4_hczz_processor.py::JET_TAG_BOUNDARIES_BY_ERA` --
copied from there below rather than re-derived. The 2024 UParT boundaries are NOT needed here
(this corrector is gated off for nano_version=="15" in correction_manager.py) and are not
included.

The (x, y) coordinate transform itself was NOT the bug and needed no fix: `lz4_hczz_processor.
py::compute_jet_categories` derives x=pBplusC=B+(1-B)*CvL, y=pBvsC=1-CvB from the B/CvB/CvL
softmax outputs; this file computes x=CvL/(CvL+CvB*(1-CvL)), y=1-CvB from CvL/CvB alone --
algebraically identical (substitute CvB=C/(B+C), CvL=C/(C+L) into either and both reduce to
(B+C)/(B+C+L) for x), confirmed against the actual pasted etsai.web.cern.ch UParT-doc HFvLF
formula too. Only the per-era EDGE VALUES were wrong, not the transform.

SCOPE DECISION (differs from hww-analysis): that repo applies this to a single "event
candidate c-jet" (their MVA is keyed on one leading-CvsL jet). This analysis's existing 1D
CTagCorrector (ctag.py) instead multiplies a per-jet SF across every jet in
`events.selected_jets` (all flavors, tagged + untagged formula). To keep scope identical to
the corrector being replaced -- so yields are comparable before/after and c-tagging remains
tied to the full selected-jet collection, not one jet -- this version evaluates a 2D category
per jet and takes ak.prod across ALL selected jets in the event. A pseudo-continuous SF does
not need ctag.py's separate tagged/untagged eff-based formula: every jet (including L0,
"fails the tag") gets its own category's data/MC correction factor.

CONFIRMED 2026-08-18: this all-jets scope is the source of the CMS_ctag2d Down-variation
asymmetry (placeholder central=1.000/up_Total=3.000/down_Total=0.300 calibration values in
statistically-empty (flavor, WP) corners of the official JSON, amplified by ak.prod across
many jets). User sign-off to keep this scope regardless -- yield-comparability with the
corrector being replaced outweighs the smoother-but-incomparable one-jet alternative. Not a
bug; documented caveat only (see README_HcZZ.md "Known traps").
"""
import numpy as np
import awkward as ak
import correctionlib
from typing import Type
from coffea.analysis_tools import Weights
from analysis.corrections.utils import correction_files

CATS = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
CID = {n: i for i, n in enumerate(CATS)}
WP_ID = {"L0": 0, "C0": 40, "C1": 41, "C2": 42, "C3": 43, "C4": 44,
         "B0": 50, "B1": 51, "B2": 52, "B3": 53, "B4": 54}
CAT_TO_WP = np.array([WP_ID[c] for c in CATS], dtype=np.int64)

# Per-era PNet 2D-category boundaries in (pBplusC=HFvLF, pBvsC=BvC=1-CvB) space, i.e.
# {"x": (lo, hi), "y": (lo, hi)}. Source: HiggsDNA metaconditions Era2022_v1.json
# "HPC_ctag_WPs" block, cross-confirmed 2026-08-14 against
# b-hive_ttcc/utils/coffea_processors/lz4_hczz_processor.py::JET_TAG_BOUNDARIES_BY_ERA
# (byte-for-byte match). wp_id -> category: 0=L0, 40-44=C0-C4, 50-54=B0-B4.
BOUNDARIES_BY_ERA = {
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

CORRECTION_NAME = "ParticleNetAK4_pseudocontinuous"
NUIS_YEAR = {"2022preEE": "2022", "2022postEE": "2022",
             "2023preBPix": "2023", "2023postBPix": "2023"}
# single-nuisance source keys, matching the hww-analysis reference (decision there: 2026-07-23,
# Total; per-source decorrelation deferred to a whole-card decorrelation pass -- same deferral
# applies here, not yet revisited).
SYST_UP = "up_Total"
SYST_DN = "down_Total"
PT_BINS = (20.0001, 9999.0)


def _category_np(cvsl, cvsb, era):
    """Map (CvL, CvB) per jet to one of the 11 WP-code categories, using ERA-SPECIFIC
    boundaries (they differ meaningfully between 2022*/2023* -- see BOUNDARIES_BY_ERA).
    -1 = out of range (shouldn't happen: every era's boundaries tile [0,1]x[0,1])."""
    den = cvsl + cvsb * (1.0 - cvsl)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(den != 0, cvsl / den, np.nan)
    y = 1.0 - cvsb
    good = np.isfinite(x) & np.isfinite(y)
    cat = np.full(x.shape, -1, dtype=np.int64)
    boundaries = BOUNDARIES_BY_ERA[era]
    for name, bounds in boundaries.items():
        x_lo, x_hi = bounds["x"]
        y_lo, y_hi = bounds["y"]
        m = good & (x >= x_lo) & (x < x_hi) & (y >= y_lo) & (y < y_hi)
        cat[m] = CID[name]
    # closed upper corner (x=1, y=1) falls just outside every "< hi" box -- assign it to
    # whichever category owns the (x_hi, y_hi)=(1,1) corner (B4 always, per the tiling above).
    edge = good & (cat < 0) & (x >= boundaries["B4"]["x"][0]) & (y >= boundaries["B4"]["y"][0])
    cat[edge] = CID["B4"]
    return cat


class CTag2DCorrector:
    """
    2D pseudo-continuous c-tag corrector, applied to every jet in `events.selected_jets`
    (product across the event) -- same scope as ctag.py::CTagCorrector.

    Parameters:
    -----------
        events:
            Events collection (must carry `selected_jets` with btagPNetCvL/CvB/hadronFlavour)
        weights:
            Weights container from coffea.analysis_tools
        year:
            dataset year {2022preEE, 2022postEE, 2023preBPix, 2023postBPix}
        variation:
            if 'nominal' (default), add nominal + up/down variations; else nominal only
            (matches ctag.py's convention in this repo -- NOT hww-analysis's `shift is None`
            polarity, which is inverted)
    """

    def __init__(self, events, weights: Type[Weights], year: str, variation: str = "nominal") -> None:
        if year not in BOUNDARIES_BY_ERA:
            raise ValueError(
                f"CTag2DCorrector has no PNet category boundaries for year '{year}' "
                f"(only {list(BOUNDARIES_BY_ERA)} are wired -- 2024/UParTAK4 needs its own "
                f"boundary table, not yet added)."
            )
        self._year = year
        self._weights = weights
        self._variation = variation
        self._corr = correctionlib.CorrectionSet.from_file(
            correction_files["ctagging_2d"][year]
        )[CORRECTION_NAME]
        self._nuis = f"CMS_ctag2d_{NUIS_YEAR[year]}"

        jets = events.selected_jets
        self._nj = ak.num(jets)
        j = ak.flatten(jets)
        self._cvsl = ak.to_numpy(ak.fill_none(j.btagPNetCvL, np.nan)).astype(np.float64)
        self._cvsb = ak.to_numpy(ak.fill_none(j.btagPNetCvB, np.nan)).astype(np.float64)
        self._pt = ak.to_numpy(ak.fill_none(j.pt, np.nan)).astype(np.float64)
        flav = ak.fill_none(getattr(j, "hadronFlavour", ak.zeros_like(j.pt)), 0)
        self._flav = ak.to_numpy(flav).astype(np.int64)

    def _eval_flat(self, syst):
        n = len(self._flav)
        sf = np.ones(n, dtype=np.float64)
        cat = _category_np(self._cvsl, self._cvsb, self._year)
        wp = np.where(cat >= 0, CAT_TO_WP[np.clip(cat, 0, 10)], -1)
        ok = (wp >= 0) & np.isfinite(self._pt)
        if ok.any():
            sf[ok] = self._corr.evaluate(
                syst,
                self._flav[ok],
                wp[ok],
                np.zeros(ok.sum(), dtype=np.float64),  # abseta -- inclusive, per reference impl
                np.clip(self._pt[ok], *PT_BINS),
            )
        return sf

    def _eval_event(self, syst):
        """Per-jet SF -> per-event product (ak.prod over jets, matching ctag.py's
        tagged_sf/untagged_sf multiplicative combination)."""
        flat = self._eval_flat(syst)
        return ak.to_numpy(ak.prod(ak.unflatten(flat, self._nj), axis=1))

    def add_weights(self) -> None:
        sf_c = self._eval_event("central")
        if self._variation == "nominal":
            sf_up = self._eval_event(SYST_UP)
            sf_dn = self._eval_event(SYST_DN)
            r_up = np.where(sf_c != 0, sf_up / sf_c, 1.0)
            r_dn = np.where(sf_c != 0, sf_dn / sf_c, 1.0)
            self._weights.add(
                name=self._nuis,
                weight=sf_c,
                weightUp=sf_c * r_up,
                weightDown=sf_c * r_dn,
            )
        else:
            self._weights.add(name=self._nuis, weight=sf_c)
