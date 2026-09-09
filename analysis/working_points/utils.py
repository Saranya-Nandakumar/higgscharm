import numpy as np
import correctionlib
import awkward as ak
from analysis.filesets.utils import get_nano_version
from analysis.corrections.utils import correction_files


def get_ctag_mask(jets: ak.Array, year: str, wp: str):
    """
    Parameters:
    -----------
      jets: Jet collection
      year: {2016preVFP, 2016postVFP, 2017, 2018, 2022preEE, 2022postEE, 2023preBPix, 2023postBPix, 2024}
      wp: {loose, medium, tight}
    """
    nano_version = get_nano_version(year)
    if nano_version == "9":
        tagger = "deepjet"
    elif nano_version == "12":
        tagger = "pnet"
    elif nano_version == "15":
        tagger = "upart"

    wp_map = {"loose": "L", "medium": "M", "tight": "T"}
    tagger_map = {
        "deepjet": "deepJet_wp_values",
        "pnet": "particleNet_wp_values",
        "upart": "UParTAK4_wp_values",
    }

    cset = correctionlib.CorrectionSet.from_file(correction_files["ctagging"][year])
    ctag_wps_evaluator = cset[tagger_map[tagger]]
    cvsb_wp = ctag_wps_evaluator.evaluate(wp_map[wp], "CvB")
    cvsl_wp = ctag_wps_evaluator.evaluate(wp_map[wp], "CvL")

    if tagger == "deepjet":
        pass_ctag_wp = (jets.btagDeepFlavCvB > cvsb_wp) & (
            jets.btagDeepFlavCvL > cvsl_wp
        )
    elif tagger == "pnet":
        pass_ctag_wp = (jets.btagPNetCvB > cvsb_wp) & (jets.btagPNetCvL > cvsl_wp)
    elif tagger == "upart":
        pass_ctag_wp = (jets.btagUParTAK4CvB > cvsb_wp) & (
            jets.btagUParTAK4CvL > cvsl_wp
        )

    return pass_ctag_wp


def get_ctag_mask_2d(jets: ak.Array, year: str, wp: str = "loose"):
    """
    2D pseudo-continuous c-tag SELECTION mask -- jet passes if its (CvL, CvB)
    2D category is NOT L0 ("fails the tag" bin), using the SAME era-specific
    category boundaries (BOUNDARIES_BY_ERA) as CTag2DCorrector's SF weighting
    (analysis/corrections/ctag2d.py), so object selection and the c-tag SF
    correction are on a consistent 2D scheme end-to-end -- unlike
    get_ctag_mask() above, whose fixed 1D BTV Loose-WP cut (still the
    production/comparison default) is defined independently of the 2D
    category grid. Boundaries independently cross-checked 2026-09-09 against
    the public HiggsDNA metaconditions Era2022_v1.json "HPC_ctag_WPs" block on
    gitlab.cern.ch -- byte-for-byte match including the 2022preEE/postEE
    B1/B2 boundary quirk (0.933 vs 0.932).

    Parameters:
    -----------
      jets: Jet collection (jagged, per-event) -- must carry
            btagPNetCvL/btagPNetCvB (PNet only; this reuses ctag2d.py's
            category logic, which is not wired for deepjet/UParT eras).
      year: {2022preEE, 2022postEE, 2023preBPix, 2023postBPix} -- 2024/UParT
            not included, matching CTag2DCorrector's own year gating.
      wp: only "loose" is meaningful (= "not L0"). Kept as a parameter for
          call-signature compatibility with get_ctag_mask/jet_ctagging, not
          because other 2D-category selection cuts are implemented yet.
    """
    from analysis.corrections.ctag2d import _category_np, CID, BOUNDARIES_BY_ERA

    if wp != "loose":
        raise ValueError(
            f"get_ctag_mask_2d only supports wp='loose' (= not L0), got '{wp}'"
        )
    if year not in BOUNDARIES_BY_ERA:
        raise ValueError(
            f"get_ctag_mask_2d has no PNet 2D category boundaries for year "
            f"'{year}' (only {list(BOUNDARIES_BY_ERA)} are wired)."
        )

    n = ak.num(jets)
    cvsl = ak.to_numpy(
        ak.fill_none(ak.flatten(jets.btagPNetCvL), np.nan)
    ).astype(np.float64)
    cvsb = ak.to_numpy(
        ak.fill_none(ak.flatten(jets.btagPNetCvB), np.nan)
    ).astype(np.float64)
    cat = _category_np(cvsl, cvsb, year)
    pass_flat = cat != CID["L0"]  # NOT L0; unresolved category (-1) also fails
    return ak.unflatten(pass_flat, n)
