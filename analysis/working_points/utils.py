import correctionlib
import awkward as ak
import numpy as np
from analysis.filesets.utils import get_nano_version
from analysis.corrections.utils import correction_files


# MVA-based c-tagging boundaries from HcZZ training
# Derived from: law run TrainingTask --config hc_zzto4l_mw_training_enhanced
#               --dataset-version hcZZ_2era_new60 --model-name MLP_HcZZ_MW_Deep
#               --epochs 20 --training-version train_MLP_new
# Coordinate system:
#   x-axis (pBplusC): B + (1-B)*CvL  (combined b+c probability)
#   y-axis (pBvsC): 1.0 - CvB        (b vs c separation)
MVA_JET_TAG_BOUNDARIES = {
    "C4": {"x": (0.7339, 1.0000), "y": (0.0000, 0.0382)},  # High-purity charm
    "C3": {"x": (0.7339, 1.0000), "y": (0.0382, 0.1851)},  # Charm (medium-high)
    "C2": {"x": (0.7339, 1.0000), "y": (0.1851, 0.2688)},  # Charm (medium)
    "C1": {"x": (0.4, 0.7339), "y": (0.0000, 0.9000)},  # Charm (loose)
    "C0": {"x": (0.15, 0.4), "y": (0.0000, 0.9000)},  # Charm (very loose)
    "L0": {"x": (0.0000, 0.15), "y": (0.0000, 0.9000)},  # Light jets
    "B0": {"x": (0.7339, 1.0000), "y": (0.2688, 0.4057)},  # B-tag (loose)
    "B1": {"x": (0.7339, 1.0000), "y": (0.4057, 0.4068)},  # B-tag (medium-loose)
    "B2": {"x": (0.7339, 1.0000), "y": (0.4068, 0.6788)},  # B-tag (medium)
    "B3": {"x": (0.7339, 1.0000), "y": (0.6788, 0.8406)},  # B-tag (medium-tight)
    "B4": {"x": (0.7339, 1.0000), "y": (0.8406, 1.0000)},  # B-tag (tight)
}

# Mapping from MVA categories to c-tagging selection
# Any jets in C0-C4 categories are considered c-tagged
MVA_CHARM_CATEGORIES = ["C0", "C1", "C2", "C3", "C4"]
MVA_BTAG_CATEGORIES = ["B0", "B1", "B2", "B3", "B4"]
MVA_LIGHT_CATEGORIES = ["L0"]


def get_mva_coordinates(jets: ak.Array, year: str):
    """
    Compute the 2D MVA coordinates for jet tagging.

    Parameters:
    -----------
      jets: Jet collection
      year: Year string to determine which tagger to use

    Returns:
    --------
      pBplusC: Combined b+c probability (x-axis)
      pBvsC: B vs C separation (y-axis)
    """
    nano_version = get_nano_version(year)

    if nano_version == "9":
        # DeepJet - need to compute B from individual scores
        # For DeepJet, btagDeepFlavB is the b-probability
        B = jets.btagDeepFlavB
        CvB = jets.btagDeepFlavCvB
        CvL = jets.btagDeepFlavCvL
    elif nano_version == "12":
        # ParticleNet
        B = jets.btagPNetB
        CvB = jets.btagPNetCvB
        CvL = jets.btagPNetCvL
    elif nano_version == "15":
        # UParT
        B = jets.btagUParTAK4B
        CvB = jets.btagUParTAK4CvB
        CvL = jets.btagUParTAK4CvL
    else:
        raise ValueError(f"Unknown NanoAOD version: {nano_version}")

    # Compute 2D coordinates
    pBvsC = 1.0 - CvB  # y-axis
    pBplusC = B + (1 - B) * CvL  # x-axis

    # Clip to valid range [0, 1]
    pBplusC = ak.where(pBplusC < 0, 0, ak.where(pBplusC > 1, 1, pBplusC))
    pBvsC = ak.where(pBvsC < 0, 0, ak.where(pBvsC > 1, 1, pBvsC))

    return pBplusC, pBvsC


def get_mva_ctag_category_mask(jets: ak.Array, year: str, category: str):
    """
    Get mask for jets in a specific MVA category.

    Parameters:
    -----------
      jets: Jet collection
      year: Year string
      category: One of C0, C1, C2, C3, C4, B0, B1, B2, B3, B4, L0

    Returns:
    --------
      Boolean mask for jets in the specified category
    """
    if category not in MVA_JET_TAG_BOUNDARIES:
        raise ValueError(
            f"Invalid MVA category '{category}'. "
            f"Valid categories: {list(MVA_JET_TAG_BOUNDARIES.keys())}"
        )

    pBplusC, pBvsC = get_mva_coordinates(jets, year)
    bounds = MVA_JET_TAG_BOUNDARIES[category]

    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]

    # Use inclusive-exclusive logic to avoid overlaps
    mask = (
        (pBplusC >= x_min) & (pBplusC < x_max) &
        (pBvsC >= y_min) & (pBvsC < y_max)
    )

    return mask


def get_mva_ctag_mask(jets: ak.Array, year: str, categories: list = None):
    """
    Get mask for jets that pass MVA-based c-tagging selection.

    Parameters:
    -----------
      jets: Jet collection
      year: Year string
      categories: List of categories to include (default: all charm categories C0-C4)

    Returns:
    --------
      Boolean mask for c-tagged jets
    """
    if categories is None:
        categories = MVA_CHARM_CATEGORIES

    # Start with all False
    mask = ak.zeros_like(jets.pt, dtype=bool)

    # OR together all requested categories
    for cat in categories:
        cat_mask = get_mva_ctag_category_mask(jets, year, cat)
        mask = mask | cat_mask

    return mask


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
