# Higgs + heavy-flavour composition uncertainty, ported 2026-08-18.
#
# SOURCE: HiggsDNA higgs_dna/systematics/event_weight_systematics.py ::
#   Higgs_plus_HF_syst (https://gitlab.cern.ch/cms-analysis/general/HiggsDNA);
#   scope per AN-23-102 section 7.1 / Table 16: "a conservative uncertainty of
#   heavy flavor modeling of ggH is assigned, 50% uncertainty on the
#   normalisation of the yield". HiggsDNA's own docstring warns "make sure you
#   apply it only on ggH or VBF samples".
#
# WHY THIS EXISTS: replaces a flat lnN on the whole pooled Other_Higgs group
# (ggH+VBF+WH+ZH+ttH+bbH) with a per-event weight that only touches events
# that actually contain a heavy-flavour gen jet. A flat lnN on the pooled
# group either over-penalises the non-ggH/VBF majority or has to be diluted
# to an average that doesn't reflect any single component correctly, and
# can't produce a real shape effect either way.
#
# SCOPE, deliberately narrow, matching the source exactly: ggH and VBF only.
# Does NOT cover WH/ZH/ttH/bbH (no heavy-flavour-modeling uncertainty
# documented for those in AN-23-102) or HPlusBottom (a different, private
# signal-adjacent process this analysis added on top of the pooled
# Other_Higgs group -- not something AN-23-102 covers, and not extended here
# without a separate physics justification).
import awkward as ak
import numpy as np

_HIGGS_PREFIXES = ("GluGluHtoZZ", "VBFHto")


def add_higgs_hf_weight(
    events,
    weights_container,
    dataset,
    flav="c",
    pt_min=25.0,
    eta_max=2.5,
    rel_unc=0.5,
):
    """Flat +-rel_unc on ggH/VBF events containing >=1 heavy-flavour GEN jet.

    Events with no such jet get exactly 1.0, so the nuisance is automatically
    confined to the phase space it is meant to cover.

    flav: "c" -> hadronFlavour == 4, "b" -> hadronFlavour == 5. This analysis's
    signal is H+c, so "c" is the relevant variant (matching the
    ggH+heavy-flavor-jets composition this search is most sensitive to).
    """
    if not dataset.startswith(_HIGGS_PREFIXES):
        return
    if "GenJet" not in events.fields:
        print("No GenJet in dataset, skip systematic: Higgs+HF composition")
        return

    flav_id = {"c": 4, "b": 5}.get(flav)
    if flav_id is None:
        raise ValueError("flav must be either 'b' or 'c'")

    try:
        gj = events.GenJet
        gj = gj[(gj.pt > pt_min) & (abs(gj.eta) < eta_max)]
        n_hf = ak.sum(gj.hadronFlavour == flav_id, axis=-1)

        has_hf = ak.to_numpy(n_hf > 0)
        up = np.where(has_hf, 1.0 + rel_unc, 1.0)
        down = np.where(has_hf, 1.0 - rel_unc, 1.0)

        weights_container.add(
            name=f"higgs_plus_{flav}",
            weight=np.ones(len(events)),
            weightUp=up,
            weightDown=down,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to build Higgs+HF weight, skipping systematic: {exc}")
