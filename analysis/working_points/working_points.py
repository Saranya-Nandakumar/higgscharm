import numpy as np
import awkward as ak
from analysis.filesets.utils import get_nano_version
from analysis.working_points.utils import (
    get_ctag_mask,
    get_mva_ctag_mask,
    get_mva_ctag_category_mask,
    MVA_CHARM_CATEGORIES,
    MVA_BTAG_CATEGORIES,
    MVA_LIGHT_CATEGORIES,
)

    
class WorkingPoints:

    # -----------------------------------------------------------------
    # Electrons
    # -----------------------------------------------------------------
    def electron_id(self, events, wp):
        wps = {
            "wp80iso": (
                events.Electron.mvaFall17V2Iso_WP80
                if hasattr(events.Electron, "mvaFall17V2Iso_WP80")
                else events.Electron.mvaIso_WP80
            ),
            "wp90iso": (
                events.Electron.mvaFall17V2Iso_WP90
                if hasattr(events.Electron, "mvaFall17V2Iso_WP90")
                else events.Electron.mvaIso_WP90
            ),
            "fail": events.Electron.cutBased == 0,
            "veto": events.Electron.cutBased == 1,
            "loose": events.Electron.cutBased == 2,
            "medium": events.Electron.cutBased == 3,
            "tight": events.Electron.cutBased == 4,
        }

        # add electron BDT ID wp
        if hasattr(events.Electron, "mvaHZZIso"):
            bdt_id = (
                (
                    (np.abs(events.Electron.eta) < 0.8)
                    & (
                        (
                            (events.Electron.pt > 5)
                            & (events.Electron.pt < 10)
                            & (events.Electron.mvaHZZIso > 0.9044)
                        )
                        | (
                            (events.Electron.pt > 10)
                            & (events.Electron.mvaHZZIso > 0.1969)
                        )
                    )
                )
                | (
                    (np.abs(events.Electron.eta) > 0.8)
                    & (np.abs(events.Electron.eta) < 1.479)
                    & (
                        (
                            (events.Electron.pt > 5)
                            & (events.Electron.pt < 10)
                            & (events.Electron.mvaHZZIso > 0.9094)
                        )
                        | (
                            (events.Electron.pt > 10)
                            & (events.Electron.mvaHZZIso > 0.0759)
                        )
                    )
                )
                | (
                    (np.abs(events.Electron.eta) > 1.479)
                    & (
                        (
                            (events.Electron.pt > 5)
                            & (events.Electron.pt < 10)
                            & (events.Electron.mvaHZZIso > 0.9444)
                        )
                        | (
                            (events.Electron.pt > 10)
                            & (events.Electron.mvaHZZIso > -0.5169)
                        )
                    )
                )
            )
            wps["bdt"] = bdt_id

        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for electron ID working point. Please specify {list(wps.keys())}"
            )

        return wps[wp]

    # -----------------------------------------------------------------
    # Muons
    # -----------------------------------------------------------------
    def muon_id(self, events, wp):
        wps = {
            "loose": events.Muon.looseId,
            "medium": events.Muon.mediumId,
            "tight": events.Muon.tightId,
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value for muon ID working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def muon_iso(self, events, wp):
        wps = {
            "loose": (
                events.Muon.pfRelIso04_all < 0.25
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.25
            ),
            "medium": (
                events.Muon.pfRelIso04_all < 0.20
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.20
            ),
            "tight": (
                events.Muon.pfRelIso04_all < 0.15
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.15
            ),
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for muon ISO working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    # -----------------------------------------------------------------
    # Jets
    # -----------------------------------------------------------------
    def jet_id(self, events, year, wp):
        # Run 3 NanoAODs have a bug in jetId
        # Implement fix from:
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetID13p6TeV#nanoAOD_Flags
        nano_version = get_nano_version(year)
        if nano_version == "15":
            barrel = (
                (events.Jet.neHEF < 0.99)
                & (events.Jet.neEmEF < 0.9)
                & (events.Jet.chMultiplicity + events.Jet.neMultiplicity > 1)
                & (events.Jet.chHEF > 0.01)
                & (events.Jet.chMultiplicity > 0)
            )
            t1 = (events.Jet.neHEF < 0.9) & (events.Jet.neEmEF < 0.99)
            t2 = events.Jet.neHEF < 0.99
            endcap = (events.Jet.neMultiplicity >= 2) & (events.Jet.neEmEF < 0.4)

            jetid_tight = ak.where(
                abs(events.Jet.eta) <= 2.6,
                barrel,
                ak.where(
                    (abs(events.Jet.eta) > 2.6) & (abs(events.Jet.eta) <= 2.7),
                    t1,
                    ak.where(
                        (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                        t2,
                        ak.where(
                            (abs(events.Jet.eta) > 3.0),
                            endcap,
                            ak.zeros_like(events.Jet.pt, dtype=bool),
                        ),
                    ),
                ),
            )
            jetid_tightlepveto = ak.where(
                np.abs(events.Jet.eta) <= 2.7,
                jetid_tight & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
                jetid_tight,
            )
        elif nano_version == "12":
            jetid_tight = ak.where(
                np.abs(events.Jet.eta) <= 2.7,
                (events.Jet.jetId >= 2)
                & (events.Jet.muEF < 0.8)
                & (events.Jet.chEmEF < 0.8),
                ak.where(
                    (abs(events.Jet.eta) > 2.7) & (abs(events.Jet.eta) <= 3.0),
                    (events.Jet.jetId >= 2) & (events.Jet.neHEF < 0.99),
                    ak.where(
                        (abs(events.Jet.eta) > 3.0),
                        (events.Jet.jetId & (1 << 1)) & (events.Jet.neEmEF < 0.4),
                        ak.zeros_like(events.Jet.pt, dtype=bool),
                    ),
                ),
            )
            jetid_tightlepveto = ak.where(
                np.abs(events.Jet.eta) <= 2.7,
                jetid_tight & (events.Jet.muEF < 0.8) & (events.Jet.chEmEF < 0.8),
                jetid_tight,
            )
        elif nano_version == "9":
            jetid_tight = events.Jet.jetId == 2
            jetid_tightlepveto = events.Jet.jetId == 6

        wps = {
            "tight": ak.values_astype(jetid_tight, bool),
            "tightlepveto": ak.values_astype(jetid_tightlepveto, bool),
        }
        if wp not in wps:
            raise ValueError(
                f"Invalid value {wp} for jet ID working point. Please specify {list(wps.keys())}"
            )
        return wps[wp]

    def jet_pileup_id(self, events, year, wp):
        nano_version = get_nano_version(year)
        if nano_version == "9":
            wps = {
                "2016preVFP": {
                    "loose": events.Jet.puId == 1,
                    "medium": events.Jet.puId == 3,
                    "tight": events.Jet.puId == 7,
                },
                "2016postVFP": {
                    "loose": events.Jet.puId == 1,
                    "medium": events.Jet.puId == 3,
                    "tight": events.Jet.puId == 7,
                },
                "2017": {
                    "loose": events.Jet.puId == 4,
                    "medium": events.Jet.puId == 6,
                    "tight": events.Jet.puId == 7,
                },
                "2018": {
                    "loose": events.Jet.puId == 4,
                    "medium": events.Jet.puId == 6,
                    "tight": events.Jet.puId == 7,
                },
            }
            if wp not in wps[year]:
                raise ValueError(
                    f"Invalid value {wp} for jet pileup ID working point. Please specify {list(wps[year].keys())}"
                )
            # break up selection for low and high pT jets
            # to apply jets_pileup only to jets with pT < 50 GeV
            return ak.where(
                events.Jet.pt < 50,
                wps[year][wp],
                events.Jet.pt >= 50,
            )
        else:
            return np.ones_like(events.Jet.pt, dtype=bool)

    def jet_ctagging(self, events, wp, year, use_mva_boundaries=False, mva_categories=None):
        """
        Apply c-tagging selection to jets.

        Parameters:
        -----------
          events: Event collection
          wp: Working point (loose, medium, tight) - ignored if use_mva_boundaries=True
          year: Year string
          use_mva_boundaries: If True, use MVA-derived boundaries instead of standard WPs
          mva_categories: List of MVA categories to include when use_mva_boundaries=True
                          Default: all charm categories (C0-C4)
                          Options: C0, C1, C2, C3, C4 (charm), B0-B4 (b-tag), L0 (light)

        Returns:
        --------
          Boolean mask for c-tagged jets
        """
        if use_mva_boundaries:
            return get_mva_ctag_mask(events.Jet, year, categories=mva_categories)
        else:
            return get_ctag_mask(events.Jet, year, wp)

    def jet_mva_category(self, events, year, category):
        """
        Get mask for jets in a specific MVA tagging category.

        Parameters:
        -----------
          events: Event collection
          year: Year string
          category: MVA category (C0-C4, B0-B4, L0)

        Returns:
        --------
          Boolean mask for jets in the specified category
        """
        return get_mva_ctag_category_mask(events.Jet, year, category)