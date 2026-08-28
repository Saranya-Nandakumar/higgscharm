import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from typing import Type
from pathlib import Path
from coffea.analysis_tools import Weights
from analysis.filesets.utils import get_nano_version
from analysis.corrections.utils import correction_files
from analysis.working_points.utils import get_ctag_mask


class CTagCorrector:
    """
    CTag corrector class.

    Parameters:
    -----------
        events:
            Events collection
        weights:
            Weights container from coffea.analysis_tools
        worging_point:
            worging point {loose, medium, tight}
        year:
            dataset year {2022preEE, 2022postEE, 2023preBPix, 2023PostBPix, 2024}
        category:
            selection category name
        variation:
            if 'nominal' (default) add 'nominal', 'up' and 'down' variations to weights container
    """

    def __init__(
        self,
        events,
        weights: Type[Weights],
        worging_point: str,
        year: str,
        category: str = "base",
        variation: str = "nominal",
    ) -> None:
        self._year = year
        self._wp = worging_point
        self._weights = weights
        self._variation = variation

        # set tagger
        self._nano_version = get_nano_version(year)
        if self._nano_version == "9":
            tagger = "deepjet"
        elif self._nano_version == "12":
            tagger = "pnet"
        elif self._nano_version == "15":
            tagger = "upart"

        # check available ctag SFs
        self._wp_map = {"tight": "T", "medium": "M", "loose": "L"}
        if self._wp not in self._wp_map:
            raise ValueError(
                f"There are no available c-tag SFs for the working point. Please specify {list(self._wp_map.keys())}"
            )

        # check available c-tag efficiencies (ctag_eff_<wp>_<category>_<year>.coffea)
        ctag_eff_file = (
            Path.cwd()
            / "analysis"
            / "data"
            / "ctag_efficiencies"
            / f"ctag_eff_{self._wp}_{category}_{self._year}.coffea"
        )
        if not ctag_eff_file.exists():
            raise ValueError(
                f"There are no available c-tag efficiencies for tagger '{tagger}', year '{self._year}', and wp '{self._wp}'"
            )
        with importlib.resources.path(
            "analysis.data.ctag_efficiencies",
            f"ctag_eff_{self._wp}_{category}_{self._year}.coffea",
        ) as filename:
            self._efflookup = util.load(str(filename))

        # load correction set
        self._cset = correctionlib.CorrectionSet.from_file(
            correction_files["ctagging"][year]
        )

        # select b, c and light jets
        self._flavors = {"b": 5, "c": 4, "light": 0}
        self._c_jets = events.selected_jets[
            events.selected_jets.hadronFlavour == self._flavors["c"]
        ]
        self._b_jets = events.selected_jets[
            events.selected_jets.hadronFlavour == self._flavors["b"]
        ]
        self._light_jets = events.selected_jets[
            events.selected_jets.hadronFlavour == self._flavors["light"]
        ]
        self._jet_map = {
            "c": self._c_jets,
            "b": self._b_jets,
            "light": self._light_jets,
        }
        self._jet_pass_ctag = {
            "c": get_ctag_mask(self._jet_map["c"], self._year, self._wp),
            "b": get_ctag_mask(self._jet_map["b"], self._year, self._wp),
            "light": get_ctag_mask(self._jet_map["light"], self._year, self._wp),
        }
        self.var_naming_map = {
            "c": "CMS_ctag_c",
            "b": "CMS_ctag_b",
            "light": "CMS_ctag_light",
        }

    def add_ctag_weights(self, flavor: str) -> None:
        """
        Add c-tagging weights (nominal, up and down) to weights container for b, c or light jets
        """
        # efficiencies
        eff = self.efficiency(flavor=flavor)

        # mask with events that pass the ctag working point
        pass_ctag = self._jet_pass_ctag[flavor]

        # nominal scale factors
        ctag_sf = self.get_scale_factors(flavor=flavor, syst="central")

        # nominal weights
        ctag_weight = self.get_ctag_weight(eff, ctag_sf, pass_ctag)

        if self._variation == "nominal":
            # systematics
            # up and down scale factors
            ctag_sf_up = self.get_scale_factors(flavor=flavor, syst="up")
            ctag_sf_down = self.get_scale_factors(flavor=flavor, syst="down")
            ctag_weight_up = self.get_ctag_weight(eff, ctag_sf_up, pass_ctag)
            ctag_weight_down = self.get_ctag_weight(eff, ctag_sf_down, pass_ctag)
            # add weights to Weights container
            self._weights.add(
                name=self.var_naming_map[flavor],
                weight=ctag_weight,
                weightUp=ctag_weight_up,
                weightDown=ctag_weight_down,
            )
        else:
            self._weights.add(
                name=self.var_naming_map[flavor],
                weight=ctag_weight,
            )

    def efficiency(self, flavor: str, fill_value=1) -> ak.Array:
        """compute the c-tagging efficiency"""
        return self._efflookup(
            self._jet_map[flavor].pt,
            np.abs(self._jet_map[flavor].eta),
            self._jet_map[flavor].hadronFlavour,
        )

    def get_scale_factors(self, flavor: str, syst="central", fill_value=1) -> ak.Array:
        """
        compute c-taggging scale factors
        """
        return self.get_sf(flavor=flavor, syst=syst)

    def get_sf(self, flavor: str, syst: str = "central") -> ak.Array:
        """
        compute the c-tagging scale factors for b, c or light jets

        Parameters:
        -----------
            flavor:
                hadron flavor {b, c, light}
            syst:
                Name of the systematic {central, up, down}
        """
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(self._jet_map[flavor]), ak.num(self._jet_map[flavor])

        # get 'in-limits' jets
        jet_eta_mask = np.abs(j.eta) < 2.4
        in_jet_mask = jet_eta_mask
        in_jets = j.mask[in_jet_mask]

        # get jet transverse momentum, abs pseudorapidity and hadron flavour (replace None values with some 'in-limit' value)
        jets_pt = ak.fill_none(in_jets.pt, 0.0)
        jets_eta = ak.fill_none(np.abs(in_jets.eta), 0.0)
        jets_hadron_flavour = ak.fill_none(in_jets.hadronFlavour, self._flavors[flavor])

        # set keys to access correction set
        cset_keys = {
            "9": {
                "b": "deepJet_wp",
                "c": "deepJet_wp",
                "light": "deepJet_wp",
            },
            "12": {
                "b": "particleNet_tnp",
                "c": "particleNet_wc",
                "light": "particleNet_light",
            },
        }
        if self._nano_version == "9":
            # 'incl' for light jets, 'wcharm' for b/c jets
            method = "wcharm" if flavor != "light" else "incl"
            # The uncertainties are to be decorrelated between c jets ('wcharm'), b jets ('TnP') and light jets ('incl')
            if syst != "central":
                if flavor == "b":
                    method = "TnP"

            sf = self._cset[cset_keys[self._nano_version][flavor]].evaluate(
                syst,
                method,
                self._wp_map[self._wp],
                np.array(jets_hadron_flavour),
                np.array(jets_eta),
                np.array(jets_pt),
            )
        if self._nano_version == "12":
            if flavor == "c":
                if syst != "central":
                    syst += "_stat"

            sf = self._cset[cset_keys[self._nano_version][flavor]].evaluate(
                syst,
                self._wp_map[self._wp],
                np.array(jets_hadron_flavour),
                np.array(jets_eta),
                np.array(jets_pt),
            )
        sf = ak.where(in_jet_mask, sf, ak.ones_like(sf))
        return ak.unflatten(sf, nj)

    @staticmethod
    def get_ctag_weight(eff: ak.Array, sf: ak.Array, pass_ctag: ak.Array) -> ak.Array:
        """
        compute c-tagging weights

        Parameters:
        -----------
            eff:
                c-tagging efficiencies
            sf:
                jet's scale factors
            pass_ctag:
                mask with jets that pass the c-tagging working point
        """
        # tagged SF = SF * eff / eff = SF
        tagged_sf = ak.prod(sf.mask[pass_ctag], axis=-1)

        # untagged SF = (1 - SF * eff) / (1 - eff)
        # Clamp SF*eff to not exceed 1: standard BTV-POG mitigation for the
        # known numerical instability where a large SF variation (typically
        # the light-flavor mistag "up" uncertainty, which can be large in
        # some pT/eta bins) combined with efficiency pushes SF*eff above 1,
        # making the per-jet untagged weight negative. Confirmed 2026-08-03:
        # this flipped ~0.5-0.6% of qqZZ/Other_Higgs events' sign specifically
        # for CMS_ctag_lightUp (not _b or _c, whose SFs don't hit this regime).
        sf_eff = ak.where(sf * eff > 1, 1.0, sf * eff)
        untagged_sf = ak.prod(((1 - sf_eff) / (1 - eff)).mask[~pass_ctag], axis=-1)

        return ak.fill_none(tagged_sf * untagged_sf, 1.0)
