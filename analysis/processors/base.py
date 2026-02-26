import numpy as np
import awkward as ak
from copy import deepcopy
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.vector import LorentzVector
from analysis.utils import dump_lumi, dump_pa_table
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.histograms import HistBuilder, fill_histograms
from analysis.corrections.jetvetomaps import apply_jetvetomaps
from analysis.corrections.correction_manager import (
    object_corrector_manager,
    weight_manager,
)
from analysis.selections import (
    ObjectSelector,
    get_lumi_mask,
    get_trigger_mask,
    get_zzto4l_trigger_mask,
    get_metfilters_mask,
    get_trigger_match_mask,
    get_stitching_mask,
)

# Optional MVA inference
try:
    from analysis.utils.mva_inference import MVAInference
    HAS_MVA = True
except ImportError:
    HAS_MVA = False
    MVAInference = None

NanoAODSchema.warn_missing_crossrefs = False


class BaseProcessor(processor.ProcessorABC):
    def __init__(
        self,
        workflow: str,
        year: str,
        output_format: str,
        output_location: str,
        use_6class: bool = False,
        mva_model_path: str = None,
        use_pytorch: bool = False,
        mva_hidden_dim: int = 64,
        mva_num_layers: int = 2,
        model_type: str = "mlp",
        mass_window: bool = False,
    ):
        self.year = year
        self.workflow = workflow
        self.output_format = output_format
        self.output_location = output_location
        self.use_6class = use_6class  # True for 6-class (separate gg->ZZ/qq->ZZ), False for 5-class
        self.mva_model_path = mva_model_path
        self.use_pytorch = use_pytorch
        self.mva_hidden_dim = mva_hidden_dim
        self.mva_num_layers = mva_num_layers
        self.model_type = model_type  # 'mlp' or 'part'
        self.mass_window = mass_window  # Apply 100 < m4l < 150 GeV cut

        config_builder = WorkflowConfigBuilder(workflow)
        self.workflow_config = config_builder.build_workflow_config()
        self.histogram_config = self.workflow_config.histogram_config
        self.histograms = HistBuilder(self.workflow_config).build_histogram()

        # Note: MVA model is loaded lazily in process() to avoid pickling issues
        # with multiprocessing (ONNX InferenceSession cannot be pickled)
        self._mva_initialized = False
        self._mva = None

    @property
    def mva(self):
        """Lazy initialization of MVA model to avoid pickling issues."""
        if not self._mva_initialized:
            self._mva_initialized = True
            if self.mva_model_path is not None and HAS_MVA:
                try:
                    self._mva = MVAInference(
                        model_path=self.mva_model_path,
                        use_6class=self.use_6class,
                        use_pytorch=self.use_pytorch,
                        hidden_dim=self.mva_hidden_dim,
                        num_layers=self.mva_num_layers,
                        model_type=self.model_type,
                    )
                    model_desc = f"PyTorch ({self.model_type.upper()})" if self.use_pytorch else "ONNX"
                    mass_window_str = " [mass window: 100-150 GeV]" if self.mass_window else ""
                    print(f"MVA inference enabled ({model_desc}){mass_window_str}: {self.mva_model_path}")
                except Exception as e:
                    print(f"Warning: Failed to load MVA model: {e}")
                    self._mva = None
            elif self.mva_model_path is not None:
                print("Warning: MVA inference requested but required libraries not installed")
        return self._mva

    def add_cutflow(
        self, events, output, objects, selection_manager, weight_manager, dataset
    ):
        sumw = ak.sum(events.genWeight) if hasattr(events, "genWeight") else len(events)
        for category, category_cuts in self.workflow_config.event_selection[
            "categories"
        ].items():
            output["metadata"].update({category: {"cutflow": {"initial": sumw}}})
            selections = []
            for cut_name in category_cuts:
                selections.append(cut_name)
                current_selection = selection_manager.all(*selections)
                if ak.sum(current_selection) != 0:
                    """
                    pruned_ev_cutflow = events[current_selection]
                    for obj in objects:
                        pruned_ev_cutflow[f"selected_{obj}"] = objects[obj][
                            current_selection
                        ]
                    weights_container_cutflow = weight_manager(
                        pruned_ev=pruned_ev_cutflow,
                        year=self.year,
                        workflow_config=self.workflow_config,
                        variation="nominal",
                        dataset=dataset,
                    )
                    output["metadata"][category]["cutflow"][cut_name] = ak.sum(
                        weights_container_cutflow.weight()
                    )
                    """
                    sumw_cutflow = (
                        ak.sum(events.genWeight[current_selection])
                        if hasattr(events, "genWeight")
                        else len(events[current_selection])
                    )
                    output["metadata"][category]["cutflow"][cut_name] = sumw_cutflow
                else:
                    output["metadata"][category]["cutflow"][cut_name] = 0

    def process(self, events):
        year = self.year
        dataset = events.metadata["dataset"]

        object_selections = self.workflow_config.object_selection
        event_selection = self.workflow_config.event_selection
        hlt_paths = event_selection["hlt_paths"]
        histograms = deepcopy(self.histograms)

        if self.workflow_config.corrections_config["objects"]:
            if "jet_vetomaps" in self.workflow_config.corrections_config["objects"]:
                events = apply_jetvetomaps(events, year)

        # check if dataset is MC or Data
        is_mc = hasattr(events, "genWeight")
        if not is_mc:
            events["Jet", "hadronFlavour"] = ak.zeros_like(events.Jet.pt)

        # initialize output dictionary
        output = {}

        # initialize metadata info with sumw before selection
        output["metadata"] = {}
        sumw = ak.sum(events.genWeight) if is_mc else len(events)
        output["metadata"].update({"sumw": sumw})

        # --------------------------------------------------------------
        # Object corrections
        # --------------------------------------------------------------
        object_corrector_manager(
            events=events,
            year=year,
            dataset=dataset,
            workflow_config=self.workflow_config,
        )
        # --------------------------------------------------------------
        # Object selection
        # --------------------------------------------------------------
        object_selector = ObjectSelector(object_selections, year)
        objects = object_selector.select_objects(events)

        # --------------------------------------------------------------
        # Event selection
        # --------------------------------------------------------------
        if not is_mc:
            # save (run, luminosityBlock) pairs to metadata
            lumi_mask = eval(event_selection["selections"]["lumimask"])
            dump_lumi(events[lumi_mask], output)

        # initialize selection manager
        selection_manager = PackedSelection()
        # add all selections to selector manager
        for selection, mask in event_selection["selections"].items():
            selection_manager.add(selection, eval(mask))

        # add cutflow to metadata
        self.add_cutflow(
            events, output, objects, selection_manager, weight_manager, dataset
        )
        # --------------------------------------------------------------
        # Histogram filling / array dumping
        # --------------------------------------------------------------
        categories = event_selection["categories"]
        for category, category_cuts in categories.items():
            # get selection mask by category
            category_mask = selection_manager.all(*category_cuts)
            nevents_after = ak.sum(category_mask)
            if nevents_after > 0:
                # get pruned events
                pruned_ev = events[category_mask]
                # add each selected object to 'pruned_ev' as a new field
                for obj in objects:
                    pruned_ev[f"selected_{obj}"] = objects[obj][category_mask]
                # get weights container
                weights_container = weight_manager(
                    pruned_ev=pruned_ev,
                    year=year,
                    dataset=dataset,
                    workflow_config=self.workflow_config,
                )
                # save number of events after selection to metadata
                weighted_final_nevents = ak.sum(weights_container.weight())
                output["metadata"][category].update(
                    {
                        "weighted_final_nevents": weighted_final_nevents,
                        "raw_final_nevents": nevents_after,
                    }
                )
                # get analysis variables map
                variables_map = {}
                for variable, axis in self.histogram_config.axes.items():
                    variables_map[variable] = eval(axis.expression)[category_mask]

                # --- Truth label mapping for H+c MVA analysis ---
                process_name = dataset.lower() if isinstance(dataset, str) else str(dataset)
                n_events = len(events.event[category_mask]) if hasattr(events, "event") else len(events[category_mask])
                def label_array(match):
                    return np.ones(n_events, dtype=np.int8) if match else np.zeros(n_events, dtype=np.int8)

                # MVA truth labels - configurable between 5-class and 6-class
                # Set use_6class=True to separate gg->ZZ and qq->ZZ, False to combine them
                use_6class = getattr(self, 'use_6class', False)

                if use_6class:
                    # 6-category MVA truth labels (separate gg->ZZ and qq->ZZ)
                    truth_labels = ["is_Signal", "is_HPlusB", "is_ggZZ", "is_qqZZ", "is_Other", "is_ggH"]
                else:
                    # 5-category MVA truth labels (combined gg->ZZ and qq->ZZ)
                    truth_labels = ["is_qqgg_toZZ", "is_ggH", "is_Signal", "is_HPlusB", "is_Other"]

                for label in truth_labels:
                    variables_map[label] = label_array(False)

                # Determine the correct label based on dataset name
                # NOTE: Order matters! Check more specific patterns first.
                # "glugluhtozz" must come before "zzto4l" because GluGluHtoZZto4L contains both substrings
                if "glugluhtozz" in process_name:
                    # ggH -> ZZ -> 4l (GluGluHtoZZto4L)
                    variables_map["is_ggH"] = label_array(True)
                elif any(s in process_name for s in ["gluglutocontinto2zto", "gluglucontinto"]):
                    # gg->ZZ continuum (GluGluToContinto*, GluGlutoContinto*)
                    if use_6class:
                        variables_map["is_ggZZ"] = label_array(True)
                    else:
                        variables_map["is_qqgg_toZZ"] = label_array(True)
                elif "zzto4l" in process_name:
                    # qq->ZZ (ZZto4L)
                    if use_6class:
                        variables_map["is_qqZZ"] = label_array(True)
                    else:
                        variables_map["is_qqgg_toZZ"] = label_array(True)
                elif any(s in process_name for s in ["somesmsignal", "smsignal"]):
                    # Signal (SomeSMSignal with c-jet via hplusc workflow)
                    variables_map["is_Signal"] = label_array(True)
                elif "hb" in process_name:
                    # H+b MC sample
                    variables_map["is_HPlusB"] = label_array(True)
                else:
                    # Everything else (other Higgs, DY, tt, WZ, diboson, EW, etc.)
                    variables_map["is_Other"] = label_array(True)

                if self.output_format == "coffea":
                    fill_histograms(
                        histogram_config=self.histogram_config,
                        weights_container=weights_container,
                        variables_map=variables_map,
                        histograms=histograms,
                        variation="nominal",
                        category=category,
                        is_mc=is_mc,
                        flow=True,
                    )
                elif self.output_format == "parquet":
                    # add weights to variables map
                    if is_mc:
                        variations = ["nominal"] + list(weights_container.variations)
                        for variation in variations:
                            if variation == "nominal":
                                variables_map[f"weight_nominal"] = (
                                    weights_container.weight()
                                )
                                for (
                                    partial_weight
                                ) in weights_container.weightStatistics:
                                    variables_map[f"weight_{partial_weight}"] = (
                                        weights_container.partial_weight(
                                            include=[partial_weight]
                                        )
                                    )
                            else:
                                variables_map[f"weight_{variation}"] = (
                                    weights_container.weight(modifier=variation)
                                )

                    # ============================================================
                    # Add MVA training features for H+c analysis
                    # ============================================================
                    best_zz = pruned_ev['selected_best_zzcandidate']
                    selected_jets = pruned_ev['selected_jets']

                    # --- Global features ---
                    variables_map['n_jet'] = ak.num(selected_jets)

                    # 4-lepton system kinematics (Higgs candidate)
                    zz_p4 = best_zz.z1.p4 + best_zz.z2.p4
                    variables_map['m4l'] = zz_p4.mass
                    variables_map['pT_4l'] = zz_p4.pt
                    variables_map['eta_4l'] = zz_p4.eta
                    variables_map['phi_4l'] = zz_p4.phi

                    # Z candidate masses and pT
                    variables_map['m_Z1'] = best_zz.z1.p4.mass
                    variables_map['m_Z2'] = best_zz.z2.p4.mass
                    variables_map['pT_Z1'] = best_zz.z1.p4.pt
                    variables_map['pT_Z2'] = best_zz.z2.p4.pt
                    # Z candidate eta and phi (for ParT model 4-vectors)
                    variables_map['eta_Z1'] = best_zz.z1.p4.eta
                    variables_map['phi_Z1'] = best_zz.z1.p4.phi
                    variables_map['eta_Z2'] = best_zz.z2.p4.eta
                    variables_map['phi_Z2'] = best_zz.z2.p4.phi

                    # deltaR between Z candidates
                    variables_map['deltaR_ZZ'] = best_zz.z1.p4.delta_r(best_zz.z2.p4)

                    # ZZ flavor: 0=4mu, 1=4e, 2=2mu2e, 3=2e2mu
                    z1_is_muon = np.abs(best_zz.z1.l1.pdgId) == 13
                    z2_is_muon = np.abs(best_zz.z2.l1.pdgId) == 13
                    zz_flavor = ak.where(z1_is_muon & z2_is_muon, 0,
                                ak.where(~z1_is_muon & ~z2_is_muon, 1,
                                ak.where(z1_is_muon & ~z2_is_muon, 2, 3)))
                    variables_map['zz_flavor'] = zz_flavor

                    # Jet HT (scalar sum of jet pT)
                    variables_map['Jet_HT'] = ak.sum(selected_jets.pt, axis=1)

                    # Number of leptons (always 4 in this selection)
                    variables_map['n_lepton'] = ak.ones_like(variables_map['m4l']) * 4

                    # --- Jet features as jagged arrays (for b-hive) ---
                    # These will be padded by b-hive processor to n_cpf_candidates
                    variables_map['jet_pt'] = selected_jets.pt
                    variables_map['jet_eta'] = selected_jets.eta
                    variables_map['jet_phi'] = selected_jets.phi
                    variables_map['jet_mass'] = selected_jets.mass
                    variables_map['jet_btagPNetB'] = selected_jets.btagPNetB
                    variables_map['jet_btagPNetCvL'] = selected_jets.btagPNetCvL
                    variables_map['jet_btagPNetCvB'] = selected_jets.btagPNetCvB
                    variables_map['jet_btagPNetQvG'] = selected_jets.btagPNetQvG
                    variables_map['jet_btagRobustParTAK4B'] = selected_jets.btagRobustParTAK4B
                    variables_map['jet_btagRobustParTAK4CvB'] = selected_jets.btagRobustParTAK4CvB
                    variables_map['jet_hadronFlavour'] = selected_jets.hadronFlavour

                    # --- Leading/subleading jet PNet scores ---
                    # Pad jets to 2 for leading/subleading extraction
                    jets_padded = ak.pad_none(selected_jets, 2, clip=True)
                    variables_map['leadingjet_pnet_cvsb'] = ak.fill_none(jets_padded[:, 0].btagPNetCvB, 0.0)
                    variables_map['leadingjet_pnet_cvsl'] = ak.fill_none(jets_padded[:, 0].btagPNetCvL, 0.0)
                    variables_map['subleadingjet_pnet_cvsb'] = ak.fill_none(jets_padded[:, 1].btagPNetCvB, 0.0)
                    variables_map['subleadingjet_pnet_cvsl'] = ak.fill_none(jets_padded[:, 1].btagPNetCvL, 0.0)

                    # --- C-tag categories for leading jet ---
                    # Compute which MVA category (C0-C4, B0-B4, L0) the leading jet falls into
                    from analysis.working_points.utils import MVA_JET_TAG_BOUNDARIES

                    lead_B = ak.fill_none(jets_padded[:, 0].btagPNetB, 0.0)
                    lead_CvB = ak.fill_none(jets_padded[:, 0].btagPNetCvB, 0.0)
                    lead_CvL = ak.fill_none(jets_padded[:, 0].btagPNetCvL, 0.0)

                    # Compute 2D coordinates
                    lead_pBvsC = 1.0 - lead_CvB
                    lead_pBplusC = lead_B + (1.0 - lead_B) * lead_CvL
                    lead_pBplusC = np.clip(ak.to_numpy(lead_pBplusC), 0, 1)
                    lead_pBvsC = np.clip(ak.to_numpy(lead_pBvsC), 0, 1)

                    # Determine category for each event
                    CATEGORY_ORDER = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
                    n_ev = len(lead_pBplusC)
                    leadjet_category = np.full(n_ev, -1, dtype=np.int32)  # -1 = no category

                    for cat_idx, cat_name in enumerate(CATEGORY_ORDER):
                        bounds = MVA_JET_TAG_BOUNDARIES[cat_name]
                        x_min, x_max = bounds["x"]
                        y_min, y_max = bounds["y"]
                        mask = (
                            (lead_pBplusC >= x_min) & (lead_pBplusC < x_max) &
                            (lead_pBvsC >= y_min) & (lead_pBvsC < y_max) &
                            (leadjet_category == -1)  # Only assign if not already assigned
                        )
                        leadjet_category[mask] = cat_idx

                    variables_map['leadjet_ctag_category'] = leadjet_category

                    # Also add boolean flags for each category (useful for selection)
                    for cat_idx, cat_name in enumerate(CATEGORY_ORDER):
                        variables_map[f'leadjet_is_{cat_name}'] = (leadjet_category == cat_idx).astype(np.int8)

                    # --- Z boson features as arrays (for b-hive vtx_candidates) ---
                    # Stack Z1 and Z2 into length-2 arrays
                    variables_map['z_pt'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(best_zz.z1.p4.pt, 0.0)),
                        ak.to_numpy(ak.fill_none(best_zz.z2.p4.pt, 0.0))
                    ])
                    variables_map['z_eta'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(best_zz.z1.p4.eta, 0.0)),
                        ak.to_numpy(ak.fill_none(best_zz.z2.p4.eta, 0.0))
                    ])
                    variables_map['z_phi'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(best_zz.z1.p4.phi, 0.0)),
                        ak.to_numpy(ak.fill_none(best_zz.z2.p4.phi, 0.0))
                    ])
                    variables_map['z_mass'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(best_zz.z1.p4.mass, 0.0)),
                        ak.to_numpy(ak.fill_none(best_zz.z2.p4.mass, 0.0))
                    ])

                    # --- Lepton features as jagged arrays (for b-hive) ---
                    # Collect all 4 leptons from the best ZZ candidate into arrays
                    l1_z1 = best_zz.z1.l1
                    l2_z1 = best_zz.z1.l2
                    l1_z2 = best_zz.z2.l1
                    l2_z2 = best_zz.z2.l2

                    # Stack leptons into jagged arrays (4 leptons per event)
                    # Use ak.fill_none to convert optional types to simple types for PyArrow compatibility
                    variables_map['lepton_charge'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.charge, 0)),
                        ak.to_numpy(ak.fill_none(l2_z1.charge, 0)),
                        ak.to_numpy(ak.fill_none(l1_z2.charge, 0)),
                        ak.to_numpy(ak.fill_none(l2_z2.charge, 0))
                    ])
                    variables_map['lepton_pdgId'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.pdgId, 0)),
                        ak.to_numpy(ak.fill_none(l2_z1.pdgId, 0)),
                        ak.to_numpy(ak.fill_none(l1_z2.pdgId, 0)),
                        ak.to_numpy(ak.fill_none(l2_z2.pdgId, 0))
                    ])
                    variables_map['lepton_sip3d'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.sip3d, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z1.sip3d, 0.0)),
                        ak.to_numpy(ak.fill_none(l1_z2.sip3d, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z2.sip3d, 0.0))
                    ])
                    variables_map['lepton_pfRelIso03_all'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.pfRelIso03_all, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z1.pfRelIso03_all, 0.0)),
                        ak.to_numpy(ak.fill_none(l1_z2.pfRelIso03_all, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z2.pfRelIso03_all, 0.0))
                    ])
                    variables_map['lepton_pt'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.pt, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z1.pt, 0.0)),
                        ak.to_numpy(ak.fill_none(l1_z2.pt, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z2.pt, 0.0))
                    ])
                    variables_map['lepton_eta'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.eta, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z1.eta, 0.0)),
                        ak.to_numpy(ak.fill_none(l1_z2.eta, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z2.eta, 0.0))
                    ])
                    variables_map['lepton_phi'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.phi, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z1.phi, 0.0)),
                        ak.to_numpy(ak.fill_none(l1_z2.phi, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z2.phi, 0.0))
                    ])
                    variables_map['lepton_mass'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.mass, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z1.mass, 0.0)),
                        ak.to_numpy(ak.fill_none(l1_z2.mass, 0.0)),
                        ak.to_numpy(ak.fill_none(l2_z2.mass, 0.0))
                    ])
                    # Lepton ID arrays (for npf_candidates in b-hive)
                    variables_map['lepton_is_tight'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.is_tight, False)),
                        ak.to_numpy(ak.fill_none(l2_z1.is_tight, False)),
                        ak.to_numpy(ak.fill_none(l1_z2.is_tight, False)),
                        ak.to_numpy(ak.fill_none(l2_z2.is_tight, False))
                    ])
                    variables_map['lepton_mvaHZZIso'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.mvaHZZIso, -999.0)),
                        ak.to_numpy(ak.fill_none(l2_z1.mvaHZZIso, -999.0)),
                        ak.to_numpy(ak.fill_none(l1_z2.mvaHZZIso, -999.0)),
                        ak.to_numpy(ak.fill_none(l2_z2.mvaHZZIso, -999.0))
                    ])
                    variables_map['lepton_isPFcand'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.isPFcand, False)),
                        ak.to_numpy(ak.fill_none(l2_z1.isPFcand, False)),
                        ak.to_numpy(ak.fill_none(l1_z2.isPFcand, False)),
                        ak.to_numpy(ak.fill_none(l2_z2.isPFcand, False))
                    ])
                    variables_map['lepton_highPtId'] = np.column_stack([
                        ak.to_numpy(ak.fill_none(l1_z1.highPtId, 0)),
                        ak.to_numpy(ak.fill_none(l2_z1.highPtId, 0)),
                        ak.to_numpy(ak.fill_none(l1_z2.highPtId, 0)),
                        ak.to_numpy(ak.fill_none(l2_z2.highPtId, 0))
                    ])

                    # ============================================================
                    # End MVA features
                    # ============================================================

                    # ============================================================
                    # MVA Inference (if model loaded)
                    # ============================================================
                    if self.mva is not None:
                        try:
                            n_ev = len(pruned_ev)

                            # Apply mass window cut if enabled
                            if self.mass_window:
                                # Compute m4l for mass window selection
                                m4l_vals = variables_map.get('m4l', None)
                                if m4l_vals is None:
                                    # Compute m4l if not already in variables_map
                                    z1_p4 = best_zz.z1.p4
                                    z2_p4 = best_zz.z2.p4
                                    m4l_vals = ak.to_numpy(ak.fill_none((z1_p4 + z2_p4).mass, 0))

                                # Ensure m4l_vals is 1D (fix for shape (n,1) vs (n,) issue)
                                m4l_vals = np.asarray(m4l_vals).flatten()

                                # Mass window: 100 < m4l < 150 GeV
                                mass_window_mask = (m4l_vals >= 100) & (m4l_vals < 150)
                                n_in_window = np.sum(mass_window_mask)

                                if n_in_window > 0:
                                    # Filter events and objects for MVA inference
                                    pruned_ev_mva = pruned_ev[mass_window_mask]
                                    best_zz_mva = best_zz[mass_window_mask]
                                    selected_jets_mva = selected_jets[mass_window_mask]

                                    objects_for_mva = {
                                        'best_zzcandidate': best_zz_mva,
                                        'jets': selected_jets_mva,
                                        'leptons': None,
                                    }
                                    mva_result = self.mva.predict(pruned_ev_mva, objects_for_mva)

                                    # Initialize full arrays with defaults
                                    signal_score_full = np.full(n_ev, -1.0, dtype=np.float32)
                                    class_pred_full = np.full(n_ev, -1, dtype=np.int32)

                                    # Fill in values for events in mass window
                                    signal_score = mva_result['signal_score']
                                    if signal_score.ndim > 1:
                                        signal_score = signal_score.flatten()
                                    signal_score_full[mass_window_mask] = signal_score

                                    class_pred = mva_result['class_prediction']
                                    if hasattr(class_pred, 'flatten') and class_pred.ndim > 1:
                                        class_pred = class_pred.flatten()
                                    class_pred_full[mass_window_mask] = class_pred

                                    variables_map['mva_signal_score'] = signal_score_full
                                    variables_map['mva_class_prediction'] = class_pred_full

                                    # Add all class probabilities
                                    scores = mva_result['scores']
                                    for i, class_name in enumerate(mva_result['class_names']):
                                        score_full = np.full(n_ev, -1.0, dtype=np.float32)
                                        if scores.ndim == 1:
                                            score_full[mass_window_mask] = scores
                                        else:
                                            if i < scores.shape[1]:
                                                score_full[mass_window_mask] = scores[:, i]
                                        variables_map[f'mva_score_{class_name}'] = score_full
                                else:
                                    # No events in mass window
                                    variables_map['mva_signal_score'] = np.full(n_ev, -1.0, dtype=np.float32)
                                    variables_map['mva_class_prediction'] = np.full(n_ev, -1, dtype=np.int32)
                            else:
                                # No mass window cut - run on all events
                                objects_for_mva = {
                                    'best_zzcandidate': best_zz,
                                    'jets': selected_jets,
                                    'leptons': None,
                                }
                                mva_result = self.mva.predict(pruned_ev, objects_for_mva)

                                signal_score = mva_result['signal_score']
                                if signal_score.ndim > 1:
                                    signal_score = signal_score.flatten()
                                variables_map['mva_signal_score'] = signal_score

                                class_pred = mva_result['class_prediction']
                                if hasattr(class_pred, 'flatten') and class_pred.ndim > 1:
                                    class_pred = class_pred.flatten()
                                variables_map['mva_class_prediction'] = class_pred

                                # Add all class probabilities
                                scores = mva_result['scores']
                                for i, class_name in enumerate(mva_result['class_names']):
                                    if scores.ndim == 1:
                                        variables_map[f'mva_score_{class_name}'] = scores
                                    else:
                                        if i < scores.shape[1]:
                                            variables_map[f'mva_score_{class_name}'] = scores[:, i]

                        except Exception as e:
                            import traceback
                            print(f"Warning: MVA inference failed for {dataset}: {e}")
                            print(f"Full traceback:\n{traceback.format_exc()}")
                            # Set default values if inference fails
                            n_ev = len(pruned_ev)
                            variables_map['mva_signal_score'] = np.zeros(n_ev, dtype=np.float32)
                            variables_map['mva_class_prediction'] = np.zeros(n_ev, dtype=np.int32)

                    # save parquet files
                    fname = (
                        events.behavior["__events_factory__"]._partition_key.replace(
                            "/", "_"
                        )
                        + ".parquet"
                    )
                    subdirs = [self.workflow, self.year, dataset, category]
                    dump_pa_table(variables_map, fname, self.output_location, subdirs)

        # add histograms to output dictionary
        if self.output_format == "coffea":
            output["histograms"] = histograms
        return output

    def postprocess(self, accumulator):
        pass
