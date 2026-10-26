import numpy as np
import awkward as ak
from copy import deepcopy
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods.vector import LorentzVector
from analysis.utils import dump_lumi, dump_pa_table, dump_sumw_json
from analysis.filesets.utils import get_nano_version
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

NanoAODSchema.warn_missing_crossrefs = False

# 3FS/4FS stitching (LHCHWG-2026-007 sec. 4.4, added 2026-09-27): events are split
# by the number of GEN c-jets (GenJet hadronFlavour == 4, pT > 10 GeV, no eta cut).
# 0 c-jets are taken from 4FS FxFx, >= 1 from 3FS; 1 and >= 2 are stored separately
# so the "massless 1c" variant (0/1 from 4FS, >= 2 from 3FS) stays possible.
# See second-brain Notes/hczz-fs-stitching-recipe-2026-09-25.md.
GENC_PT_MIN = 10.0
GENC_BINS = ("0", "1", "2p")


def count_gen_cjets(events):
    """Per-event number of GEN c-jets (hadronFlavour == 4, pT > GENC_PT_MIN)."""
    genjets = events.GenJet
    is_c = (genjets.hadronFlavour == 4) & (genjets.pt > GENC_PT_MIN)
    return ak.to_numpy(ak.sum(is_c, axis=1))


# GEN c-hadron variables (added 2026-10-02) for the FS stitching by reweighting the 4FS
# FxFx sample with weights derived on the official H->gg 3FS/4FS samples (FS.pdf,
# T. Bevilacqua; c-hadron definition as the HWW FS study, C. Gupta). Unlike GenJets,
# c-hadrons cannot contain the Higgs decay products, so the weights transfer between decays.
# c-hadron: status 2, open charm (|pdgId| 4xx, 4xxx, 104xx, 204xx; charmonium excluded),
# weakly decaying (no c-hadron daughter), not from a b-hadron, |eta| < 2.5.
# See second-brain Notes/hczz-fs-3fs-gridpack-mismatch-2026-10-02.md.
CHAD_ETA_MAX = 2.5
CHAD_PT_THRESHOLDS = (5, 10)


def _is_open_charm_hadron(apid):
    openc = (
        ((apid > 400) & (apid < 500))
        | ((apid > 4000) & (apid < 5000))
        | ((apid > 10400) & (apid < 10500))
        | ((apid > 20400) & (apid < 20500))
    )
    return openc & (((apid // 10) % 100) != 44)


def _is_b_hadron(apid):
    return ((apid > 500) & (apid < 600)) | ((apid > 5000) & (apid < 6000))


def gen_chadron_info(events):
    """Leading GEN c-hadron pT (-1 if none) and the number of c-hadrons with
    pT > T for each T in CHAD_PT_THRESHOLDS."""
    gp = events.GenPart
    apid = abs(gp.pdgId)
    mom = gp.genPartIdxMother
    is_c = _is_open_charm_hadron(apid)
    idx = ak.local_index(apid)
    c_moms = mom[is_c]
    has_c_daughter = ak.any(idx[:, :, None] == c_moms[:, None, :], axis=-1)
    mom_pid = ak.fill_none(apid[ak.mask(mom, mom >= 0)], 0)
    sel = (
        is_c
        & (gp.status == 2)
        & ~has_c_daughter
        & ~_is_b_hadron(mom_pid)
        & (abs(gp.eta) < CHAD_ETA_MAX)
    )
    cpt = gp.pt[sel]
    out = {"gen_chad1_pt": ak.to_numpy(ak.fill_none(ak.max(cpt, axis=1), -1.0))}
    for t in CHAD_PT_THRESHOLDS:
        out[f"n_chad{t}"] = ak.to_numpy(ak.sum(cpt > t, axis=1))
    return out


def sumw_by_genc(events, n_genc, prefix="genc"):
    """
    Pre-selection sum of genWeight per GEN c-jet bin, nominal and multiplied by
    each of the 9 LHEScaleWeight entries (as the ratio to index 4, the same
    convention as analysis/corrections/lhescale.py). The stitching normalises
    each piece with these, and renormalises every scale variation separately.
    """
    genw = ak.to_numpy(events.genWeight)
    masks = {"0": n_genc == 0, "1": n_genc == 1, "2p": n_genc >= 2}
    out = {f"sumw_{prefix}{b}": float(genw[m].sum()) for b, m in masks.items()}

    lhe = events.LHEScaleWeight if "LHEScaleWeight" in events.fields else None
    if lhe is None or len(events) == 0 or not ak.all(ak.num(lhe) == 9):
        return out
    lhe = ak.to_numpy(ak.to_regular(lhe, axis=1))
    ratio = lhe / lhe[:, 4:5]
    for b, m in masks.items():
        out[f"sumw_{prefix}{b}_scalevar"] =(genw[m][:, None] * ratio[m]).sum(axis=0).tolist()
    return out


class BaseProcessor(processor.ProcessorABC):
    def __init__(
        self,
        workflow: str,
        year: str,
        output_format: str,
        output_location: str,
    ):
        self.year = year
        self.workflow = workflow
        self.output_format = output_format
        self.output_location = output_location

        config_builder = WorkflowConfigBuilder(workflow)
        self.workflow_config = config_builder.build_workflow_config()
        self.histogram_config = self.workflow_config.histogram_config
        self.histograms = HistBuilder(self.workflow_config).build_histogram()

    @staticmethod
    def _partition_fname(events, suffix):
        return (
            events.behavior["__events_factory__"]._partition_key.replace("/", "_")
            + suffix
        )

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

        # genWeight sum BEFORE the event-level jet veto below (added 2026-10-02). The
        # sidecar "sumw" is computed after the veto, so it misses the vetoed events
        # (memory hczz_jetveto_before_sumw); this extra key lets that bias be measured
        # and corrected without changing what the builders read today.
        sumw_preveto = float(ak.sum(events.genWeight)) if hasattr(events, "genWeight") else None
        # Same GEN c-jet split BEFORE the veto (added 2026-10-05, test for the direct 3FS+4FS
        # stitching): sumw_preveto_genc{0,1,2p} (+ _scalevar). Purely additive: the existing
        # post-veto keys (sumw, sumw_genc*, sumw_chad*) are unchanged, so builders that do not
        # ask for the new keys behave exactly as before. GenJets do not depend on the veto.
        # Revert: base.py.bak_20261005_preveto_genc; second-brain
        # Notes/hczz-preveto-genc-sumw-and-jec-pershift-sumw-2026-10-05.md.
        sumw_preveto_genc = {}
        if hasattr(events, "genWeight") and "GenJet" in events.fields:
            sumw_preveto_genc = sumw_by_genc(events, count_gen_cjets(events), prefix="preveto_genc")
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

        # GEN c-jet count for the 3FS/4FS stitching (MC with GenJet only)
        n_genc = None
        sumw_extra = {}
        if is_mc and "GenJet" in events.fields:
            n_genc = count_gen_cjets(events)
            sumw_extra = sumw_by_genc(events, n_genc)
            output["metadata"].update(sumw_extra)
        # GEN c-hadron variables (MC with GenPart only); the sidecar also gets the
        # pre-selection sumw split by c-hadron multiplicity (sumw_chad<T>{0,1,2p})
        chad = None
        if is_mc and "GenPart" in events.fields:
            chad = gen_chadron_info(events)
            for t in CHAD_PT_THRESHOLDS:
                sumw_extra.update(sumw_by_genc(events, chad[f"n_chad{t}"], prefix=f"chad{t}_"))
            output["metadata"].update(sumw_extra)
        if is_mc and sumw_preveto is not None:
            sumw_extra["sumw_preveto"] = sumw_preveto
            output["metadata"]["sumw_preveto"] = sumw_preveto
        if is_mc and sumw_preveto_genc:
            sumw_extra.update(sumw_preveto_genc)
            output["metadata"].update(sumw_preveto_genc)

        # --------------------------------------------------------------
        # Object corrections
        # --------------------------------------------------------------
        object_shifts = object_corrector_manager(
            events=events,
            year=year,
            dataset=dataset,
            workflow_config=self.workflow_config,
        )
        # object_shifts is None for every workflow that doesn't request "jec_shifts"
        # (JES/JER shift production, added 2026-08-18) -- in that case run exactly
        # one pass with events as object_corrector_manager already left them
        # (nominal-only, mutated in place, unchanged from before shift support
        # existed). When shifts ARE requested, object_corrector_manager instead
        # returns [(collections, shift_name), ...] (nominal first, then one entry
        # per JES/JER Up/Down variant) and every entry below reruns the FULL
        # selection/weight/output block -- required because shifted jet kinematics
        # can change which events pass selection and every jet-derived MVA feature,
        # not just a per-event weight.
        if object_shifts is None:
            object_shifts = [({}, None)]

        met_field_key = "PuppiMET" if int(get_nano_version(year)) >= 12 else "MET"

        for shift_collections, shift_name in object_shifts:
            if shift_collections:
                if "Jet" in shift_collections:
                    events["Jet"] = shift_collections["Jet"]
                if "MET" in shift_collections:
                    events[met_field_key] = shift_collections["MET"]
                if "Muon" in shift_collections:
                    # added 2026-08-19 for CMS_scale_m/CMS_res_m shift
                    # production (see correction_manager.py's jec_shifts
                    # branch) -- every entry from that branch carries Muon
                    # explicitly (nominal or varied) for the same
                    # no-stale-inheritance reason Jet/MET already do.
                    events["Muon"] = shift_collections["Muon"]

            # --------------------------------------------------------------
            # Object selection
            # --------------------------------------------------------------
            object_selector = ObjectSelector(object_selections, year)
            objects = object_selector.select_objects(events)

            # --------------------------------------------------------------
            # Event selection
            # --------------------------------------------------------------
            if not is_mc and shift_name is None:
                # save (run, luminosityBlock) pairs to metadata (nominal pass only --
                # identical across shifts, no point repeating it)
                lumi_mask = eval(event_selection["selections"]["lumimask"])
                dump_lumi(events[lumi_mask], output)

            # initialize selection manager
            selection_manager = PackedSelection()
            # add all selections to selector manager
            for selection, mask in event_selection["selections"].items():
                selection_manager.add(selection, eval(mask))

            # add cutflow to metadata (nominal pass only)
            if shift_name is None:
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
                    # save number of events after selection to metadata (nominal only
                    # -- per-shift yields aren't tracked in the cutflow/metadata dict)
                    if shift_name is None:
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
                            if shift_name is None:
                                # nominal pass: keep every per-event weight-systematic
                                # variation, exactly as before shift support existed
                                variations = ["nominal"] + list(
                                    weights_container.variations
                                )
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
                            else:
                                # object-shift pass (CMS_scale_j/CMS_res_j Up/Down):
                                # only the nominal per-event weight is needed -- this
                                # pass's own shifted kinematics/selection ARE the
                                # systematic, not a reweighting of the nominal sample
                                variables_map[f"weight_nominal"] = (
                                    weights_container.weight()
                                )
                        if n_genc is not None:
                            variables_map["n_genc10"] = n_genc[
                                ak.to_numpy(category_mask)
                            ]
                        if chad is not None:
                            for key, val in chad.items():
                                variables_map[key] = val[ak.to_numpy(category_mask)]
                        # save parquet files
                        fname = self._partition_fname(events, ".parquet")
                        subdirs = [self.workflow, self.year, dataset, category]
                        if shift_name is not None:
                            subdirs = subdirs + [shift_name]
                        dump_pa_table(
                            variables_map, fname, self.output_location, subdirs
                        )

        # pre-selection sumw sidecar, written for every MC chunk (even when no event
        # passes the selection): the datacard builders read <dataset>/sumw/*.json as the
        # xs*lumi denominator. The call was lost when this method was reworked for
        # jec_shifts on 2026-08-18 and restored 2026-09-27 (see the 2026-07-31 entry in
        # second-brain Tasks/HcZZ-fake-rate.md). Written LAST, after every parquet of
        # this chunk, so that a chunk which crashes part-way leaves no sidecar and its
        # sumw can't enter the denominator without its events.
        if is_mc and self.output_format == "parquet":
            dump_sumw_json(
                sumw,
                dataset,
                self._partition_fname(events, ".json"),
                self.output_location,
                [self.workflow, self.year, dataset, "sumw"],
                extra=sumw_extra,
            )

        # add histograms to output dictionary
        if self.output_format == "coffea":
            output["histograms"] = histograms
        return output

    def postprocess(self, accumulator):
        pass
