import awkward as ak
from pathlib import Path


# summary of pog scale factors: https://cms-analysis-corrections.docs.cern.ch/
correction_files = {
    "jerc": {
        # dated snapshots verified 2026-08-18 to contain exactly the correction
        # names jec_params_correctionlib.yaml expects for each era (Total JES +
        # JER ScaleFactor/PtResolution) -- CVMFS "latest" drifts per-era on an
        # independent cadence, so don't swap these for "latest" without
        # re-verifying against jec_params_correctionlib.yaml's pinned tags.
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/latest/jet_jerc.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2026-04-13/jet_jerc.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2026-04-13/jet_jerc.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2026-04-13/jet_jerc.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-07-17/jet_jerc.json.gz",
    },
    "pileup": {
        "2016preVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2016preVFP-UL-NanoAODv9/2021-09-10/puWeights.json.gz",
        "2016postVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2016postVFP-UL-NanoAODv9/2021-09-10/puWeights.json.gz",
        "2017": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2017-UL-NanoAODv9/2021-09-10/puWeights.json.gz",
        "2018": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2018-UL-NanoAODv9/2021-09-10/puWeights.json.gz",
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-22CDSep23-Summer22-NanoAODv12/2024-01-31/puWeights.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-22EFGSep23-Summer22EE-NanoAODv12/2024-01-31/puWeights.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-23CSep23-Summer23-NanoAODv12/2024-01-31/puWeights.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run3-23DSep23-Summer23BPix-NanoAODv12/2024-01-31/puWeights.json.gz",
        "2024": f"{Path.cwd()}/analysis/data/pileup/2024_puWeights.json.gz",  # taken from /eos/cms/store/user/evourlio/pileupJson_forHiggsDNA
    },
    "muon": {
        "2016preVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2016preVFP-UL-NanoAODv9/2024-07-02/muon_Z.json.gz",
        "2016postVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2016postVFP-UL-NanoAODv9/2024-07-02/muon_Z.json.gz",
        "2017": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2017-UL-NanoAODv9/2024-07-02/muon_Z.json.gz",
        "2018": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2018-UL-NanoAODv9/2024-07-02/muon_Z.json.gz",
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-14/muon_Z.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-14/muon_Z.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2025-08-14/muon_Z.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-14/muon_Z.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-10-17/muon_Z.json.gz",
    },
    "muon_hlt": {
        "2016preVFP": f"{Path.cwd()}/analysis/data/muon_hlt/2016preVFP_Muon_HLT_Eff.json",
        "2016postVFP": f"{Path.cwd()}/analysis/data/muon_hlt/2016postVFP_Muon_HLT_Eff.json",
        "2017": f"{Path.cwd()}/analysis/data/muon_hlt/2017_Muon_HLT_Eff.json",
        "2018": f"{Path.cwd()}/analysis/data/muon_hlt/2018_Muon_HLT_Eff.json",
        "2022preEE": f"{Path.cwd()}/analysis/data/muon_hlt/2022preEE_Muon_HLT_Eff.json",
        "2022postEE": f"{Path.cwd()}/analysis/data/muon_hlt/2022postEE_Muon_HLT_Eff.json",
        "2023preBPix": f"{Path.cwd()}/analysis/data/muon_hlt/2023preBPix_Muon_HLT_Eff.json",
        "2023postBPix": f"{Path.cwd()}/analysis/data/muon_hlt/2023postBPix_Muon_HLT_Eff.json",
        "2024": f"{Path.cwd()}/analysis/data/muon_hlt/2024_Muon_HLT_Eff.json",
    },
    "muon_ss": {
        "2016preVFP": f"{Path.cwd()}/analysis/data/rochester/RoccoR2016preVFPUL.txt",
        "2016postVFP": f"{Path.cwd()}/analysis/data/rochester/RoccoR2016postVFPUL.txt",
        "2017": f"{Path.cwd()}/analysis/data/rochester/RoccoR2017UL.txt",
        "2018": f"{Path.cwd()}/analysis/data/rochester/RoccoR2018UL.txt",
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-14/muon_scalesmearing.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-10-17/muon_scalesmearing.json.gz",
    },
    "electron_id": {
        "2016preVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016preVFP-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2016postVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016postVFP-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2017": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2017-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2018": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2018-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/2025-10-22/electron.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-22/electron.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23CSep23-Summer23-NanoAODv12/2025-10-22/electron.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-22/electron.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-10-22/electronID.json.gz",
    },
    "electron_reco": {
        "2016preVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016preVFP-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2016postVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2016postVFP-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2017": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2017-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2018": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-2018-UL-NanoAODv9/2024-07-02/electron.json.gz",
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/2025-10-22/electron.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-22/electron.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23CSep23-Summer23-NanoAODv12/2025-10-22/electron.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-22/electron.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-10-22/electron.json.gz",
    },
    "electron_hlt": {
        "2016preVFP": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_SF_2016preVFP.json.gz",
        "2016postVFP": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_SF_2016postVFP.json.gz",
        "2017": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_SF_2017.json.gz",
        "2018": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_SF_2018.json.gz",
        "2022preEE": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2022_Summer22/electronHlt.json.gz",
        "2022postEE": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2022_Summer22EE/electronHlt.json.gz",
        "2023preBPix": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2023_Summer23/electronHlt.json.gz",
        "2023postBPix": "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/EGM/2023_Summer23BPix/electronHlt.json.gz",
    },
    "electron_hlt_mc_eff": {
        "2016preVFP": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_MCEff_2016preVFP.json.gz",
        "2016postVFP": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_MCEff_2016postVFP.json.gz",
        "2017": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_MCEff_2017.json.gz",
        "2018": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_MCEff_2018.json.gz",
    },
    "electron_hlt_data_eff": {
        "2016preVFP": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_DataEff_2016preVFP.json.gz",
        "2016postVFP": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_DataEff_2016postVFP.json.gz",
        "2017": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_DataEff_2017.json.gz",
        "2018": f"{Path.cwd()}/analysis/data/electron_hlt/EGamma_HLT_DataEff_2018.json.gz",
    },
    "electron_ss": {
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22EFGSep23-Summer22EE-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23CSep23-Summer23-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/latest/electronSS_EtDependent.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/latest/electronSS_EtDependent.json.gz",
    },
    "jetvetomaps": {
        "2016preVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016preVFP-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz",
        "2016postVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016postVFP-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz",
        "2017": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2017-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz",
        "2018": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2018-UL-NanoAODv9/2025-04-11/jetvetomaps.json.gz",
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22CDSep23-Summer22-NanoAODv12/2025-09-23/jetvetomaps.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-10-07/jetvetomaps.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23CSep23-Summer23-NanoAODv12/2025-10-07/jetvetomaps.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-10-07/jetvetomaps.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-07-17/jetvetomaps.json.gz",
    },
    "ctagging": {
        "2016preVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2016preVFP-UL-NanoAODv9/2025-08-19/ctagging.json.gz",
        "2016postVFP": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2016postVFP-UL-NanoAODv9/2025-08-19/ctagging.json.gz",
        "2017": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2017-UL-NanoAODv9/2025-08-19/ctagging.json.gz",
        "2018": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2018-UL-NanoAODv9/2025-08-19/ctagging.json.gz",
        "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-20/ctagging.json.gz",
        "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-20/ctagging.json.gz",
        "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-23CSep23-Summer23-NanoAODv12/2025-08-20/ctagging.json.gz",
        "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-20/ctagging.json.gz",
        "2024": "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/2025-08-19/ctagging.json.gz",
    },
    # Preliminary 2D pseudo-continuous HF-tagging SF ("2D_HF_Tagging"), distinct scheme from
    # "ctagging" above -- see analysis/corrections/ctag2d.py. Paths confirmed readable
    # 2026-08-10 (second-brain memory hczz_pnet_ctag_sf_migration); correction name
    # ParticleNetAK4_pseudocontinuous for 2022/2023, UParTAK4_pseudocontinuous for 2024.
    "ctagging_2d": {
        "2022preEE": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/2D_HF_Tagging/flavTaggingSF_2022preEE.json.gz",
        "2022postEE": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2022/2D_HF_Tagging/flavTaggingSF_2022postEE.json.gz",
        "2023preBPix": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/2D_HF_Tagging/flavTaggingSF_2023preBPix.json.gz",
        "2023postBPix": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2023/2D_HF_Tagging/flavTaggingSF_2023postBPix.json.gz",
        "2024": "/eos/cms/store/group/phys_higgs/cmshgg/ingredients/2024/2D_HF_Tagging/flavTaggingSF_2024.json.gz",
    },
}


def unflat_sf(sf: ak.Array, in_limit_mask: ak.Array, n: ak.Array):
    """
    get scale factors for in-limit objects (otherwise assign 1).
    Unflat array to original shape and multiply scale factors event-wise

    Parameters:
    -----------
        sf:
            Array with 1D scale factors
        in_limit_mask:
            Array mask for events with objects within correction limits
        n:
            Array with number of objects per event
    """
    sf = ak.where(in_limit_mask, sf, ak.ones_like(sf))
    return ak.fill_none(ak.prod(ak.unflatten(sf, n), axis=1), value=1)
