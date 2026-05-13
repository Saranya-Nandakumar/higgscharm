# H+c

Python package for analyzing H+c events. The package uses a columnar framework to process input tree-based NanoAOD V12 files using [Coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries.

- [Input filesets](#Input-filesets)
- [Workflows](#Workflows)
- [Local run](#Local-run)
- [Submit Condor jobs](#Submit-Condor-jobs)
- [Postprocessing](#Postprocessing)

### Development workflow
Please make a dedicated branch with your commits and make a PR to the main.

### Setup
- Lxplus
- Images
  - For filesets generation (used in make_filesets.py):
    - /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.1-py3.10
    - python version: 3.10
  - For running analysis:
    - /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.9
    - python version: 3.9

### Input filesets
 

Each data-taking year or campaign has a corresponding config file in `analysis/filesets/<year>_<nano version>.yaml`, which defines the input datasets to process

```yaml
EGammaD:
  era: D
  query: EGamma/Run2022D-22Sep2023-v1/NANOAOD
  process: Data
  key: electron
  xsec: None
DYJetsToLL:
  era: mc
  query: DYJetsToLL_M-50_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Summer22NanoAODv12-forPOG_130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM
  process: DY+Jets
  key: dy_lo
  xsec: 5558.0
```

- **`era`**: Dataset era. For data, this is typically the run letter (e.g. `B`, `F`, etc.). For MC datasets use `"mc"`, and for signal datasets use `"signal"`.
- **`process`**: Process name used during postprocessing to group multiple datasets into the same physical process.
- **`key`**: Label that associates the dataset with a group defined in the workflow configuration under `datasets` (see next section).
- **`query`**: DAS path used to fetch the files for this dataset.
- **`xsec`**: Cross section in picobarns (pb). Should be `null` for data. Required for MC and signal samples in order to compute event weights.

The input filesets are generated automatically from these configuration files. This process ensures that the list of ROOT files used in each campaign reflects the datasets currently available in CMS Rucio/DAS. The central script is [`fetch.py`](https://github.com/ua-cms/higgscharm/blob/main/fetch.py), which orchestrates the creation of the filesets
```
usage: fetch.py [-h] [--year {2022preEE,2022postEE,2023preBPix,2023postBPix}] [--image IMAGE] [--samples [SAMPLES ...]]

options:
  -h, --help            show this help message and exit
  --year {2022preEE,2022postEE,2023preBPix,2023postBPix}
  --image IMAGE         Specifies the container image used to run the dataset discovery tools
  --samples [SAMPLES ...]
                        (Optional) List of samples to use. If omitted, all available samples will be used
```
We use [Coffea's dataset discovery tools](https://coffea-hep.readthedocs.io/en/latest/dataset_tools.html) to build the input filesets. The container (default is [/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest-py3.10](https://hub.docker.com/layers/coffeateam/coffea-dask/latest-py3.10/images/sha256-110d080a5ec9155f56cabf8303a2e4e87768111abea99d9898b67d0aff370c1f)) provides a reproducible environment with all required dependencies. The script executes [`analysis/filesets/make_filesets.py`](https://github.com/ua-cms/higgscharm/blob/main/analysis/filesets/make_filesets.py) inside the container (using `Singularity`), which reads the corresponding dataset configuration file to extract the DAS queries and builds a dictionary of dataset definitions. It then calls `coffea.dataset_tools.DataDiscoveryCLI` to query Rucio/DAS, collecting the available file replicas across the allowed sites (listed [here](https://github.com/ua-cms/higgscharm/blob/main/analysis/filesets/sites.yaml)). The files are reformatted into a dictionary mapping dataset names to lists of ROOT files and saved to `analysis/filesets/fileset_<year>_NANO_lxplus.json`

### Workflows

Workflows define the configuration of an analysis: object and event selections, triggers, correctionsm variabbles and histograms. They are stored as YAML files in the `analysis/workflows` directory. 




Each workflow also specifies which data, mc or signal samples to run over for a given workflow through the `datasets` field:
```yaml
datasets:
  data: 
    - electron
  mc:
    - dy_inclusive
    - singletop
    - tt
    - wjets_ht
    - diboson
```
These entries (e.g. `muon`, `dy_inclusive`, etc) refer directly to the key fields defined in the input fileset configuration. **You can find a detailed explanation of the structure of a workflow** [here](https://github.com/ua-cms/higgscharm/blob/dev/signal/analysis/workflows/README.md) 

The available workflows are:

* $Z\rightarrow \ell\ell$
    * `ztomumu`: Select events in a $Z\rightarrow \mu \mu$ region
    * `ztoee`: Select events in a $Z\rightarrow ee$ region
* $H\rightarrow ZZ \rightarrow 4\ell$
    * Signal region 
        * `zzto4l`: Select $H \rightarrow ZZ \rightarrow 4\ell$ events
    * Z+X background estimation
        * `zplusl_os`: Select $Z+\ell$ events (lepton fake rate estimation with the OS Method)
        * `zplusll_os`: Select $Z+\ell\ell$ events (Z+X estimation with the OS Method)
        * `zplusl_ss`: Select $Z+\ell$ events (lepton fake rate estimation with the SS Method)
        * `zplusll_ss`: Select $Z+\ell\ell$ events (Z+X estimation with the SS Method)


The workflows (selections, variables, output histograms, triggers, etc) are defined through a configuration file located in `analysis/workflows/workflow/<workflow>.yaml`. [Here](https://github.com/deoache/higgscharm/blob/lxplus/analysis/workflows/README.md) you can find a detailed description on how to create the config file.


### Local run

To run locally, you can use the Coffea-Casa tool, which you can accessd [here](https://coffea.casa/hub/login?next=%2Fhub%2F) (**make sure to select the coffea 0.7.26 image**) (more info on coffea-casa [here](https://coffea-casa.readthedocs.io/en/latest/)). You can use the `tester.ipynb` notebook to test a workflow. There, you can select the year, dataset, output format (`coffea` for histograms or `parquet` for pandas DataFrames) and executor (`iterative` or `futures`). Feel free to add more datasets in case you need to run a particular workflow (and don't forget to use `root://xcache//` in order to be able to access the dataset).

This way, you can check that the workflow is running without issues before submitting batch jobs. It also allows you to interact with the output (whether histograms or DataFrames) to check that it makes sense and contains the expected information.



### Submit Condor jobs

**1. Log in to lxplus and clone the repository**

If you haven't done so already:
```bash
ssh <your_username>@lxplus.cern.ch

git clone https://github.com/ua-cms/higgscharm.git
cd higgscharm
```
**2. Initialize a valid CMS grid proxy**

To access remote datasets via xrootd, you need a valid grid proxy.

To generate the proxy:
```bash
voms-proxy-init --voms cms
```
If you're not registered in the CMS VO, follow these [instructions](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideLcgAccess) to request access.


**3. Submit all datasets from a workflow**

Use [runner.py](https://github.com/deoache/higgscharm/blob/lxplus/runner.py) to submit jobs for a specific workflow and year. 
```bash
python3 runner.py --workflow <workflow> --year <year> --submit --eos
``` 

**Note**: It's recommended to add the `--eos` flag to save the outputs to your `/eos` area, so the postprocessing step can be done from [SWAN](https://swan-k8s.cern.ch/hub/spawn). In this case, **you need to clone the repo before submitting jobs** in [SWAN](https://swan-k8s.cern.ch/hub/spawn) (select the 105a release) in order to be able to run the postprocess.

You could use [submit_condor.py](https://github.com/deoache/bsm3g_coffea/blob/main/submit_condor.py) to submit jobs for a specific dataset:
```bash
python3 submit_condor.py --workflow <workflow> --dataset <dataset> --year <campaign> --submit --eos
```

**4. Monitor job status**

To continuously monitor your Condor jobs:
```bash
watch condor_q
```
To get a summary of missing, failed, or incomplete jobs, and resubmit them, use:
```bash
python3 jobs_status.py --workflow <workflow> --year <campaign> --eos
```
This command will:
- Report total expected, completed, and missing jobs per dataset.
- Analyze recent xrootd-related errors on a per-dataset basis.
- Optionally regenerate input filesets for datasets with failing sites. By default, datasets are grouped by identical failing sites to minimize repeated fetches.
- Use the `--single_fetch` flag to merge all datasets with failing sites into a single group such that `fetch.py` is run only once.
- Optionally resubmit missing jobs.


### Postprocessing

Once the Condor jobs are completed and all outputs are saved under the `outputs/` directory, you can run `run_postprocess.py` to aggregate results, compute cutflows, and generate plots
```bash
python3 run_postprocess.py --workflow <workflow> --year <year> --postprocess --plot --log
``` 

After running post-processing for the two campaigns of a particular year, you can use the same command (with `--year 2022` or `--year 2023`) to automatically combine both campaigns and compute joint results and plots.

Results will be saved to the same directory as the output files.

#### Automated Postprocessing
Instead of manually running `run_postprocess.py` for each workflow and year, you can use the helper script `run_full_postprocess.py` to automatically execute the postprocessing pipelines.

This script:

* Iterates over all years unless a specific year is provided.
* Runs all steps of `run_postprocess.py` depending on the workflow type.
* Adds special configurations (e.g. `--group_by` for the $Z\rightarrow \ell\ell$ workflows and `--pass_axis` for the `zplusl_X` workflows).
* Automatically produces plots and merged results for multi-campaign years (e.g. 2022, 2023).

#### MVA inference

python3 run_postprocess.py --workflow hplusc_mva_4class --year 2022preEE --postprocess --plot --log --mva-inference --mva-config /path/to/hc_zzto4l_config.yml --mva-model /path/to/best_model.pt

model and config is stored in the folder models/
