# Saved Combine templates, [90,160] GeV, 16-systematic ctag2d card -- 2026-08-18

Permanent snapshot, kept separate from `combine_run3_90160_ctag2d_pipeline_run/`
(which `run_pipeline.sh` reuses and overwrites on every invocation -- do not treat
that directory as a stable reference; that's exactly what caused the `--with-zx`
mixup during this session's CMSSW-rebuild validation).

Each subdirectory contains: the datacard, the ROOT histogram file (nominal + every
Up/Down shape-systematic template, as fed to `combine`), a matplotlib PDF plotting
every template (nominal/Up/Down overlay + ratio panel, from
`scripts/plot_systematic_templates.py`), and a flat CSV dump of every bin
(`process,systematic,variation,bin,edge_lo,edge_hi,value`) for inspection without
ROOT.

## Subdirectories

- **`no_zx_surgical/`** -- production default: surgical rebin (`--merge-bin-ranges
  2-4,7-10`), Z+X excluded. Zero negative bins anywhere (nominal or any
  variation). `combine` median r = 324.5 (full-syst).
- **`surgical_withzx/`** -- same surgical rebin, Z+X included. Only `h_ZX` itself
  has (small, expected) negative bins -- standard AN-18-340 fake-rate-subtraction
  behavior, not a defect. `combine` median r = 329.0 (full-syst).
- **`original_unmerged_withzx/`** -- deliberately-unmerged 20-bin comparison
  variant, Z+X included, no rebin. Signal carries the known negative bin (bin 9,
  ~-1e-5) that motivated the surgical rebin in the first place. `combine` median
  r = 329.0 (full-syst) -- bit-identical to the surgical version, because the
  merged bins carry negligible signal (confirmed, not a bug); also includes
  `impact_ranking.json`/`.png` from the freeze-and-compare scan run against this
  variant.

## Provenance

Built 2026-08-18 immediately after `setup_cmssw.sh --force` (proper `cmsrel`
rebuild, replacing the prior per-shell `sed` path-patch workaround) and the
`run_pipeline.sh` stale-datacard-selection bugfix. Datacard builder:
`scripts/create_datacards_ctag2d.py`. See `second-brain/Tasks/HcZZ-fake-rate.md`
and `second-brain/slides/combine_summary_hczz.tex` (slides 4b/5/10/11) for the
full narrative -- `CMS_ctag2d` Down-variation asymmetry root cause, private-sample
`lhe_pdf`/`lhe_alphaS`/`scalevar` numbers, negative-bin sweep, and the POI-range
boundary-clipping fix.
