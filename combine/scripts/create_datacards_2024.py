#!/usr/bin/env python3
"""
2024 (NanoAOD v15, UParTAK4) datacards (2026-10-25): create_datacards_ctag2d_100150_stitched_decorr.py
with the era list opened to 2024. The 2022/2023 builders are untouched by this file; the shared modules
only gained era-aware switches whose defaults reproduce the 2022/2023 cards:
  - create_datacards_ctag2d_100150.py: FILESET_YAML (2024 -> 2024_nanov15.yaml), "GluGlu2Zto" -> ggZZ
    (2024 sample names), era_year("2024") = "2024", SUMW_KEY (sidecar key of load_sumw);
  - _rawsf / _decorr: correction_name(era) (UParTAK4_pseudocontinuous for 2024) and per-era jet columns
    rs.jet_score_cols(era) (jet_btagUParTAK4CvL/CvB for 2024);
  - _stitched: per-era jet columns, dc.SUMW_KEY in the single-sample signal sumw, and --signal-mode 4fs
    allowed without a 3FS sample.

Set here, for 2024:
  - LUMINOSITY_PB["2024"] = 108960.0 (analysis/postprocess/luminosity.yaml);
  - sumw = the sidecars' PRE-jet-veto "sumw_preveto" (the 2022/23 cards get the same quantity from
    --sumw-table; the 2024 sidecar coverage was checked against the input files, final_check.py);
  - muon ID SF: inside weight_nominal in the 2024 production (workflow has the muon block), so the flat
    signal MUON_ID_SF is 1.0 and --recompute-muon-id-sf is not used.

Scope decisions (user, 2026-10-23): plain 4FS signal (--signal-mode 4fs), no Z+X (--skip-zx) until
pkatris has 2024 fake rates. Only --eras 2024 is accepted (combination with 2022/23: later, by card).

Usage: as create_datacards_ctag2d_100150_stitched_decorr.py, with --eras 2024, without --sumw-table,
--preveto-genc-4fs, --fs-prescribed and --recompute-muon-id-sf.
"""
import sys

sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/scripts")
sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm")

ERAS_2024 = ["2024"]
LUMI_2024_PB = 108960.0

_a = sys.argv
if "--eras" not in _a:
    raise SystemExit("create_datacards_2024.py: pass --eras 2024")
_i = _a.index("--eras") + 1
_eras = []
while _i < len(_a) and not _a[_i].startswith("--"):
    _eras.append(_a[_i])
    _i += 1
if _eras != ERAS_2024:
    raise SystemExit(f"create_datacards_2024.py: only --eras 2024 is supported, got {_eras}")
for _bad in ("--sumw-table", "--preveto-genc-4fs", "--recompute-muon-id-sf", "--fs-prescribed"):
    if _bad in _a:
        raise SystemExit(f"create_datacards_2024.py: {_bad} is not supported for 2024 (see docstring)")
if "--skip-zx" not in _a:
    raise SystemExit("create_datacards_2024.py: no 2024 Z+X yet, pass --skip-zx")

# Open the era list BEFORE the other modules import: decorr builds its per-era stat sources at import.
import create_datacards_ctag2d_100150 as dc
dc.ERAS[:] = ERAS_2024
dc.LUMINOSITY_PB["2024"] = LUMI_2024_PB
dc.TOTAL_LUMI_PB = LUMI_2024_PB
dc.SUMW_KEY = "sumw_preveto"

import create_datacards_ctag2d_100150_rawsf as rs
import create_datacards_ctag2d_100150_decorr as dec
import create_datacards_ctag2d_100150_stitched as st
import create_datacards_ctag2d_100150_stitched_decorr  # noqa: F401  (installs the decorr signal/background loaders)

if st.CFG.signal_mode != "4fs":
    raise SystemExit("create_datacards_2024.py: 2024 has no 3FS sample, use --signal-mode 4fs")
st.MUON_ID_SF["2024"] = 1.0   # already in weight_nominal

if __name__ == "__main__":
    print("=" * 70)
    print(f"2024 BUILD: eras {dc.ERAS}, lumi {dc.TOTAL_LUMI_PB:.1f} pb^-1, sumw key {dc.SUMW_KEY}, "
          f"signal {st.CFG.signal_mode}, c-tag {rs.correction_name('2024')} "
          f"({len(dec.STAT_GROUPS['2024'])} stat sources), jet columns {rs.jet_score_cols('2024')[:2]}")
    print("=" * 70)
    dc.main()
