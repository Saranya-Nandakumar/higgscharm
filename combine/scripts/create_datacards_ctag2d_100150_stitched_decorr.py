#!/usr/bin/env python3
"""
Stitched-signal datacards (create_datacards_ctag2d_100150_stitched.py) with CMS_ctag2d
decorrelated into the 33 per-flavour/WP stat sources per era plus CMS_ctag2d_resid
(create_datacards_ctag2d_100150_decorr.py, Livio Redard-Jacot's recommendation), for
backgrounds AND the stitched H+c signal (2026-10-15: port of the 09-25 decorr scheme to the
[105,160) stitched nominal).

Composition, by import order (each module monkey-patches dc):
  1. decorr: swaps CMS_ctag2d for the new nuisances in dc.SYSTEMATICS/USABLE_SYST, wraps
     dc.build_histograms (pruning) and dc.write_datacard (column fix);
  2. stitched: wraps those again (FS envelope, pdf_cH) and replaces the MC loader;
  3. here: the stitched background loader is pointed to the decorr loader (signal samples
     skipped), and the stitched signal's c-tag variations to the decorr per-event ratios.
--ctag2d-capfix (rawsf) acts inside rs._raw_event_sf, so it reaches both the residual's
Total and the stat sources' central value.

Usage: the stitched CLI plus --prune-threshold (decorr) and --ctag2d-capfix (rawsf).
Optional (2026-10-22) --recompute-muon-id-sf: per-event muon ID SF for the background files of
--scored-dir/--jecshifts-dir that lack it (musf_recompute.py), so the background nominal can be taken
from the JES/JER shift production (JEC V4/V3, as the signal and the shifts); the flat per-era
MUON_ID_SF on the shift sumw is then switched off.
"""
import sys

MUSF = "--recompute-muon-id-sf" in sys.argv
if MUSF:
    sys.argv.remove("--recompute-muon-id-sf")

sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/scripts")
sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm")
import create_datacards_ctag2d_100150 as dc
import create_datacards_ctag2d_100150_rawsf as rs
import create_datacards_ctag2d_100150_decorr as dec
import create_datacards_ctag2d_100150_stitched as st


def _decorr_backgrounds(scored_dir, sumw_by_group):
    orig = dc.get_process_from_path
    dc.get_process_from_path = lambda p: None if orig(p) == "Signal" else orig(p)
    try:
        return dec.load_mc_scored_parquets_decorr(scored_dir, sumw_by_group)
    finally:
        dc.get_process_from_path = orig


def _decorr_signal(df, era, usable):
    if dec.RESID not in usable:
        return {}
    w = df["weight_nominal"].fillna(0).values
    return {k: (w * u, w * d) for k, (u, d) in dec._event_ratios(df, era).items() if k in usable}


rs.load_mc_scored_parquets_rawsf = _decorr_backgrounds
st.ctag2d_weights = _decorr_signal

if MUSF:
    import musf_recompute
    _a = sys.argv
    _dirs = [_a[_a.index(f) + 1] for f in ("--scored-dir", "--jecshifts-dir") if f in _a]
    musf_recompute.install(dc.E, _dirs)
    st.MUON_ID_SF = {era: 1.0 for era in st.MUON_ID_SF}   # replaced by the per-event SF


if __name__ == "__main__":
    print("=" * 70)
    print(f"STITCHED-SIGNAL + CTAG2D DECORRELATED BUILD ({st.CFG.signal_mode}), "
          f"prune {dec.PRUNE_THRESHOLD}, capfix {rs.CAPFIX}")
    print("=" * 70)
    dc.main()
    if MUSF:
        print(f"  [musf] patched {musf_recompute.STATS['files']} files, {musf_recompute.STATS['events']} events")
