#!/bin/bash
# One-command HcZZ ctag2d pipeline: build datacard -> run combine -> print r/kappa_c.
# Everything this needs (scripts, CMSSW combine build, outputs) lives under this
# same directory -- see README.md for the full layout.
#
# Usage:
#   ./run_pipeline.sh                       # DEFAULT WINDOW = [100,150] GeV (since 2026-08-18),
#                                            # recommended settings (surgical rebin, no Z+X)
#   ./run_pipeline.sh --with-zx             # recommended settings, WITH Z+X
#                                            # -- NOT AVAILABLE at [100,150]: Z+X is deliberately
#                                            #    excluded from scope for now (decision 2026-08-18),
#                                            #    see the message below.
#   ./run_pipeline.sh --original             # original 20-bin, no rebin (may have negative bins)
#   ./run_pipeline.sh --window 90160        # switch to the OLDER [90,160] window instead (any
#                                            # preset above still applies, e.g. combine with
#                                            # '--window 90160 --with-zx' for its real with-ZX number)
#   ./run_pipeline.sh --n-bins 20 --merge-bin-ranges "2-4,7-10" --skip-zx --output <dir>
#     (any flag after -- is passed straight to the underlying create_datacards_ctag2d*.py,
#      overriding the presets below)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMSSW_DIR="$HERE/CMSSW_14_1_0_pre4"

# Pull --window out of the argument list wherever it appears. Default changed
# to 100150 on 2026-08-18 (analysis decision: [100,150] is now the production
# default mass window) -- see second-brain/Tasks/HcZZ-fake-rate.md and
# Notes/HcZZ-combine-results-reproducibility.md for the reasoning (tighter
# S/sqrt(B), 0.0054 vs 0.0043). Z+X (fake-rate background) is deliberately
# excluded from scope at this window for now (decision 2026-08-18, same date)
# -- not a blocker to chase, a scope choice -- see ZX_AVAILABLE below.
WINDOW="100150"
RAW_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--window" ]]; then
    WINDOW="$2"; shift 2
  else
    RAW_ARGS+=("$1"); shift
  fi
done
set -- "${RAW_ARGS[@]}"

# NOTE: each script's own --zx-dir default points at the SS (same-sign) ZX
# estimate -- but the documented reference results were built against the OS
# (opposite-sign, featmajor) estimate. Every with-ZX preset below pins that
# explicitly; don't rely on the script's own default if reproducing a
# documented number matters (confirmed the hard way 2026-08-17: the SS
# default gives r=364.5, not 317.0, for the same --original settings
# otherwise, at [90,160]).
case "$WINDOW" in
  100150)
    SCRIPT="$HERE/scripts/create_datacards_ctag2d_100150.py"
    OUTDIR="$HERE/outputs/combine_run3_100150_ctag2d_pipeline_run"
    ZX_DIR="/eos/user/s/snandaku/Analysis/zx_background_mva_os_100150_featmajor"
    ZX_AVAILABLE=0  # Z+X deliberately out of scope at this window for now (decision 2026-08-18)
    ;;
  90160)
    SCRIPT="$HERE/scripts/create_datacards_ctag2d.py"
    OUTDIR="$HERE/outputs/combine_run3_90160_ctag2d_pipeline_run"
    ZX_DIR="/eos/user/s/snandaku/Analysis/zx_background_mva_os_90160_featmajor"
    ZX_AVAILABLE=1
    ;;
  *)
    echo "[run_pipeline] unknown --window '$WINDOW' (expected 100150 or 90160)" >&2
    exit 1
    ;;
esac

if [[ "${1:-}" == "--with-zx" ]]; then
  if [[ "$ZX_AVAILABLE" -eq 0 ]]; then
    echo "[run_pipeline] --with-zx requested at [100,150] GeV -- Z+X is deliberately" >&2
    echo "  out of scope at this window for now (decision 2026-08-18, same date the" >&2
    echo "  default window changed). Use '--window 90160 --with-zx' if you want the" >&2
    echo "  older window's with-ZX number instead." >&2
    exit 1
  fi
  DC_ARGS=(--merge-bin-ranges "2-4,7-10" --zx-dir "$ZX_DIR")
  shift
elif [[ "${1:-}" == "--original" ]]; then
  if [[ "$ZX_AVAILABLE" -eq 0 ]]; then
    DC_ARGS=(--skip-zx)
    echo "[run_pipeline] NOTE: --original normally includes Z+X, but Z+X is" >&2
    echo "  deliberately out of scope at [100,150] for now -- running MC-only" >&2
    echo "  (no-ZX, unmerged binning) instead. Use '--window 90160 --original'" >&2
    echo "  for the older window's with-ZX number." >&2
  else
    DC_ARGS=(--zx-dir "$ZX_DIR")
  fi
  shift
elif [[ $# -eq 0 ]]; then
  DC_ARGS=(--merge-bin-ranges "2-4,7-10" --skip-zx)
else
  DC_ARGS=()
fi

# Any remaining/raw args override or extend the preset (e.g. a custom --output).
DC_ARGS+=("$@")

# If the caller already passed --output explicitly (raw passthrough), honor
# it and don't also inject our own default -- otherwise argparse silently
# takes the LAST --output (the caller's), but this script would still go
# looking for the resulting datacard under the hardcoded default OUTDIR
# below, and fail to find it (or worse, pick up a stale file left over from
# an earlier run at that default path).
for i in "${!DC_ARGS[@]}"; do
  if [[ "${DC_ARGS[$i]}" == "--output" ]]; then
    OUTDIR="${DC_ARGS[$((i + 1))]}"
    unset 'DC_ARGS[i]' 'DC_ARGS[i+1]'
    DC_ARGS=("${DC_ARGS[@]}")  # re-index after unset leaves a gap
    break
  fi
done

if [[ ! -d "$CMSSW_DIR" ]]; then
  echo "[run_pipeline] $CMSSW_DIR not found -- bootstrapping (this can take a while)."
  "$HERE/setup_cmssw.sh"
fi

mkdir -p "$OUTDIR"

echo "=================================================================="
echo "[1/2] Building datacard"
echo "  output: $OUTDIR"
echo "  args:   ${DC_ARGS[*]}"
echo "=================================================================="
# Run in its OWN child subshell -- sourcing LCG_105 in run_pipeline.sh's own
# shell would leak LD_LIBRARY_PATH/PATH/PYTHONHOME into every later step
# (including the combine subshell below), which is exactly the "mixing
# environments corrupts the interpreter" gotcha this project has hit
# repeatedly (second-brain/Notes/HcZZ-combine-results-reproducibility.md
# section 1). Keeping it confined to a child process is what makes this
# script's own environment behave like two genuinely separate shells.
bash -c '
  source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
  python3 "$1" --output "$2" "${@:3}"
' _ "$SCRIPT" "$OUTDIR" "${DC_ARGS[@]}"

# Select the datacard this preset actually asked for, not just "whichever
# exists" -- all presets share the same OUTDIR, so a stale file from a
# PREVIOUS invocation (e.g. a plain no-ZX run) can otherwise silently look
# like a valid result for a WITH-ZX run that never touched it (found the hard
# way 2026-08-18: --with-zx picked up a leftover datacard_no_zx.txt and
# reported the no-ZX number instead of rebuilding/reading the ZX one).
SKIP_ZX=0
for a in "${DC_ARGS[@]}"; do
  [[ "$a" == "--skip-zx" ]] && SKIP_ZX=1
done
if [[ "$SKIP_ZX" -eq 1 ]]; then
  DATACARD="$OUTDIR/datacard_no_zx.txt"
else
  DATACARD="$OUTDIR/datacard_mva_with_zx.txt"
  [[ -f "$DATACARD" ]] || DATACARD="$OUTDIR/datacard_no_zx.txt"
fi
LABEL="HcZZ_ctag2d_${WINDOW}_pipeline"

echo
echo "=================================================================="
echo "[2/2] Running combine on $DATACARD"
echo "=================================================================="
# Separate shell invocation for the CMSSW environment -- mixing it with the
# LCG_105 env sourced above corrupts the python interpreter path (established
# gotcha throughout this project, see second-brain/Notes/HcZZ-combine-results-
# reproducibility.md section 1).
# Own child subshell too, same reasoning as the datacard-build step above --
# this script's own shell never sourced LCG_105 (that was confined to its
# own subshell), so this one starts clean.
bash -c "
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  cd '$CMSSW_DIR/src'
  eval \$(scramv1 runtime -sh)
  cd '$OUTDIR'
  combine -M AsymptoticLimits -m 120 --run blind --rAbsAcc 0.00001 --rRelAcc 0.00001 \
    '$(basename "$DATACARD")' -n $LABEL
" | tee "$OUTDIR/combine_result.log"

echo
echo "=================================================================="
echo "[kappa_c] converting r -> kappa_c (Felix Heyen thesis Eq. 2.28, closed-form"
echo "          inverse: kappa_c = sqrt(0.97*r + 0.03*r^2), validated against his"
echo "          worked example r=1088 -> kappa_c=191.23)"
echo "=================================================================="
python3 - "$OUTDIR/combine_result.log" << 'PYEOF'
import re, sys, math
log = open(sys.argv[1]).read()
for m in re.finditer(r"Expected\s+([\d.]+)%:\s*r\s*<\s*([\d.]+)", log):
    q, r = m.group(1), float(m.group(2))
    kc = math.sqrt(0.97 * r + 0.03 * r * r)
    print(f"  {q:>5}%: r < {r:>10.4f}   kappa_c = {kc:.2f}")
PYEOF

echo
echo "All output in: $OUTDIR"
