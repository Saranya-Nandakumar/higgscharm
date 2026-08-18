#!/bin/bash
# Bootstrap CMSSW_14_1_0_pre4 + HiggsAnalysis/CombinedLimit (Combine v10.5.0) from
# scratch. This is the git-trackable representation of "the CMSSW area" -- the
# built CMSSW_14_1_0_pre4/ tree itself (~560MB of scram-compiled, architecture-
# specific binaries) is deliberately NOT committed to this repo (see .gitignore);
# git already tracks the actual Combine source upstream at
# https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit -- this script just
# reruns the standard install recipe against a pinned tag.
#
# Idempotent: if CMSSW_14_1_0_pre4/ already exists next to this script, does
# nothing (rerun with --force to wipe and rebuild).
#
# Usage:
#   ./setup_cmssw.sh            # build if missing
#   ./setup_cmssw.sh --force    # wipe and rebuild even if present
set -eo pipefail
# (no -u: CVMFS/CMSSW environment scripts sourced below reference unset
# variables internally -- that's expected, not an error in this script)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMSSW_VERSION="CMSSW_14_1_0_pre4"
COMBINE_TAG="v10.5.0"

if [[ "${1:-}" == "--force" ]] && [[ -d "$HERE/$CMSSW_VERSION" ]]; then
  echo "[setup_cmssw] --force given, removing existing $HERE/$CMSSW_VERSION"
  rm -rf "$HERE/$CMSSW_VERSION"
fi

if [[ -d "$HERE/$CMSSW_VERSION" ]]; then
  echo "[setup_cmssw] $HERE/$CMSSW_VERSION already exists, nothing to do (use --force to rebuild)."
  exit 0
fi

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$HERE"
cmsrel "$CMSSW_VERSION"
cd "$CMSSW_VERSION/src"
cmsenv

git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin --tags
git checkout "$COMBINE_TAG"
cd "$HERE/$CMSSW_VERSION/src"

scramv1 b clean
scramv1 b -j 8

echo
echo "[setup_cmssw] Done. Verify with:"
echo "  cd $HERE/$CMSSW_VERSION/src && eval \$(scramv1 runtime -sh) && combine --help | head -3"
