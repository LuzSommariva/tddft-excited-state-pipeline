#!/usr/bin/env bash
# Tier 1 — single vertical absorption peak. Coumarin 102 only (the only molecule
# with a Tier 1 run in this repo; 01/02 hardcode "coumarin102.xyz").
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$REPO_ROOT/molecules/coumarin102/results"
export PYTHONPATH="$REPO_ROOT/src/common:${PYTHONPATH:-}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

python3 "$REPO_ROOT/src/tier1/00_sanity_check.py"
python3 "$REPO_ROOT/src/tier1/01_production_cpu_reference.py" | tee cpu_reference.log
python3 "$REPO_ROOT/src/tier1/02_production_gpu.py"            | tee gpu_run.log
python3 "$REPO_ROOT/src/tier1/03_compare_cpu_gpu.py" cpu_reference.log gpu_run.log
