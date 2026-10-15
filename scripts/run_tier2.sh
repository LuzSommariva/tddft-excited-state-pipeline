#!/usr/bin/env bash
# Tier 2 — Nuclear Ensemble Approach (Wigner sampling) emission band. Coumarin 102
# only (the only molecule with a Tier 2 run in this repo). The production run only
# ever exercised the GPU S1-optimization path, so 07 onward is pointed explicitly
# at coumarin102_s1opt_gpu.xyz rather than 05's CPU-path output.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$REPO_ROOT/molecules/coumarin102/results"
export PYTHONPATH="$REPO_ROOT/src/common:${PYTHONPATH:-}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

[ -f coumarin102.xyz ] || python3 "$REPO_ROOT/src/common/gen_geometry.py"

python3 "$REPO_ROOT/src/tier2/04_s0_optimize.py"     | tee s0_optimize.log
python3 "$REPO_ROOT/src/tier2/06_s1_optimize_gpu.py" | tee s1_optimize_gpu.log
python3 "$REPO_ROOT/src/tier2/07_s1_frequencies.py" --geometry coumarin102_s1opt_gpu.xyz \
    | tee s1_frequencies.log
python3 "$REPO_ROOT/src/tier2/08_wigner_sample.py" --n 100 --seed 42 | tee wigner_sample.log
python3 "$REPO_ROOT/src/tier2/09_ensemble_emission.py" --gpu         | tee ensemble_emission.log
python3 "$REPO_ROOT/src/common/10_broaden_and_plot.py" --weight-e2f  | tee broaden_and_plot.log
