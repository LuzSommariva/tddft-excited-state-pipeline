#!/usr/bin/env bash
# Tier 3 — excited-state AIMD emission band, PCM + full TDDFT throughout.
# Usage: run_tier3.sh <coumarin102|p_terphenyl> [--gpu]
#
# For resuming an interrupted MD run, or for a custom step count, call
# src/tier3/11_s1_md_run*.py directly with --resume / --burnin-steps / etc.
# instead of this wrapper.
set -euo pipefail

MOLECULE="${1:?Usage: run_tier3.sh <coumarin102|p_terphenyl> [--gpu]}"
GPU_FLAG="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src/common:${PYTHONPATH:-}"

case "$MOLECULE" in
  coumarin102)
    SUFFIX=""
    ;;
  p_terphenyl)
    SUFFIX="_pterphenyl"
    ;;
  *)
    echo "Unknown molecule: $MOLECULE (expected coumarin102 or p_terphenyl)" >&2
    exit 1
    ;;
esac

WORKDIR="$REPO_ROOT/molecules/$MOLECULE/results"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

python3 "$REPO_ROOT/src/tier3/04_s0_optimize_pcm${SUFFIX}.py"     | tee s0_optimize_pcm.log
python3 "$REPO_ROOT/src/tier3/06_s1_optimize_gpu_pcm${SUFFIX}.py" | tee s1_optimize_gpu_pcm.log

python3 "$REPO_ROOT/src/tier3/11_s1_md_run${SUFFIX}.py" $GPU_FLAG | tee s1_md_run.log

python3 "$REPO_ROOT/src/tier3/12_sample_trajectory.py" \
    --traj "s1_md_trajectory${SUFFIX}.traj" \
    --out "s1_md_samples${SUFFIX}.xyz" \
    --burnin-steps 100 --interval-steps 20

python3 "$REPO_ROOT/src/tier3/13_ensemble_emission_pcm${SUFFIX}.py" $GPU_FLAG \
    | tee ensemble_emission_pcm_tier3.log

python3 "$REPO_ROOT/src/tier3/14_plot_emission_final.py" \
    --molecule "$MOLECULE" \
    --csv "ensemble_emission_pcm_tier3${SUFFIX}.csv" \
    --out-prefix "emission_spectrum_tier3${SUFFIX}_final" \
    --weight-e2f
