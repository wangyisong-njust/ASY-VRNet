#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export EXP_NAME=${EXP_NAME:-detection_baseline_phi_l_5frames_bs64_300e_320}
export TASK_LOSS=${TASK_LOSS:-sum}
export PHI=${PHI:-l}
export CONFIDENCE=${CONFIDENCE:-0.001}
export MAX_BOXES=${MAX_BOXES:-100}
export BEST_OUT=${BEST_OUT:-results/detection_baseline_best}
export LAST_OUT=${LAST_OUT:-results/detection_baseline_last}

exec bash scripts/after_train_eval_and_diagnose.sh "$@"
