#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export EXP_NAME=${EXP_NAME:-legacy_highscore_phi_l_5frames_bs64_300e_320}
export TASK_LOSS=${TASK_LOSS:-sum}
export PHI=${PHI:-l}
export CONFIDENCE=${CONFIDENCE:-0.001}
export MAX_BOXES=${MAX_BOXES:-100}
export RADAR_PRESERVE_POINTS=${RADAR_PRESERVE_POINTS:-0}
export RADAR_SOURCE_ORDER=${RADAR_SOURCE_ORDER:-range,doppler,elevation,power}
export RADAR_TARGET_ORDER=${RADAR_TARGET_ORDER:-range,doppler,elevation,power}
export RADAR_LEGACY_PREPROCESS=${RADAR_LEGACY_PREPROCESS:-1}
export BEST_OUT=${BEST_OUT:-results/legacy_highscore_best}
export LAST_OUT=${LAST_OUT:-results/legacy_highscore_last}

exec bash scripts/after_train_eval_and_diagnose.sh "$@"
