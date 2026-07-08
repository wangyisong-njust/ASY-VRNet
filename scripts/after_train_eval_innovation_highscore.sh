#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INNOVATION_VARIANT=${INNOVATION_VARIANT:-i3_full}
export EXP_NAME=${EXP_NAME:-innovation_highscore_${INNOVATION_VARIANT}_phi_l_5frames_bs64_e220_320}

if [[ "${INNOVATION_VARIANT}" == "i4_uncertainty" ]]; then
    export TASK_LOSS=${TASK_LOSS:-uncertainty}
else
    export TASK_LOSS=${TASK_LOSS:-sum}
fi

export PHI=${PHI:-l}
export CONFIDENCE=${CONFIDENCE:-0.001}
export MAX_BOXES=${MAX_BOXES:-100}
export RADAR_PRESERVE_POINTS=${RADAR_PRESERVE_POINTS:-0}
export RADAR_SOURCE_ORDER=${RADAR_SOURCE_ORDER:-range,doppler,elevation,power}
export RADAR_TARGET_ORDER=${RADAR_TARGET_ORDER:-range,doppler,elevation,power}
export RADAR_LEGACY_PREPROCESS=${RADAR_LEGACY_PREPROCESS:-1}
export DARK_TIMES=${DARK_TIMES:-night}
export DIM_LIGHTINGS=${DIM_LIGHTINGS:-dim}
export DIM_TIMES=${DIM_TIMES:-daytime,night}
export DIM_WEATHERS=${DIM_WEATHERS:-overcast,rainy}
export SMALL_AREA=${SMALL_AREA:-4096}
export SMALL_AREA_SPACE=${SMALL_AREA_SPACE:-original}
export BEST_OUT=${BEST_OUT:-results/${EXP_NAME}_best}
export LAST_OUT=${LAST_OUT:-results/${EXP_NAME}_last}

case "${INNOVATION_VARIANT}" in
    i1_reliability)
        export FUSION_MODE=${FUSION_MODE:-reliability}
        export ASY_RADAR_DROPOUT=${ASY_RADAR_DROPOUT:-0}
        ;;
    i2_reliability_rdrop|i3_full|i4_uncertainty)
        export FUSION_MODE=${FUSION_MODE:-reliability}
        export ASY_RADAR_DROPOUT=${ASY_RADAR_DROPOUT:-0.03}
        ;;
    *)
        echo "Unknown INNOVATION_VARIANT=${INNOVATION_VARIANT}"
        echo "Supported: i1_reliability, i2_reliability_rdrop, i3_full, i4_uncertainty"
        exit 2
        ;;
esac

exec bash scripts/after_train_eval_and_diagnose.sh "$@"
