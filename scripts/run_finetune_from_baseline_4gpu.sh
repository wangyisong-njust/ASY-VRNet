#!/usr/bin/env bash
set -euo pipefail

# Low-LR fine-tune from the reproduced high-score baseline weights.
#
# Two modes (FT_MODE):
#   baseline_control  - same baseline fusion, just continue training at low LR.
#                       This is the CONTROL: it tells us whether "continue from
#                       the high-score checkpoint" by itself moves mAP up or down,
#                       so any reliability gain can be attributed to the gate and
#                       not merely to extra fine-tuning.
#   reliability_fixed - reliability gate fusion with the FIXED identity init
#                       (gate starts ~= 1, i.e. exactly the baseline fusion) and
#                       GroupNorm gate. Fine-tunes the gate on top of the frozen
#                       high-score feature distribution.
#
# Usage:
#   FT_MODE=baseline_control  bash scripts/run_finetune_from_baseline_4gpu.sh
#   FT_MODE=reliability_fixed bash scripts/run_finetune_from_baseline_4gpu.sh

cd "$(dirname "$0")/.."

PROJECT_ROOT=$(pwd)
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${HOME}/anaconda3/envs/PDPP/bin/python" ]]; then
        PYTHON="${HOME}/anaconda3/envs/PDPP/bin/python"
    else
        PYTHON=$(command -v python3 || command -v python || true)
    fi
fi
if [[ -z "${PYTHON}" ]]; then
    echo "No Python interpreter found. Set PYTHON=/path/to/python before running."
    exit 1
fi

FT_MODE=${FT_MODE:-reliability_fixed}
BASELINE_WEIGHTS=${BASELINE_WEIGHTS:-${PROJECT_ROOT}/weights/baseline_best.pth}

if [[ ! -f "${BASELINE_WEIGHTS}" ]]; then
    echo "Baseline weights not found: ${BASELINE_WEIGHTS}"
    echo "Set BASELINE_WEIGHTS=/path/to/best_epoch_weights.pth"
    exit 1
fi

case "${FT_MODE}" in
    baseline_control)
        export ASY_FUSION_MODE=baseline
        DEFAULT_EXP=ft_baseline_control_from_highscore_e60_lr1e4
        ;;
    reliability_fixed)
        export ASY_FUSION_MODE=reliability
        DEFAULT_EXP=ft_reliability_fixed_from_highscore_e60_lr1e4
        ;;
    *)
        echo "Unknown FT_MODE=${FT_MODE} (use baseline_control or reliability_fixed)"
        exit 2
        ;;
esac

EXP_NAME=${EXP_NAME:-${DEFAULT_EXP}}
MASTER_PORT=${MASTER_PORT:-29530}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=${ASY_DISTRIBUTED:-1}
export ASY_SYNC_BN=${ASY_SYNC_BN:-1}
export ASY_FP16=${ASY_FP16:-1}

export ASY_MODEL_PATH=${ASY_MODEL_PATH:-${BASELINE_WEIGHTS}}

export ASY_INPUT_SHAPE=${ASY_INPUT_SHAPE:-320,320}
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-64}
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-8}
# Short low-LR fine-tune: 60 epochs is enough for the gate to adapt.
export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-60}
export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-5}
export ASY_PHI=${ASY_PHI:-l}

# Very low init LR to prevent catastrophic forgetting from a converged checkpoint.
# 0.001 destroyed features in 5 epochs; 1e-4 is ~1/100 of original training LR.
export ASY_INIT_LR=${ASY_INIT_LR:-0.0001}
export ASY_LR_DECAY=${ASY_LR_DECAY:-cos}
export ASY_OPTIMIZER=${ASY_OPTIMIZER:-sgd}
export ASY_MOMENTUM=${ASY_MOMENTUM:-0.937}
export ASY_WEIGHT_DECAY=${ASY_WEIGHT_DECAY:-0.0005}
export ASY_FREEZE_TRAIN=${ASY_FREEZE_TRAIN:-0}
export ASY_YOLO_BOX_WEIGHT=${ASY_YOLO_BOX_WEIGHT:-1.0}
export ASY_YOLO_OBJ_WEIGHT=${ASY_YOLO_OBJ_WEIGHT:-2.0}
export ASY_YOLO_CLS_WEIGHT=${ASY_YOLO_CLS_WEIGHT:-2.0}

export ASY_EVAL=${ASY_EVAL:-0}
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-5}
export ASY_BEST_METRIC=${ASY_BEST_METRIC:-det}

export ASY_SAVE_DIR=${ASY_SAVE_DIR:-logs_${EXP_NAME}}
export ASY_SAVE_DIR_SEG=${ASY_SAVE_DIR_SEG:-logs_seg_${EXP_NAME}}
export ASY_VOCDEVKIT=${ASY_VOCDEVKIT:-${PROJECT_ROOT}/dataset/VOCdevkit}
export ASY_RADAR_ROOT=${ASY_RADAR_ROOT:-${PROJECT_ROOT}/dataset/VOCradar_5_frames}
export ASY_TASK_LOSS=${ASY_TASK_LOSS:-sum}
export ASY_RADAR_DROPOUT=${ASY_RADAR_DROPOUT:-0}
export ASY_RADAR_CHANNELS=${ASY_RADAR_CHANNELS:-4}
export ASY_RADAR_ALIGN_MODE=${ASY_RADAR_ALIGN_MODE:-letterbox}
export ASY_RADAR_NORMALIZE=${ASY_RADAR_NORMALIZE:-0}
export ASY_RADAR_PRESERVE_POINTS=${ASY_RADAR_PRESERVE_POINTS:-0}
export ASY_RADAR_SOURCE_ORDER=${ASY_RADAR_SOURCE_ORDER:-range,doppler,elevation,power}
export ASY_RADAR_TARGET_ORDER=${ASY_RADAR_TARGET_ORDER:-range,doppler,elevation,power}
export ASY_RADAR_LEGACY_PREPROCESS=${ASY_RADAR_LEGACY_PREPROCESS:-1}
export ASY_WEATHER_AUG=${ASY_WEATHER_AUG:-0}

mkdir -p "${ASY_SAVE_DIR}" "${ASY_SAVE_DIR_SEG}"

printf 'Fine-tune from baseline config:\n'
printf '  mode=%s exp=%s\n' "${FT_MODE}" "${EXP_NAME}"
printf '  init_weights=%s\n' "${ASY_MODEL_PATH}"
printf '  fusion=%s lr=%s epochs=%s batch=%s\n' "${ASY_FUSION_MODE}" "${ASY_INIT_LR}" "${ASY_UNFREEZE_EPOCH}" "${ASY_BATCH_SIZE}"

if [[ "${ASY_SKIP_STARTUP_CHECKS:-0}" =~ ^(0|false|FALSE|no|NO|off|OFF)$ ]]; then
    "${PYTHON}" scripts/check_dataset.py
    "${PYTHON}" scripts/audit_detection_pipeline.py --sample_limit 256 --skip_model
fi

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
