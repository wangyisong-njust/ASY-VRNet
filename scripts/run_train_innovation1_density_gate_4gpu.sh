#!/usr/bin/env bash
set -euo pipefail

# Innovation 1 v2: reliability gate + radar-density-aware gate input.
#
# Extends the original reliability gate by appending an explicit radar-occupancy
# density map (sigmoid of per-pixel L2-norm across radar channels) to the gate
# input in both ImageEnhanceByRadar and RadarEnhanceByImage.  The gate can now
# directly see "is radar present at this spatial location?" rather than having
# to infer it from feature magnitudes, giving it a stronger physics signal to
# decide how much to trust the radar branch.
#
# Key differences from fine-tune v1 (60e, lr=1e-4):
#   - 300 epochs from scratch (same schedule as the reproduced highscore baseline)
#   - ASY_GATE_DENSITY=1 enables the extra density channel in both gate modules
#   - No pre-trained init required; identity init (bias=+4 → sigmoid≈0.982) is
#     preserved regardless of the extra channel (last-conv weight=0 dominates)
#
# Usage:
#   bash scripts/run_train_innovation1_density_gate_4gpu.sh

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

# --- Innovation 1 v2 switches ---
export ASY_FUSION_MODE=${ASY_FUSION_MODE:-reliability}
export ASY_GATE_DENSITY=${ASY_GATE_DENSITY:-1}

EXP_NAME=${EXP_NAME:-innovation1_density_gate_phi_l_5frames_bs64_300e_320}
MASTER_PORT=${MASTER_PORT:-29560}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=${ASY_DISTRIBUTED:-1}
export ASY_SYNC_BN=${ASY_SYNC_BN:-1}
export ASY_FP16=${ASY_FP16:-1}

export ASY_INPUT_SHAPE=${ASY_INPUT_SHAPE:-320,320}
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-64}
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-8}
export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-300}
export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-20}
export ASY_PHI=${ASY_PHI:-l}

export ASY_INIT_LR=${ASY_INIT_LR:-0.01}
export ASY_LR_DECAY=${ASY_LR_DECAY:-cos}
export ASY_OPTIMIZER=${ASY_OPTIMIZER:-sgd}
export ASY_MOMENTUM=${ASY_MOMENTUM:-0.937}
export ASY_WEIGHT_DECAY=${ASY_WEIGHT_DECAY:-0.0005}
export ASY_FREEZE_TRAIN=${ASY_FREEZE_TRAIN:-0}
export ASY_YOLO_BOX_WEIGHT=${ASY_YOLO_BOX_WEIGHT:-1.0}
export ASY_YOLO_OBJ_WEIGHT=${ASY_YOLO_OBJ_WEIGHT:-2.0}
export ASY_YOLO_CLS_WEIGHT=${ASY_YOLO_CLS_WEIGHT:-2.0}

export ASY_EVAL=${ASY_EVAL:-0}
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-10}
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

printf 'Innovation-1 density-gate training config:\n'
printf '  exp=%s\n' "${EXP_NAME}"
printf '  fusion=%s gate_density=%s lr=%s epochs=%s batch=%s\n' \
    "${ASY_FUSION_MODE}" "${ASY_GATE_DENSITY}" "${ASY_INIT_LR}" "${ASY_UNFREEZE_EPOCH}" "${ASY_BATCH_SIZE}"
printf '  init_weights=%s\n' "${ASY_MODEL_PATH:-<none, training from scratch>}"

if [[ "${ASY_SKIP_STARTUP_CHECKS:-0}" =~ ^(0|false|FALSE|no|NO|off|OFF)$ ]]; then
    "${PYTHON}" scripts/check_dataset.py
    "${PYTHON}" scripts/audit_detection_pipeline.py --sample_limit 256 --skip_model
fi

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
