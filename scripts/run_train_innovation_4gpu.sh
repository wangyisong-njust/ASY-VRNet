#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONNOUSERSITE=1
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON=$(command -v python3 || command -v python || true)
fi
if [[ -z "${PYTHON}" ]]; then
    echo "No Python interpreter found. Set PYTHON=/path/to/python before running."
    exit 1
fi
export PYTHON

export EXP_NAME=${EXP_NAME:-innovation_reliability_gate_phi_nano_5frames_bs16_100e_320}
export MASTER_PORT=${MASTER_PORT:-29501}

export ASY_DISTRIBUTED=${ASY_DISTRIBUTED:-1}
export ASY_SYNC_BN=${ASY_SYNC_BN:-1}
export ASY_FP16=${ASY_FP16:-1}

export ASY_INPUT_SHAPE=${ASY_INPUT_SHAPE:-320,320}
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-16}
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-8}
export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-100}
export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-10}
export ASY_PHI=${ASY_PHI:-nano}

# train.py scales Init_lr by global_batch / 64. With global batch 16,
# ASY_INIT_LR=0.04 gives an effective initial SGD LR of 0.01.
export ASY_INIT_LR=${ASY_INIT_LR:-0.04}
export ASY_LR_DECAY=${ASY_LR_DECAY:-cos}
export ASY_OPTIMIZER=${ASY_OPTIMIZER:-sgd}
export ASY_FREEZE_TRAIN=${ASY_FREEZE_TRAIN:-0}

export ASY_EVAL=${ASY_EVAL:-0}
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-10}

export ASY_SAVE_DIR=${ASY_SAVE_DIR:-logs_${EXP_NAME}}
export ASY_SAVE_DIR_SEG=${ASY_SAVE_DIR_SEG:-logs_seg_${EXP_NAME}}
export ASY_VOCDEVKIT=${ASY_VOCDEVKIT:-${PROJECT_ROOT}/dataset/VOCdevkit}
export ASY_RADAR_ROOT=${ASY_RADAR_ROOT:-${PROJECT_ROOT}/dataset/VOCradar_5_frames}
export ASY_RADAR_ALIGN_MODE=${ASY_RADAR_ALIGN_MODE:-letterbox}
export ASY_RADAR_NORMALIZE=${ASY_RADAR_NORMALIZE:-0}

# Innovation switches. Set each one back to its baseline value for ablations.
export ASY_RADAR_CHANNELS=${ASY_RADAR_CHANNELS:-4}
export ASY_FUSION_MODE=${ASY_FUSION_MODE:-reliability}
export ASY_RADAR_DROPOUT=${ASY_RADAR_DROPOUT:-0.05}
export ASY_TASK_LOSS=${ASY_TASK_LOSS:-uncertainty}

mkdir -p "${ASY_SAVE_DIR}" "${ASY_SAVE_DIR_SEG}"
stamp=$(date +%Y%m%d_%H%M%S)

"$PYTHON" scripts/check_dataset.py

"$PYTHON" -m torch.distributed.run --master_port="${MASTER_PORT}" --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
