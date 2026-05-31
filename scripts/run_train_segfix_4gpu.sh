#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT=$(pwd)
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON=$(command -v python3 || command -v python || true)
fi
if [[ -z "${PYTHON}" ]]; then
    echo "No Python interpreter found. Set PYTHON=/path/to/python before running."
    exit 1
fi
EXP_NAME=${EXP_NAME:-paper_baseline_segfix_4task_bs128_e130_320}
MASTER_PORT=${MASTER_PORT:-29500}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=${ASY_DISTRIBUTED:-1}
export ASY_SYNC_BN=${ASY_SYNC_BN:-1}
export ASY_FP16=${ASY_FP16:-1}

export ASY_INPUT_SHAPE=${ASY_INPUT_SHAPE:-320,320}
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-128}
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-16}
export ASY_INIT_EPOCH=${ASY_INIT_EPOCH:-100}
export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-130}
export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-10}

# Fine-tune from the completed bs128 run with a lower LR. train.py scales by
# global_batch / 64, so 0.0005 becomes an effective peak LR of 0.001 at bs128.
export ASY_MODEL_PATH=${ASY_MODEL_PATH:-logs_paper_baseline_latestfix_ddpfix_5frames_uncert_bs128_100e_320/last_epoch_weights.pth}
export ASY_INIT_LR=${ASY_INIT_LR:-0.0005}
export ASY_LR_DECAY=${ASY_LR_DECAY:-cos}
export ASY_OPTIMIZER=${ASY_OPTIMIZER:-sgd}
export ASY_FREEZE_TRAIN=${ASY_FREEZE_TRAIN:-0}

export ASY_EVAL=${ASY_EVAL:-0}
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-10}

export ASY_SAVE_DIR=${ASY_SAVE_DIR:-logs_${EXP_NAME}}
export ASY_SAVE_DIR_SEG=${ASY_SAVE_DIR_SEG:-logs_seg_${EXP_NAME}}
export ASY_VOCDEVKIT=${ASY_VOCDEVKIT:-${PROJECT_ROOT}/dataset/VOCdevkit}
export ASY_RADAR_ROOT=${ASY_RADAR_ROOT:-${PROJECT_ROOT}/dataset/VOCradar_5_frames}
export ASY_TASK_LOSS=${ASY_TASK_LOSS:-uncertainty}
export ASY_FUSION_MODE=${ASY_FUSION_MODE:-baseline}
export ASY_RADAR_DROPOUT=${ASY_RADAR_DROPOUT:-0}
export ASY_RADAR_CHANNELS=${ASY_RADAR_CHANNELS:-4}
export ASY_RADAR_ALIGN_MODE=${ASY_RADAR_ALIGN_MODE:-letterbox}
export ASY_RADAR_NORMALIZE=${ASY_RADAR_NORMALIZE:-0}

# Slightly emphasize drivable-area pixels to close the mIoU_d gap while keeping
# object classes unchanged.
export ASY_SEG_CLASS_WEIGHTS=${ASY_SEG_CLASS_WEIGHTS:-1,1,1,1,1,1,1,1,2}
export ASY_DICE_LOSS=${ASY_DICE_LOSS:-1}
export ASY_FOCAL_LOSS=${ASY_FOCAL_LOSS:-1}

mkdir -p "${ASY_SAVE_DIR}" "${ASY_SAVE_DIR_SEG}" logs

"$PYTHON" scripts/check_dataset.py

stamp=$(date +%Y%m%d_%H%M%S)
"$PYTHON" -m torch.distributed.run --master_port="${MASTER_PORT}" --nproc_per_node=4 train.py 2>&1 | tee "logs/segfix_4task_${stamp}.log"
