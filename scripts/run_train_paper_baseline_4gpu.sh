#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/kaixin/anaconda3/envs/PDPP/bin/python}
EXP_NAME=${EXP_NAME:-paper_baseline_lr001_300e_320}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=${ASY_DISTRIBUTED:-1}
export ASY_SYNC_BN=${ASY_SYNC_BN:-1}
export ASY_FP16=${ASY_FP16:-1}

export ASY_INPUT_SHAPE=${ASY_INPUT_SHAPE:-320,320}
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-16}
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-8}
export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-300}
export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-20}

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
export ASY_VOCDEVKIT=${ASY_VOCDEVKIT:-/home/kaixin/code/ASY-VRNet/dataset/VOCdevkit}
export ASY_RADAR_ROOT=${ASY_RADAR_ROOT:-/home/kaixin/code/ASY-VRNet/dataset/VOCradar}

mkdir -p "${ASY_SAVE_DIR}" "${ASY_SAVE_DIR_SEG}"

"$PYTHON" scripts/check_dataset.py

stamp=$(date +%Y%m%d_%H%M%S)
"$PYTHON" -m torch.distributed.run --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
