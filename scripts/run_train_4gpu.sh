#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/kaixin/anaconda3/envs/PDPP/bin/python}
export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=${ASY_DISTRIBUTED:-1}
export ASY_SYNC_BN=${ASY_SYNC_BN:-1}
export ASY_FP16=${ASY_FP16:-1}
export ASY_INPUT_SHAPE=${ASY_INPUT_SHAPE:-320,320}
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-16}
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-8}
export ASY_EVAL=${ASY_EVAL:-0}
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-5}
export ASY_VOCDEVKIT=${ASY_VOCDEVKIT:-/home/kaixin/code/ASY-VRNet/dataset/VOCdevkit}
export ASY_RADAR_ROOT=${ASY_RADAR_ROOT:-/home/kaixin/code/ASY-VRNet/dataset/VOCradar}

mkdir -p logs

"$PYTHON" scripts/check_dataset.py

stamp=$(date +%Y%m%d_%H%M%S)
"$PYTHON" -m torch.distributed.run --nproc_per_node=4 train.py 2>&1 | tee "logs/train_${stamp}.log"
