#!/usr/bin/env bash
# Innovation 3 (v2): scale-aware fine-tuning of the innov2 (QFL+radar) model.
#
# The TTA study showed the detector is over-fitted to a single 320 scale, so
# multi-scale test-time fusion hurts. Here we fix the cause: fine-tune innov2
# with the input size cycling over {320,384,448} per epoch (ASY_MULTISCALE).
# This makes the model scale-robust, which both lifts the base-320 result and
# finally lets the radar-aware multi-scale TTA add on top.
#
# Eval / validation stay at 320 (paper protocol), so the result is directly
# comparable to baseline 42.570 / innov2 49.958.
#
# Usage (3 free GPUs on L40):
#   CUDA_VISIBLE_DEVICES=0,2,3 nohup bash scripts/run_finetune_innov3_multiscale.sh \
#       > logs/innov3_multiscale_ft.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
PYTHON=${PYTHON:-${HOME}/anaconda3/envs/PDPP/bin/python}

EXP_NAME=${EXP_NAME:-innovation3_multiscale_ft_phi_l_5frames_bs48_e50_320}
MASTER_PORT=${MASTER_PORT:-29555}
NPROC=${NPROC:-3}

# --- start from the innov2 best checkpoint (the current strongest model) ---
export ASY_MODEL_PATH=${ASY_MODEL_PATH:-${PROJECT_ROOT}/logs_innovation2_qfl_radar_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth}

# --- innov2 head settings (QFL + radar prior), unchanged ---
export ASY_QFL=1
export ASY_RADAR_PRIOR=1
export ASY_RADAR_PRIOR_WEIGHT=${ASY_RADAR_PRIOR_WEIGHT:-0.5}
export ASY_QFL_BETA=${ASY_QFL_BETA:-2.0}

# --- scale-aware fine-tuning (the innovation) ---
export ASY_MULTISCALE=1
export ASY_MULTISCALE_SCALES=${ASY_MULTISCALE_SCALES:-320,384,448}

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3}
export ASY_DISTRIBUTED=1
export ASY_SYNC_BN=1
export ASY_FP16=1
export ASY_INPUT_SHAPE=320,320          # base/val/eval scale (paper protocol)
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-48}   # 16 per GPU x 3 GPUs
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-8}
export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-50}
export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-10}
export ASY_PHI=l
export ASY_INIT_LR=${ASY_INIT_LR:-0.001}      # conservative fine-tune LR
export ASY_LR_DECAY=cos
export ASY_OPTIMIZER=sgd
export ASY_MOMENTUM=0.937
export ASY_WEIGHT_DECAY=0.0005
export ASY_FREEZE_TRAIN=0
export ASY_YOLO_BOX_WEIGHT=1.0
export ASY_YOLO_OBJ_WEIGHT=2.0
export ASY_YOLO_CLS_WEIGHT=2.0
export ASY_EVAL=${ASY_EVAL:-0}                # mid-training mAP eval OFF by default (DDP-safe)
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-10}
export ASY_BEST_METRIC=det
export ASY_SAVE_DIR=logs_${EXP_NAME}
export ASY_SAVE_DIR_SEG=logs_seg_${EXP_NAME}
export ASY_VOCDEVKIT=${ASY_VOCDEVKIT:-${PROJECT_ROOT}/dataset/VOCdevkit}
export ASY_RADAR_ROOT=${ASY_RADAR_ROOT:-${PROJECT_ROOT}/dataset/VOCradar_5_frames}
export ASY_TASK_LOSS=sum
export ASY_FUSION_MODE=baseline
export ASY_RADAR_DROPOUT=0
export ASY_RADAR_CHANNELS=4
export ASY_RADAR_ALIGN_MODE=letterbox
export ASY_RADAR_NORMALIZE=0
export ASY_RADAR_PRESERVE_POINTS=0
export ASY_RADAR_SOURCE_ORDER=range,doppler,elevation,power
export ASY_RADAR_TARGET_ORDER=range,doppler,elevation,power
export ASY_RADAR_LEGACY_PREPROCESS=1
export ASY_WEATHER_AUG=0
export ASY_SKIP_STARTUP_CHECKS=1

mkdir -p "${ASY_SAVE_DIR}" "${ASY_SAVE_DIR_SEG}" logs

printf 'Innovation-3 multi-scale fine-tune config:\n'
printf '  init=%s\n  scales=%s lr=%s epochs=%s batch=%s gpus=%s\n' \
    "${ASY_MODEL_PATH}" "${ASY_MULTISCALE_SCALES}" "${ASY_INIT_LR}" \
    "${ASY_UNFREEZE_EPOCH}" "${ASY_BATCH_SIZE}" "${CUDA_VISIBLE_DEVICES}"

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NPROC}" train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
echo "TRAIN DONE (exit=$?): innov3 multiscale fine-tune"
