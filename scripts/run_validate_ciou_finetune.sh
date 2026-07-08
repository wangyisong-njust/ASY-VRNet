#!/usr/bin/env bash
# Validation run: CIoU loss fine-tune of innov2 (20 epochs, L40).
# Goal: verify CIoU improves over innov2's plain-IoU baseline before
# committing to a full 300e A100 run.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2 nohup bash scripts/run_validate_ciou_finetune.sh \
#       > logs/validate_ciou_ft.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
PYTHON=${PYTHON:-${HOME}/anaconda3/envs/PDPP/bin/python}

EXP_NAME=validate_ciou_ft_innov2_e20
MASTER_PORT=29576
NPROC=${NPROC:-3}

export ASY_MODEL_PATH=${PROJECT_ROOT}/weights/innov1_qfl_radar_best.pth
export ASY_QFL=1
export ASY_RADAR_PRIOR=1
export ASY_RADAR_PRIOR_WEIGHT=0.5
export ASY_QFL_BETA=2.0
export ASY_FUSION_MODE=baseline

# CIoU regression loss (the innovation being validated)
export ASY_IOU_LOSS_TYPE=ciou

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export ASY_DISTRIBUTED=1
export ASY_SYNC_BN=1
export ASY_FP16=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=1800
export OMP_NUM_THREADS=4

export ASY_INPUT_SHAPE=320,320
export ASY_BATCH_SIZE=48           # 16/GPU x 3
export ASY_NUM_WORKERS=8
export ASY_UNFREEZE_EPOCH=20       # quick validation
export ASY_SAVE_PERIOD=20
export ASY_PHI=l
export ASY_INIT_LR=0.0001
export ASY_LR_DECAY=cos
export ASY_OPTIMIZER=sgd
export ASY_MOMENTUM=0.937
export ASY_WEIGHT_DECAY=0.0005
export ASY_FREEZE_TRAIN=0
export ASY_YOLO_BOX_WEIGHT=1.0
export ASY_YOLO_OBJ_WEIGHT=2.0
export ASY_YOLO_CLS_WEIGHT=2.0
export ASY_EVAL=${ASY_EVAL:-0}                # mid-training mAP eval OFF by default (DDP-safe)
export ASY_EVAL_PERIOD=5
export ASY_BEST_METRIC=det
export ASY_SAVE_DIR=logs_${EXP_NAME}
export ASY_SAVE_DIR_SEG=logs_seg_${EXP_NAME}
export ASY_VOCDEVKIT=${PROJECT_ROOT}/dataset/VOCdevkit
export ASY_RADAR_ROOT=${PROJECT_ROOT}/dataset/VOCradar_5_frames
export ASY_TASK_LOSS=sum
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

echo "=== CIoU validation fine-tune: 20e from innov2 best ==="
echo "  iou_loss=ciou  lr=${ASY_INIT_LR}  gpus=${CUDA_VISIBLE_DEVICES}"

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NPROC}" train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
train_exit=$?
echo "TRAIN DONE (exit=${train_exit})"
[[ "${train_exit}" -ne 0 ]] && { echo "Training failed"; exit "${train_exit}"; }

sleep 10
echo "=== EVAL best checkpoint (paper protocol) ==="
ASY_IOU_LOSS_TYPE=ciou CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} \
    "${PYTHON}" eval_paper_metrics.py \
    --model_path "${ASY_SAVE_DIR}/best_epoch_weights.pth" \
    --fusion_mode baseline --phi l \
    --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
    --radar_root "${ASY_RADAR_ROOT}" --vocdevkit_path "${ASY_VOCDEVKIT}" \
    --radar_legacy_preprocess --no_radar_preserve_points \
    --radar_source_order range,doppler,elevation,power \
    --radar_target_order range,doppler,elevation,power \
    --task_loss sum \
    --dark_times night --dim_lightings dim --dim_times daytime,night \
    --dim_weathers overcast,rainy --small_area 4096 --small_area_space original \
    --out_dir results/validate_ciou_best 2>&1 | tail -20

echo ""
echo "=== RESULT (innov2 baseline: 49.958) ==="
"${PYTHON}" -c "
import json, os
f = 'results/validate_ciou_best/paper_metrics.json'
if os.path.exists(f):
    d = json.load(open(f))
    print(f'  mAP50-95 : {d[\"mAP50-95\"]:.3f}')
    print(f'  AP50     : {d[\"AP50\"]:.3f}')
    print(f'  AP_small : {d[\"AP_small\"]:.3f}')
    print(f'  dark_mAP : {d[\"dark_mAP50-95\"]:.3f}')
    delta = d['mAP50-95'] - 49.958
    print(f'  vs innov2: {delta:+.3f}')
"
