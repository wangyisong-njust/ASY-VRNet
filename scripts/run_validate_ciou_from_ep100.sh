#!/usr/bin/env bash
# CIoU validation: continue innov2 training from ep100 with CIoU loss.
#
# Key insight: innov2 val_loss minimum is ep160 (1.395), not ep300 (1.986).
# Fine-tuning from the converged best always fails because the model is locked
# in a sharp minimum. Starting from ep100 (loss still falling, model plastic)
# and running 100 more epochs with CIoU at lr=1e-3 is a genuine test of
# whether CIoU improves over plain-IoU.
#
# Quick mode  (EPOCHS=30, ~2h on L40, early signal):
#   EPOCHS=30 CUDA_VISIBLE_DEVICES=0,1,2 nohup bash scripts/run_validate_ciou_from_ep100.sh \
#       > logs/validate_ciou_ep100.out 2>&1 &
#
# Full mode   (EPOCHS=100, ~6h on L40, definitive result):
#   EPOCHS=100 CUDA_VISIBLE_DEVICES=0,1,2 nohup bash scripts/run_validate_ciou_from_ep100.sh \
#       > logs/validate_ciou_ep100.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
PYTHON=${PYTHON:-${HOME}/anaconda3/envs/PDPP/bin/python}

EPOCHS=${EPOCHS:-30}
EXP_NAME=validate_ciou_from_ep100_e${EPOCHS}
MASTER_PORT=29577
NPROC=${NPROC:-3}

# Init from the quality-aligned model. The original experiment started from the
# ep100 checkpoint; the shipped weights/ keep only best + ep140/160/180/200, so
# point INNOV2_INIT at one of those (override with INNOV2_INIT=... if you have ep100).
INNOV2_INIT=${INNOV2_INIT:-${PROJECT_ROOT}/weights/innov1_qfl_radar_best.pth}
[[ -f "${INNOV2_INIT}" ]] || { echo "ERROR: init checkpoint not found: ${INNOV2_INIT}"; exit 1; }

export ASY_MODEL_PATH=${INNOV2_INIT}
export ASY_QFL=1
export ASY_RADAR_PRIOR=1
export ASY_RADAR_PRIOR_WEIGHT=0.5
export ASY_QFL_BETA=2.0
export ASY_FUSION_MODE=baseline

# CIoU — the innovation being tested
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
export ASY_BATCH_SIZE=48           # 16/GPU × 3
export ASY_NUM_WORKERS=8
export ASY_UNFREEZE_EPOCH=${EPOCHS}
export ASY_SAVE_PERIOD=${EPOCHS}   # only save final checkpoint
export ASY_PHI=l
export ASY_INIT_LR=0.001           # same order as innov2 ep100 LR (not too small)
export ASY_LR_DECAY=cos
export ASY_OPTIMIZER=sgd
export ASY_MOMENTUM=0.937
export ASY_WEIGHT_DECAY=0.0005
export ASY_FREEZE_TRAIN=0
export ASY_YOLO_BOX_WEIGHT=1.0
export ASY_YOLO_OBJ_WEIGHT=2.0
export ASY_YOLO_CLS_WEIGHT=2.0
# Mid-training mAP eval defaults OFF (it blocks rank 0 for ~13 min and is the
# historical cause of the DDP SIGABRT). "best" is picked by val loss; the full
# mAP eval runs once at the end. Override with ASY_EVAL=1 to force it on.
export ASY_EVAL=${ASY_EVAL:-0}
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-10}
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

echo "=== CIoU validation: ep100 init, ${EPOCHS} epochs, lr=${ASY_INIT_LR} ==="
echo "  init=${ASY_MODEL_PATH}"

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NPROC}" train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
train_exit=$?
echo "TRAIN DONE (exit=${train_exit})"
[[ "${train_exit}" -ne 0 ]] && exit "${train_exit}"

sleep 10
echo "=== EVAL best checkpoint ==="
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
    --out_dir results/${EXP_NAME} 2>&1 | tail -20

echo ""
echo "=== RESULT (innov2 ep160-best=49.958 | ep100 start point) ==="
"${PYTHON}" -c "
import json, os
f = 'results/${EXP_NAME}/paper_metrics.json'
if os.path.exists(f):
    d = json.load(open(f))
    print(f'  mAP50-95 : {d[\"mAP50-95\"]:.3f}')
    print(f'  AP50     : {d[\"AP50\"]:.3f}')
    print(f'  AP_small : {d[\"AP_small\"]:.3f}')
    print(f'  dark_mAP : {d.get(\"dark_mAP50-95\", 0):.3f}')
    delta = d['mAP50-95'] - 49.958
    print(f'  vs innov2: {delta:+.3f}  ({\"positive\" if delta > 0 else \"negative\"})')
    if delta > 0:
        print('  => POSITIVE: commit to full 300e on A100')
    else:
        print('  => Try ep120 init or longer epochs before committing to A100')
"
