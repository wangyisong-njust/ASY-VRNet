#!/usr/bin/env bash
# Combined full model from scratch: QFL + radar prior + reliability gate (all three innovations).
# Trains 300e from scratch on A100 GPU 0-3.
# Goal: definitive ablation entry showing all three innovations together.
#
# Usage (on A100):
#   nohup bash scripts/a100_run_combined_3in1_scratch.sh \
#       > logs/a100_combined_3in1_scratch.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
WORK_DIR=$(pwd)
PYTHON=${PYTHON:-~/miniconda3/envs/torch21new/bin/python}
PYTHON=$(eval echo "${PYTHON}")
log() { printf '\n[A100_3IN1 %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

EXP_NAME=combined_3in1_qfl_radar_gate_phi_l_5frames_bs64_300e_320
MASTER_PORT=29571

# --- three innovations: QFL + radar prior + reliability gate ---
export ASY_QFL=1
export ASY_RADAR_PRIOR=1
export ASY_RADAR_PRIOR_WEIGHT=0.5
export ASY_QFL_BETA=2.0
export ASY_FUSION_MODE=reliability
export ASY_GATE_DENSITY=0

# no pretrained model — full 300e training from scratch
unset ASY_MODEL_PATH 2>/dev/null || true

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=1
export ASY_SYNC_BN=1
export ASY_FP16=1

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=1800
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

export ASY_INPUT_SHAPE=320,320
export ASY_BATCH_SIZE=64
export ASY_NUM_WORKERS=8
export ASY_UNFREEZE_EPOCH=300
export ASY_SAVE_PERIOD=20
export ASY_PHI=l
export ASY_INIT_LR=0.01               # same as innov2 from-scratch LR
export ASY_LR_DECAY=cos
export ASY_OPTIMIZER=sgd
export ASY_MOMENTUM=0.937
export ASY_WEIGHT_DECAY=0.0005
export ASY_FREEZE_TRAIN=0
export ASY_YOLO_BOX_WEIGHT=1.0
export ASY_YOLO_OBJ_WEIGHT=2.0
export ASY_YOLO_CLS_WEIGHT=2.0
export ASY_EVAL=0
export ASY_BEST_METRIC=det
export ASY_SAVE_DIR=logs_${EXP_NAME}
export ASY_SAVE_DIR_SEG=logs_seg_${EXP_NAME}
export ASY_VOCDEVKIT=${WORK_DIR}/dataset/VOCdevkit
export ASY_RADAR_ROOT=${WORK_DIR}/dataset/VOCradar_5_frames
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

log "TRAIN START: combined 3-in-1 from scratch 300e"
log "  qfl=${ASY_QFL} radar_prior=${ASY_RADAR_PRIOR} fusion=${ASY_FUSION_MODE} lr=${ASY_INIT_LR} gpus=${CUDA_VISIBLE_DEVICES}"

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" \
    --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
train_exit=$?
log "TRAIN DONE (exit=${train_exit})"
if [[ "${train_exit}" -ne 0 ]]; then
    log "ERROR: training exited ${train_exit}; skipping eval."
    [[ ! -f "${ASY_SAVE_DIR}/best_epoch_weights.pth" ]] && exit "${train_exit}"
    log "best ckpt exists -> proceeding to eval anyway."
fi

sleep 20
log "EVAL best checkpoint"
ASY_GATE_DENSITY=0 "${PYTHON}" eval_paper_metrics.py \
    --model_path "${ASY_SAVE_DIR}/best_epoch_weights.pth" \
    --fusion_mode reliability --phi l \
    --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
    --radar_root "${ASY_RADAR_ROOT}" --vocdevkit_path "${ASY_VOCDEVKIT}" \
    --radar_legacy_preprocess --no_radar_preserve_points \
    --radar_source_order range,doppler,elevation,power \
    --radar_target_order range,doppler,elevation,power \
    --task_loss sum \
    --dark_times night --dim_lightings dim --dim_times daytime,night \
    --dim_weathers overcast,rainy --small_area 4096 --small_area_space original \
    --out_dir paper_metrics_combined_3in1_best 2>&1 | tail -15

echo "DONE" > "${WORK_DIR}/logs/a100_combined_3in1_COMPLETE.flag"
log "ALL DONE. baseline 42.570 | innov2 49.958"
