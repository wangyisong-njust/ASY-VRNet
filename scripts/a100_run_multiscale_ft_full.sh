#!/usr/bin/env bash
# Proper multi-scale fine-tune of innov2 (the 创新点二 soup ingredient).
# The original multiscale-ft only ran 9 epochs (crashed). This trains it to
# 50 epochs properly, producing a solid scale-robust checkpoint that:
#   1. removes the 9-epoch reproducibility risk for 创新点二 (soup),
#   2. is a stronger soup ingredient -> soup likely > 50.114,
#   3. is a better TTA base -> 创新点三 likely > 51.186.
#
# Runs on A100 GPU 0-3 (shared with another user: ~27GB free per card).
# Uses conservative batch (12/GPU) and ASY_EVAL=0 to avoid shared-GPU OOM
# during a mid-training eval. Final eval runs once at the end.
#
# Usage (on A100):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash scripts/a100_run_multiscale_ft_full.sh \
#       > logs/a100_multiscale_ft_full.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
WORK_DIR=$(pwd)
PYTHON=${PYTHON:-~/miniconda3/envs/torch21new/bin/python}
PYTHON=$(eval echo "${PYTHON}")
log() { printf '\n[A100_MS %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

EXP_NAME=multiscale_ft_full_phi_l_5frames_bs48_e50_320
MASTER_PORT=29581

# --- init from innov2 best (the strongest single-scale model) ---
export ASY_MODEL_PATH=${WORK_DIR}/logs_innovation2_qfl_radar_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth

# --- innov2 head settings (QFL + radar prior), unchanged ---
export ASY_QFL=1
export ASY_RADAR_PRIOR=1
export ASY_RADAR_PRIOR_WEIGHT=0.5
export ASY_QFL_BETA=2.0
export ASY_FUSION_MODE=baseline

# --- the multi-scale fine-tune itself ---
export ASY_MULTISCALE=1
export ASY_MULTISCALE_SCALES=320,384,448

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on shared GPU

export ASY_INPUT_SHAPE=320,320          # base/val/eval scale (paper protocol)
export ASY_BATCH_SIZE=48                # 12/GPU x 4 -- conservative for shared GPU
export ASY_NUM_WORKERS=6
export ASY_UNFREEZE_EPOCH=50
export ASY_SAVE_PERIOD=10
export ASY_PHI=l
export ASY_INIT_LR=0.001                # conservative fine-tune LR
export ASY_LR_DECAY=cos
export ASY_OPTIMIZER=sgd
export ASY_MOMENTUM=0.937
export ASY_WEIGHT_DECAY=0.0005
export ASY_FREEZE_TRAIN=0
export ASY_YOLO_BOX_WEIGHT=1.0
export ASY_YOLO_OBJ_WEIGHT=2.0
export ASY_YOLO_CLS_WEIGHT=2.0
export ASY_EVAL=0                        # no mid-training eval (avoid shared-GPU OOM)
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

log "TRAIN START: multiscale fine-tune 50e (scales=${ASY_MULTISCALE_SCALES})"
log "  init=${ASY_MODEL_PATH} bs=${ASY_BATCH_SIZE} lr=${ASY_INIT_LR} gpus=${CUDA_VISIBLE_DEVICES}"

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" \
    --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
train_exit=$?
log "TRAIN DONE (exit=${train_exit})"
if [[ "${train_exit}" -ne 0 ]]; then
    log "ERROR: training exited ${train_exit}."
    [[ ! -f "${ASY_SAVE_DIR}/best_epoch_weights.pth" ]] && exit "${train_exit}"
    log "best ckpt exists -> proceeding to eval anyway."
fi

sleep 20
log "EVAL multiscale-ft best at single-scale 320 (paper protocol)"
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
    --out_dir paper_metrics_multiscale_ft_full_best 2>&1 | tail -15

echo "DONE" > "${WORK_DIR}/logs/a100_multiscale_ft_full_COMPLETE.flag"
log "ALL DONE. innov2 single-scale=49.958 | soup(old 9e)=50.114"
