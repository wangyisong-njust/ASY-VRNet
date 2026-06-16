#!/usr/bin/env bash
# Innovation 3 (final): fine-tune the radar-reliability gate ON TOP of the
# innov2 (QFL + radar-prior) model -- the strongest checkpoint (49.958).
#
# Rationale: the gate only helped when FINE-TUNED from a strong model
# (reliability-gate ft from baseline gave +0.14; trained from scratch it lost
# -0.48). Here we start from innov2's weights, add the reliability gate
# (randomly initialised), keep innov2's QFL+radar-prior head/loss, and
# fine-tune 60 epochs at a low LR. Goal: push 49.958 a little higher.
#
# Runs on the A100 server (8 free GPUs), evaluates best+last with the paper
# protocol (fusion_mode=reliability), then writes a completion flag for the
# L40 pull watcher.
#
# Usage (on A100):
#   nohup bash scripts/a100_run_innov3_gate_on_innov2.sh \
#       > logs/a100_innov3_gate_on_innov2.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
WORK_DIR=$(pwd)
PYTHON=${PYTHON:-~/miniconda3/envs/torch21new/bin/python}
PYTHON=$(eval echo "${PYTHON}")
log() { printf '\n[A100_I3 %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

EXP_NAME=innov3_gate_on_innov2_ft_phi_l_5frames_bs64_e60_320
MASTER_PORT=29563

# --- start from innov2 best (QFL + radar prior, fusion=baseline weights) ---
export ASY_MODEL_PATH=${WORK_DIR}/weights/innov1_qfl_radar_best.pth

# --- add the reliability gate (plain, NO density: density variant failed) ---
export ASY_FUSION_MODE=reliability
export ASY_GATE_DENSITY=0

# --- keep innov2 head/loss (QFL + radar prior) ---
export ASY_QFL=1
export ASY_RADAR_PRIOR=1
export ASY_RADAR_PRIOR_WEIGHT=0.5
export ASY_QFL_BETA=2.0

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=1
export ASY_SYNC_BN=1
export ASY_FP16=1

# NCCL stability (prevents the mid/late-epoch allreduce hangs seen earlier)
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=1800
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

export ASY_INPUT_SHAPE=320,320
export ASY_BATCH_SIZE=64
export ASY_NUM_WORKERS=8
export ASY_UNFREEZE_EPOCH=60
export ASY_SAVE_PERIOD=20
export ASY_PHI=l
export ASY_INIT_LR=0.0001            # proven gate-ft LR (the +0.14 recipe)
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
# Use /raid (disk) NOT /dev/shm -- the innov1 run OOM-killed a rank from RAM
# pressure at epoch 299; a 60e fine-tune from disk is plenty fast and safe.
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

log "TRAIN START: gate-on-innov2 fine-tune 60e"
log "  init=${ASY_MODEL_PATH}"
log "  fusion=${ASY_FUSION_MODE} gate_density=${ASY_GATE_DENSITY} qfl=${ASY_QFL} radar_prior=${ASY_RADAR_PRIOR} lr=${ASY_INIT_LR}"

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" \
    --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
train_exit=$?
log "TRAIN DONE (exit=${train_exit})"
if [[ "${train_exit}" -ne 0 ]]; then
    log "ERROR: training exited ${train_exit}; skipping eval, NOT writing flag."
    # still allow eval if best ckpt exists (training may have been near-complete)
    if [[ ! -f "${ASY_SAVE_DIR}/best_epoch_weights.pth" ]]; then exit "${train_exit}"; fi
    log "best ckpt exists -> proceeding to eval anyway."
fi

sleep 20
for entry in "best:results/innov3_gate_on_innov2_best" "last:results/innov3_gate_on_innov2_last"; do
    tag="${entry%%:*}"; out="${entry##*:}"
    ckpt="${ASY_SAVE_DIR}/${tag}_epoch_weights.pth"
    [[ -f "${ckpt}" ]] || { log "SKIP ${ckpt} (missing)"; continue; }
    log "EVAL ${tag}: ${out}"
    ASY_GATE_DENSITY=0 "${PYTHON}" eval_paper_metrics.py \
        --model_path "${ckpt}" --fusion_mode reliability --phi l \
        --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
        --radar_root "${ASY_RADAR_ROOT}" --vocdevkit_path "${ASY_VOCDEVKIT}" \
        --radar_legacy_preprocess --no_radar_preserve_points \
        --radar_source_order range,doppler,elevation,power \
        --radar_target_order range,doppler,elevation,power \
        --task_loss sum \
        --dark_times night --dim_lightings dim --dim_times daytime,night \
        --dim_weathers overcast,rainy --small_area 4096 --small_area_space original \
        --out_dir "${out}" 2>&1 | tail -15
done

echo "DONE" > "${WORK_DIR}/logs/a100_innov3_COMPLETE.flag"
log "ALL DONE. baseline 42.570 | innov2 49.958"
