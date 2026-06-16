#!/usr/bin/env bash
# Self-contained training script; run from the repo root on any CUDA host.
# Runs Innovation 1 v2 (reliability gate + radar-density gate) 300e training,
# evaluates best/last checkpoints, then rsyncs results back to the L40 machine.
#
# Usage (run on A100 server):
#   nohup bash scripts/a100_run_innov1_density_gate.sh \
#     > logs/a100_innov1_density_gate.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
WORK_DIR=$(pwd)

log() { printf '\n[A100_I1 %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

PYTHON=${PYTHON:-~/miniconda3/envs/torch21new/bin/python}
PYTHON=$(eval echo "${PYTHON}")   # expand ~

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python not found: ${PYTHON}"; exit 1
fi

log "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"
log "Working dir: ${WORK_DIR}"
log "CUDA devices: $(nvidia-smi -L 2>/dev/null | wc -l) GPUs"

mkdir -p logs

# ===========================================================================
# Innovation 1 v2: reliability gate + radar-density gate, 300e from scratch.
# ===========================================================================
EXP_NAME=innovation1_density_gate_phi_l_5frames_bs64_300e_320
MASTER_PORT=29561

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=1
export ASY_SYNC_BN=1
export ASY_FP16=1

# Innovation 1 v2 switches
export ASY_FUSION_MODE=reliability
export ASY_GATE_DENSITY=1

# NCCL reliability settings for A100 (prevents mid-epoch allreduce hangs).
# ASYNC_ERROR_HANDLING: raise error instead of hanging indefinitely.
# BLOCKING_WAIT: synchronous barrier so errors surface immediately.
# TIMEOUT: 30-minute cap on any collective op (default is infinite).
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

export ASY_INIT_LR=0.01
export ASY_LR_DECAY=cos
export ASY_OPTIMIZER=sgd
export ASY_MOMENTUM=0.937
export ASY_WEIGHT_DECAY=0.0005
export ASY_FREEZE_TRAIN=0
export ASY_YOLO_BOX_WEIGHT=1.0
export ASY_YOLO_OBJ_WEIGHT=2.0
export ASY_YOLO_CLS_WEIGHT=2.0

export ASY_EVAL=0
export ASY_EVAL_PERIOD=10
export ASY_BEST_METRIC=det

export ASY_SAVE_DIR=logs_${EXP_NAME}
export ASY_SAVE_DIR_SEG=logs_seg_${EXP_NAME}
# Use /dev/shm (RAM disk) for dataset to eliminate RAID I/O latency.
# Dataset was pre-copied there (12 GB, instant access from RAM).
export ASY_VOCDEVKIT=/dev/shm/ASY_dataset/VOCdevkit
export ASY_RADAR_ROOT=/dev/shm/ASY_dataset/VOCradar_5_frames
export ASY_TASK_LOSS=sum
export ASY_FUSION_MODE=reliability
export ASY_RADAR_DROPOUT=0
export ASY_RADAR_CHANNELS=4
export ASY_RADAR_ALIGN_MODE=letterbox
export ASY_RADAR_NORMALIZE=0
export ASY_RADAR_PRESERVE_POINTS=0
export ASY_RADAR_SOURCE_ORDER=range,doppler,elevation,power
export ASY_RADAR_TARGET_ORDER=range,doppler,elevation,power
export ASY_RADAR_LEGACY_PREPROCESS=1
export ASY_WEATHER_AUG=0

mkdir -p "${ASY_SAVE_DIR}" "${ASY_SAVE_DIR_SEG}" logs

log "TRAIN START: innov1 density gate 300e"
log "  exp=${EXP_NAME}"
log "  fusion=${ASY_FUSION_MODE} gate_density=${ASY_GATE_DENSITY} lr=${ASY_INIT_LR} epochs=${ASY_UNFREEZE_EPOCH} batch=${ASY_BATCH_SIZE}"
log "  GPUs=${CUDA_VISIBLE_DEVICES}"

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=4 \
    train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
train_exit=$?
log "TRAIN DONE (exit=${train_exit}): innov1 density gate"

if [[ "${train_exit}" -ne 0 ]]; then
    log "ERROR: training exited with code ${train_exit}. Skipping eval and NOT writing COMPLETE flag."
    exit "${train_exit}"
fi

sleep 20

# ===========================================================================
# Evaluation: best and last checkpoints with paper protocol.
# ===========================================================================
eval_ckpts=(
    "${ASY_SAVE_DIR}/best_epoch_weights.pth:results/innovation1_density_gate_best"
    "${ASY_SAVE_DIR}/last_epoch_weights.pth:results/innovation1_density_gate_last"
)

for entry in "${eval_ckpts[@]}"; do
    ckpt="${entry%%:*}"
    out_dir="${entry##*:}"
    if [[ ! -f "${ckpt}" ]]; then
        log "SKIP eval: ${ckpt} not found"; continue
    fi
    log "EVAL START: ${out_dir}"
    "${PYTHON}" eval_paper_metrics.py \
        --model_path "${ckpt}" \
        --fusion_mode reliability \
        --phi l \
        --input_shape 320 320 \
        --confidence 0.001 \
        --max_boxes 100 \
        --out_dir "${out_dir}" \
        --radar_root "${ASY_RADAR_ROOT}" \
        --radar_legacy_preprocess \
        --no_radar_preserve_points \
        --radar_source_order range,doppler,elevation,power \
        --radar_target_order range,doppler,elevation,power \
        --dark_times night \
        --dim_lightings dim \
        --dim_times daytime,night \
        --dim_weathers overcast,rainy \
        --small_area 4096 \
        --small_area_space original 2>&1 | tee "logs/eval_${out_dir}_${stamp}.log"
    log "EVAL DONE: ${out_dir}"
done

# ===========================================================================
# Mark completion (L40 will pull results via poll script).
# ===========================================================================
echo "DONE" > "${WORK_DIR}/logs/a100_innov1_COMPLETE.flag"

log "ALL DONE. Results ready for pull:"
log "  results/innovation1_density_gate_best/paper_metrics.json"
log "  results/innovation1_density_gate_last/paper_metrics.json"
log "  ${ASY_SAVE_DIR}/best_epoch_weights.pth"
log ""
log "  Pull example: rsync -az <user@host>:<remote_repo>/results/innovation1_density_gate_best/ ./results/innovation1_density_gate_best/"
