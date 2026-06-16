#!/usr/bin/env bash
# Wait for the innov3 multi-scale fine-tune to finish, then evaluate the best
# checkpoint two ways on the FULL val set (paper protocol, comparable to
# baseline 42.570 / innov2 49.958):
#   1. single-scale 320           -> did scale-aware fine-tuning lift the base?
#   2. radar multi-scale TTA      -> does TTA now help on a scale-robust model?
#
# Usage:
#   TRAIN_PID=<ddp_pid> nohup bash scripts/watch_eval_innov3_multiscale.sh \
#       > logs/innov3_multiscale_eval.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
PYTHON=${PYTHON:-${HOME}/anaconda3/envs/PDPP/bin/python}
TRAIN_PID=${TRAIN_PID:-}
EXP=logs_innovation3_multiscale_ft_phi_l_5frames_bs48_e50_320
CKPT=${PROJECT_ROOT}/${EXP}/best_epoch_weights.pth
EVAL_GPU=${EVAL_GPU:-0}

log() { printf '\n[INNOV3MS_EVAL %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

if [[ -n "${TRAIN_PID}" ]]; then
    log "Waiting for training PID ${TRAIN_PID} to finish..."
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do sleep 60; done
    log "Training PID ${TRAIN_PID} has exited."
fi
sleep 20

if [[ ! -f "${CKPT}" ]]; then
    log "ERROR: best checkpoint not found at ${CKPT}"; exit 1
fi

COMMON=(--val_txt 2007_val.txt --model_path "${CKPT}" --fusion_mode baseline
        --phi l --input_shape 320 320 --confidence 0.001 --max_boxes 100
        --radar_legacy_preprocess --no_radar_preserve_points
        --radar_source_order range,doppler,elevation,power
        --radar_target_order range,doppler,elevation,power
        --task_loss uncertainty
        --dark_times night --dim_lightings dim --dim_times daytime,night
        --dim_weathers overcast,rainy --small_area 4096 --small_area_space original)

# 1) single-scale 320 (paper protocol)
log "EVAL 1/2: single-scale 320"
CUDA_VISIBLE_DEVICES=${EVAL_GPU} "${PYTHON}" eval_paper_metrics.py "${COMMON[@]}" \
    --out_dir paper_metrics_innov3_multiscale_320 2>&1 | tail -20

# 2) radar multi-scale TTA (NMS merge keeps localization)
log "EVAL 2/2: radar multi-scale TTA (320/384/448, NMS merge)"
CUDA_VISIBLE_DEVICES=${EVAL_GPU} "${PYTHON}" eval_paper_metrics.py "${COMMON[@]}" \
    --tta --tta_scales 320,384,448 --tta_fusion nms --tta_radar_alpha 0.0 \
    --out_dir paper_metrics_innov3_multiscale_tta 2>&1 | tail -20

log "=== RESULTS (baseline 42.570 | innov2 49.958) ==="
for d in paper_metrics_innov3_multiscale_320 paper_metrics_innov3_multiscale_tta; do
    if [[ -f "${d}/paper_metrics.json" ]]; then
        v=$(${PYTHON} -c "import json;print(round(json.load(open('${d}/paper_metrics.json'))['mAP50-95'],3))" 2>/dev/null)
        log "  ${d}: mAP50-95 = ${v}"
    fi
done
log "DONE"
