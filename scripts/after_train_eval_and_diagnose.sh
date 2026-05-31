#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT=$(pwd)
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "${HOME}/anaconda3/envs/PDPP/bin/python" ]]; then
        PYTHON="${HOME}/anaconda3/envs/PDPP/bin/python"
    else
        PYTHON=$(command -v python3 || command -v python || true)
    fi
fi
if [[ -z "${PYTHON}" ]]; then
    echo "No Python interpreter found. Set PYTHON=/path/to/python before running."
    exit 1
fi
TRAIN_PID=${TRAIN_PID:-3280801}
POLL_SECONDS=${POLL_SECONDS:-60}
EXP_NAME=${EXP_NAME:-paper_baseline_latestfix_ddpfix_5frames_uncert_bs128_100e_320}
DET_DIR=${DET_DIR:-logs_${EXP_NAME}}
SEG_DIR=${SEG_DIR:-logs_seg_${EXP_NAME}}
RADAR_ROOT=${RADAR_ROOT:-${PROJECT_ROOT}/dataset/VOCradar_5_frames}
VOCDEVKIT=${VOCDEVKIT:-${PROJECT_ROOT}/dataset/VOCdevkit}
INFO_CSV=${INFO_CSV:-${PROJECT_ROOT}/dataset/WaterScenes_Full/information_list.csv}
VAL_TXT=${VAL_TXT:-2007_val.txt}
CLASSES_PATH=${CLASSES_PATH:-model_data/waterscenes.txt}
INPUT_H=${INPUT_H:-320}
INPUT_W=${INPUT_W:-320}
PHI=${PHI:-${ASY_PHI:-l}}
RADAR_CHANNELS=${RADAR_CHANNELS:-4}
RADAR_ALIGN_MODE=${RADAR_ALIGN_MODE:-${ASY_RADAR_ALIGN_MODE:-letterbox}}
RADAR_NORMALIZE=${RADAR_NORMALIZE:-${ASY_RADAR_NORMALIZE:-0}}
FUSION_MODE=${FUSION_MODE:-baseline}
TASK_LOSS=${TASK_LOSS:-uncertainty}
CONFIDENCE=${CONFIDENCE:-0.05}
NMS_IOU=${NMS_IOU:-0.5}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export CUDA_VISIBLE_DEVICES
export ASY_RADAR_ROOT="${RADAR_ROOT}"
export ASY_VOCDEVKIT="${VOCDEVKIT}"
export ASY_INFO_CSV="${INFO_CSV}"
export ASY_RADAR_CHANNELS="${RADAR_CHANNELS}"
export ASY_RADAR_ALIGN_MODE="${RADAR_ALIGN_MODE}"
export ASY_RADAR_NORMALIZE="${RADAR_NORMALIZE}"
export ASY_FUSION_MODE="${FUSION_MODE}"
export ASY_RADAR_DROPOUT=${ASY_RADAR_DROPOUT:-0}
export ASY_TASK_LOSS="${TASK_LOSS}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/after_train_eval_${EXP_NAME}_${STAMP}.log"
REPORT_DIR=${REPORT_DIR:-reproduction_reports}
mkdir -p "${REPORT_DIR}"

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

wait_for_training() {
    log "Waiting for training PID ${TRAIN_PID} to finish."
    while kill -0 "${TRAIN_PID}" 2>/dev/null; do
        log "Training is still running; polling again in ${POLL_SECONDS}s."
        sleep "${POLL_SECONDS}"
    done
    log "Training PID ${TRAIN_PID} has exited."
}

run_eval() {
    local name=$1
    local weight=$2
    local out_dir=$3
    local radar_normalize_arg=--no_radar_normalize
    if [[ "${RADAR_NORMALIZE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        radar_normalize_arg=--radar_normalize
    fi

    if [[ ! -f "${weight}" ]]; then
        log "Missing checkpoint: ${weight}"
        return 2
    fi

    log "Evaluating ${name}: ${weight}"
    "${PYTHON}" eval_paper_metrics.py \
        --val_txt "${VAL_TXT}" \
        --model_path "${weight}" \
        --classes_path "${CLASSES_PATH}" \
        --radar_root "${RADAR_ROOT}" \
        --vocdevkit_path "${VOCDEVKIT}" \
        --info_csv "${INFO_CSV}" \
        --out_dir "${out_dir}" \
        --phi "${PHI}" \
        --input_shape "${INPUT_H}" "${INPUT_W}" \
        --num_seg_classes 9 \
        --radar_channels "${RADAR_CHANNELS}" \
        --radar_align_mode "${RADAR_ALIGN_MODE}" \
        "${radar_normalize_arg}" \
        --fusion_mode "${FUSION_MODE}" \
        --task_loss "${TASK_LOSS}" \
        --confidence "${CONFIDENCE}" \
        --nms_iou "${NMS_IOU}" \
        --cuda
    log "Finished ${name} evaluation."
}

main() {
    {
        log "After-train evaluation script started."
        log "Run log: ${RUN_LOG}"
        wait_for_training

        BEST_WEIGHT="${DET_DIR}/best_epoch_weights.pth"
        LAST_WEIGHT="${DET_DIR}/last_epoch_weights.pth"
        BEST_OUT=${BEST_OUT:-paper_metrics_bs128_best_final}
        LAST_OUT=${LAST_OUT:-paper_metrics_bs128_last_final}

        run_eval best "${BEST_WEIGHT}" "${BEST_OUT}"
        run_eval last "${LAST_WEIGHT}" "${LAST_OUT}"

        REPORT_MD="${REPORT_DIR}/paper_gap_${EXP_NAME}_${STAMP}.md"
        REPORT_JSON="${REPORT_DIR}/paper_gap_${EXP_NAME}_${STAMP}.json"
        "${PYTHON}" scripts/analyze_paper_gap.py \
            --best "${BEST_OUT}/paper_metrics.json" \
            --last "${LAST_OUT}/paper_metrics.json" \
            --out "${REPORT_MD}" \
            --json_out "${REPORT_JSON}"

        log "Gap report: ${REPORT_MD}"
        log "JSON summary: ${REPORT_JSON}"
        log "If the report still shows gaps, inspect the listed code-level checks first before starting innovation experiments."
    } 2>&1 | tee -a "${RUN_LOG}"
}

main "$@"
