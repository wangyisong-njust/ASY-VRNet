#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CURRENT_PID=${CURRENT_PID:?Set CURRENT_PID to the current training wrapper PID.}
POLL_SECONDS=${POLL_SECONDS:-60}
EXP_NAME=${EXP_NAME:-detection_baseline_phi_l_5frames_bs64_300e_320}
MASTER_PORT=${MASTER_PORT:-29651}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "${LOG_DIR}"

STAMP=$(date +%Y%m%d_%H%M%S)
WATCH_LOG=${WATCH_LOG:-${LOG_DIR}/after_current_start_${EXP_NAME}_${STAMP}.log}
LOCK_FILE=${LOCK_FILE:-${LOG_DIR}/after_current_start_${EXP_NAME}.lock}
TRAIN_STDOUT=${TRAIN_STDOUT:-${LOG_DIR}/${EXP_NAME}_train_launcher_${STAMP}.log}
EVAL_STDOUT=${EVAL_STDOUT:-${LOG_DIR}/${EXP_NAME}_eval_launcher_${STAMP}.log}

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

main() {
    exec >> "${WATCH_LOG}" 2>&1

    exec 9>"${LOCK_FILE}"
    if ! flock -n 9; then
        log "Another continuation watcher is already active for ${EXP_NAME}; exiting."
        exit 1
    fi

    log "Continuation watcher started."
    log "Waiting for current training PID ${CURRENT_PID}."
    while kill -0 "${CURRENT_PID}" 2>/dev/null; do
        log "Current training is still running; polling again in ${POLL_SECONDS}s."
        sleep "${POLL_SECONDS}"
    done
    log "Current training PID ${CURRENT_PID} has exited."

    export EXP_NAME
    export MASTER_PORT
    export ASY_TASK_LOSS=${ASY_TASK_LOSS:-sum}
    export ASY_BEST_METRIC=${ASY_BEST_METRIC:-det}
    export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-64}
    export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-300}
    export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-20}
    export ASY_PHI=${ASY_PHI:-l}
    export ASY_INIT_LR=${ASY_INIT_LR:-0.01}
    export ASY_OPTIMIZER=${ASY_OPTIMIZER:-sgd}
    export ASY_MOMENTUM=${ASY_MOMENTUM:-0.937}
    export ASY_WEIGHT_DECAY=${ASY_WEIGHT_DECAY:-0.0005}
    export ASY_YOLO_BOX_WEIGHT=${ASY_YOLO_BOX_WEIGHT:-1.0}
    export ASY_YOLO_OBJ_WEIGHT=${ASY_YOLO_OBJ_WEIGHT:-2.0}
    export ASY_YOLO_CLS_WEIGHT=${ASY_YOLO_CLS_WEIGHT:-2.0}
    export ASY_RADAR_NORMALIZE=${ASY_RADAR_NORMALIZE:-0}
    export ASY_RADAR_PRESERVE_POINTS=${ASY_RADAR_PRESERVE_POINTS:-1}
    export ASY_RADAR_ALIGN_MODE=${ASY_RADAR_ALIGN_MODE:-letterbox}
    export ASY_RADAR_SOURCE_ORDER=${ASY_RADAR_SOURCE_ORDER:-range,doppler,elevation,power}
    export ASY_RADAR_TARGET_ORDER=${ASY_RADAR_TARGET_ORDER:-range,elevation,velocity,power}

    log "Starting detection-first training: EXP_NAME=${EXP_NAME}, MASTER_PORT=${MASTER_PORT}."
    log "Training launcher output: ${TRAIN_STDOUT}"
    bash scripts/run_train_detection_baseline_4gpu.sh >> "${TRAIN_STDOUT}" 2>&1 &
    TRAIN_PID=$!
    printf '%s\n' "${TRAIN_PID}" > "${LOG_DIR}/${EXP_NAME}.pid"
    log "New training wrapper PID: ${TRAIN_PID}."

    log "Evaluation launcher output: ${EVAL_STDOUT}"
    TRAIN_PID="${TRAIN_PID}" \
    EXP_NAME="${EXP_NAME}" \
    TASK_LOSS=sum \
    PHI="${ASY_PHI}" \
    CONFIDENCE="${CONFIDENCE:-0.001}" \
    MAX_BOXES="${MAX_BOXES:-100}" \
    BEST_OUT="${BEST_OUT:-results/detection_baseline_best}" \
    LAST_OUT="${LAST_OUT:-results/detection_baseline_last}" \
        bash scripts/after_train_eval_detection_baseline.sh >> "${EVAL_STDOUT}" 2>&1 &
    EVAL_PID=$!
    printf '%s\n' "${EVAL_PID}" > "${LOG_DIR}/${EXP_NAME}_after_eval.pid"
    log "New after-train evaluation watcher PID: ${EVAL_PID}."

    set +e
    wait "${TRAIN_PID}"
    TRAIN_STATUS=$?
    log "Detection-first training exited with status ${TRAIN_STATUS}."

    wait "${EVAL_PID}"
    EVAL_STATUS=$?
    log "Detection-first evaluation watcher exited with status ${EVAL_STATUS}."
    set -e
    if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
        exit "${TRAIN_STATUS}"
    fi
    exit "${EVAL_STATUS}"
}

main "$@"
