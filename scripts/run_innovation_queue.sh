#!/usr/bin/env bash
# Serial queue: wait for the currently-running job (WAIT_PID), then train +
# evaluate innovation 1 (control + reliability) and innovation 3 one after
# another. Each 4-GPU job runs to completion before the next starts.
#
# Robustness: NOT using `set -e` so one failed step does not abort the whole
# queue. Every step is logged and timestamped.
#
# Usage:
#   WAIT_PID=231700 nohup bash scripts/run_innovation_queue.sh > logs/innovation_queue.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

QUEUE_LOG_DIR=logs
mkdir -p "${QUEUE_LOG_DIR}"

log() { printf '\n[QUEUE %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------------------#
# 0. Wait for the in-flight job to finish.
# ---------------------------------------------------------------------------#
WAIT_PID=${WAIT_PID:-}
if [[ -n "${WAIT_PID}" ]]; then
    log "Waiting for in-flight job PID=${WAIT_PID} to finish before starting the queue..."
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
        sleep 60
    done
    log "PID=${WAIT_PID} finished. Giving GPUs 30s to release, then starting queue."
    sleep 30
fi

run_train() {
    local desc="$1"; shift
    log "TRAIN START: ${desc}"
    log "  cmd: $*"
    "$@"
    log "TRAIN DONE (exit=$?): ${desc}"
    sleep 20  # let DDP / port release
}

run_eval() {
    local desc="$1"; shift
    log "EVAL START: ${desc}"
    "$@"
    log "EVAL DONE (exit=$?): ${desc}"
    sleep 10
}

BASE_WEIGHTS=weights/baseline_best.pth

# ===========================================================================#
# 0.5 Evaluate innovation 2 (just finished training when WAIT_PID died).
# ===========================================================================#
run_eval "innov2 qfl_radar" \
    env EXP_NAME=innovation2_qfl_radar_phi_l_5frames_bs64_300e_320 \
        FUSION_MODE=baseline \
        BEST_OUT=results/innovation2_qfl_radar_best \
        LAST_OUT=results/innovation2_qfl_radar_last \
        bash scripts/after_train_eval_and_diagnose.sh

# ===========================================================================#
# A. Innovation 1 control: baseline fusion, low-LR fine-tune from high score.
# ===========================================================================#
run_train "innov1 baseline_control finetune" \
    env FT_MODE=baseline_control bash scripts/run_finetune_from_baseline_4gpu.sh

run_eval "innov1 baseline_control" \
    env EXP_NAME=ft_baseline_control_from_highscore_e60_lr1e3 \
        FUSION_MODE=baseline \
        BEST_OUT=results/ft_baseline_control_best \
        LAST_OUT=results/ft_baseline_control_last \
        bash scripts/after_train_eval_and_diagnose.sh

# ===========================================================================#
# B. Innovation 1: reliability gate (fixed identity init), fine-tune.
# ===========================================================================#
run_train "innov1 reliability_fixed finetune" \
    env FT_MODE=reliability_fixed bash scripts/run_finetune_from_baseline_4gpu.sh

run_eval "innov1 reliability_fixed" \
    env EXP_NAME=ft_reliability_fixed_from_highscore_e60_lr1e3 \
        FUSION_MODE=reliability \
        BEST_OUT=results/ft_reliability_fixed_best \
        LAST_OUT=results/ft_reliability_fixed_last \
        bash scripts/after_train_eval_and_diagnose.sh

# ===========================================================================#
# C. Innovation 3: modality-dropout consistency regularization (300e).
# ===========================================================================#
run_train "innov3 consistency 300e" \
    bash scripts/run_train_innovation3_consistency_4gpu.sh

run_eval "innov3 consistency" \
    env EXP_NAME=innovation3_consistency_phi_l_5frames_bs64_300e_320 \
        FUSION_MODE=baseline \
        BEST_OUT=results/innovation3_consistency_best \
        LAST_OUT=results/innovation3_consistency_last \
        bash scripts/after_train_eval_and_diagnose.sh

log "QUEUE COMPLETE. Results:"
log "  results/ft_baseline_control_best / _last"
log "  results/ft_reliability_fixed_best / _last"
log "  results/innovation3_consistency_best / _last"
