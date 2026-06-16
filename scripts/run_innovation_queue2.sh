#!/usr/bin/env bash
# Restarted queue (v2) after fixing fine-tune LR: 1e-3 → 1e-4.
#
# Skips innov2 eval (already done, result saved to paper_metrics_innovation2_qfl_radar_best).
# Runs:
#   A. innov1 baseline_control  (60e, lr=1e-4, new exp name *_lr1e4)
#   B. innov1 reliability_fixed (60e, lr=1e-4)
#   C. innov3 consistency       (300e, from scratch — not affected by LR fix)
#
# Usage:
#   nohup bash scripts/run_innovation_queue2.sh > logs/innovation_queue2.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

log() { printf '\n[QUEUE2 %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

run_train() {
    local desc="$1"; shift
    log "TRAIN START: ${desc}"
    log "  cmd: $*"
    "$@"
    local exit_code=$?
    log "TRAIN DONE (exit=${exit_code}): ${desc}"
    sleep 20
}

run_eval() {
    local desc="$1"; shift
    log "EVAL START: ${desc}"
    "$@"
    local exit_code=$?
    log "EVAL DONE (exit=${exit_code}): ${desc}"
    sleep 10
}

# ===========================================================================
# A. Innovation 1 control: baseline fusion, low-LR fine-tune, lr=1e-4.
# ===========================================================================
run_train "innov1 baseline_control finetune (lr=1e-4)" \
    env FT_MODE=baseline_control bash scripts/run_finetune_from_baseline_4gpu.sh

run_eval "innov1 baseline_control (lr=1e-4)" \
    env EXP_NAME=ft_baseline_control_from_highscore_e60_lr1e4 \
        FUSION_MODE=baseline \
        RADAR_LEGACY_PREPROCESS=1 \
        RADAR_PRESERVE_POINTS=0 \
        BEST_OUT=paper_metrics_ft_baseline_control_best \
        LAST_OUT=paper_metrics_ft_baseline_control_last \
        bash scripts/after_train_eval_and_diagnose.sh

# ===========================================================================
# B. Innovation 1: reliability gate (fixed identity init), fine-tune, lr=1e-4.
# ===========================================================================
run_train "innov1 reliability_fixed finetune (lr=1e-4)" \
    env FT_MODE=reliability_fixed bash scripts/run_finetune_from_baseline_4gpu.sh

run_eval "innov1 reliability_fixed (lr=1e-4)" \
    env EXP_NAME=ft_reliability_fixed_from_highscore_e60_lr1e4 \
        FUSION_MODE=reliability \
        RADAR_LEGACY_PREPROCESS=1 \
        RADAR_PRESERVE_POINTS=0 \
        BEST_OUT=paper_metrics_ft_reliability_fixed_best \
        LAST_OUT=paper_metrics_ft_reliability_fixed_last \
        bash scripts/after_train_eval_and_diagnose.sh

# ===========================================================================
# C. Innovation 3: modality-dropout consistency regularization (300e).
# ===========================================================================
run_train "innov3 consistency 300e" \
    bash scripts/run_train_innovation3_consistency_4gpu.sh

run_eval "innov3 consistency" \
    env EXP_NAME=innovation3_consistency_phi_l_5frames_bs64_300e_320 \
        FUSION_MODE=baseline \
        RADAR_LEGACY_PREPROCESS=1 \
        RADAR_PRESERVE_POINTS=0 \
        BEST_OUT=paper_metrics_innovation3_consistency_best \
        LAST_OUT=paper_metrics_innovation3_consistency_last \
        bash scripts/after_train_eval_and_diagnose.sh

log "QUEUE2 COMPLETE. Results in:"
log "  paper_metrics_ft_baseline_control_best / _last  (innov1 control, lr=1e-4)"
log "  paper_metrics_ft_reliability_fixed_best / _last (innov1 reliability, lr=1e-4)"
log "  paper_metrics_innovation3_consistency_best / _last"
