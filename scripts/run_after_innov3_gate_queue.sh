#!/usr/bin/env bash
# Queue: wait for the running innov3 job (master_port 29550) to finish,
# then train Innovation 1 v2 (reliability gate + radar-density gate, 300e)
# and eval it.
#
# Usage:
#   nohup bash scripts/run_after_innov3_gate_queue.sh \
#     > logs/innov1_density_gate_queue.out 2>&1 &
#
# To check progress:
#   tail -f logs/innov1_density_gate_queue.out

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

log() { printf '\n[GATE_QUEUE %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

# --------------------------------------------------------------------------- #
# Step 1: wait for innov3 (and its eval) to release the GPUs.                  #
# We poll the master_port=29550 launcher — that PID outlives training + eval.  #
# --------------------------------------------------------------------------- #
INNOV3_PORT=29550
INNOV3_PID=$(pgrep -f "master_port=${INNOV3_PORT}" | head -1 || true)

if [[ -n "${INNOV3_PID}" ]]; then
    log "Waiting for innov3 (PID=${INNOV3_PID}, port=${INNOV3_PORT}) to finish..."
    while kill -0 "${INNOV3_PID}" 2>/dev/null; do
        sleep 60
    done
    log "innov3 queue finished (PID=${INNOV3_PID} gone)."
else
    log "No innov3 job found on port ${INNOV3_PORT} — proceeding immediately."
fi

# Extra grace period: eval may still be running even after the train PID exits.
# Wait until GPU memory use drops below 5GB total across all GPUs.
log "Waiting for GPUs to clear (all GPU mem < 5000 MiB each)..."
while true; do
    # get per-GPU used mem in MiB
    max_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk 'BEGIN{m=0} {if($1>m)m=$1} END{print m}')
    if [[ "${max_mem}" -lt 5000 ]]; then
        log "GPUs clear (max used=${max_mem} MiB). Starting innov1 density gate."
        break
    fi
    log "GPU max mem still ${max_mem} MiB — waiting 60s..."
    sleep 60
done

# --------------------------------------------------------------------------- #
# Step 2: Innovation 1 v2 — reliability gate + radar-density gate, 300e.      #
# --------------------------------------------------------------------------- #
log "TRAIN START: innov1 density gate 300e (from scratch)"
bash scripts/run_train_innovation1_density_gate_4gpu.sh
train_exit=$?
log "TRAIN DONE (exit=${train_exit}): innov1 density gate"

sleep 20

# --------------------------------------------------------------------------- #
# Step 3: Evaluate best and last checkpoints.                                  #
# --------------------------------------------------------------------------- #
log "EVAL START: innov1 density gate (best checkpoint)"
EXP_NAME=innovation1_density_gate_phi_l_5frames_bs64_300e_320 \
    FUSION_MODE=reliability \
    RADAR_LEGACY_PREPROCESS=1 \
    RADAR_PRESERVE_POINTS=0 \
    BEST_OUT=paper_metrics_innovation1_density_gate_best \
    LAST_OUT=paper_metrics_innovation1_density_gate_last \
    bash scripts/after_train_eval_and_diagnose.sh
eval_exit=$?
log "EVAL DONE (exit=${eval_exit}): innov1 density gate"

log "GATE_QUEUE COMPLETE."
log "  Results: paper_metrics_innovation1_density_gate_best / _last"
log "  Compare mAP50-95 against:"
log "    baseline:          42.570  (paper_metrics_legacy_highscore_best)"
log "    innov1 ft-60e v1:  42.714  (paper_metrics_ft_reliability_fixed_best)"
log "    innov2 qfl_radar:  49.958  (paper_metrics_innovation2_qfl_radar_best)"
