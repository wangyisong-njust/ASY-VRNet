#!/usr/bin/env bash
# Poll a remote training host until innov1 density gate training is done,
# then pull results back to this machine.
#
# Configure the remote via env vars (no hardcoded host/paths):
#   REMOTE_HOST   user@host of the training server (e.g. user@10.0.0.1)
#   REMOTE_BASE   project path on the remote (e.g. /data/ASY-VRNet)
#   LOCAL_BASE    project path locally (defaults to this repo root)
#
# Usage:
#   REMOTE_HOST=user@host REMOTE_BASE=/data/ASY-VRNet \
#     nohup bash scripts/pull_a100_results.sh > logs/pull_a100_results.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

log() { printf '\n[PULL_A100 %s] %s\n' "$(date +%Y-%m-%d_%H:%M:%S)" "$*"; }

A100="${REMOTE_HOST:?set REMOTE_HOST=user@host}"
REMOTE_BASE="${REMOTE_BASE:?set REMOTE_BASE=/path/to/ASY-VRNet on remote}"
FLAG="${REMOTE_BASE}/logs/a100_innov1_COMPLETE.flag"
LOCAL_BASE="${LOCAL_BASE:-$(pwd)}"
EXP="innovation1_density_gate_phi_l_5frames_bs64_300e_320"

log "Waiting for A100 to finish innov1 density gate training..."
log "  polling flag: ${A100}:${FLAG}"

while true; do
    result=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
        "${A100}" "test -f '${FLAG}' && echo DONE || echo WAIT" 2>/dev/null)
    if [[ "${result}" == "DONE" ]]; then
        log "Flag found! Pulling results..."
        break
    fi
    log "Still training... (check: ssh ${A100} 'tail -3 ${REMOTE_BASE}/logs/a100_innov1_density_gate_v4.out')"
    sleep 300  # check every 5 minutes
done

# Pull evaluation results
for dir in results/innovation1_density_gate_best results/innovation1_density_gate_last; do
    log "Pulling ${dir}..."
    rsync -az -e "ssh -o StrictHostKeyChecking=no" \
        "${A100}:${REMOTE_BASE}/${dir}/" \
        "${LOCAL_BASE}/${dir}/" && log "  OK: ${dir}" || log "  WARN: ${dir} not found"
done

# Pull best checkpoint
log "Pulling best checkpoint..."
rsync -az -e "ssh -o StrictHostKeyChecking=no" \
    "${A100}:${REMOTE_BASE}/logs_${EXP}/best_epoch_weights.pth" \
    "${LOCAL_BASE}/logs_${EXP}_a100/" 2>/dev/null && \
    log "  OK: best_epoch_weights.pth → logs_${EXP}_a100/" || \
    log "  WARN: checkpoint pull skipped"

# Print results
log "=== PULL COMPLETE ==="
if [[ -f "${LOCAL_BASE}/results/innovation1_density_gate_best/paper_metrics.json" ]]; then
    log "Best checkpoint mAP50-95:"
    python3 -c "
import json
d = json.load(open('${LOCAL_BASE}/results/innovation1_density_gate_best/paper_metrics.json'))
print(f'  mAP50-95: {d.get(\"mAP50-95\", d.get(\"mAP_50_95\", \"N/A\"))}')
print(f'  mAP50:    {d.get(\"mAP50\", d.get(\"mAP_50\", \"N/A\"))}')
" 2>/dev/null || cat "${LOCAL_BASE}/results/innovation1_density_gate_best/paper_metrics.json"
fi
log "  Baseline: 42.570  |  Innov1 ft-60e: 42.714  |  Innov2: 49.958"
