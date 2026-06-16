#!/usr/bin/env bash
# Final improvement pipeline (run AFTER multiscale_ft_full finishes):
#   1. Greedy model soup over {innov2 ep140/160/180/200/best, multiscale-ft-full best}
#      -- selection on the 400-img subset.
#   2. Full-val eval of the greedy soup           -> 创新点二 (target > 50.114)
#   3. Full-val 2-scale radar Soft-NMS TTA        -> 创新点三 (target > 51.186)
#   4. Full-val 3-scale (320,384,448) TTA         -> only meaningful now that the
#      soup is scale-robust; previously 448 hurt on the fixed-scale model.
#
# Usage:
#   EVAL_GPU=0 nohup bash scripts/run_soup_and_tta_pipeline.sh \
#       > logs/soup_tta_pipeline.out 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PYTHON:-${HOME}/anaconda3/envs/PDPP/bin/python}
GPU=${EVAL_GPU:-0}
log() { printf '\n[SOUP_TTA %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

INNOV2=logs_innovation2_qfl_radar_phi_l_5frames_bs64_300e_320
MSFT=logs_multiscale_ft_full_phi_l_5frames_bs48_e50_320
GREEDY_OUT=logs_innov2_soup/greedy_soup_ms_full.pth

COMMON=(--fusion_mode baseline --phi l --input_shape 320 320
        --confidence 0.001 --max_boxes 100
        --radar_root dataset/VOCradar_5_frames --vocdevkit_path dataset/VOCdevkit
        --radar_legacy_preprocess --no_radar_preserve_points
        --radar_source_order range,doppler,elevation,power
        --radar_target_order range,doppler,elevation,power
        --task_loss sum
        --dark_times night --dim_lightings dim --dim_times daytime,night
        --dim_weathers overcast,rainy --small_area 4096 --small_area_space original)

[[ -f "${MSFT}/best_epoch_weights.pth" ]] || { log "ERROR: multiscale-ft best missing: ${MSFT}"; exit 1; }

# ---- 1) greedy soup (subset selection) ----
log "STEP 1: greedy soup selection on 400-img subset"
"${PY}" scripts/greedy_soup.py \
    --out "${GREEDY_OUT}" \
    --val_txt 2007_val_subset400.txt \
    --python "${PY}" --gpu "${GPU}" \
    --candidates \
        "${MSFT}/best_epoch_weights.pth" \
        "${INNOV2}/best_epoch_weights.pth" \
        "${INNOV2}"/ep160-*.pth \
        "${INNOV2}"/ep180-*.pth \
        "${INNOV2}"/ep140-*.pth \
        "${INNOV2}"/ep200-*.pth \
    2>&1 | tee logs/greedy_soup_select.log

[[ -f "${GREEDY_OUT}" ]] || { log "ERROR: greedy soup not produced"; exit 1; }

# ---- 2) full-val eval of greedy soup (创新点二) ----
log "STEP 2: full-val eval of greedy soup (创新点二, target > 50.114)"
CUDA_VISIBLE_DEVICES=${GPU} "${PY}" eval_paper_metrics.py \
    --model_path "${GREEDY_OUT}" "${COMMON[@]}" \
    --out_dir paper_metrics_greedy_soup_full 2>&1 | tail -4

# ---- 3) 2-scale radar Soft-NMS TTA (创新点三) ----
log "STEP 3: full-val 2-scale radar Soft-NMS TTA (创新点三, target > 51.186)"
CUDA_VISIBLE_DEVICES=${GPU} "${PY}" eval_paper_metrics.py \
    --model_path "${GREEDY_OUT}" "${COMMON[@]}" \
    --tta --tta_scales 320,384 --no_tta_flip --tta_fusion softnms --tta_radar_alpha 0.5 \
    --out_dir paper_metrics_greedy_soup_tta_320_384 2>&1 | tail -4

# ---- 4) 3-scale TTA (now that soup is scale-robust) ----
log "STEP 4: full-val 3-scale radar Soft-NMS TTA (320,384,448)"
CUDA_VISIBLE_DEVICES=${GPU} "${PY}" eval_paper_metrics.py \
    --model_path "${GREEDY_OUT}" "${COMMON[@]}" \
    --tta --tta_scales 320,384,448 --no_tta_flip --tta_fusion softnms --tta_radar_alpha 0.5 \
    --out_dir paper_metrics_greedy_soup_tta_320_384_448 2>&1 | tail -4

# ---- summary ----
log "=== SUMMARY (baseline 42.570 | innov2 49.958 | old soup 50.114 | old TTA 51.186) ==="
for pair in \
    "paper_metrics_greedy_soup_full:创新点二 greedy-soup" \
    "paper_metrics_greedy_soup_tta_320_384:创新点三 TTA-2scale" \
    "paper_metrics_greedy_soup_tta_320_384_448:TTA-3scale"; do
    d="${pair%%:*}"; name="${pair##*:}"
    if [[ -f "${d}/paper_metrics.json" ]]; then
        v=$("${PY}" -c "import json;d=json.load(open('${d}/paper_metrics.json'));print(f\"mAP={d['mAP50-95']:.3f} AP75={d['AP75']:.3f} AR={d.get('AR50-95',0):.3f} small={d['AP_small']:.3f}\")" 2>/dev/null)
        log "  ${name}: ${v}"
    fi
done
echo "PIPELINE_DONE" > logs/soup_tta_pipeline_COMPLETE.flag
log "ALL DONE"
