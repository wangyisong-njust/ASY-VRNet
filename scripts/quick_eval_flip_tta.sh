#!/usr/bin/env bash
# Quick eval: flip-only TTA on innov2 best (scale=320, horizontal flip only).
# No training needed. 30 min. Compare to innov2 single-scale 49.958.
# Usage: bash scripts/quick_eval_flip_tta.sh

set -uo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-${HOME}/anaconda3/envs/PDPP/bin/python}
CKPT=weights/innov1_qfl_radar_best.pth
GPU=${EVAL_GPU:-0}

[[ -f "${CKPT}" ]] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }

echo "[flip-TTA] single-scale 320, horizontal flip, NMS merge"
CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" eval_paper_metrics.py \
    --model_path "${CKPT}" --fusion_mode baseline --phi l \
    --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
    --radar_legacy_preprocess --no_radar_preserve_points \
    --radar_source_order range,doppler,elevation,power \
    --radar_target_order range,doppler,elevation,power \
    --task_loss sum \
    --tta --tta_scales 320 --tta_flip --tta_fusion nms --tta_radar_alpha 0.0 \
    --dark_times night --dim_lightings dim --dim_times daytime,night \
    --dim_weathers overcast,rainy --small_area 4096 --small_area_space original \
    --out_dir results/innov2_flip_tta 2>&1 | tail -20

echo ""
echo "[384-eval] single-scale 384 (no flip, no TTA)"
CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON}" eval_paper_metrics.py \
    --model_path "${CKPT}" --fusion_mode baseline --phi l \
    --input_shape 384 384 --confidence 0.001 --max_boxes 100 \
    --radar_legacy_preprocess --no_radar_preserve_points \
    --radar_source_order range,doppler,elevation,power \
    --radar_target_order range,doppler,elevation,power \
    --task_loss sum \
    --dark_times night --dim_lightings dim --dim_times daytime,night \
    --dim_weathers overcast,rainy --small_area 4096 --small_area_space original \
    --out_dir results/innov2_384eval 2>&1 | tail -20

echo ""
echo "=== SUMMARY (baseline 42.570 | innov2-320 49.958) ==="
for d in results/innov2_flip_tta results/innov2_384eval; do
    if [[ -f "${d}/paper_metrics.json" ]]; then
        v=$(${PYTHON} -c "import json; d=json.load(open('${d}/paper_metrics.json')); print(round(d['mAP50-95'],3))" 2>/dev/null)
        echo "  ${d}: mAP50-95 = ${v}"
    fi
done
