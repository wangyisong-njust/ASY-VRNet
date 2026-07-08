#!/usr/bin/env bash
set -euo pipefail

IMAGE_ID="${1:-15239}"
INPUT_SHAPE="${2:-512}"
CONFIDENCE="${3:-0.3}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" compare_detection.py \
  --image "image/${IMAGE_ID}.jpg" \
  --radar-root "dataset/VOCradar_5_frames" \
  --baseline "weights/baseline_best.pth" \
  --ours "weights/final_greedy_soup.pth" \
  --gt-xml "/root/datasets/WaterScenes_data/detection/xml/${IMAGE_ID}.xml" \
  --input-shape "${INPUT_SHAPE},${INPUT_SHAPE}" \
  --confidence "${CONFIDENCE}" \
  --radar-legacy-preprocess --no-radar-preserve-points \
  --radar-source-order "range,doppler,elevation,power" \
  --radar-target-order "range,doppler,elevation,power" \
  --out "outputs/compare_${IMAGE_ID}_gt_shape_${INPUT_SHAPE}_conf_${CONFIDENCE}.jpg"
