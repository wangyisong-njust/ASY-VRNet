#!/usr/bin/env bash
set -euo pipefail

# Innovation 3: modality-dropout consistency regularization.
#
# A second forward with the radar modality dropped is pulled toward the
# full-input prediction (detached teacher), so the model degrades gracefully
# when radar is unreliable/missing. Training-time only; inference is unchanged.
#
# Built on the reproduced high-score BASELINE fusion + loss form so the result
# is directly comparable to paper_metrics_legacy_highscore_best.
#
# Key env (all read by utils/utils_fit.py):
#   ASY_CONSISTENCY=1                 enable
#   ASY_CONSISTENCY_PROB=0.5          per-iter probability of the extra forward
#   ASY_CONSISTENCY_WEIGHT=1.0        overall consistency weight
#   ASY_CONSISTENCY_DET_WEIGHT=1.0    detection-branch consistency weight
#   ASY_CONSISTENCY_SEG_WEIGHT=0.5    segmentation-branch consistency weight
#   ASY_CONSISTENCY_RADAR_DROP=1.0    1.0 = full radar drop; <1.0 = random mask

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

export ASY_CONSISTENCY=${ASY_CONSISTENCY:-1}
export ASY_CONSISTENCY_PROB=${ASY_CONSISTENCY_PROB:-0.5}
export ASY_CONSISTENCY_WEIGHT=${ASY_CONSISTENCY_WEIGHT:-1.0}
export ASY_CONSISTENCY_DET_WEIGHT=${ASY_CONSISTENCY_DET_WEIGHT:-1.0}
export ASY_CONSISTENCY_SEG_WEIGHT=${ASY_CONSISTENCY_SEG_WEIGHT:-0.5}
export ASY_CONSISTENCY_RADAR_DROP=${ASY_CONSISTENCY_RADAR_DROP:-1.0}

EXP_NAME=${EXP_NAME:-innovation3_consistency_phi_l_5frames_bs64_300e_320}
MASTER_PORT=${MASTER_PORT:-29550}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export ASY_DISTRIBUTED=${ASY_DISTRIBUTED:-1}
export ASY_SYNC_BN=${ASY_SYNC_BN:-1}
export ASY_FP16=${ASY_FP16:-1}

export ASY_INPUT_SHAPE=${ASY_INPUT_SHAPE:-320,320}
export ASY_BATCH_SIZE=${ASY_BATCH_SIZE:-64}
export ASY_NUM_WORKERS=${ASY_NUM_WORKERS:-8}
export ASY_UNFREEZE_EPOCH=${ASY_UNFREEZE_EPOCH:-300}
export ASY_SAVE_PERIOD=${ASY_SAVE_PERIOD:-20}
export ASY_PHI=${ASY_PHI:-l}

export ASY_INIT_LR=${ASY_INIT_LR:-0.01}
export ASY_LR_DECAY=${ASY_LR_DECAY:-cos}
export ASY_OPTIMIZER=${ASY_OPTIMIZER:-sgd}
export ASY_MOMENTUM=${ASY_MOMENTUM:-0.937}
export ASY_WEIGHT_DECAY=${ASY_WEIGHT_DECAY:-0.0005}
export ASY_FREEZE_TRAIN=${ASY_FREEZE_TRAIN:-0}
export ASY_YOLO_BOX_WEIGHT=${ASY_YOLO_BOX_WEIGHT:-1.0}
export ASY_YOLO_OBJ_WEIGHT=${ASY_YOLO_OBJ_WEIGHT:-2.0}
export ASY_YOLO_CLS_WEIGHT=${ASY_YOLO_CLS_WEIGHT:-2.0}

export ASY_EVAL=${ASY_EVAL:-0}
export ASY_EVAL_PERIOD=${ASY_EVAL_PERIOD:-10}
export ASY_BEST_METRIC=${ASY_BEST_METRIC:-det}

export ASY_SAVE_DIR=${ASY_SAVE_DIR:-logs_${EXP_NAME}}
export ASY_SAVE_DIR_SEG=${ASY_SAVE_DIR_SEG:-logs_seg_${EXP_NAME}}
export ASY_VOCDEVKIT=${ASY_VOCDEVKIT:-${PROJECT_ROOT}/dataset/VOCdevkit}
export ASY_RADAR_ROOT=${ASY_RADAR_ROOT:-${PROJECT_ROOT}/dataset/VOCradar_5_frames}
export ASY_TASK_LOSS=${ASY_TASK_LOSS:-sum}
export ASY_FUSION_MODE=${ASY_FUSION_MODE:-baseline}
export ASY_RADAR_DROPOUT=${ASY_RADAR_DROPOUT:-0}
export ASY_RADAR_CHANNELS=${ASY_RADAR_CHANNELS:-4}
export ASY_RADAR_ALIGN_MODE=${ASY_RADAR_ALIGN_MODE:-letterbox}
export ASY_RADAR_NORMALIZE=${ASY_RADAR_NORMALIZE:-0}
export ASY_RADAR_PRESERVE_POINTS=${ASY_RADAR_PRESERVE_POINTS:-0}
export ASY_RADAR_SOURCE_ORDER=${ASY_RADAR_SOURCE_ORDER:-range,doppler,elevation,power}
export ASY_RADAR_TARGET_ORDER=${ASY_RADAR_TARGET_ORDER:-range,doppler,elevation,power}
export ASY_RADAR_LEGACY_PREPROCESS=${ASY_RADAR_LEGACY_PREPROCESS:-1}
export ASY_WEATHER_AUG=${ASY_WEATHER_AUG:-0}

mkdir -p "${ASY_SAVE_DIR}" "${ASY_SAVE_DIR_SEG}"

printf 'Innovation-3 consistency training config:\n'
printf '  exp=%s\n' "${EXP_NAME}"
printf '  consistency=%s prob=%s weight=%s det_w=%s seg_w=%s radar_drop=%s\n' \
    "${ASY_CONSISTENCY}" "${ASY_CONSISTENCY_PROB}" "${ASY_CONSISTENCY_WEIGHT}" \
    "${ASY_CONSISTENCY_DET_WEIGHT}" "${ASY_CONSISTENCY_SEG_WEIGHT}" "${ASY_CONSISTENCY_RADAR_DROP}"
printf '  fusion=%s lr=%s epochs=%s batch=%s init_weights=%s\n' \
    "${ASY_FUSION_MODE}" "${ASY_INIT_LR}" "${ASY_UNFREEZE_EPOCH}" "${ASY_BATCH_SIZE}" "${ASY_MODEL_PATH:-<none>}"

if [[ "${ASY_SKIP_STARTUP_CHECKS:-0}" =~ ^(0|false|FALSE|no|NO|off|OFF)$ ]]; then
    "${PYTHON}" scripts/check_dataset.py
    "${PYTHON}" scripts/audit_detection_pipeline.py --sample_limit 256 --skip_model
fi

stamp=$(date +%Y%m%d_%H%M%S)
"${PYTHON}" -m torch.distributed.run --master_port="${MASTER_PORT}" --nproc_per_node=4 train.py 2>&1 | tee "${ASY_SAVE_DIR}/train_${stamp}.log"
