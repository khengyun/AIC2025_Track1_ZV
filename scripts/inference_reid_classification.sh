#!/bin/bash
CUR_DIR=$(pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SCENE_NAME="${SCENE_NAME:-}"
export PYTHONPATH="${CUR_DIR}:${PYTHONPATH}"

cd ReIdModel

CMD=(
    "${PYTHON_BIN}" embedding_similarity_classification.py
    --split_name "${SPLIT_NAME}"
    --det_root "${CUR_DIR}/outputs/detection"
    --reid_feat_root "${CUR_DIR}/outputs/reid_feat"
    --model_path "${CUR_DIR}/weights/best_reid_model.pth"
    --FourierGR1T2_dir "${CUR_DIR}/dataset/obj_crop_pcd_dataset/train/0_9998"
    --AgilityDigit_dir "${CUR_DIR}/dataset/obj_crop_pcd_dataset/train/0_9999"
)
if [ -n "${SCENE_NAME}" ]; then
    CMD+=(--scene_name "${SCENE_NAME}")
fi
"${CMD[@]}"


cd "${CUR_DIR}"
