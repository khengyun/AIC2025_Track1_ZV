#!/bin/bash
CUR_DIR=$(pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT_NAME="${SPLIT_NAME:-test}"
export PYTHONPATH="${CUR_DIR}:${PYTHONPATH}"

cd ReIdModel

"${PYTHON_BIN}" extract_reid_feat.py \
    --model_path "${CUR_DIR}/weights/best_reid_model.pth" \
    --split_name "${SPLIT_NAME}" \
    --det_root "${CUR_DIR}/outputs/detection" \
    --output_dir "${CUR_DIR}/outputs/reid_feat" \
    --data_root "${CUR_DIR}/dataset/pcd_dataset"

cd "${CUR_DIR}"
