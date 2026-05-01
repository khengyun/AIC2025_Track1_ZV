#!/bin/bash
CUR_DIR=$(pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT_NAME="${SPLIT_NAME:-test}"
export PYTHONPATH="${CUR_DIR}:${PYTHONPATH}"

cd Tracker

"${PYTHON_BIN}" short_term_tracking.py \
    --split_name "${SPLIT_NAME}" \
    --detection_folder "${CUR_DIR}/outputs/detection" \
    --use_reid_feat \
    --reid_feat_folder "${CUR_DIR}/outputs/reid_feat" \
    --output_folder "${CUR_DIR}/outputs/tracking"

cd "${CUR_DIR}"
