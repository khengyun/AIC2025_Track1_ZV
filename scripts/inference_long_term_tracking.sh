#!/bin/bash
CUR_DIR=$(pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT_NAME="${SPLIT_NAME:-test}"
export PYTHONPATH="${CUR_DIR}:${PYTHONPATH}"

cd Tracker

"${PYTHON_BIN}" long_term_tracking.py \
    --split_name "${SPLIT_NAME}" \
    --track_res_folder "${CUR_DIR}/outputs/tracking" \
    --reid_feat_folder "${CUR_DIR}/outputs/reid_feat" \
    --output_folder "${CUR_DIR}/outputs/tracking"

cd "${CUR_DIR}"

echo "Long-term tracking completed for all chunks."
