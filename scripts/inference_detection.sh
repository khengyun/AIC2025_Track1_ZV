#!/bin/bash
CUR_DIR=$(pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SCENE_NAME="${SCENE_NAME:-}"
export PYTHONPATH="${CUR_DIR}:${CUR_DIR}/V-DETR/third_party/pointnet2:${PYTHONPATH}"

cd V-DETR
CMD=(
    "${PYTHON_BIN}" inference.py
    --config configs/best.yaml
    --test_ckpt checkpoints/best/checkpoint_best.pth
    --output_dir "${CUR_DIR}/outputs/detection"
    --data_root "${CUR_DIR}/dataset/pcd_dataset"
    --split_name "${SPLIT_NAME}"
)
if [ -n "${SCENE_NAME}" ]; then
    CMD+=(--scene_name "${SCENE_NAME}")
fi
"${CMD[@]}" # "test" or "val"
