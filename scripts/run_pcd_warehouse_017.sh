#!/bin/bash
#
# Script to generate point cloud (.ply) files for a single scene (Warehouse_017)
# from multi-camera RGB-D frames.
#
# Usage:
#   bash scripts/run_pcd_warehouse_017.sh [OPTIONS]
#
# Options (all optional, with sensible defaults):
#   --data-root PATH          Input dataset root (default: /home/niran/Documents/data/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026)
#   --scene NAME              Scene name to process (default: Warehouse_017)
#   --split SPLIT             Dataset split: train/val/test (default: train)
#   --output-dir PATH         Output directory for .ply files (default: /home/niran/Documents/data/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026/train_debug)
#   --out-layout LAYOUT       Output layout: scene or split (default: scene)
#   --conda-env NAME          Conda environment name (default: aic2025-zv-py38)
#   --include-gt              Also export ground-truth txt files
#   --frame-stride N          Frame sampling stride (default: 1 - process every frame; use 10 for every 10th frame, etc.)
#   --max-seconds N           Max seconds to process per scene (default: none)
#   --max-cameras N           Max cameras to use (default: none - use all)
#   --fps N                   Dataset frame rate (default: 30)
#   --help                    Show this help message
#

set -e

# Default values
DATA_ROOT="/home/niran/Documents/data/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026"
SCENE_NAME="Warehouse_017"
SPLIT="train"
OUTPUT_DIR="/home/niran/Documents/data/vsw-mini/Data/3_dataset/AI_City_Challenge/MTMC_Tracking_2026/train_debug"
OUT_LAYOUT="scene"
CONDA_ENV="aic2025-zv-py38"
INCLUDE_GT="false"
FRAME_STRIDE="1"
MAX_SECONDS=""
MAX_CAMERAS=""
FPS="30"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --scene)
            SCENE_NAME="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --out-layout)
            OUT_LAYOUT="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --include-gt)
            INCLUDE_GT="true"
            shift 1
            ;;
        --frame-stride)
            FRAME_STRIDE="$2"
            shift 2
            ;;
        --max-seconds)
            MAX_SECONDS="$2"
            shift 2
            ;;
        --max-cameras)
            MAX_CAMERAS="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | tail -n +2 | head -n 30
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Point Cloud Generation for Warehouse_017"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Data Root:      $DATA_ROOT"
echo "  Scene:          $SCENE_NAME"
echo "  Split:          $SPLIT"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Out Layout:     $OUT_LAYOUT"
echo "  Conda Env:      $CONDA_ENV"
echo "  Frame Stride:   $FRAME_STRIDE"
echo "  Max Seconds:    ${MAX_SECONDS:-none}"
echo "  Max Cameras:    ${MAX_CAMERAS:-all}"
echo "  FPS:            $FPS"
echo "  Export GT:      $INCLUDE_GT"
echo ""

# Verify input directory exists
SCENE_INPUT_PATH="$DATA_ROOT/$SPLIT/$SCENE_NAME"
if [ ! -d "$SCENE_INPUT_PATH" ]; then
    echo "ERROR: Input scene directory not found: $SCENE_INPUT_PATH"
    exit 1
fi

if [ ! -f "$SCENE_INPUT_PATH/calibration.json" ]; then
    echo "ERROR: calibration.json not found in $SCENE_INPUT_PATH"
    exit 1
fi

if [ ! -d "$SCENE_INPUT_PATH/depth_maps" ]; then
    echo "ERROR: depth_maps directory not found in $SCENE_INPUT_PATH"
    exit 1
fi

if [ ! -d "$SCENE_INPUT_PATH/videos" ]; then
    echo "ERROR: videos directory not found in $SCENE_INPUT_PATH"
    exit 1
fi

echo "OK: Input scene directory verified: $SCENE_INPUT_PATH"
echo ""

# Create output directory
if [ "$OUT_LAYOUT" = "scene" ]; then
    mkdir -p "$OUTPUT_DIR/$SCENE_NAME"
else
    mkdir -p "$OUTPUT_DIR/$SPLIT/pcd"
fi
echo "OK: Output directory created: $OUTPUT_DIR"
echo ""

# Build the command to run generate_pcd_data.py
echo "Running point cloud generation..."
echo ""

PYTHON_RUN=(python)
if command -v conda >/dev/null 2>&1; then
    PYTHON_RUN=(conda run -n "$CONDA_ENV" python)
fi

CMD=("${PYTHON_RUN[@]}" tools/generate_pcd_data.py)
CMD+=(--data-root "$DATA_ROOT")
CMD+=(--out-dir "$OUTPUT_DIR")
CMD+=(--out-layout "$OUT_LAYOUT")
CMD+=(--splits "$SPLIT")
CMD+=(--scene-name "$SCENE_NAME")
CMD+=(--frame-stride "$FRAME_STRIDE")
CMD+=(--fps "$FPS")

if [ -n "$MAX_SECONDS" ]; then
    CMD+=(--max-seconds "$MAX_SECONDS")
fi

if [ -n "$MAX_CAMERAS" ]; then
    CMD+=(--max-cameras "$MAX_CAMERAS")
fi

if [ "$INCLUDE_GT" != "true" ]; then
    CMD+=(--no-gt)
fi

# Change to repo root to ensure relative paths work
cd "$REPO_ROOT"

# Run the command
printf "Command: "
printf "%q " "${CMD[@]}"
echo ""
"${CMD[@]}"

echo ""
echo "=========================================="
echo "Point Cloud Generation Complete"
echo "=========================================="
echo ""

# Count output files
if [ "$OUT_LAYOUT" = "scene" ]; then
    GENERATED_DIR="$OUTPUT_DIR/$SCENE_NAME"
else
    GENERATED_DIR="$OUTPUT_DIR/$SPLIT/pcd"
fi
if [ -d "$GENERATED_DIR" ]; then
    PLY_COUNT=$(find "$GENERATED_DIR" -name "${SCENE_NAME}_*.ply" -type f | wc -l)
    echo "Output directory: $GENERATED_DIR"
    echo "Total .ply files generated: $PLY_COUNT"
    echo ""
    
    if [ $PLY_COUNT -gt 0 ]; then
        echo "First 5 generated files:"
        find "$GENERATED_DIR" -name "${SCENE_NAME}_*.ply" -type f | sort | head -5 | sed 's/^/  /'
        echo ""
        
        # Validate first .ply file
        FIRST_PLY=$(find "$GENERATED_DIR" -name "${SCENE_NAME}_*.ply" -type f | sort | head -1)
        if [ -n "$FIRST_PLY" ]; then
            FILE_SIZE=$(du -h "$FIRST_PLY" | awk '{print $1}')
            echo "Sample .ply file: $(basename "$FIRST_PLY")"
            echo "  Size: $FILE_SIZE"
            echo "  Path: $FIRST_PLY"
        fi
    else
        echo "WARNING: No .ply files were generated!"
        exit 1
    fi
else
    echo "ERROR: Output directory not created: $GENERATED_DIR"
    exit 1
fi

echo ""
echo "SUCCESS: Point cloud generation completed for $SCENE_NAME"
echo ""
