#!/bin/bash
# ============================================================================
# Batch video processing: inference (batch_demo.py) + rendering (rgbd_scan_render.py)
#
# Usage:
#   bash examples/process_videos.sh
#
# Skips videos that already have npz output (safe to re-run).
# ============================================================================

set -e

# ======================== CONFIG ========================

# Input / Output
VIDEO_DIR="/data2/datasets/demo_videos_new"
OUTPUT_DIR="/data2/clz/batch_outputs_c"
MODEL_PATH="/data1/clz/logs/longseq_baseline_1_dinov2_s3_mb/ckpts/checkpoint.pt"

# GPU
CUDA_DEVICE=0

# Inference (batch_demo.py)
TARGET_FRAMES=4000
MODE="windowed"
WINDOW_SIZE=64
FLOW_THRESHOLD=25.0
MAX_NON_KEYFRAME_GAP=100
VIS_THRESHOLD_INFER=2.0
IMAGE_STRIDE=1
SKY_MASK_DIR="${OUTPUT_DIR}/sky_masks"
SKY_MASK_VIZ_DIR="${OUTPUT_DIR}/sky_mask_viz"

# Rendering (rgbd_scan_render.py)
VOXEL_SIZE=0.001
VIS_THRESHOLD_RENDER=2.0
BACK_OFFSET=0.6
UP_OFFSET=0.3
LOOK_OFFSET=0.3
BIRDEYE_START="-120"
BIRDEYE_DURATION="120"
BIRDEYE_TRANSITION=30
NUM_WORKERS=16
DRAW_TRAJ="--draw_traj"
MASK_SKY_RENDER="--mask_sky"
FPS=60

# ======================== MAIN ========================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Find all video files
shopt -s nullglob nocaseglob
VIDEOS=("$VIDEO_DIR"/*.{mp4,mov,avi,mkv})
shopt -u nullglob nocaseglob

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "No video files found in $VIDEO_DIR"
    exit 1
fi

echo "============================================"
echo "Found ${#VIDEOS[@]} videos in $VIDEO_DIR"
echo "Output: $OUTPUT_DIR"
echo "============================================"

TOTAL=${#VIDEOS[@]}
FAILED=0

# ======================== PHASE 1: Inference ========================
echo ""
echo "============================================"
echo "PHASE 1: Inference (${TOTAL} videos)"
echo "============================================"

CURRENT=0
for VIDEO_PATH in "${VIDEOS[@]}"; do
    CURRENT=$((CURRENT + 1))
    VIDEO_NAME=$(basename "$VIDEO_PATH")
    SCENE_NAME="${VIDEO_NAME%.*}"
    NPZ_DIR="${OUTPUT_DIR}/${SCENE_NAME}"

    echo ""
    echo "[$CURRENT/$TOTAL] $VIDEO_NAME"

    if [ -d "$NPZ_DIR" ] && ls "$NPZ_DIR"/frame_*.npz &>/dev/null; then
        FRAME_COUNT=$(ls "$NPZ_DIR"/frame_*.npz 2>/dev/null | wc -l)
        echo "  [skip] NPZ already exists ($FRAME_COUNT frames)"
    else
        echo "  [inference] Running batch_demo.py..."
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python examples/batch_demo.py \
            --video_path "$VIDEO_PATH" \
            --output_folder "$OUTPUT_DIR" \
            --model_path "$MODEL_PATH" \
            --mode "$MODE" \
            --window_size $WINDOW_SIZE \
            --flow_threshold $FLOW_THRESHOLD \
            --max_non_keyframe_gap $MAX_NON_KEYFRAME_GAP \
            --vis_threshold $VIS_THRESHOLD_INFER \
            --image_stride $IMAGE_STRIDE \
            --target_frames $TARGET_FRAMES \
            --mask_sky \
            --sky_mask_dir "$SKY_MASK_DIR" \
            --sky_mask_visualization_dir "$SKY_MASK_VIZ_DIR" \
            --camera_mode zoom_out \
            --save_predictions --no_render \
        || { echo "  [FAILED] inference for $VIDEO_NAME"; FAILED=$((FAILED + 1)); continue; }
    fi
done

echo ""
echo "============================================"
echo "PHASE 1 done. Inference failures: $FAILED"
echo "============================================"

# ======================== PHASE 2: Rendering ========================
echo ""
echo "============================================"
echo "PHASE 2: Rendering (${TOTAL} videos)"
echo "============================================"

CURRENT=0
for VIDEO_PATH in "${VIDEOS[@]}"; do
    CURRENT=$((CURRENT + 1))
    VIDEO_NAME=$(basename "$VIDEO_PATH")
    SCENE_NAME="${VIDEO_NAME%.*}"
    NPZ_DIR="${OUTPUT_DIR}/${SCENE_NAME}"
    RENDER_OUTPUT="${OUTPUT_DIR}/${SCENE_NAME}.mp4"

    echo ""
    echo "[$CURRENT/$TOTAL] $VIDEO_NAME"

    if ! [ -d "$NPZ_DIR" ] || ! ls "$NPZ_DIR"/frame_*.npz &>/dev/null; then
        echo "  [skip] No NPZ found, skipping render"
        continue
    fi

    if [ -f "$RENDER_OUTPUT" ]; then
        echo "  [skip] Render already exists: $RENDER_OUTPUT"
    else
        echo "  [render] Running rgbd_scan_render.py..."
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python examples/rgbd_scan_render.py \
            --input_npz "$NPZ_DIR" \
            --output_video "$RENDER_OUTPUT" \
            --voxel_size $VOXEL_SIZE \
            --vis_threshold $VIS_THRESHOLD_RENDER \
            --back_offset $BACK_OFFSET \
            --up_offset $UP_OFFSET \
            --look_offset $LOOK_OFFSET \
            --birdeye_start "$BIRDEYE_START" \
            --birdeye_duration "$BIRDEYE_DURATION" \
            --birdeye_transition $BIRDEYE_TRANSITION \
            --num_workers $NUM_WORKERS \
            --fps $FPS \
            $DRAW_TRAJ $MASK_SKY_RENDER \
        || { echo "  [FAILED] render for $VIDEO_NAME"; FAILED=$((FAILED + 1)); continue; }
    fi
done

echo ""
echo "============================================"
echo "Finished: $TOTAL videos, $FAILED total failures"
echo "============================================"
