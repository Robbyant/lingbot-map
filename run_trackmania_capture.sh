#!/usr/bin/env bash
# Run lingbot-map on the left_right video clips (bash equiv of run_trackmania_capture.bat).
# Processes 0.mp4 by default; change the glob to *.mp4 for the whole folder.
source "$(dirname "$0")/_env.sh"

[ -n "${1:-}" ] && MODEL_PATH="$1"

for V in "$LEFT_RIGHT_DIR"/0.mp4; do
    [ -e "$V" ] || { echo "no clip: $V"; continue; }
    base="$(basename "$V" .mp4)"
    EXPORT_DIR="$LEFT_RIGHT_DIR/results/$base"
    echo "Running lingbot-map on $(basename "$V") -> $EXPORT_DIR"
    if [ -f "$EXPORT_DIR/poses.json" ]; then
        echo "  SKIP -- already done (poses.json exists)"
    else
        "$PY" demo.py --model_path "$MODEL_PATH" --video_path "$V" --fps 10 --use_sdpa \
            --export_results "$EXPORT_DIR" --kv_cache_sliding_window "$KV_WINDOW" --camera_num_iterations "$CAM_ITERS"
    fi
    echo "  Results: $EXPORT_DIR"
done
