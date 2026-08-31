#!/usr/bin/env bash
# Run lingbot-map on every left_right .mp4 (bash equiv of run_left_right.bat).
source "$(dirname "$0")/_env.sh"

[ -n "${1:-}" ] && MODEL_PATH="$1"

for V in "$LEFT_RIGHT_DIR"/*.mp4; do
    [ -e "$V" ] || { echo "no clips in $LEFT_RIGHT_DIR"; break; }
    base="$(basename "$V" .mp4)"
    EXPORT_DIR="$LEFT_RIGHT_DIR/${base}_results"
    echo "Running lingbot-map on $(basename "$V") -> $EXPORT_DIR"
    "$PY" demo.py --model_path "$MODEL_PATH" --video_path "$V" --fps 10 --use_sdpa \
        --export_results "$EXPORT_DIR" --kv_cache_sliding_window "$KV_WINDOW" --camera_num_iterations "$CAM_ITERS"
    echo "Results saved to: $EXPORT_DIR"
done
