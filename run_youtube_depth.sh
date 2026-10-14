#!/usr/bin/env bash
# Extract depth from every .mp4 in a folder (bash equiv of run_youtube_depth.bat).
#   Arg 1: data dir (default $YOUTUBE_DIR). Arg 2: fps (5). Arg 3: minutes limit (1; 0 = full clip).
source "$(dirname "$0")/_env.sh"

DATA_DIR="${1:-$YOUTUBE_DIR}"
FPS="${2:-5}"
MAX_MINUTES="${3:-1}"

FIRST_K_FLAG=()
if [ "$MAX_MINUTES" != "0" ]; then
    FIRST_K=$(( FPS * MAX_MINUTES * 60 ))
    FIRST_K_FLAG=(--first_k "$FIRST_K")
fi

for V in "$DATA_DIR"/*.mp4; do
    [ -e "$V" ] || { echo "no clips in $DATA_DIR"; break; }
    base="$(basename "$V" .mp4)"
    EXPORT_DIR="$DATA_DIR/${base}_results"
    echo "Extracting depth from $(basename "$V") (fps=$FPS, first ${MAX_MINUTES}min) -> $EXPORT_DIR"
    if [ -f "$EXPORT_DIR/poses.json" ]; then
        echo "  SKIP -- already done (poses.json exists)"
    else
        "$PY" demo.py --model_path "$MODEL_PATH" --video_path "$V" --fps "$FPS" "${FIRST_K_FLAG[@]}" \
            --use_sdpa --export_results "$EXPORT_DIR" --kv_cache_sliding_window "$KV_WINDOW" --camera_num_iterations "$CAM_ITERS" \
            || echo "  FAILED: $(basename "$V")"
    fi
    echo "  Results: $EXPORT_DIR"
done
echo "Done. Depth + poses in $DATA_DIR/<clip>_results"
