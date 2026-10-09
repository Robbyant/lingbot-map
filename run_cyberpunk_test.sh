#!/usr/bin/env bash
# Cyberpunk gameplay test: extract depth (skip intro/menu) + self-test + overlays + 10-frame row.
# (bash equiv of run_cyberpunk_test.bat)
#   Arg 1: start_frame (default 3600 ~= 60s at 59.76fps). Arg 2: fps (5). Arg 3: first_k (300).
source "$(dirname "$0")/_env.sh"

VIDEO="$YOUTUBE_DIR/cyberpunk.mp4"
EXPORT_DIR="$YOUTUBE_DIR/cyberpunk_results"
START_FRAME="${1:-3600}"
FPS="${2:-5}"
FIRST_K="${3:-300}"

echo "Extracting cyberpunk depth: start_frame=$START_FRAME fps=$FPS first_k=$FIRST_K -> $EXPORT_DIR"
"$PY" demo.py --model_path "$MODEL_PATH" --video_path "$VIDEO" --fps "$FPS" --first_k "$FIRST_K" \
    --start_frame "$START_FRAME" --use_sdpa --export_results "$EXPORT_DIR" --comparison_stride 1 --no_viser

echo "Running reprojection self-test..."
"$PY" self_test_reprojection.py "$EXPORT_DIR" --frame 0

echo "Rendering reprojection overlays + 10-frame reprojection row..."
"$PY" show_reprojection.py "$EXPORT_DIR" --source 0
"$PY" show_reprojection.py "$EXPORT_DIR" --source 0 --target 20
"$PY" draw_reproj_row.py "$EXPORT_DIR" --source 0 --count 10
echo "Done. Results + overlays in $EXPORT_DIR"
