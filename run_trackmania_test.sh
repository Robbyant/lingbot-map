#!/usr/bin/env bash
# Trackmania test: extract depth from a left_right clip + self-test + overlays + 10-frame row.
# (bash equiv of run_trackmania_test.bat)
#   Arg 1: clip name (default 1). Arg 2: fps (10).
source "$(dirname "$0")/_env.sh"

CLIP="${1:-1}"
FPS="${2:-10}"
VIDEO="$LEFT_RIGHT_DIR/$CLIP.mp4"
EXPORT_DIR="$LEFT_RIGHT_DIR/${CLIP}_results"

echo "Extracting trackmania depth: clip=$CLIP fps=$FPS -> $EXPORT_DIR"
"$PY" demo.py --model_path "$MODEL_PATH" --video_path "$VIDEO" --fps "$FPS" --first_k 21 \
    --use_sdpa --export_results "$EXPORT_DIR" --comparison_stride 1 --no_viser

echo "Running reprojection self-test..."
"$PY" self_test_reprojection.py "$EXPORT_DIR" --frame 0

echo "Rendering reprojection overlays + 10-frame reprojection row..."
"$PY" show_reprojection.py "$EXPORT_DIR" --source 0
"$PY" show_reprojection.py "$EXPORT_DIR" --source 0 --target 20
"$PY" draw_reproj_row.py "$EXPORT_DIR" --source 0 --count 10
echo "Done. Results + overlays in $EXPORT_DIR"
