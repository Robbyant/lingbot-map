#!/usr/bin/env bash
# 21-frame CS:GO validation + reprojection self-test + overlays (bash equiv of run_self_test.bat).
# Arg 1: start_frame (default 0 -> frames 0-20). e.g. ./run_self_test.sh 250 -> frames 250-270.
source "$(dirname "$0")/_env.sh"

START="${1:-0}"
IMAGE_FOLDER="$CSGO_DIR/csgo_dm_1"
if [ "$START" = "0" ]; then
    EXPORT_DIR="$CSGO_DIR/selftest_results"
else
    EXPORT_DIR="$CSGO_DIR/selftest_results_$START"
fi

echo "Running 21-frame validation inference (start_frame=$START)..."
"$PY" demo.py --model_path "$MODEL_PATH" --image_folder "$IMAGE_FOLDER" --start_frame "$START" --first_k 21 \
    --use_sdpa --export_results "$EXPORT_DIR" --comparison_stride 1 \
    --kv_cache_sliding_window 32 --camera_num_iterations 1 --no_viser

echo "Running self-consistency reprojection test..."
"$PY" self_test_reprojection.py "$EXPORT_DIR" --frame 0

echo "Rendering reprojection overlay + 10-frame reprojection row..."
"$PY" show_reprojection.py "$EXPORT_DIR" --source 0
"$PY" draw_reproj_row.py "$EXPORT_DIR" --source 0 --count 10
echo "Done. Results in $EXPORT_DIR"
