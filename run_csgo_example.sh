#!/usr/bin/env bash
# Run lingbot-map on a CS:GO gameplay clip (bash equivalent of run_csgo_example.bat).
source "$(dirname "$0")/_env.sh"

[ -n "${1:-}" ] && MODEL_PATH="$1"
IMAGE_FOLDER="$CSGO_DIR/csgo_dm_1"
EXPORT_DIR="$CSGO_DIR/csgo_dm_1_results"

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "Extracting CS:GO frames ..."
    "$PY" extract_csgo_frames.py --hdf5_path /mnt/c/csgo_data/hdf5_dm_july2021_1.hdf5 --output_folder "$IMAGE_FOLDER" --num_frames 500
fi

echo "Running lingbot-map on: $IMAGE_FOLDER"
"$PY" demo.py --model_path "$MODEL_PATH" --image_folder "$IMAGE_FOLDER" --use_sdpa \
    --export_results "$EXPORT_DIR" --kv_cache_sliding_window "$KV_WINDOW" --camera_num_iterations "$CAM_ITERS" "${@:2}"
echo "Results saved to: $EXPORT_DIR"
