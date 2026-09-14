#!/usr/bin/env bash
# Download TUM RGB-D sequences into the layout benchmark/datasets/tum.py expects:
#   {raw_data_root}/{scene_name}/{rgb/, rgb.txt, depth/, groundtruth.txt}
#
# Usage:
#   ./download_tum.sh /path/to/TUM-RGBD [scene ...]
# If no scenes are given, downloads two small, commonly-used ones.

set -euo pipefail

RAW_DATA_ROOT="${1:?Usage: $0 <raw_data_root> [scene ...]}"
shift || true
SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
    SCENES=(rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_xyz)
fi

mkdir -p "$RAW_DATA_ROOT"
cd "$RAW_DATA_ROOT"

for scene in "${SCENES[@]}"; do
    # Freiburg camera id is embedded in the scene name (freiburg1/2/3).
    fr=$(echo "$scene" | grep -oE 'freiburg[0-9]')
    url="https://cvg.cit.tum.de/rgbd/dataset/${fr}/${scene}.tgz"

    if [ -d "$scene" ]; then
        echo "Already present: $scene (skipping)"
        continue
    fi

    echo "Downloading $scene from $url ..."
    curl -L --fail -o "${scene}.tgz" "$url"
    echo "Extracting $scene ..."
    tar xzf "${scene}.tgz"
    rm "${scene}.tgz"
    echo "Done: $RAW_DATA_ROOT/$scene"
done

echo
echo "raw_data_root is ready at: $RAW_DATA_ROOT"
echo "Set this in benchmark/configs/datasets/tum.yaml as raw_data_root: $RAW_DATA_ROOT"
