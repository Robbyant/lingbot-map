#!/usr/bin/env bash
# End-to-end: download TUM RGB-D (has ground-truth trajectories), run
# lingbot-map on it, evaluate against GT, and inspect the per-frame error.
# Everything runs in the currently-active conda env (expected: lingbot-map,
# which already has the benchmark-side deps installed: plyfile, open3d, evo, ...).
#
# Run from the benchmark/ directory, inside the lingbot-map env:
#   ./run_tum_eval.sh                                  # default scene (freiburg1_desk)
#   ./run_tum_eval.sh rgbd_dataset_freiburg1_room       # any other TUM scene name
#
# Note: prepare.py/run.py/evaluate.py process every scene already downloaded
# into data/TUM-RGBD/ (not just $SCENE) -- already-processed scenes are
# skipped automatically, so running this repeatedly with different scene
# names accumulates results rather than replacing them.

set -euo pipefail
cd "$(dirname "$0")"

SCENE="${1:-rgbd_dataset_freiburg1_desk}"

# 1. Download a GT-bearing sequence (skips if already present)
./download_tum.sh ./data/TUM-RGBD "$SCENE"

# 2. Convert data -> run the model -> compute error vs GT (for every scene present)
python prepare.py  --config configs/tum.yaml
python run.py      --config configs/tum.yaml
python evaluate.py --config configs/tum.yaml

# 3. Per-frame error breakdown + plots (aggregate numbers, worst frames, error curve)
python inspect_trajectory_error.py \
    --scene_dir "data/workspace/tum/tum/${SCENE}"
