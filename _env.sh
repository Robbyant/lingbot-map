#!/usr/bin/env bash
# Shared config for the lingbot-map *.sh runners (bash/WSL/Linux equivalents of the .bat files).
# Paths default to WSL mappings of the Windows data (/mnt/c/...). Override any of them by
# exporting the variable before calling a runner, or edit here for a Linux box like horde.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Model checkpoint (Windows HF cache under WSL). Override with MODEL_PATH=... for another host.
: "${MODEL_PATH:=/mnt/c/Users/kschmid/.cache/huggingface/hub/models--robbyant--lingbot-map/snapshots/204754b72bb24f561f8d7e7e1e4e4cd9e809adf9/lingbot-map.pt}"

# Data roots (WSL mappings of the Windows drives).
: "${CSGO_DIR:=/mnt/c/csgo_data/lingbot_map_example}"
: "${LEFT_RIGHT_DIR:=/mnt/c/workspace/data/left_right/left_right}"
: "${YOUTUBE_DIR:=/mnt/c/workspace/data/youtube}"

# 32GB-safe defaults (expandable_segments is a no-op off-Windows but harmless).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
: "${KV_WINDOW:=32}"
: "${CAM_ITERS:=1}"

# Activate a Linux venv if one exists (Windows .venv/Scripts won't work under bash).
if [ -f "$REPO/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO/.venv/bin/activate"
fi
PY="${PY:-python}"
