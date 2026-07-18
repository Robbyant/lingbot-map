#!/usr/bin/env bash
# Install script for the `bench` env.
# Purpose: prepare.py, evaluate.py, report.py, and run.py (the dispatcher).
# Usage: bash envs/install_bench.sh [--force]
set -euo pipefail

ENV_NAME="bench"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--force" ]]; then
    echo "[INFO] Removing existing env $ENV_NAME ..."
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
fi

if conda env list | grep -qw "$ENV_NAME"; then
    echo "[INFO] $ENV_NAME already exists, skipping. Use --force to recreate."
    exit 0
fi

echo "[INFO] Creating $ENV_NAME ..."

conda create -n "$ENV_NAME" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

pip install numpy opencv-python-headless open3d evo matplotlib pyyaml tqdm scipy \
    imageio trimesh plyfile OpenEXR Imath Pillow onnxruntime-gpu==1.23.2

echo "[INFO] $ENV_NAME installed successfully."
