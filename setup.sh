#!/usr/bin/env bash
# Create a Linux venv and install lingbot-map (bash equiv of setup.bat).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    uv venv --python 3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing torch 2.8.0 (cu128)..."
uv pip install --python .venv torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installing lingbot-map (+ vis, csgo_example extras)..."
uv pip install --python .venv -e ".[vis,csgo_example]"

echo "Setup complete. Next:"
echo "  ./download_model.sh"
echo "  ./run_csgo_example.sh"
