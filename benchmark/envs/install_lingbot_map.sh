#!/usr/bin/env bash
# Add benchmark-side dependencies to the `lingbot_map` conda env.
#
# Recommended flow:
#   1. Install the upstream lingbot-map env per
#      https://github.com/robbyant/lingbot-map
#      (creates a conda env named `lingbot_map`).
#   2. Run this script to install benchmark-side deps (numpy, opencv,
#      open3d, evo, ...) into that same env.
#
# Usage:
#   bash envs/install_lingbot_map.sh           # interactive
#   bash envs/install_lingbot_map.sh --append  # non-interactive append to existing env
#   bash envs/install_lingbot_map.sh --force   # recreate env from scratch
set -euo pipefail

ENV_NAME="lingbot_map"
BENCH_DEPS="numpy opencv-python Pillow matplotlib open3d plyfile tqdm scipy evo pyyaml OpenEXR Imath"

MODE=""
for arg in "$@"; do
    case "$arg" in
        --force)  MODE="force" ;;
        --append) MODE="append" ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "[ERROR] Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

env_exists() {
    conda env list | awk '{print $1}' | grep -qw "$ENV_NAME"
}

install_bench_deps() {
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    pip install $BENCH_DEPS
}

create_from_scratch() {
    echo "[INFO] Creating $ENV_NAME from scratch ..."
    conda create -n "$ENV_NAME" python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    # torch >= 2.5 required: lingbot_map imports torch.nn.attention.flex_attention.
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    # FlashInfer: paged KV-cache attention used when _use_sdpa=false.
    pip install --index-url https://pypi.org/simple flashinfer-python
    pip install $BENCH_DEPS
    echo ""
    echo "[NOTE] This script created a fresh env but did NOT install the upstream"
    echo "       lingbot-map package. Install it per:"
    echo "       https://github.com/robbyant/lingbot-map"
}

if env_exists; then
    case "$MODE" in
        force)
            echo "[INFO] Removing existing env $ENV_NAME ..."
            conda env remove -n "$ENV_NAME" -y
            create_from_scratch
            ;;
        append)
            echo "[INFO] Appending bench deps to existing $ENV_NAME ..."
            install_bench_deps
            ;;
        "")
            echo "[INFO] Env '$ENV_NAME' already exists."
            echo "  [1] Install bench deps into the existing env (default)"
            echo "  [2] Recreate the env from scratch"
            echo "  [3] Skip"
            read -rp "Choice [1]: " choice
            case "${choice:-1}" in
                1) install_bench_deps ;;
                2) conda env remove -n "$ENV_NAME" -y; create_from_scratch ;;
                3) echo "[INFO] Skipped."; exit 0 ;;
                *) echo "[ERROR] Invalid choice"; exit 1 ;;
            esac
            ;;
    esac
else
    if [[ "$MODE" == "append" ]]; then
        echo "[ERROR] Env '$ENV_NAME' does not exist; nothing to append to." >&2
        echo "[HINT]  Install the upstream lingbot-map env first per" >&2
        echo "        https://github.com/robbyant/lingbot-map" >&2
        echo "        or rerun with --force to create from scratch." >&2
        exit 1
    fi
    create_from_scratch
fi

echo ""
echo "[INFO] $ENV_NAME is ready."
echo "[NOTE] Set _checkpoint in configs/methods/lingbot_map.yaml to your"
echo "       actual weights path (default placeholder: /path/to/lingbot-map.pt)."
