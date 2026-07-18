#!/usr/bin/env bash
# Install all benchmark environments
# Usage: bash envs/install_all.sh [--force]
# Each script is idempotent: skips if env exists, use --force to recreate all.
# Individual failures do not block subsequent installs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAG="${1:-}"

# Auto-discover sibling install_*.sh scripts (excluding this dispatcher).
# Alphabetical order ensures install_bench.sh runs before any method env.
SCRIPTS=()
for f in "$SCRIPT_DIR"/install_*.sh; do
    name="$(basename "$f")"
    [[ "$name" == "install_all.sh" ]] && continue
    SCRIPTS+=("$name")
done

FAILED=()

for script in "${SCRIPTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "[install_all] Running $script ..."
    echo "=========================================="
    if bash "$SCRIPT_DIR/$script" $FLAG; then
        echo "[install_all] $script OK"
    else
        echo "[install_all] $script FAILED"
        FAILED+=("$script")
    fi
done

echo ""
echo "=========================================="
echo "[install_all] Summary"
echo "=========================================="
echo "Total: ${#SCRIPTS[@]}"
echo "Failed: ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
echo "All environments installed successfully."
