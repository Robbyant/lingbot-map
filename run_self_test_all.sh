#!/usr/bin/env bash
# Reprojection self-test across all available results dirs (bash equiv of run_self_test_all.bat).
# Pure validation -- no GPU/inference; skips dirs not exported yet.
source "$(dirname "$0")/_env.sh"

DIRS=(
    "$CSGO_DIR/selftest_results"
    "$LEFT_RIGHT_DIR/1_results"
    "$YOUTUBE_DIR/cyberpunk_results"
)

for D in "${DIRS[@]}"; do
    echo
    echo "================================================================"
    echo "Self-test: $D"
    echo "================================================================"
    if [ -f "$D/poses.json" ] && [ -d "$D/depth_raw" ]; then
        "$PY" self_test_reprojection.py "$D" --frame 0
    else
        echo "  SKIP -- no poses.json/depth_raw yet (run not finished / not exported)"
    fi
done
echo "Done."
