#!/usr/bin/env bash
# View a point-cloud PLY in the viser viewer (bash equiv of view_ply.bat).
#   ./view_ply.sh <path/to/point_cloud.ply> [port]
source "$(dirname "$0")/_env.sh"

PLY_PATH="${1:-}"
[ -z "$PLY_PATH" ] && { echo "Usage: ./view_ply.sh <path/to/point_cloud.ply> [port]"; exit 1; }
PORT="${2:-8080}"
echo "Loading $PLY_PATH ... viewer at http://localhost:$PORT"
"$PY" view_ply.py "$PLY_PATH" --port "$PORT"
