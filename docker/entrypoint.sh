#!/bin/bash
set -e

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/model}"
HF_MODEL_NAME="${HF_MODEL_NAME:-lingbot-map}"
MODEL_PATH="${MODEL_PATH:-}"

# ── Resolve model path ────────────────────────────────────────────────────────
if [ -z "$MODEL_PATH" ]; then
    # Look for a matching .pt file already in the cache volume
    MODEL_FILE=$(find "$MODEL_CACHE_DIR" -name "${HF_MODEL_NAME}.pt" -print -quit 2>/dev/null || true)
    if [ -z "$MODEL_FILE" ]; then
        MODEL_FILE=$(find "$MODEL_CACHE_DIR" -name "*.pt" -print -quit 2>/dev/null || true)
    fi

    if [ -z "$MODEL_FILE" ]; then
        echo "Model not found in ${MODEL_CACHE_DIR}. Downloading '${HF_MODEL_NAME}' from HuggingFace..."
        python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="robbyant/lingbot-map",
    local_dir="${MODEL_CACHE_DIR}",
    ignore_patterns=["*.pdf", "*.md", "*.txt", "*.gitattributes"],
)
print("Download complete.")
PYEOF
        MODEL_FILE=$(find "$MODEL_CACHE_DIR" -name "${HF_MODEL_NAME}.pt" -print -quit 2>/dev/null || \
                     find "$MODEL_CACHE_DIR" -name "*.pt" -print -quit 2>/dev/null || true)
    fi

    MODEL_PATH="$MODEL_FILE"
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: No model .pt file found. Either:"
    echo "  - Mount a pre-downloaded model: -v /path/to/model.pt:/model/lingbot-map.pt"
    echo "  - Or let auto-download run (requires internet access)"
    exit 1
fi

echo "Using model: ${MODEL_PATH}"

# ── Check FlashInfer availability ─────────────────────────────────────────────
EXTRA_ARGS=""
python -c "import flashinfer" 2>/dev/null || EXTRA_ARGS="--use_sdpa"
if [ -n "$EXTRA_ARGS" ]; then
    echo "FlashInfer not available, using SDPA backend."
fi

# ── Launch demo ───────────────────────────────────────────────────────────────
exec python /app/demo.py \
    --model_path "$MODEL_PATH" \
    $EXTRA_ARGS \
    "$@"
