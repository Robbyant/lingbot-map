#!/usr/bin/env bash
# Download robbyant/lingbot-map :: <name>.pt from HuggingFace (bash equiv of download_model.bat).
source "$(dirname "$0")/_env.sh"

MODEL_NAME="${1:-lingbot-map}"
echo "Downloading robbyant/lingbot-map :: $MODEL_NAME.pt ..."
"$PY" -c "from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='robbyant/lingbot-map', filename='$MODEL_NAME.pt'); print('Downloaded to:', p)"
echo "Done."
