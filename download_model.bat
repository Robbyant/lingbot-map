@echo off
setlocal enabledelayedexpansion

REM Downloads the lingbot-map checkpoint(s) from Hugging Face into the local HF cache.
REM Usage: download_model.bat [lingbot-map|lingbot-map-long|lingbot-map-stage1]
REM Defaults to the balanced "lingbot-map" checkpoint (used in paper/benchmark/demo).

set "MODEL_NAME=%~1"
if "%MODEL_NAME%"=="" set "MODEL_NAME=lingbot-map"

call .\.venv\Scripts\activate.bat

echo.
echo Downloading robbyant/lingbot-map :: %MODEL_NAME%.pt ...
echo.

python -c "from huggingface_hub import hf_hub_download; p = hf_hub_download(repo_id='robbyant/lingbot-map', filename='%MODEL_NAME%.pt'); print('Downloaded to:', p)"

echo.
echo Done.
