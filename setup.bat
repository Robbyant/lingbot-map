@echo off
setlocal enabledelayedexpansion

REM Sets up the lingbot-map venv: Python 3.11, torch 2.8.0 cu128, the package itself
REM (with vis + csgo_example extras). Does NOT install FlashInfer or the render
REM pipeline extras (Kaolin/ffmpeg/CUDA ext) -- see README.md for those if needed.

if not exist ".venv" (
    echo Creating venv...
    uv venv --python 3.11 .venv
)

call .\.venv\Scripts\activate.bat

echo.
echo Installing torch 2.8.0 (cu128)...
uv pip install --python .venv torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error

echo.
echo Installing lingbot-map (+ vis, csgo_example extras)...
uv pip install --python .venv -e ".[vis,csgo_example]"
if errorlevel 1 goto :error

echo.
echo Setup complete. Next steps:
echo   download_model.bat            (fetch lingbot-map.pt)
echo   run_csgo_example.bat ^<ckpt^>   (run on CS:GO gameplay frames)
goto :eof

:error
echo.
echo Setup failed -- see error above.
exit /b 1
