@echo off
setlocal enabledelayedexpansion

REM Same as run_csgo_example.bat, but for the left_right video clips: runs lingbot-map
REM streaming reconstruction on every .mp4 in C:\workspace\data\left_right\left_right and
REM writes poses.json + depth PNGs to <clip>_results next to each video.

cd /d C:\workspace\world\lingbot-map

set "DATA_DIR=C:\workspace\data\left_right\left_right"
set "MODEL_PATH=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-map\snapshots\204754b72bb24f561f8d7e7e1e4e4cd9e809adf9\lingbot-map.pt"
if not "%~1"=="" set "MODEL_PATH=%~1"

REM 32GB-GPU memory defaults (identical to run_csgo_example.bat).
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "KV_WINDOW=32"
set "CAM_ITERS=1"

call .\.venv\Scripts\activate.bat

for %%V in ("%DATA_DIR%\*.mp4") do (
    set "EXPORT_DIR=%DATA_DIR%\%%~nV_results"
    echo.
    echo Running lingbot-map on %%~nxV  -^>  !EXPORT_DIR!
    echo.
    python demo.py --model_path "%MODEL_PATH%" --video_path "%%~fV" --fps 10 --use_sdpa --export_results "!EXPORT_DIR!" --kv_cache_sliding_window %KV_WINDOW% --camera_num_iterations %CAM_ITERS%
    echo.
    echo Results saved to: !EXPORT_DIR!
)

endlocal
