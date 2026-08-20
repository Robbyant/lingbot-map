@echo off
setlocal enabledelayedexpansion

REM Extract depth + camera poses with lingbot-map from every .mp4 in
REM C:\workspace\data\youtube (e.g. the downloaded YouTube clip). Same 32GB-safe
REM defaults as run_csgo_example.bat. Writes poses.json + depth PNGs to
REM <clip>_results next to each video.
REM
REM Arg 1: optional data dir (default C:\workspace\data\youtube)
REM Arg 2: optional --fps sampling value (default 10)

cd /d C:\workspace\world\lingbot-map

set "DATA_DIR=%~1"
if "%DATA_DIR%"=="" set "DATA_DIR=C:\workspace\data\youtube"
set "FPS=%~2"
if not defined FPS set "FPS=10"

set "MODEL_PATH=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-map\snapshots\204754b72bb24f561f8d7e7e1e4e4cd9e809adf9\lingbot-map.pt"

REM 32GB-GPU memory defaults (identical to run_csgo_example.bat).
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "KV_WINDOW=32"
set "CAM_ITERS=1"

call .\.venv\Scripts\activate.bat

for %%V in ("%DATA_DIR%\*.mp4") do (
    set "EXPORT_DIR=%DATA_DIR%\%%~nV_results"
    echo.
    echo Extracting depth from %%~nxV  ^(fps=%FPS%^)  -^>  !EXPORT_DIR!
    echo.
    if exist "!EXPORT_DIR!\poses.json" (
        echo   SKIP -- already done ^(poses.json exists^)
    ) else (
        python demo.py --model_path "%MODEL_PATH%" --video_path "%%~fV" --fps %FPS% --use_sdpa --export_results "!EXPORT_DIR!" --kv_cache_sliding_window %KV_WINDOW% --camera_num_iterations %CAM_ITERS%
        if errorlevel 1 echo   FAILED: %%~nxV
    )
    echo   Results: !EXPORT_DIR!
)

echo.
echo Done. Depth + poses in %DATA_DIR%\^<clip^>_results
endlocal
