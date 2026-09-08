@echo off
setlocal enabledelayedexpansion

REM Same as run_youtube_depth.bat, but limited to witcher.mp4 specifically. Writes
REM poses.json + depth PNGs to witcher_results next to the video.
REM
REM Arg 1: optional data dir (default C:\workspace\data\youtube)
REM Arg 2: optional --fps sampling value (default 5)
REM Arg 3: optional minutes limit -- only process the first N minutes of the clip
REM        (default 1; pass 0 to disable and process the full clip)
REM Arg 4: optional output dir override (default <data dir>\witcher_results)
REM Arg 5: optional start minute -- skip into the source clip before sampling (default 0).
REM        witcher.mp4 is native 60fps, so this converts to --start_frame = minute*60*60.

cd /d C:\workspace\world\lingbot-map

set "DATA_DIR=%~1"
if "%DATA_DIR%"=="" set "DATA_DIR=C:\workspace\data\youtube"
set "FPS=%~2"
if not defined FPS set "FPS=5"
set "MAX_MINUTES=%~3"
if not defined MAX_MINUTES set "MAX_MINUTES=1"
set "OUT_OVERRIDE=%~4"
set "START_MINUTE=%~5"
if not defined START_MINUTE set "START_MINUTE=0"
REM A literal "" passed through some callers (e.g. cmd /c from a non-cmd shell) can
REM arrive as the two-character string ""  rather than truly empty -- treat that the
REM same as "no override" instead of using it as a garbage path.
if "%OUT_OVERRIDE%"=="""" set "OUT_OVERRIDE="

set "MODEL_PATH=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-map\snapshots\204754b72bb24f561f8d7e7e1e4e4cd9e809adf9\lingbot-map.pt"

REM 32GB-GPU memory defaults (identical to run_csgo_example.bat).
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "KV_WINDOW=32"
set "CAM_ITERS=1"

set "FIRST_K_FLAG="
if not "%MAX_MINUTES%"=="0" (
    set /a "FIRST_K=%FPS% * %MAX_MINUTES% * 60"
    set "FIRST_K_FLAG=--first_k !FIRST_K!"
)

set "START_FRAME_FLAG="
if not "%START_MINUTE%"=="0" (
    set /a "START_FRAME=%START_MINUTE% * 60 * 60"
    set "START_FRAME_FLAG=--start_frame !START_FRAME!"
)

call .\.venv\Scripts\activate.bat

set "V=%DATA_DIR%\witcher.mp4"
if not exist "%V%" (
    echo Missing %V%
    exit /b 1
)
set "EXPORT_DIR=%DATA_DIR%\witcher_results"
if not "%START_MINUTE%"=="0" set "EXPORT_DIR=!EXPORT_DIR!_from_min%START_MINUTE%"
if not "%OUT_OVERRIDE%"=="" set "EXPORT_DIR=%OUT_OVERRIDE%"
echo.
echo Extracting depth from witcher.mp4  (fps=%FPS%, first %MAX_MINUTES% min, start_minute=%START_MINUTE%)  -^>  !EXPORT_DIR!
echo.
if exist "!EXPORT_DIR!\poses.json" (
    echo   SKIP -- already done ^(poses.json exists^)
) else (
    python demo.py --model_path "%MODEL_PATH%" --video_path "%V%" --fps %FPS% !FIRST_K_FLAG! !START_FRAME_FLAG! --use_sdpa --export_results "!EXPORT_DIR!" --kv_cache_sliding_window %KV_WINDOW% --camera_num_iterations %CAM_ITERS%
    if errorlevel 1 (
        echo   FAILED: witcher.mp4
    ) else (
        python C:\workspace\world\lingbot-map\self_test_reprojection.py "!EXPORT_DIR!" --frame 0
    )
)
echo   Results: !EXPORT_DIR!

echo.
echo Done. Depth + poses in !EXPORT_DIR!
endlocal
