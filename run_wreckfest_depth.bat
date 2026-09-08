@echo off
setlocal enabledelayedexpansion

REM Same as run_youtube_depth.bat, but limited to wreckfest.mp4 specifically. Writes
REM poses.json + depth PNGs to wreckfest_results next to the video.
REM
REM Arg 1: optional data dir (default C:\workspace\data\youtube)
REM Arg 2: optional --fps sampling value (default 5)
REM Arg 3: optional minutes limit -- only process the first N minutes of the clip
REM        (default 1; pass 0 to disable and process the full clip)
REM Arg 4: optional output dir override (default <data dir>\wreckfest_results)
REM Arg 5: optional --start_frame value -- skip this many source frames before sampling
REM        (default 0; e.g. 1000 to skip past an intro)
REM Arg 6: optional start minute -- skip into the source clip before sampling, in minutes
REM        (default 4; overrides arg 5 if nonzero; pass 0 for the true start of the clip).
REM        wreckfest.mp4 is native ~59.94fps (60000/1001), so this converts to
REM        --start_frame = minute*3600000/1001.

cd /d C:\workspace\world\lingbot-map

set "DATA_DIR=%~1"
if "%DATA_DIR%"=="" set "DATA_DIR=C:\workspace\data\youtube"
set "FPS=%~2"
if not defined FPS set "FPS=5"
set "MAX_MINUTES=%~3"
if not defined MAX_MINUTES set "MAX_MINUTES=1"
set "OUT_OVERRIDE=%~4"
REM A literal "" passed through some callers (e.g. cmd /c from a non-cmd shell) can
REM arrive as the two-character string ""  rather than truly empty -- treat that the
REM same as "no override" instead of using it as a garbage path.
if "%OUT_OVERRIDE%"=="""" set "OUT_OVERRIDE="
set "START_FRAME=%~5"
if not defined START_FRAME set "START_FRAME=0"
set "START_MINUTE=%~6"
if not defined START_MINUTE set "START_MINUTE=4"
if not "%START_MINUTE%"=="0" set /a "START_FRAME=%START_MINUTE% * 3600000 / 1001"

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

call .\.venv\Scripts\activate.bat

set "V=%DATA_DIR%\wreckfest.mp4"
if not exist "%V%" (
    echo Missing %V%
    exit /b 1
)
set "EXPORT_DIR=%DATA_DIR%\wreckfest_results"
if not "%START_MINUTE%"=="4" set "EXPORT_DIR=!EXPORT_DIR!_from_min%START_MINUTE%"
if not "%OUT_OVERRIDE%"=="" set "EXPORT_DIR=%OUT_OVERRIDE%"
echo.
echo Extracting depth from wreckfest.mp4  (fps=%FPS%, first %MAX_MINUTES% min, start_frame=%START_FRAME%)  -^>  !EXPORT_DIR!
echo.
if exist "!EXPORT_DIR!\poses.json" (
    echo   SKIP -- already done ^(poses.json exists^)
) else (
    python demo.py --model_path "%MODEL_PATH%" --video_path "%V%" --fps %FPS% !FIRST_K_FLAG! --start_frame %START_FRAME% --use_sdpa --export_results "!EXPORT_DIR!" --kv_cache_sliding_window %KV_WINDOW% --camera_num_iterations %CAM_ITERS%
    if errorlevel 1 (
        echo   FAILED: wreckfest.mp4
    ) else (
        python C:\workspace\world\lingbot-map\self_test_reprojection.py "!EXPORT_DIR!" --frame 0
    )
)
echo   Results: !EXPORT_DIR!

echo.
echo Done. Depth + poses in !EXPORT_DIR!
endlocal
