@echo off
setlocal enabledelayedexpansion

REM Full wreckfest test in one shot: extract depth+poses from wreckfest.mp4 (skipping the
REM intro/menu via --start_frame), run the reprojection self-test, and render the reprojection
REM overlays. Uses --no_viser so it exports and exits (no blocking viewer). All memory-safe
REM defaults (kv window 32 + camera 1 iter) come from demo.py.
REM
REM Defaults: start at minute 4 (past the main menu / car-select screens), fps 5, 300 frames.
REM wreckfest.mp4 is native ~59.94fps (60000/1001), so minute 4 = 14385 source frames.
REM   Arg 1: start_frame (source frames to skip). Default 14385 (~minute 4).
REM   Arg 2: fps sampling. Default 5.
REM   Arg 3: first_k (frame budget). Default 300.

cd /d C:\workspace\world\lingbot-map

set "MODEL_PATH=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-map\snapshots\204754b72bb24f561f8d7e7e1e4e4cd9e809adf9\lingbot-map.pt"
set "VIDEO=C:\workspace\data\youtube\wreckfest.mp4"
set "EXPORT_DIR=C:\workspace\data\youtube\wreckfest_results"

set "START_FRAME=%~1"
if not defined START_FRAME set "START_FRAME=14385"
set "FPS=%~2"
if not defined FPS set "FPS=5"
set "FIRST_K=%~3"
if not defined FIRST_K set "FIRST_K=300"

call .\.venv\Scripts\activate.bat

echo.
echo Extracting wreckfest depth: start_frame=%START_FRAME% fps=%FPS% first_k=%FIRST_K%
echo   -^> %EXPORT_DIR%
echo.
python demo.py --model_path "%MODEL_PATH%" --video_path "%VIDEO%" --fps %FPS% --first_k %FIRST_K% --start_frame %START_FRAME% --use_sdpa --export_results "%EXPORT_DIR%" --comparison_stride 1 --no_viser
if errorlevel 1 ( echo demo.py FAILED & exit /b 1 )

echo.
echo Running reprojection self-test...
python self_test_reprojection.py "%EXPORT_DIR%" --frame 0

echo.
echo Rendering reprojection overlays + 10-frame reprojection row...
python show_reprojection.py "%EXPORT_DIR%" --source 0
python show_reprojection.py "%EXPORT_DIR%" --source 0 --target 20
python draw_reproj_row.py "%EXPORT_DIR%" --source 0 --count 10

echo.
echo Done. Results + overlays in %EXPORT_DIR%
endlocal
