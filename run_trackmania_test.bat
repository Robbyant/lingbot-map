@echo off
setlocal enabledelayedexpansion

REM Full trackmania test in one shot: extract depth+poses from a left_right clip,
REM run the reprojection self-test, and render the reprojection overlays + a 10-frame
REM reprojection row. Exports every frame (--comparison_stride 1) so the row has 10
REM consecutive frames. Headless (--no_viser). Memory-safe defaults come from demo.py.
REM
REM   Arg 1: clip name (default 1 -> C:\workspace\data\left_right\left_right\1.mp4)
REM   Arg 2: fps sampling (default 10)

cd /d C:\workspace\world\lingbot-map

set "CLIP=%~1"
if not defined CLIP set "CLIP=1"
set "FPS=%~2"
if not defined FPS set "FPS=10"

set "MODEL_PATH=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-map\snapshots\204754b72bb24f561f8d7e7e1e4e4cd9e809adf9\lingbot-map.pt"
set "VIDEO=C:\workspace\data\left_right\left_right\%CLIP%.mp4"
set "EXPORT_DIR=C:\workspace\data\left_right\left_right\%CLIP%_results"

call .\.venv\Scripts\activate.bat

echo.
echo Extracting trackmania depth: clip=%CLIP% fps=%FPS%
echo   -^> %EXPORT_DIR%
echo.
python demo.py --model_path "%MODEL_PATH%" --video_path "%VIDEO%" --fps %FPS% --first_k 21 --use_sdpa --export_results "%EXPORT_DIR%" --comparison_stride 1 --no_viser
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
