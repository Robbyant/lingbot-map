@echo off
setlocal enabledelayedexpansion

REM Small, fast validation run (21 CS:GO frames) purely to produce a depth_raw/
REM export, then runs self_test_reprojection.py to check that unprojecting a frame's
REM own depth and reprojecting it back onto itself lands near-exactly on the original
REM pixel grid -- validates the extrinsic/intrinsic convention used by
REM reproject_pointcloud.py before trusting any cross-frame reprojection result.

cd /d C:\workspace\world\lingbot-map

REM Arg 1: start_frame -- test 21 CS:GO frames starting this far into csgo_dm_1
REM        (default 0 = frames 0-20). e.g. run_self_test.bat 250  -> frames 250-270.
set "START=%~1"
if not defined START set "START=0"

set "MODEL_PATH=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-map\snapshots\204754b72bb24f561f8d7e7e1e4e4cd9e809adf9\lingbot-map.pt"
set "IMAGE_FOLDER=C:\csgo_data\lingbot_map_example\csgo_dm_1"
if "%START%"=="0" (
    set "EXPORT_DIR=C:\csgo_data\lingbot_map_example\selftest_results"
) else (
    set "EXPORT_DIR=C:\csgo_data\lingbot_map_example\selftest_results_%START%"
)

call .\.venv\Scripts\activate.bat

echo.
echo Running 21-frame validation inference (start_frame=%START%)...
echo.
python demo.py --model_path "%MODEL_PATH%" --image_folder "%IMAGE_FOLDER%" --start_frame %START% --first_k 21 --use_sdpa --export_results "%EXPORT_DIR%" --comparison_stride 1 --kv_cache_sliding_window 32 --camera_num_iterations 1 --no_viser

echo.
echo Running self-consistency reprojection test...
echo.
python self_test_reprojection.py "%EXPORT_DIR%" --frame 0

echo.
echo Rendering reprojection overlay + 10-frame reprojection row...
python show_reprojection.py "%EXPORT_DIR%" --source 0
python draw_reproj_row.py "%EXPORT_DIR%" --source 0 --count 10

echo.
echo Done. Results in %EXPORT_DIR%
