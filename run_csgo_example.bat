@echo off
setlocal enabledelayedexpansion

REM Runs lingbot-map SLAM/reconstruction on a CS:GO gameplay clip extracted from
REM C:\csgo_data (frame_i_x RGB frames inside the dataset's HDF5 files), as a test
REM of running lingbot-map on video-game footage instead of a real-world capture.
REM
REM Requires: .venv set up (torch cu128 + pip install -e . + pip install -e ".[vis]"),
REM lingbot-map.pt downloaded (see download_model.bat).

set "HDF5_PATH=C:\csgo_data\hdf5_dm_july2021_1.hdf5"
set "IMAGE_FOLDER=C:\csgo_data\lingbot_map_example\csgo_dm_1"
set "MODEL_PATH=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-map\snapshots\204754b72bb24f561f8d7e7e1e4e4cd9e809adf9\lingbot-map.pt"
if not "%~1"=="" set "MODEL_PATH=%~1"
set "EXPORT_DIR=C:\csgo_data\lingbot_map_example\csgo_dm_1_results"

REM --- 32GB-GPU memory defaults (500-frame streaming OOM'd at 64-window/4-iter) ---
REM expandable_segments cuts allocator fragmentation; the two flags lower the forward
REM peak: sliding KV window 64->32 (half the resident cache) and camera head 4->1 iter
REM (drops 3 refinement passes). Override any of them by exporting the var / passing
REM the flag as %2..%5 before calling this bat.
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "KV_WINDOW=32"
set "CAM_ITERS=1"

call .\.venv\Scripts\activate.bat

if not exist "%IMAGE_FOLDER%" (
    echo Extracting CS:GO frames from %HDF5_PATH% ...
    python extract_csgo_frames.py --hdf5_path "%HDF5_PATH%" --output_folder "%IMAGE_FOLDER%" --num_frames 500
)

echo.
echo Running lingbot-map on CS:GO gameplay frames: %IMAGE_FOLDER%
echo.

REM Optional %1 = path to lingbot-map.pt override (defaults to the snapshot downloaded
REM via download_model.bat -- see its printed "Downloaded to:" path if it moves).
REM --use_sdpa: FlashInfer isn't installed (setup.bat skips it); SDPA is PyTorch's
REM native attention fallback -- slower but works with no extra install.
REM --export_results: always saves poses.json + depth PNGs to %EXPORT_DIR%, since
REM demo.py's viser viewer is otherwise in-memory only and gives you nothing on disk.
python demo.py --model_path "%MODEL_PATH%" --image_folder "%IMAGE_FOLDER%" --use_sdpa --export_results "%EXPORT_DIR%" --kv_cache_sliding_window %KV_WINDOW% --camera_num_iterations %CAM_ITERS% %2 %3 %4 %5

echo.
echo Results saved to: %EXPORT_DIR%
