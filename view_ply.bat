@echo off
setlocal enabledelayedexpansion

REM Opens a saved point_cloud.ply (from demo.py's --export_results) in a viser
REM browser viewer, same idea as demo.py's live viewer but for a file already on disk.
REM Usage: view_ply.bat <path\to\point_cloud.ply> [port]

set "PLY_PATH=%~1"
if "%PLY_PATH%"=="" (
    echo Usage: view_ply.bat ^<path\to\point_cloud.ply^> [port]
    exit /b 1
)
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"

cd /d C:\workspace\world\lingbot-map
call .\.venv\Scripts\activate.bat

echo.
echo Loading %PLY_PATH% ...
echo Viewer will be at http://localhost:%PORT%
echo.

python view_ply.py "%PLY_PATH%" --port %PORT%
