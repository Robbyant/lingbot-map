@echo off
setlocal enabledelayedexpansion

REM Download a YouTube video to C:\workspace\data\youtube\<name>.mp4 via yt-dlp
REM (run through uvx, so nothing needs to be pre-installed).
REM
REM Usage:
REM   download_youtube.bat <url>                 -> saves as <video_id>.mp4
REM   download_youtube.bat <url> cyberpunk       -> saves as cyberpunk.mp4
REM   download_youtube.bat <url> cyberpunk 1080  -> cap height at 1080p

REM NOTE: cmd.exe splits %1/%2/%3 on space, comma, semicolon, AND "=" -- YouTube
REM URLs contain "=" (e.g. watch?v=XXXX), which silently shifts NAME/MAXH into the
REM wrong slots if the URL isn't quoted at the call site. Parse the raw tail via %*
REM instead (FOR /F's default delimiters are space/tab only, so "=" survives intact),
REM so this works whether or not the caller quotes the URL.
set "ARGS=%*"
if "%ARGS%"=="" (
    echo Usage: download_youtube.bat ^<youtube_url^> [name] [max_height]
    exit /b 1
)
for /f "tokens=1*" %%A in ("%ARGS%") do (
    set "URL=%%~A"
    set "REST=%%B"
)
set "NAME="
set "MAXH="
if defined REST (
    for /f "tokens=1,2" %%A in ("%REST%") do (
        set "NAME=%%~A"
        set "MAXH=%%~B"
    )
)

set "OUT_DIR=C:\workspace\data\youtube"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

if "%NAME%"=="" (
    set "OUT_TMPL=%OUT_DIR%\%%(id)s.%%(ext)s"
) else (
    set "OUT_TMPL=%OUT_DIR%\%NAME%.%%(ext)s"
)

REM Prefer mp4 video+audio muxed; optional height cap keeps the file smaller.
if "%MAXH%"=="" (
    set "FMT=bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
) else (
    set "FMT=bestvideo[ext=mp4][height<=%MAXH%]+bestaudio[ext=m4a]/best[ext=mp4][height<=%MAXH%]/best"
)

echo Downloading %URL%
echo   -^> %OUT_DIR%\%NAME%.mp4
echo.
uvx yt-dlp -f "%FMT%" --merge-output-format mp4 -o "%OUT_TMPL%" "%URL%"

echo.
echo Done. Saved under %OUT_DIR%
endlocal
