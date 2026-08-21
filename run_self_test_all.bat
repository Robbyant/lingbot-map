@echo off
setlocal enabledelayedexpansion

REM Run the reprojection self-consistency test on ALL available results dirs
REM (CS:GO, trackmania/left_right, cyberpunk). Skips any that haven't exported a
REM depth_raw/ + poses.json yet. Pure validation -- no GPU / no inference needed;
REM it reads each run's existing export. To (re)generate a run's export first, use
REM its own runner (run_csgo_example.bat / run_trackmania_capture.bat / run_youtube_depth.bat).

cd /d C:\workspace\world\lingbot-map
call .\.venv\Scripts\activate.bat

set "DIRS=C:\csgo_data\lingbot_map_example\selftest_results C:\workspace\data\left_right\left_right\1_results C:\workspace\data\youtube\cyberpunk_results"

for %%D in (%DIRS%) do (
    echo.
    echo ================================================================
    echo Self-test: %%D
    echo ================================================================
    if exist "%%D\poses.json" (
        if exist "%%D\depth_raw" (
            python self_test_reprojection.py "%%D" --frame 0
        ) else (
            echo   SKIP -- no depth_raw\ ^(run was exported without raw depth^)
        )
    ) else (
        echo   SKIP -- no poses.json yet ^(run not finished / not exported^)
    )
)

echo.
echo Done.
endlocal
