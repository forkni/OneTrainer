@echo off
setlocal
rem Phase 1 -- pre-flight screen of a training image folder, before captioning.
rem Wraps course_lora_kit\scripts\dataset_hygiene_profiler.py.
rem
rem Usage:
rem   01_dataset_hygiene.cmd <image_folder> [extra profiler flags]
rem
rem Defaults passed for you (your own flags override them, last flag wins):
rem   --recursive        also walk subfolders and check subfolder balance
rem   --min-side 1024    flag images under the SDXL floor (pass --min-side 512 for SD1.5)

if "%~1"=="" (
    echo Usage: 01_dataset_hygiene.cmd ^<image_folder^> [extra profiler flags]
    echo.
    echo Screens a training folder for outliers ^(tone, sharpness, saturation^),
    echo undersized images, letterbox/pillarbox borders, and subfolder imbalance.
    echo See course_lora_kit\README.md, phase 1.
    pause
    exit /b 1
)

call "%~dp0_env.cmd" || (pause & exit /b 1)

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\dataset_hygiene_profiler.py" --recursive --min-side 1024 %*
set "EC=%errorlevel%"
pause
exit /b %EC%
