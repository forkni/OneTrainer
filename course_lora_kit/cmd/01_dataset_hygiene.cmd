@echo off
setlocal
rem Phase 1 -- pre-flight screen of a training image folder, before captioning.
rem Wraps course_lora_kit\scripts\dataset_hygiene_profiler.py.
rem
rem Terminal usage:
rem   01_dataset_hygiene.cmd <image_folder> [extra profiler flags]
rem
rem Defaults passed for you (your own flags override them, last flag wins):
rem   --recursive        also walk subfolders and check subfolder balance
rem   --min-side 1024    flag images under the SDXL floor
rem
rem The real floor is the aspect bucket, not the nominal resolution: at
rem resolution 1024 a 4:3 image trains in the 1152x896 bucket, so a uniform
rem 1280x960 set is a pure downscale and passes with --min-side 896. Pass
rem your own --min-side (it overrides the default) when the set is not 1:1.
rem
rem Double-clicked with no arguments, the script asks for everything instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\dataset_hygiene_profiler.py" --recursive --min-side 1024 %*
goto :finish

:interactive
echo [course_lora_kit] Dataset hygiene screen.
echo Screens a training folder for outliers ^(tone, sharpness, saturation^),
echo undersized images, letterbox/pillarbox borders, and subfolder imbalance.
echo.
echo Tip: in Explorer, right-click your folder and choose "Copy as path", then
echo right-click in this window to paste it.
echo.

:ask_folder
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "FOLDER="
set /p "FOLDER=Image folder to screen: "
if not defined FOLDER goto :ask_folder
set "FOLDER=%FOLDER:"=%"
if not exist "%FOLDER%" (
    echo   Not found: "%FOLDER%" -- try again.
    goto :ask_folder
)

echo.
echo Size floor: 1024 on the short side ^(SDXL^). For a non-square set, run from a
echo terminal and pass the bucket's short side instead, e.g. --min-side 896 for 4:3.
echo.

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\dataset_hygiene_profiler.py" --recursive --min-side 1024 "%FOLDER%"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
