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
rem   --min-side 1024    flag images under the SDXL floor (512 for SD1.5)
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
choice /c 12 /m "Model family: [1] SDXL (min side 1024)  [2] SD1.5 (min side 512)"
if errorlevel 2 (set "MINSIDE=512") else (set "MINSIDE=1024")
echo.

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\dataset_hygiene_profiler.py" --recursive --min-side %MINSIDE% "%FOLDER%"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
