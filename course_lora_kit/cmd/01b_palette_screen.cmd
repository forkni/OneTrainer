@echo off
setlocal
rem Phase 1b -- palette screen of a training image folder: mean CIELAB L*/a*/b* per
rem image, z-flags against the folder's own mean, a hue read per image, per-subfolder
rem means, and a comparison line when you pass several folders (kept set vs dropped set).
rem Wraps course_lora_kit\scripts\palette_ab_stats.py. No GPU, no base model.
rem
rem Terminal usage:
rem   01b_palette_screen.cmd <image_folder> [more folders] [extra flags]
rem
rem Defaults passed for you (your own flags override them, last flag wins):
rem   --recursive        also walk subfolders and print per-subfolder means
rem
rem Extra flags: --z-threshold 2.0   --sort a^|b^|chroma^|name   --top N   --json out.json
rem
rem Examples:
rem   01b_palette_screen.cmd D:\sets\train
rem   01b_palette_screen.cmd D:\sets\train D:\sets\dropped      (second folder vs the first)
rem
rem The numbers describe the DATASET; they do not predict what a trained LoRA renders.
rem Double-clicked with no arguments, the script asks for the folder instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\palette_ab_stats.py" --recursive %*
goto :finish

:interactive
echo [course_lora_kit] Palette screen.
echo Mean a* ^(red/green^) and b* ^(yellow/blue^) per image, flagged relative to the
echo folder's own mean, with a hue read and per-subfolder means.
echo.
echo Tip: in Explorer, right-click your folder and choose "Copy as path", then
echo right-click in this window to paste it. To compare two folders, run this
echo script from a terminal with both folders as arguments.
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
"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\palette_ab_stats.py" --recursive "%FOLDER%"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
