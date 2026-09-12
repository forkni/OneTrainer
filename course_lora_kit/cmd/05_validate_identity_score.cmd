@echo off
setlocal
rem Phase 5 (character LoRAs) -- same identity metric as
rem 05_validate_identity_consistency.cmd, but for renders you already have from
rem anywhere else (ComfyUI output, a StreamDiffusionTD capture). No diffusion
rem pipeline, no GPU required -- it embeds and scores existing files.
rem Wraps course_lora_kit\scripts\test_lora_identity_score.py.
rem
rem Terminal usage:
rem   05_validate_identity_score.cmd --render LABEL=<file^|folder^|glob> ^
rem       --reference-dir <held_out_images> [flags]
rem   Repeatable: --render A2="out\A2_*.png" --render A4="out\A4_*.png"
rem
rem Scores are directly comparable to the consistency script's "identity" column
rem as long as --reference-dir is the same folder.
rem
rem Double-clicked with no arguments, the script asks for one render set and the
rem reference folder. To score several sets in one run, use the terminal form.
rem
rem Relative --output-dir / --json paths resolve against the directory you run
rem this from (the wrapper does not cd), not against the script's folder.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_identity_score.py" %*
goto :finish

:interactive
echo [course_lora_kit] Identity scoring of existing renders ^(no GPU needed^).
echo.
echo The reference folder is the 8-12 images you HELD BACK from training.
echo Use the same reference folder as the consistency script to keep the
echo identity numbers comparable.
echo.
echo Tip: right-click a file or folder in Explorer, "Copy as path", then
echo right-click in this window to paste it. Globs work too, e.g. out\A2_*.png
echo.

:ask_render
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "RENDER="
set /p "RENDER=Renders to score (file, folder, or glob): "
if not defined RENDER goto :ask_render
set "RENDER=%RENDER:"=%"

set "LABEL=myrenders"
set /p "LABEL=Short label for this render set [myrenders]: "
set "LABEL=%LABEL:"=%"

:ask_refdir
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "REFDIR="
set /p "REFDIR=Held-out reference image folder: "
if not defined REFDIR goto :ask_refdir
set "REFDIR=%REFDIR:"=%"
if not exist "%REFDIR%" (
    echo   Not found: "%REFDIR%" -- try again.
    goto :ask_refdir
)
echo.

echo Output folder: %LORA_ROOT%\outputs\identity_score
echo.
"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_identity_score.py" --render "%LABEL%=%RENDER%" --reference-dir "%REFDIR%" --output-dir "%LORA_ROOT%\outputs\identity_score"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
