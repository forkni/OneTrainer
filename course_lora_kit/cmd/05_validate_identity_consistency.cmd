@echo off
setlocal
rem Phase 5 (character LoRAs) -- the measurement pixel-diff can't make: is it the
rem SAME subject across prompts and seeds. Renders a prompt x seed grid per arm
rem and scores identity (DINOv2 cosine vs a held-out reference set), cross-seed
rem consistency, flexibility, and a CLIP prompt-adherence score.
rem Wraps course_lora_kit\scripts\test_lora_identity_consistency.py.
rem
rem Terminal usage:
rem   05_validate_identity_consistency.cmd --lora LABEL=<path> [--lora LABEL2=<path>] ^
rem       --reference-dir <held_out_images> [flags]
rem   Repeatable: --lora A2=a.safetensors --lora A4=b.safetensors  (LABEL=none for a
rem   no-LoRA baseline); per-label scale: --lora-scale LABEL=0.0
rem
rem Double-clicked with no arguments, the script asks for one LoRA and the
rem reference folder. To compare several LoRAs in one run, use the terminal form.
rem
rem Relative --output-dir / --json paths resolve against the directory you run
rem this from (the wrapper does not cd), not against the script's folder.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_identity_consistency.py" %*
goto :finish

:interactive
echo [course_lora_kit] Identity / consistency scoring ^(character LoRAs^).
echo.
echo The reference folder is the 8-12 images you HELD BACK from training --
echo images the LoRA has never seen. Scoring against training images measures
echo memorization, not identity.
echo To compare several LoRAs in one run, use the terminal form instead
echo ^(see the usage lines at the top of this script^).
echo.
echo Tip: right-click a file or folder in Explorer, "Copy as path", then
echo right-click in this window to paste it.
echo.

:ask_lora
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "LORA="
set /p "LORA=Path to your LoRA .safetensors: "
if not defined LORA goto :ask_lora
set "LORA=%LORA:"=%"
if not exist "%LORA%" (
    echo   Not found: "%LORA%" -- try again.
    goto :ask_lora
)

set "LABEL=mylora"
set /p "LABEL=Short label for this LoRA [mylora]: "
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

echo Output folder: %LORA_ROOT%\outputs\identity_consistency
echo.
"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_identity_consistency.py" --lora "%LABEL%=%LORA%" --reference-dir "%REFDIR%" --output-dir "%LORA_ROOT%\outputs\identity_consistency"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
