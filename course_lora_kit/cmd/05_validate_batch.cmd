@echo off
setlocal
rem Phase 5 -- seed batch for ONE checkpoint: the same prompt across several seeds,
rem LoRA on and LoRA off, with a per-render colour-drift number (mean CIELAB a*).
rem The grid and the sweep each render one seed; this is the check for the per-seed
rem magenta drift the course's Round 2 saw (S0: 2 clean / 1 drifting / 1 pink across
rem four seeds, judged by eye in ComfyUI; S0b: 4/4). A flag over --a-threshold means
rem look at that render, not reject it -- mean a* cannot tell a sunset from drift.
rem Wraps course_lora_kit\scripts\test_lora_seed_batch.py.
rem
rem Terminal usage:
rem   05_validate_batch.cmd --lora <file.safetensors> --trigger <word> [--seeds 4] [--a-threshold 20] [flags]
rem
rem Output goes to scripts\outputs\seed_batch\ by default. A relative --output-dir
rem resolves against the directory you run this from, not the script's folder.
rem
rem Double-clicked with no arguments, the script asks for everything instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_seed_batch.py" %*
goto :finish

:interactive
echo [course_lora_kit] Seed batch ^(rendered, LoRA on vs off, colour drift^).
echo.
echo Tip: right-click a file in Explorer, "Copy as path", then right-click in
echo this window to paste it.
echo.

:ask_lora
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "LORA="
set /p "LORA=LoRA .safetensors to test: "
if not defined LORA goto :ask_lora
set "LORA=%LORA:"=%"
if not exist "%LORA%" (
    echo   Not found: "%LORA%" -- try again.
    goto :ask_lora
)

:ask_trigger
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "TRIGGER="
set /p "TRIGGER=Trigger word (copy it out of a caption file -- do not retype it): "
if not defined TRIGGER goto :ask_trigger

set "SEEDS="
set /p "SEEDS=How many seeds [4]: "
if not defined SEEDS set "SEEDS=4"
echo.

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_seed_batch.py" --lora "%LORA%" --trigger "%TRIGGER%" --seeds %SEEDS%
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
