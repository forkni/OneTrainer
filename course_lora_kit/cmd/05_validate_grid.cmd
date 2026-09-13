@echo off
setlocal
rem Phase 5 -- the cheapest render check: one 2x2 grid (LoRA on/off x trigger
rem present/absent) on SDXL base at normal step counts. Answers three questions
rem in one pass: did training take, does the trigger gate, does the style leak.
rem Wraps course_lora_kit\scripts\test_lora_grid_sdxl_base.py.
rem
rem Terminal usage:
rem   05_validate_grid.cmd --lora <path\to\lora.safetensors> --trigger <word> [flags]
rem   Optional: --prompts-file battery.json  (runs the 2x2 once per prompt)
rem
rem Double-clicked with no arguments, the script asks for everything instead.
rem
rem Relative --output-dir / --json paths resolve against the directory you run
rem this from (the wrapper does not cd), not against the script's folder.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_grid_sdxl_base.py" %*
goto :finish

:interactive
echo [course_lora_kit] 2x2 validation grid ^(LoRA on/off x trigger on/off^).
echo.
echo Tip: right-click the .safetensors file in Explorer, "Copy as path", then
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

:ask_trigger
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "TRIGGER="
set /p "TRIGGER=Trigger word (copy it out of a caption file -- do not retype it): "
if not defined TRIGGER goto :ask_trigger
echo.

echo Output folder: %LORA_ROOT%\outputs\grid
echo.
"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_grid_sdxl_base.py" --lora "%LORA%" --trigger "%TRIGGER%" --output-dir "%LORA_ROOT%\outputs\grid"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
