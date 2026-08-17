@echo off
setlocal
rem Phase 5 -- the full trigger-gating measurement: four cells per prompt
rem (LoRA on/off x trigger present/absent) with the token-insertion null
rem subtracted out, so the reported ratio measures your LoRA, not the tokenizer.
rem Wraps course_lora_kit\scripts\test_lora_gating_measure.py.
rem
rem Terminal usage:
rem   05_validate_gating.cmd --lora <path\to\lora.safetensors> --trigger <word> [flags]
rem   Optional: --prompts-file battery.json  (out-of-domain prompt battery)
rem
rem Re-run this at the checkpoint you actually ship -- gating measured at an
rem earlier checkpoint does not transfer.
rem
rem Double-clicked with no arguments, the script asks for everything instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_gating_measure.py" %*
goto :finish

:interactive
echo [course_lora_kit] Trigger-gating measurement.
echo Run this against the checkpoint you actually ship -- gating measured at an
echo earlier checkpoint does not transfer.
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

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_gating_measure.py" --lora "%LORA%" --trigger "%TRIGGER%"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
