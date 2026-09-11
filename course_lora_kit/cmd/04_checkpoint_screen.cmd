@echo off
setlocal
rem Phase 4 -- rendering-free screen of a checkpoint sweep: no GPU, no base model.
rem Wraps course_lora_kit\scripts\checkpoint_norm_analyzer.py.
rem
rem Terminal usage:
rem   04_checkpoint_screen.cmd <checkpoint_dir_or_file> [more dirs] [--json out.json]
rem
rem Each directory is analyzed as its own group, so you can compare several
rem training arms in one invocation (terminal only). Read the printed "knee"
rem flag as the config-independent signal; the absolute reference bands only
rem apply at the scale they were calibrated on (the script prints the caveat).
rem
rem Double-clicked with no arguments, the script asks for the folder instead.
rem
rem Relative --output-dir / --json paths resolve against the directory you run
rem this from (the wrapper does not cd), not against the script's folder.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\checkpoint_norm_analyzer.py" %*
goto :finish

:interactive
echo [course_lora_kit] Checkpoint screen ^(rendering-free, no GPU needed^).
echo.
echo Point it at your training run's save folder -- e.g.
echo   workspace\character_sdxl\save
echo Tip: right-click the folder in Explorer, "Copy as path", then right-click
echo here to paste. To compare several training arms at once, run this script
echo from a terminal with multiple folders instead.
echo.

:ask_dir
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "CKPTS="
set /p "CKPTS=Checkpoint folder (or a single .safetensors file): "
if not defined CKPTS goto :ask_dir
set "CKPTS=%CKPTS:"=%"
if not exist "%CKPTS%" (
    echo   Not found: "%CKPTS%" -- try again.
    goto :ask_dir
)
echo.

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\checkpoint_norm_analyzer.py" "%CKPTS%"
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
