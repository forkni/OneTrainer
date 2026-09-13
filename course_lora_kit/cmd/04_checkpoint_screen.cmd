@echo off
setlocal
rem Phase 4 -- rendering-free screen of a checkpoint sweep: no GPU, no base model.
rem Wraps course_lora_kit\scripts\checkpoint_norm_analyzer.py.
rem
rem Terminal usage:
rem   04_checkpoint_screen.cmd <checkpoint_dir_or_file> [more dirs] [--json out.json]
rem
rem Each directory is analyzed as its own group, so you can compare several
rem training arms in one invocation (terminal only). Read the printed flags:
rem KNEE (growth jumps against the run's own median), CONVENTION (the file's
rem rank/alpha is not the recipe's; pass --recipe <merged recipe json from
rem 00b_print_recipe.cmd --json>, default rank 16 / alpha 1) and HOT-START (the
rem first save already sits 4x above the recipe's reference first save and the
rem run never speeds up after). The Round 2 absolute bands print only with
rem --bands: they are local to that dataset and step count.
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
