@echo off
setlocal
rem Phase 4 -- rendering-free screen of a checkpoint sweep: no GPU, no base model.
rem Wraps course_lora_kit\scripts\checkpoint_norm_analyzer.py.
rem
rem Usage:
rem   04_checkpoint_screen.cmd <checkpoint_dir_or_file> [more dirs] [--json out.json]
rem
rem Each directory is analyzed as its own group, so you can compare several
rem training arms in one invocation. Read the printed "knee" flag as the
rem config-independent signal; the absolute reference bands only apply at the
rem scale they were calibrated on (the script prints the caveat).

if "%~1"=="" (
    echo Usage: 04_checkpoint_screen.cmd ^<checkpoint_dir_or_file^> [more dirs] [--json out.json]
    echo Example: 04_checkpoint_screen.cmd workspace\character_sdxl\save
    pause
    exit /b 1
)

call "%~dp0_env.cmd" || (pause & exit /b 1)

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\checkpoint_norm_analyzer.py" %*
set "EC=%errorlevel%"
pause
exit /b %EC%
