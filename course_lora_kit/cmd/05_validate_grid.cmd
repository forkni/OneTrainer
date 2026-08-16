@echo off
setlocal
rem Phase 5 -- the cheapest render check: one 2x2 grid (LoRA on/off x trigger
rem present/absent) on SDXL base at normal step counts. Answers three questions
rem in one pass: did training take, does the trigger gate, does the style leak.
rem Wraps course_lora_kit\scripts\test_lora_grid_sdxl_base.py.
rem
rem Usage:
rem   05_validate_grid.cmd --lora <path\to\lora.safetensors> --trigger <word> [flags]
rem
rem Copy the trigger word out of an actual caption file -- do not retype it; a
rem one-letter mismatch silently invalidates the whole batch.

if "%~1"=="" (
    echo Usage: 05_validate_grid.cmd --lora ^<lora.safetensors^> --trigger ^<word^> [flags]
    echo Optional: --prompts-file battery.json  ^(runs the 2x2 once per prompt^)
    exit /b 1
)

call "%~dp0_env.cmd" || exit /b 1

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_grid_sdxl_base.py" %*
exit /b %errorlevel%
