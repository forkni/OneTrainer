@echo off
rem _env.cmd -- shared bootstrap for every script in course_lora_kit\cmd.
rem Not meant to be run on its own: the other scripts `call` it.
rem
rem Sets:
rem   REPO_ROOT  -- the OneTrainer checkout root (resolved from this file's location)
rem   VENV_PY    -- OneTrainer's own venv python
rem   HF_HOME    -- only if not already set; keeps Hugging Face downloads in one
rem                 predictable, gitignored place instead of scattering them per-user.

set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

set "VENV_PY=%REPO_ROOT%\venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [course_lora_kit] OneTrainer's venv was not found at:
    echo     %VENV_PY%
    echo.
    echo Run install.bat in the repo root first, then re-run this script.
    exit /b 1
)

if not defined HF_HOME set "HF_HOME=%REPO_ROOT%\workspace\hf_cache"

exit /b 0
