@echo off
setlocal
rem Phase 0 -- verify the OneTrainer install before touching a dataset:
rem venv present, torch imports, CUDA visible, and the kit's own files in place.

call "%~dp0_env.cmd" || (pause & exit /b 1)

echo [course_lora_kit] repo root: %REPO_ROOT%
echo [course_lora_kit] python:    %VENV_PY%
echo [course_lora_kit] HF_HOME:   %HF_HOME%
echo.

"%VENV_PY%" -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(none)')"
if errorlevel 1 (
    echo.
    echo [course_lora_kit] torch failed to import or crashed -- the venv is broken
    echo or half-installed. Re-run install.bat and check its output.
    pause
    exit /b 1
)

echo.
set "MISSING="
for %%F in (
    "%REPO_ROOT%\scripts\train.py"
    "%REPO_ROOT%\scripts\generate_captions.py"
    "%REPO_ROOT%\training_presets\SDXL\#sdxl 1.0 LoRA.json"
    "%REPO_ROOT%\course_lora_kit\configs\character_sdxl.json"
    "%REPO_ROOT%\course_lora_kit\scripts\checkpoint_norm_analyzer.py"
) do if not exist %%F (
    echo [course_lora_kit] MISSING: %%F
    set "MISSING=1"
)
if defined MISSING (
    echo [course_lora_kit] The checkout is incomplete -- are you on the
    echo course/lora-pipeline branch of the course fork?
    pause
    exit /b 1
)

echo [course_lora_kit] Setup looks good. Next: 01_dataset_hygiene.cmd ^<your_image_folder^>
pause
exit /b 0
