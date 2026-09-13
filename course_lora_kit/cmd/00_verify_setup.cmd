@echo off
setlocal
rem Phase 0 -- verify the OneTrainer install before touching a dataset:
rem venv present, torch imports, CUDA visible, and the kit's own files in place.
rem Also the "set once" step: asks where your LoRA work should live (LORA_ROOT),
rem remembers it in course_lora_kit\local_paths.cmd, creates the dataset folders and
rem writes both concepts files pointing at them. Re-running never overwrites anything.

call "%~dp0_env.cmd" || (pause & exit /b 1)

set "LOCAL_PATHS=%REPO_ROOT%\course_lora_kit\local_paths.cmd"
if not exist "%LOCAL_PATHS%" call :ask_root

echo [course_lora_kit] repo root: %REPO_ROOT%
echo [course_lora_kit] python:    %VENV_PY%
echo [course_lora_kit] HF_HOME:   %HF_HOME%
echo [course_lora_kit] LORA_ROOT: %LORA_ROOT%   ^(set once in course_lora_kit\local_paths.cmd^)
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
    "%REPO_ROOT%\course_lora_kit\configs\style_sdxl.json"
    "%REPO_ROOT%\course_lora_kit\configs\concepts_contrastive_template_style.json"
    "%REPO_ROOT%\course_lora_kit\scripts\checkpoint_norm_analyzer.py"
    "%REPO_ROOT%\course_lora_kit\scripts\check_trigger_word.py"
    "%REPO_ROOT%\course_lora_kit\scripts\print_effective_config.py"
    "%REPO_ROOT%\course_lora_kit\scripts\init_concepts.py"
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

rem Your folders under LORA_ROOT and the two concepts files (kept if they already exist).
if not exist "%LORA_ROOT%\dataset\trigger" mkdir "%LORA_ROOT%\dataset\trigger"
if not exist "%LORA_ROOT%\dataset\notrigger" mkdir "%LORA_ROOT%\dataset\notrigger"
if not exist "%LORA_ROOT%\outputs" mkdir "%LORA_ROOT%\outputs"
pushd "%REPO_ROOT%"
"%VENV_PY%" course_lora_kit\scripts\init_concepts.py --root "%LORA_ROOT%" --track style
"%VENV_PY%" course_lora_kit\scripts\init_concepts.py --root "%LORA_ROOT%" --track character
popd
echo.
echo   Put your captioned images ^(trigger word in every caption^) in
echo     %LORA_ROOT%\dataset\trigger
echo   and the trigger-free copy of the same images in
echo     %LORA_ROOT%\dataset\notrigger
echo   ^(01_dataset_hygiene.cmd and 02_caption_auto.cmd help you build both.^)
echo   Dataset somewhere else? Edit the two "path" fields in training_concepts\*_concepts.json.
echo.
echo [course_lora_kit] Setup looks good. Next: 00b_print_recipe.cmd (see the merged recipe),
echo then 01_dataset_hygiene.cmd ^<your_image_folder^>
pause
exit /b 0

:ask_root
echo [course_lora_kit] First run: where should your LoRA work live?
echo   One folder for datasets, validation outputs and staged picks -- e.g. D:\my_lora
echo   Press Enter to keep it inside the checkout ^(%LORA_ROOT%^).
echo   You can change it later in course_lora_kit\local_paths.cmd.
set "ROOT_IN="
set /p "ROOT_IN=LORA_ROOT: "
if defined ROOT_IN set "ROOT_IN=%ROOT_IN:"=%"
if defined ROOT_IN set "LORA_ROOT=%ROOT_IN%"
> "%LOCAL_PATHS%" echo @echo off
>> "%LOCAL_PATHS%" echo rem Your machine-specific folders ^(gitignored^). Written by 00_verify_setup.cmd; edit freely.
>> "%LOCAL_PATHS%" echo if not defined LORA_ROOT set "LORA_ROOT=%LORA_ROOT%"
>> "%LOCAL_PATHS%" echo rem Optional: default destination for 06_stage_pick.cmd
>> "%LOCAL_PATHS%" echo rem if not defined COMFY_LORAS_DIR set "COMFY_LORAS_DIR=D:\ComfyUI\models\loras"
echo   Saved to %LOCAL_PATHS%
echo.
exit /b 0
