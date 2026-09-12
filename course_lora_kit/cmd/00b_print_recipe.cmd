@echo off
setlocal
rem Phase 0b -- print the recipe OneTrainer will actually train with, after the
rem same merge scripts\train.py does: defaults -> preset -> overlay -> --config-value.
rem Overlays only list the keys they change, so this is how you see what the
rem unlisted keys resolved to (train_dtype, scheduler, loss weighting, ...).
rem No GPU, no model load; runs in a second.
rem
rem Usage:
rem   00b_print_recipe.cmd                 (style SDXL recipe, the masterclass default)
rem   00b_print_recipe.cmd character       (character SDXL recipe)
rem   00b_print_recipe.cmd style --config-value epochs=30   (see an override applied)
rem Extra args after the recipe name are forwarded (--all, --json out.json, --config-value K=V).

call "%~dp0_env.cmd" || (pause & exit /b 1)

set "RECIPE=%~1"
if "%RECIPE%"=="" set "RECIPE=style"
if /i "%RECIPE%"=="style" (
    set "PRESET=training_presets\SDXL\#sdxl 1.0 LoRA.json"
    set "OVERLAY=course_lora_kit\configs\style_sdxl.json"
) else if /i "%RECIPE%"=="character" (
    set "PRESET=training_presets\SDXL\#sdxl 1.0 LoRA.json"
    set "OVERLAY=course_lora_kit\configs\character_sdxl.json"
) else (
    echo [course_lora_kit] Unknown recipe "%RECIPE%" -- use style or character.
    pause
    exit /b 1
)
if not "%~1"=="" shift

pushd "%REPO_ROOT%"
"%VENV_PY%" course_lora_kit\scripts\print_effective_config.py --preset-path "%PRESET%" --config-path "%OVERLAY%" %1 %2 %3 %4 %5 %6 %7 %8 %9
set "EC=%errorlevel%"
popd
rem Pause only when Explorer launched this file directly (same test _env.cmd uses),
rem so a terminal or a script that calls it gets the exit code back without a keypress.
set "CMDLINE=%cmdcmdline:"=%"
if not "%CMDLINE:00b_print_recipe.cmd=%"=="%CMDLINE%" pause
exit /b %EC%
