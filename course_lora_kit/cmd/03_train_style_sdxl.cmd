@echo off
setlocal
rem Phase 3 -- train a style LoRA on SDXL base 1.0.
rem Uses OneTrainer's shipped SDXL LoRA preset (rank 16 / alpha 1.0, LR 3e-4)
rem plus the course overlay: epochs sized for the measured ~40-60 views-per-image
rem window, intermediate saves every 100 steps, warmup pinned explicitly,
rem Min-SNR (gamma 5) + offset noise 0.03, and train_dtype pinned to bf16
rem (unpinned, the preset chain trains fp16). Run 00b_print_recipe.cmd to see
rem every dial after the preset/overlay merge.
rem Extra args are forwarded (e.g. --config-value epochs=30).

call "%~dp0_env.cmd" || (pause & exit /b 1)

set "CONCEPTS=%REPO_ROOT%\training_concepts\style_concepts.json"
if not exist "%CONCEPTS%" (
    echo [course_lora_kit] No concepts file at:
    echo     %CONCEPTS%
    echo.
    echo Run course_lora_kit\cmd\00_verify_setup.cmd first: it asks for your LORA_ROOT
    echo once and writes this file pointing at %%LORA_ROOT%%\dataset\trigger and \notrigger.
    echo.
    echo Or create it by hand from the shipped template:
    echo   1. copy course_lora_kit\configs\concepts_contrastive_template_style.json ^
to training_concepts\style_concepts.json
    echo   2. edit the two "path" fields: concept 1 ^(STANDARD^) points at your
    echo      captioned dataset ^(trigger word in every caption^); concept 2
    echo      ^(PRIOR_PREDICTION^) points at a COPY of the same images whose
    echo      captions never mention the trigger.
    echo See course_lora_kit\configs\README.md for why both concepts exist.
    pause
    exit /b 1
)

pushd "%REPO_ROOT%"
"%VENV_PY%" scripts\train.py --preset-path "training_presets\SDXL\#sdxl 1.0 LoRA.json" --config-path course_lora_kit\configs\style_sdxl.json %*
set "EC=%errorlevel%"
popd

if "%EC%"=="0" (
    echo.
    echo [course_lora_kit] Done. Final LoRA: models\lora_style_sdxl.safetensors
    echo Intermediate checkpoints: workspace\style_sdxl\save\
    echo Next: 04_checkpoint_screen.cmd workspace\style_sdxl\save
)
pause
exit /b %EC%
