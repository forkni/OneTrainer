@echo off
setlocal
rem Phase 3 -- train a character LoRA on SDXL base 1.0 with the course's measured
rem recipe (the Round 1 "A2" winner): rank 16 / alpha 16, LR 1e-4 AdamW, 30-step
rem warmup, MIN_SNR_GAMMA 5.0, offset noise 0.03, bf16, attn-mlp, save every 100 steps.
rem
rem Layers OneTrainer's shipped SDXL LoRA preset under the course overlay -- the
rem overlay only overrides what the recipe changes. Extra args are forwarded, so
rem one-off tweaks work without editing the config:
rem   03_train_character_sdxl.cmd --config-value epochs=30

call "%~dp0_env.cmd" || (pause & exit /b 1)

set "CONCEPTS=%REPO_ROOT%\training_concepts\character_concepts.json"
if not exist "%CONCEPTS%" (
    echo [course_lora_kit] No concepts file at:
    echo     %CONCEPTS%
    echo.
    echo Create it from the shipped template:
    echo   1. copy course_lora_kit\configs\concepts_contrastive_template.json ^
to training_concepts\character_concepts.json
    echo   2. edit the two "path" fields: concept 1 ^(STANDARD^) points at your
    echo      captioned dataset ^(trigger word in every caption^); concept 2
    echo      ^(PRIOR_PREDICTION^) points at a COPY of the same images whose
    echo      captions never mention the trigger.
    echo See course_lora_kit\configs\README.md for why both concepts exist.
    pause
    exit /b 1
)

pushd "%REPO_ROOT%"
"%VENV_PY%" scripts\train.py --preset-path "training_presets\SDXL\#sdxl 1.0 LoRA.json" --config-path course_lora_kit\configs\character_sdxl.json %*
set "EC=%errorlevel%"
popd

if "%EC%"=="0" (
    echo.
    echo [course_lora_kit] Done. Final LoRA: models\lora_character_sdxl.safetensors
    echo Intermediate checkpoints: workspace\character_sdxl\save\
    echo Next: 04_checkpoint_screen.cmd workspace\character_sdxl\save
)
pause
exit /b %EC%
