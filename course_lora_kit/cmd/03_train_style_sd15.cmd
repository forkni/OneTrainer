@echo off
setlocal
rem Phase 3 -- train a style LoRA on SD1.5 (the light path for modest GPUs;
rem pairs with openjourney-v4 and other SD1.5-family real-time models).
rem Uses OneTrainer's shipped SD1.5 LoRA preset plus the course overlay.
rem Extra args are forwarded (e.g. --config-value epochs=30).

call "%~dp0_env.cmd" || (pause & exit /b 1)

set "CONCEPTS=%REPO_ROOT%\training_concepts\style_concepts.json"
if not exist "%CONCEPTS%" (
    echo [course_lora_kit] No concepts file at:
    echo     %CONCEPTS%
    echo.
    echo Create it from the shipped template:
    echo   1. copy course_lora_kit\configs\concepts_contrastive_template.json ^
to training_concepts\style_concepts.json
    echo   2. edit the two "path" fields: concept 1 ^(STANDARD^) points at your
    echo      captioned dataset ^(trigger word in every caption^); concept 2
    echo      ^(PRIOR_PREDICTION^) points at a COPY of the same images whose
    echo      captions never mention the trigger.
    echo See course_lora_kit\configs\README.md for why both concepts exist.
    echo Note: SD1.5 trains at 512px -- screen your dataset with --min-side 512.
    pause
    exit /b 1
)

pushd "%REPO_ROOT%"
"%VENV_PY%" scripts\train.py --preset-path "training_presets\SD1.5\#sd 1.5 LoRA.json" --config-path course_lora_kit\configs\style_sd15.json %*
set "EC=%errorlevel%"
popd

if "%EC%"=="0" (
    echo.
    echo [course_lora_kit] Done. Final LoRA: models\lora_style_sd15.safetensors
    echo Intermediate checkpoints: workspace\style_sd15\save\
    echo Next: 04_checkpoint_screen.cmd workspace\style_sd15\save
)
pause
exit /b %EC%
