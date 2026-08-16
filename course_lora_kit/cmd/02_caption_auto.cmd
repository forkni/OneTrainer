@echo off
setlocal
rem Phase 2 -- auto-caption a training folder with OneTrainer's own captioner.
rem Wraps scripts\generate_captions.py (writes a .txt next to every image).
rem
rem Usage:
rem   02_caption_auto.cmd <image_folder> [BLIP ^| BLIP2 ^| WD14_VIT_2]
rem
rem Default model: WD14_VIT_2 -- booru-style tags, the right choice for anime and
rem illustration sources. Pass BLIP for natural-language captions (photo sources).

if "%~1"=="" (
    echo Usage: 02_caption_auto.cmd ^<image_folder^> [BLIP ^| BLIP2 ^| WD14_VIT_2]
    exit /b 1
)

call "%~dp0_env.cmd" || exit /b 1

set "MODEL=%~2"
if not defined MODEL set "MODEL=WD14_VIT_2"

echo [course_lora_kit] Captioning "%~1" with %MODEL% ...
echo.
echo   *** Do NOT put your trigger word in a caption prefix. ***
echo   A trigger present in 100%% of captions cannot learn to gate -- the LoRA
echo   will apply on every prompt, trigger or not. The trigger goes into the
echo   captions of the STANDARD concept only, with a trigger-free contrastive
echo   concept alongside it -- see the lora-training-pipeline skill,
echo   references\contrastive-concept-gating.md, and phase 2 of the README.
echo   (If you do use --caption-prefix for something else, end it with a comma
echo   and a space -- the captioners concatenate it with no separator.)
echo.

"%VENV_PY%" "%REPO_ROOT%\scripts\generate_captions.py" --model %MODEL% --sample-dir "%~1" --mode fill --include-subdirectories
if errorlevel 1 exit /b %errorlevel%

echo.
echo [course_lora_kit] Done. Now do the by-hand pass in a text editor -- it is
echo not optional; it's where half the quality comes from:
echo   - style LoRA:     lead every caption with your trigger word
echo   - character LoRA: caption the VARIABLE traits (outfit, pose, expression),
echo     DELETE the permanent ones (hair colour, eye colour) -- what you don't
echo     caption is what binds to the trigger
echo   - make a copy of the folder with trigger-free captions for the
echo     contrastive concept (phase 3 checks for it)
exit /b 0
