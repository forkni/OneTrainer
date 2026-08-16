@echo off
setlocal
rem Phase 2 -- auto-caption a training folder with OneTrainer's own captioner.
rem Wraps scripts\generate_captions.py (writes a .txt next to every image).
rem
rem Terminal usage:
rem   02_caption_auto.cmd <image_folder> [BLIP ^| BLIP2 ^| WD14_VIT_2]
rem
rem Default model: WD14_VIT_2 -- booru-style tags, the right choice for anime and
rem illustration sources. BLIP gives natural-language captions (photo sources).
rem
rem Double-clicked with no arguments, the script asks for everything instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

set "FOLDER=%~1"
set "MODEL=%~2"
if not defined MODEL set "MODEL=WD14_VIT_2"
goto :run

:interactive
echo [course_lora_kit] Auto-captioning.
echo.
echo Tip: in Explorer, right-click your folder and choose "Copy as path", then
echo right-click in this window to paste it.
echo.

:ask_folder
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "FOLDER="
set /p "FOLDER=Image folder to caption: "
if not defined FOLDER goto :ask_folder
set "FOLDER=%FOLDER:"=%"
if not exist "%FOLDER%" (
    echo   Not found: "%FOLDER%" -- try again.
    goto :ask_folder
)

echo.
echo Caption model:
echo   [1] WD14_VIT_2 -- booru-style tags ^(anime / illustration sources^)
echo   [2] BLIP       -- natural-language captions ^(photo sources^)
echo   [3] BLIP2      -- larger BLIP; slower, sometimes richer captions
choice /c 123 /m "Pick a model"
if errorlevel 3 (set "MODEL=BLIP2") else if errorlevel 2 (set "MODEL=BLIP") else (set "MODEL=WD14_VIT_2")

:run
echo.
echo [course_lora_kit] Captioning "%FOLDER%" with %MODEL% ...
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

"%VENV_PY%" "%REPO_ROOT%\scripts\generate_captions.py" --model %MODEL% --sample-dir "%FOLDER%" --mode fill --include-subdirectories
set "EC=%errorlevel%"
if not "%EC%"=="0" (pause & exit /b %EC%)

echo.
echo [course_lora_kit] Done. Now do the by-hand pass in a text editor -- it is
echo not optional; it's where half the quality comes from:
echo   - style LoRA:     lead every caption with your trigger word
echo   - character LoRA: caption the VARIABLE traits (outfit, pose, expression),
echo     DELETE the permanent ones (hair colour, eye colour) -- what you don't
echo     caption is what binds to the trigger
echo   - make a copy of the folder with trigger-free captions for the
echo     contrastive concept (phase 3 checks for it)
pause
exit /b 0

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1
