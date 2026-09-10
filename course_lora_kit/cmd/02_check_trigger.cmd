@echo off
setlocal
rem Phase 2 -- check a candidate trigger word for loaded meaning BEFORE you
rem caption a dataset with it. Wraps course_lora_kit\scripts\check_trigger_word.py.
rem
rem Stage 1 prints the BPE token split from both SDXL text encoders (instant).
rem Stage 2 renders the bare trigger vs empty-prompt controls at the same seeds
rem and saves a labeled contact sheet -- it needs the SDXL base model in the
rem local HF cache (your first training run populates it); before that, the
rem script prints the token split and says what it skipped.
rem
rem Terminal usage:
rem   02_check_trigger.cmd <trigger_word> [extra flags for check_trigger_word.py]
rem
rem Cache location: pass --hf-home <folder> to use a specific Hugging Face
rem cache, or export HF_HOME before running. Without either, _env.cmd points
rem HF_HOME at the repo's workspace\hf_cache.
rem
rem Double-clicked with no arguments, the script asks for everything instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

set "TRIGGER=%~1"
shift
set "EXTRA="
:collect
if "%~1"=="" goto :run
set "EXTRA=%EXTRA% %1"
shift
goto :collect

:interactive
echo [course_lora_kit] Trigger-word check.
echo.
echo A made-up token can still land on loaded ground -- close to a name, a brand,
echo or a style the base model already knows. This checks before you caption
echo several dozen files with it.
echo.

:ask_trigger
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "TRIGGER="
set /p "TRIGGER=Candidate trigger word: "
if not defined TRIGGER goto :ask_trigger
echo.

echo Optional: folder of the Hugging Face cache that holds the SDXL base model.
echo Press Enter to use the default -- HF_HOME if set, else the repo's workspace cache.
set "HFC="
set /p "HFC=HF cache folder [Enter = default]: "
if defined HFC (
    set "HFC=%HFC:"=%"
    set "EXTRA= --hf-home "%HFC%""
)
echo.

:run
"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\check_trigger_word.py" --trigger "%TRIGGER%"%EXTRA%
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
