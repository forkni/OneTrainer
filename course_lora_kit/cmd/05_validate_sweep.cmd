@echo off
setlocal
rem Phase 5 -- render the same prompt/seed across every intermediate checkpoint,
rem plus an untriggered control per checkpoint, to see where the style takes and
rem where it collapses. The rendered confirmation of 04_checkpoint_screen's
rem rendering-free read. Wraps course_lora_kit\scripts\test_lora_checkpoint_sweep.py.
rem
rem Terminal usage:
rem   05_validate_sweep.cmd --ckpt-dir <save_dir> --final <final.safetensors> ^
rem       --trigger <word> --final-step <true_step_count> [flags]
rem
rem Get --final-step from your run's own log or config -- don't guess it from a
rem filename or reuse someone else's number.
rem
rem Double-clicked with no arguments, the script asks for everything instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_checkpoint_sweep.py" %*
goto :finish

:interactive
echo [course_lora_kit] Checkpoint sweep ^(rendered^).
echo.
echo Tip: right-click a file or folder in Explorer, "Copy as path", then
echo right-click in this window to paste it.
echo.

:ask_ckptdir
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "CKPTDIR="
set /p "CKPTDIR=Intermediate-checkpoint folder (e.g. workspace\character_sdxl\save): "
if not defined CKPTDIR goto :ask_ckptdir
set "CKPTDIR=%CKPTDIR:"=%"
if not exist "%CKPTDIR%" (
    echo   Not found: "%CKPTDIR%" -- try again.
    goto :ask_ckptdir
)

:ask_final
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "FINAL="
set /p "FINAL=Final LoRA .safetensors (e.g. models\lora_character_sdxl.safetensors): "
if not defined FINAL goto :ask_final
set "FINAL=%FINAL:"=%"
if not exist "%FINAL%" (
    echo   Not found: "%FINAL%" -- try again.
    goto :ask_final
)

:ask_trigger
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "TRIGGER="
set /p "TRIGGER=Trigger word (copy it out of a caption file -- do not retype it): "
if not defined TRIGGER goto :ask_trigger

:ask_step
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "FINALSTEP="
set /p "FINALSTEP=True final step count (from your run's log/config -- do not guess): "
if not defined FINALSTEP goto :ask_step
echo.

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_checkpoint_sweep.py" --ckpt-dir "%CKPTDIR%" --final "%FINAL%" --trigger "%TRIGGER%" --final-step %FINALSTEP%
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
