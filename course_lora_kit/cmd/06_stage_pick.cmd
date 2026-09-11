@echo off
setlocal
rem Phase 6 -- stage the checkpoint you picked: recompute its ||dW||_F and module
rem count, check rank/alpha, COPY it to your component's loras folder under a
rem label, and confirm SHA-256 on both sides. The source file is never moved.
rem Wraps course_lora_kit\scripts\stage_pick.py.
rem
rem Terminal usage:
rem   06_stage_pick.cmd --lora <checkpoint.safetensors> --label <name> --dest-dir <loras_folder> [--expected-norm N] [--expected-modules 722]
rem
rem Give --expected-norm the value 04_checkpoint_screen printed for that checkpoint;
rem the copy is refused if the recomputed norm is more than 2% off (--tol).
rem
rem Double-clicked with no arguments, the script asks for everything instead.

call "%~dp0_env.cmd" || (pause & exit /b 1)

if "%~1"=="" goto :interactive

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\stage_pick.py" %*
goto :finish

:interactive
echo [course_lora_kit] Stage a picked checkpoint ^(verify, copy, hash^).
echo.
echo Tip: right-click a file or folder in Explorer, "Copy as path", then
echo right-click in this window to paste it.
echo.

:ask_lora
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "LORA="
set /p "LORA=Checkpoint .safetensors to stage: "
if not defined LORA goto :ask_lora
set "LORA=%LORA:"=%"
if not exist "%LORA%" (
    echo   Not found: "%LORA%" -- try again.
    goto :ask_lora
)

:ask_label
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "LABEL="
set /p "LABEL=Label for the staged copy (e.g. my_style_s1099): "
if not defined LABEL goto :ask_label

:ask_dest
set /a ASKED+=1
if %ASKED% gtr 20 goto :no_input
set "DEST="
set /p "DEST=Destination loras folder: "
if not defined DEST goto :ask_dest
set "DEST=%DEST:"=%"

set "NORM="
set /p "NORM=Expected norm from 04_checkpoint_screen (Enter to skip): "
echo.

if defined NORM (
    "%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\stage_pick.py" --lora "%LORA%" --label "%LABEL%" --dest-dir "%DEST%" --expected-norm %NORM%
) else (
    "%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\stage_pick.py" --lora "%LORA%" --label "%LABEL%" --dest-dir "%DEST%"
)
goto :finish

:no_input
echo [course_lora_kit] No usable input -- exiting.
pause
exit /b 1

:finish
set "EC=%errorlevel%"
pause
exit /b %EC%
