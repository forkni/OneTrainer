@echo off
setlocal
rem Phase 5 -- render the same prompt/seed across every intermediate checkpoint,
rem plus an untriggered control per checkpoint, to see where the style takes and
rem where it collapses. The rendered confirmation of 04_checkpoint_screen's
rem rendering-free read. Wraps course_lora_kit\scripts\test_lora_checkpoint_sweep.py.
rem
rem Usage:
rem   05_validate_sweep.cmd --ckpt-dir <save_dir> --final <final.safetensors> ^
rem       --trigger <word> --final-step <true_step_count> [flags]
rem
rem Get --final-step from your run's own log or config -- don't guess it from a
rem filename or reuse someone else's number.

if "%~1"=="" (
    echo Usage: 05_validate_sweep.cmd --ckpt-dir ^<save_dir^> --final ^<final.safetensors^> --trigger ^<word^> --final-step ^<steps^> [flags]
    echo Example: 05_validate_sweep.cmd --ckpt-dir workspace\character_sdxl\save --final models\lora_character_sdxl.safetensors --trigger mychar --final-step 1188
    exit /b 1
)

call "%~dp0_env.cmd" || exit /b 1

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_checkpoint_sweep.py" %*
exit /b %errorlevel%
