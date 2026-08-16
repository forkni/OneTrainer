@echo off
setlocal
rem Phase 5 -- the full trigger-gating measurement: four cells per prompt
rem (LoRA on/off x trigger present/absent) with the token-insertion null
rem subtracted out, so the reported ratio measures your LoRA, not the tokenizer.
rem Wraps course_lora_kit\scripts\test_lora_gating_measure.py.
rem
rem Usage:
rem   05_validate_gating.cmd --lora <path\to\lora.safetensors> --trigger <word> [flags]
rem
rem Re-run this at the checkpoint you actually ship -- gating measured at an
rem earlier checkpoint does not transfer (see the skill's
rem contrastive-concept-gating reference). Copy the trigger word out of an
rem actual caption file; never retype it.

if "%~1"=="" (
    echo Usage: 05_validate_gating.cmd --lora ^<lora.safetensors^> --trigger ^<word^> [flags]
    echo Optional: --prompts-file battery.json  ^(out-of-domain prompt battery^)
    pause
    exit /b 1
)

call "%~dp0_env.cmd" || (pause & exit /b 1)

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_gating_measure.py" %*
set "EC=%errorlevel%"
pause
exit /b %EC%
