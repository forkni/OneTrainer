@echo off
setlocal
rem Phase 5 (character LoRAs) -- same identity metric as
rem 05_validate_identity_consistency.cmd, but for renders you already have from
rem anywhere else (ComfyUI output, a StreamDiffusionTD capture). No diffusion
rem pipeline, no GPU required -- it embeds and scores existing files.
rem Wraps course_lora_kit\scripts\test_lora_identity_score.py.
rem
rem Usage:
rem   05_validate_identity_score.cmd --render LABEL=<file^|folder^|glob> ^
rem       --reference-dir <held_out_images> [flags]
rem
rem Scores are directly comparable to the consistency script's "identity" column
rem as long as --reference-dir is the same folder.

if "%~1"=="" (
    echo Usage: 05_validate_identity_score.cmd --render LABEL=^<file^|folder^|glob^> --reference-dir ^<held_out_images^> [flags]
    echo Repeatable: --render A2="out\A2_*.png" --render A4="out\A4_*.png"
    exit /b 1
)

call "%~dp0_env.cmd" || exit /b 1

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_identity_score.py" %*
exit /b %errorlevel%
