@echo off
setlocal
rem Phase 5 (character LoRAs) -- the measurement pixel-diff can't make: is it the
rem SAME subject across prompts and seeds. Renders a prompt x seed grid per arm
rem and scores identity (DINOv2 cosine vs a held-out reference set), cross-seed
rem consistency, flexibility, and a CLIP prompt-adherence score.
rem Wraps course_lora_kit\scripts\test_lora_identity_consistency.py.
rem
rem Usage:
rem   05_validate_identity_consistency.cmd --lora LABEL=<path> [--lora LABEL2=<path>] ^
rem       --reference-dir <held_out_images> [flags]
rem
rem --reference-dir is the 8-12 images you held back from training -- images the
rem LoRA has never seen. Scoring against training images measures memorization,
rem not identity.

if "%~1"=="" (
    echo Usage: 05_validate_identity_consistency.cmd --lora LABEL=^<path^> --reference-dir ^<held_out_images^> [flags]
    echo Repeatable: --lora A2=a.safetensors --lora A4=b.safetensors  ^(LABEL=none for a no-LoRA baseline^)
    echo Per-label scale: --lora-scale LABEL=0.0
    exit /b 1
)

call "%~dp0_env.cmd" || exit /b 1

"%VENV_PY%" "%REPO_ROOT%\course_lora_kit\scripts\test_lora_identity_consistency.py" %*
exit /b %errorlevel%
