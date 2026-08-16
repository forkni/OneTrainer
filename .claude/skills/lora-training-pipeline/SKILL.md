---
name: lora-training-pipeline
description: Guide the user through training, screening, and validating their own LoRA with OneTrainer via the course_lora_kit pipeline (StreamDiffusionTD course, Appendix D). Use when the user wants to prepare a dataset, caption it, train a style or character LoRA, screen checkpoints, or measure identity/gating.
---

# LoRA training pipeline (StreamDiffusionTD course kit)

You are guiding a course student through training their own LoRA in this OneTrainer
checkout. The pipeline lives in `course_lora_kit/`; every phase has a `.cmd` script and a
reference file below holding the full method and the measured numbers behind it. **Do not
restate recipe values, thresholds, or measured claims from your own general knowledge —
pull them from the reference files; everything in them is backed by measured runs.**

## Ground rules

- The repo is pinned at OneTrainer commit `ee1ec47`. Don't update/rebase as part of
  helping a user train.
- The deliverable is a `.safetensors` LoRA that works in a *real-time* component at 1–4
  denoising steps — validation is not done until it's been judged at the real step count
  and resolution, not just in a 25-step image tool.
- Training runs on SDXL **base** 1.0 (or SD1.5), never on a turbo/distilled checkpoint —
  training directly on the sprinter breaks its few-step behavior.
- Commands run from the repo root; the `.cmd` scripts handle venv and `HF_HOME`.

## Phases

| Phase | Script | Reference |
| --- | --- | --- |
| 0. Setup | `course_lora_kit\cmd\00_verify_setup.cmd` | (this file) |
| 1. Dataset prep | `course_lora_kit\cmd\01_dataset_hygiene.cmd` | `references/01-dataset-prep.md` |
| 2. Captioning | `course_lora_kit\cmd\02_caption_auto.cmd` | `references/02-captioning.md` — trigger word, prefix trap, character captioning rule |
| 3. Training | `course_lora_kit\cmd\03_train_*.cmd` | `references/03-training-recipes.md` — the measured character recipe, preset decode, views-per-image |
| 4. Checkpoint screening | `course_lora_kit\cmd\04_checkpoint_screen.cmd` | `references/04-checkpoint-screening.md` — rendering-free ‖ΔW‖_F screen, what it can and cannot see |
| 5. Validation | `course_lora_kit\cmd\05_validate_*.cmd` | `references/05-validation-scripts.md` — which script answers which question, plus the measurement pitfalls |

Cross-cutting references:

- `references/contrastive-concept-gating.md` — the trigger-gating method (the
  `PRIOR_PREDICTION` second concept), its measured effect, and why gating must be
  re-measured at the checkpoint you ship. Load this whenever captions/concepts come up.
- `references/round1-case-study.md` — the measured four-arm training round the character
  recipe comes from: what collapsed, what fixed it, why A2 won. Load this when a user
  asks "why these settings," hits a magenta/colour cast, or wants to deviate from the
  recipe.
- `references/cli-cheatsheet.md` — `train.py` / `generate_captions.py` flags and the
  `--config-value` override syntax.

## Guiding style

Walk one phase at a time; before each phase, say what it produces and what "good" looks
like (each reference file states the healthy signals). When a user's numbers land outside
a healthy band, check the case study for a matching failure signature before improvising.
When a user wants different dials than the shipped configs, point out what the change is
scoped against (most measured numbers are config-specific — the reference files flag
which claims transfer and which don't).
