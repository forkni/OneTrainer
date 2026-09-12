---
name: lora-training-pipeline
description: Guide a StreamDiffusionTD student through the course_lora_kit workflow for preparing, captioning, training, screening, and validating a OneTrainer style or character LoRA (StreamDiffusionTD course, Appendix D; Appendix H is the style-track companion). Use when the user wants to prepare a dataset, caption it, train a style or character LoRA, screen checkpoints, or measure identity/gating. Use only for this course kit, not generic diffusion-training advice.
compatibility: Windows (the phase wrappers are .cmd scripts). Requires the OneTrainer venv (run install.bat once first) and an NVIDIA GPU for the training and render-validation phases; phases 1 and 4 run on CPU.
metadata:
  author: StreamDiffusionTD course
  version: 1.4.3
---

# LoRA training pipeline (StreamDiffusionTD course kit)

You are guiding a course student through training their own LoRA in this OneTrainer
checkout. The pipeline lives in `course_lora_kit/`; every phase has a `.cmd` script and a
reference file below holding the full method and the measured numbers behind it. **Do not
restate recipe values, thresholds, or measured claims from your own general knowledge —
pull them from the reference files; everything in them is backed by measured runs.**

## Operating boundary (Claude Code and Codex)

This is a Windows course-kit skill. Claude Code discovers it at
`.claude/skills/lora-training-pipeline/`; Codex discovers the pointer at
`.agents/skills/lora-training-pipeline/`, whose `SKILL.md` sends it here — the body and
the `references/` files live only in this folder. Before proposing or running a phase command, identify
the OneTrainer checkout the user intends to use and verify that it contains
`course_lora_kit/` and the named wrapper. The course references describe a pinned snapshot;
they are evidence for that snapshot, not proof that an arbitrary current OneTrainer fork has
the same files, defaults, or CLI behavior. State a mismatch rather than silently translating
the recipe to a different checkout.

Do not start `install.bat`, `update.bat`, training, checkpoint rendering, or other
potentially long/GPU-consuming work unless the user explicitly asks to execute it. When
execution is requested, first report the exact command, expected output location, and the
hardware or model files it needs. Do not install Python packages outside the checkout's
existing virtual environment.

## Ground rules

- The repo is pinned at OneTrainer commit `ee1ec47`. Don't update/rebase as part of
  helping a user train. Confirm the commit before calling it the user's current revision.
- The deliverable is a `.safetensors` LoRA that works in a *real-time* component at 1–4
  denoising steps — validation is not done until it's been judged at the real step count
  and resolution, not just in a 25-step image tool.
- Training runs on SDXL **base** 1.0 (the kit's only track), never on a turbo/distilled checkpoint —
  training directly on the sprinter breaks its few-step behavior.
- Commands run from the repo root; the `.cmd` scripts handle venv and `HF_HOME`.
  Run with arguments from a terminal, or double-click with no arguments — the scripts
  then prompt interactively for paths and options.

## Phases

| Phase | Script | Reference |
| --- | --- | --- |
| 0. Setup | `course_lora_kit\cmd\00_verify_setup.cmd`, then `00b_print_recipe.cmd` | (this file) — the recipe printer shows every dial after the preset/overlay merge; check `train_dtype = BFLOAT_16` before a style run |
| 1. Dataset prep | `course_lora_kit\cmd\01_dataset_hygiene.cmd`, then `01b_palette_screen.cmd` | `references/01-dataset-prep.md` |
| 2. Captioning | `course_lora_kit\cmd\02_check_trigger.cmd`, then `02_caption_auto.cmd` | `references/02-captioning.md` — trigger word (and its pre-check script), prefix trap, character captioning rule |
| 3. Training | `course_lora_kit\cmd\03_train_*.cmd` | `references/03-training-recipes.md` — the measured character recipe, preset decode, views-per-image |
| 4. Checkpoint screening | `course_lora_kit\cmd\04_checkpoint_screen.cmd` | `references/04-checkpoint-screening.md` — rendering-free ‖ΔW‖_F screen, what it can and cannot see |
| 5. Validation | `course_lora_kit\cmd\05_validate_*.cmd` | `references/05-validation-scripts.md` — which script answers which question, plus the measurement pitfalls |
| 6. Stage the pick | `course_lora_kit\cmd\06_stage_pick.cmd` | `references/05-validation-scripts.md` (*Which checkpoint to ship*) — recomputes ‖ΔW‖_F, rank, alpha and module count from the file, copies (never moves) it under the deploy name, SHA-256 on both sides |

Cross-cutting references:

- `references/contrastive-concept-gating.md` — the trigger-gating method (the
  `PRIOR_PREDICTION` second concept), its measured effect, and why gating must be
  re-measured at the checkpoint you ship. Load this whenever captions/concepts come up.
- `references/round1-case-study.md` — the measured four-arm training round the character
  recipe comes from: what collapsed, what fixed it, why A2 won. Load this when a user
  asks "why these settings," hits a magenta/colour cast, or wants to deviate from the
  recipe.
- `references/round2-style-case-study.md` — the measured style-ablation round (five
  one-rule-violation arms vs a control): why the style overlays pin warmup 30, why the
  norm bands are step-count-relative rather than absolute, the pillarbox tool-asymmetry
  demo, and the trigger-inert failure mode of skipping the contrastive concept. Load
  this for any style-track "what does this rule protect me from" question.
- `references/cli-cheatsheet.md` — `train.py` / `generate_captions.py` flags and the
  `--config-value` override syntax.

## Examples

- *"I want to train a LoRA of my own character"* → start at phase 0, then walk 1→5 in
  order with `03_train_character_sdxl.cmd`; load `contrastive-concept-gating.md` before
  the captioning phase so the trigger-free concept copy is planned from the start.
- *"My checkpoints all look purple/magenta"* → load `round1-case-study.md` and compare
  against the colour-collapse signature before changing any dials.
- *"Which checkpoint should I use?"* → run phase 4 on the save folder, then confirm the
  shortlisted steps with `05_validate_sweep.cmd` (norms alone can't see colour casts),
  then hand the pick over with `06_stage_pick.cmd`.

## Troubleshooting

- Any `.cmd` script prints "venv not found" → OneTrainer isn't installed yet; run
  `install.bat` from the repo root and re-run `00_verify_setup.cmd`.
- Renders show a magenta/colour cast that worsens with checkpoint step → the measured
  collapse signature; see `round1-case-study.md` (the fix is the loss/noise dials, not
  the LoRA weight).
- Validation renders ignore the LoRA entirely → the component may have failed to load it
  and silently fallen back to the base model; check the log for a LoRA-load ERROR before
  trusting any numbers from that batch.

## Guiding style

Walk one phase at a time; before each phase, say what it produces and what "good" looks
like (each reference file states the healthy signals). When a user's numbers land outside
a healthy band, check the case study for a matching failure signature before improvising.
When a user wants different dials than the shipped configs, point out what the change is
scoped against (most measured numbers are config-specific — the reference files flag
which claims transfer and which don't).

Keep a short handoff after a completed phase: source folder, command run or proposed,
artifact location, observed healthy/failure signal, and the next phase. Do not imply a
checkpoint is shippable until phase 5 validates it at the target real-time step count and
resolution.
