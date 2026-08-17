# course_lora_kit — the StreamDiffusionTD course's LoRA training pipeline

A guided, scripted path through training your own LoRA with OneTrainer, built for the
StreamDiffusionTD course (Appendix D: *Training Your Own LoRA*). Everything here is an
**addition** to a pinned OneTrainer checkout — no upstream file is modified — and every
recipe and threshold in it is backed by measured training runs, not folklore.

Each script carries its usage at the top and explains itself as it runs. The full
reasoning behind every threshold and recipe — the config decodes, the Round 1 case
study, the contrastive-concept method — is part of the StreamDiffusionTD course's
materials (Appendix D and its companion references).

## Quick start

```
git clone -b course/lora-pipeline https://github.com/forkni/OneTrainer
cd OneTrainer
install.bat
course_lora_kit\cmd\00_verify_setup.cmd
```

Then work through the phases. Every script works two ways: run it from a terminal with
arguments (usage lines are at the top of each script), or just double-click it in
Explorer — with no arguments it switches to an interactive mode that asks for the paths
and options it needs (paste paths with right-click; Explorer's "Copy as path" quotes are
handled). The window pauses at the end so you can read the output.

## The pipeline

| Phase | What you do | Script |
| --- | --- | --- |
| 0. Setup | Install OneTrainer, verify venv + CUDA | `cmd\00_verify_setup.cmd` |
| 1. Dataset prep | Curate images; screen for outliers, undersized frames, letterbox bars, subfolder imbalance | `cmd\01_dataset_hygiene.cmd` |
| 2. Captioning | Auto-caption, then the by-hand pass; set up the trigger word + contrastive concept | `cmd\02_caption_auto.cmd` |
| 3. Training | Shipped preset + course overlay config | `cmd\03_train_character_sdxl.cmd`, `cmd\03_train_style_sdxl.cmd`, `cmd\03_train_style_sd15.cmd` |
| 4. Checkpoint screening | Rendering-free weight-delta screen of the whole sweep (no GPU) | `cmd\04_checkpoint_screen.cmd` |
| 5. Validation | Render checks: 2×2 grid, gating measure, checkpoint sweep, identity scoring | `cmd\05_validate_*.cmd` |
| 6. Deploy | Load into your real-time component and judge at real step counts | (in the course repo — see Appendix D) |

## What's where

- `cmd\` — one `.cmd` per phase; they all run in OneTrainer's own venv and layer the
  shipped training presets under the course configs.
- `configs\` — sparse overlay configs. `character_sdxl.json` is the course's **measured**
  character recipe (the Round 1 "A2" winner); the style configs keep the shipped preset's
  recipe and only add checkpointing/epoch dials it leaves unset. See `configs\README.md`.
- `scripts\` — the course's seven measurement scripts (dataset hygiene, checkpoint norm
  analysis, gating, identity). Canonical copies live in the course repo's
  `course_v3/appendices/assets/`; these are verbatim copies so the kit is self-contained.

## Model downloads

The `.cmd` scripts set `HF_HOME` to `workspace\hf_cache` (inside this repo, gitignored)
unless you've already set it — so the SDXL/SD1.5 base models download once, in one
predictable place. If you already have the models cached elsewhere, set `HF_HOME` to that
location before running. The validation scripts in `scripts\` load with
`local_files_only=True` — they use the cache but won't populate it; the first *training*
run (or a one-off `transformers`/`diffusers` download) is what fills it. The two base
models the presets target (`stabilityai/stable-diffusion-xl-base-1.0`,
`stable-diffusion-v1-5/stable-diffusion-v1-5`) are public; no Hugging Face token is
needed for them. The identity scripts additionally need `facebook/dinov2-base` (and
`openai/clip-vit-base-patch32` for the consistency script) cached the same way.

## Provenance

- OneTrainer checkout pinned at commit `ee1ec47` — the commit every code citation in the
  course's fact-check record was verified against. Upstream moves fast; rebasing this
  branch is a deliberate action, not routine.
- The character recipe and all thresholds come from the course's Round 1 training round
  (four competing recipes plus two isolation retrains, scored on identity, gating,
  colour neutrality, and weight-delta diagnostics). The full case study is part of the
  course materials.
