# course_lora_kit — the StreamDiffusionTD course's LoRA training pipeline

A guided, scripted path through training your own LoRA with OneTrainer, built for the
StreamDiffusionTD course (Appendix D: *Training Your Own LoRA*). Everything here is an
**addition** to a pinned OneTrainer checkout — no upstream file is modified — and every
recipe and threshold in it is backed by measured training runs, not folklore.

Each script carries its usage at the top and explains itself as it runs. The full
reasoning behind every threshold and recipe — the config decodes, the Round 1 case
study, the contrastive-concept method — is part of the StreamDiffusionTD course's
materials (Appendix D and its companion references).

If you're using Claude Code, open a session in this repo's root: the
`lora-training-pipeline` skill (in `.claude/skills/`, tracked on this branch) guides you
through the phases below with the full reasoning behind each step. Codex finds the same
skill through the pointer at `.agents/skills/lora-training-pipeline/` (Codex does not
scan `.claude/skills/`). The `references/` files inside the skill are plain markdown —
readable without any agent.

## Quick start

```
git clone -b course/lora-pipeline https://github.com/forkni/OneTrainer
cd OneTrainer
install.bat
course_lora_kit\cmd\00_verify_setup.cmd
```

The setup script asks one question on its first run: where your LoRA work should live
(`LORA_ROOT` — one folder for datasets, validation outputs and staged picks; press Enter
to keep it inside the checkout). It remembers the answer in the gitignored
`course_lora_kit\local_paths.cmd`, creates `%LORA_ROOT%\dataset\trigger` and
`\notrigger`, and writes both concepts files pointing at them, so no path is typed twice
and nothing machine-specific is ever committed. Optional: add `COMFY_LORAS_DIR` to the same
file (see `local_paths.cmd.example`) and `06_stage_pick.cmd` offers it as the default
destination.

Then work through the phases. Every script works two ways: run it from a terminal with
arguments (usage lines are at the top of each script), or just double-click it in
Explorer — with no arguments it switches to an interactive mode that asks for the paths
and options it needs (paste paths with right-click; Explorer's "Copy as path" quotes are
handled). The window pauses at the end so you can read the output.

## The pipeline

| Phase | What you do | Script |
| --- | --- | --- |
| 0. Setup | Install OneTrainer, verify venv + CUDA; set `LORA_ROOT` once (dataset folders + concepts files written for you); print the merged recipe before training | `cmd\00_verify_setup.cmd`, `cmd\00b_print_recipe.cmd` |
| 1. Dataset prep | Curate images; screen for outliers, undersized frames, letterbox bars, subfolder imbalance; then the palette screen (mean a\*/b\* per image, kept vs dropped) | `cmd\01_dataset_hygiene.cmd`, `cmd\01b_palette_screen.cmd` |
| 2. Captioning | Check the trigger word is free of loaded meaning; auto-caption, then the by-hand pass; set up the contrastive concept | `cmd\02_check_trigger.cmd`, `cmd\02_caption_auto.cmd` |
| 3. Training | Shipped SDXL preset + course overlay config (the kit trains SDXL only) | `cmd\03_train_character_sdxl.cmd`, `cmd\03_train_style_sdxl.cmd` |
| 4. Checkpoint screening | Rendering-free weight-delta screen of the whole sweep (no GPU) | `cmd\04_checkpoint_screen.cmd` |
| 5. Validation | Render checks: 2×2 grid, gating measure, checkpoint sweep, seed batch (one checkpoint, several seeds, LoRA on/off, colour-drift number per render), identity scoring; colour-drift metric over a sweep | `cmd\05_validate_*.cmd`, `scripts\color_stats.py` |
| 6. Deploy | Stage the pick (norm recompute, copy, SHA-256), then load it into your real-time component and judge at real step counts | `cmd\06_stage_pick.cmd`, then the course repo — see Appendix D |

## What's where

- `cmd\` — one `.cmd` per phase; they all run in OneTrainer's own venv and layer the
  shipped training presets under the course configs.
- `configs\` — sparse overlay configs. `character_sdxl.json` is the course's **measured**
  character recipe (the Round 1 "A2" winner); the style config keeps the shipped preset's
  rank/alpha/LR and pin the dials it leaves unset or defaults badly (warmup, Min-SNR +
  offset noise, bf16, checkpointing) — the Round 2 "S0b" recipe. Two concepts templates:
  character (flip off) and style (flip on). See `configs\README.md`.
- `scripts\` — the course's thirteen measurement scripts (trigger-word check, dataset
  hygiene, palette screen, effective-config printer, checkpoint norm analysis, gating,
  seed batch, identity, colour drift, checkpoint staging) plus `init_concepts.py`, which
  writes your concepts files from the templates. Thirteen mirror the course repo's
  `course_v3/appendices/assets/` byte for byte;
  `print_effective_config.py` is fork-only because it imports OneTrainer. The two
  StreamDiffusion-side checks (`test_lora_graph_check.py`, `test_lora_sanity.py`) stay in
  the course repo because they need a StreamDiffusion checkout, not this one. Every script
  prints its own `Column key:` (or `reading:`) block defining every field it just printed,
  at the end of its own run; the course's Appendix H collects the same definitions in
  *What the columns mean*, grouped by what the instrument reads rather than by script.
- `.claude\skills\lora-training-pipeline\` (repo root) — the agent skill + ten
  reference files holding the deep detail: full recipe decodes, the Round 1, Round 2 and
  Round 3 case studies, the contrastive-concept method, screening thresholds, CLI cheatsheet.

## Model downloads

The `.cmd` scripts set `HF_HOME` to `workspace\hf_cache` (inside this repo, gitignored)
unless you've already set it — so the SDXL base model downloads once, in one
predictable place. If you already have the models cached elsewhere, set `HF_HOME` to that
location before running. The validation scripts in `scripts\` load with
`local_files_only=True` — they use the cache but won't populate it. They also resolve the
repo id to the cached snapshot folder before loading (`resolve_local_snapshot()` in each
rendering script): `huggingface_hub` 1.22+ keeps a listing of the repo's *full* file tree
and refuses a partial snapshot offline, even one that has every file diffusers needs —
found 2026-09-11 on the cache that had rendered all of Round 2. Relative `--output-dir` /
`--json` paths resolve against the directory you run a wrapper from, not the script's
folder. The first *training*
run (or a one-off `transformers`/`diffusers` download) is what fills it. The base model
the presets target (`stabilityai/stable-diffusion-xl-base-1.0`) is public; no Hugging
Face token is needed for it. The identity scripts additionally need `facebook/dinov2-base` (and
`openai/clip-vit-base-patch32` for the consistency script) cached the same way.

## Provenance

- OneTrainer checkout pinned at commit `ee1ec47` — the commit every code citation in the
  course's fact-check record was verified against. Upstream moves fast; rebasing this
  branch is a deliberate action, not routine.
- The character recipe and all thresholds come from the course's Round 1 training round
  (four competing recipes plus two isolation retrains, scored on identity, gating,
  colour neutrality, and weight-delta diagnostics). The full story:
  `.claude/skills/lora-training-pipeline/references/round1-case-study.md`.
- The style recipe's three corrections (warmup 30, Min-SNR + offset noise, bf16 pinned)
  come from Round 2, a six-arm ablation on a 48-image style set:
  `.claude/skills/lora-training-pipeline/references/round2-style-case-study.md`.
