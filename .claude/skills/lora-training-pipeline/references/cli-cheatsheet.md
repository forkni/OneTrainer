# CLI cheatsheet

All commands run from the repo root in OneTrainer's own venv
(`venv\Scripts\python.exe`); the `.cmd` wrappers handle both plus `HF_HOME`.
Verified against OneTrainer commit `ee1ec47`.

## `scripts/train.py`

```
venv\Scripts\python.exe scripts\train.py ^
    --preset-path "training_presets\SDXL\#sdxl 1.0 LoRA.json" ^
    --config-path course_lora_kit\configs\character_sdxl.json ^
    --config-value lora_alpha=0.8 ^
    --config-value epochs=50
```

- `--preset-path` — applies a shipped preset first (loaded without config migration).
- `--config-path` — your config/overlay, applied on top.
- `--config-value KEY=VALUE` — one field per flag, repeatable, applied last. Dot
  notation walks *real nested configs only* (e.g. `optimizer.beta1=0.9`) —
  **`lora_rank` and `lora_alpha` are flat top-level keys**, so it's
  `--config-value lora_alpha=0.8`, *not* `lora.lora_alpha=0.8` (there is no `lora`
  sub-config; the nested form errors).
- Set `HF_HOME` before running headless (the `.cmd` scripts default it to
  `workspace\hf_cache`) — the headless path inherits whatever cache dir the config
  carries, which is easy to leave empty.
- `run-cmd.sh` in community docs is the Linux/WSL launcher — on Windows call
  `train.py` directly.

Sparse-overlay rules (why the shipped configs look thin): only overridden keys are
present; `concepts` stays unset so they load from `concept_file_name` at train time;
`"samples": []` must be present explicitly (a null `samples` makes the trainer try to
open `sample_definition_file_name` at startup). Details: `configs/README.md`.

## `scripts/generate_captions.py`

```
venv\Scripts\python.exe scripts\generate_captions.py ^
    --model WD14_VIT_2 --sample-dir path\to\images ^
    --mode fill --include-subdirectories
```

- `--model {BLIP,BLIP2,WD14_VIT_2}` — WD14 for anime/booru-style, BLIP/BLIP2 for
  photographic natural language.
- `--mode` — `fill` (default; caption images that don't have one) / `replace` / `add`.
- `--initial-caption`, `--caption-prefix`, `--caption-postfix` — the prefix is raw
  concatenation with **no separator** (end it with `, `), and **never put the trigger
  word in it** (`02-captioning.md`).
- `--device`, `--dtype` — usually leave alone.

## The `.cmd` scripts (all in `course_lora_kit\cmd\`)

| Script | Args |
| --- | --- |
| `00_verify_setup.cmd` | — (venv + CUDA + key-file check) |
| `00b_print_recipe.cmd` | [`style` \| `character`] [`--config-value K=V` ...] [`--all`] [`--json out.json`] — prints the merged recipe (defaults → preset → overlay → overrides), no GPU |
| `01_dataset_hygiene.cmd` | `<image_folder>` [extra profiler flags — defaults `--recursive --min-side 1024`, yours win; use the bucket's short side for non-square sets, e.g. `--min-side 896` for 4:3] |
| `01b_palette_screen.cmd` | `<image_folder>` [more folders] [`--z-threshold 2.0`] [`--sort a\|b\|chroma\|name`] [`--top N`] [`--json out.json`] — default `--recursive`; mean CIELAB a\*/b\* per image with folder-relative z-flags and hue reads; extra folders are compared against the first |
| `02_check_trigger.cmd` | `<trigger_word>` [extra flags for `check_trigger_word.py`] |
| `02_caption_auto.cmd` | `<image_folder>` `[MODEL]` (default `WD14_VIT_2`) |
| `03_train_character_sdxl.cmd` | [extra `--config-value` flags]; needs `training_concepts\character_concepts.json` |
| `03_train_style_sdxl.cmd` | same, with `style_concepts.json` |
| `04_checkpoint_screen.cmd` | `<ckpt_dir_or_file>` [...] [`--json out.json`] |
| `05_validate_grid.cmd` | `--lora X.safetensors --trigger word` [...] |
| `05_validate_gating.cmd` | `--lora X.safetensors --trigger word` [`--prompts-file battery.json`] |
| `05_validate_sweep.cmd` | `--ckpt-dir save\ --final X.safetensors --trigger word --final-step N` |
| `05_validate_identity_consistency.cmd` | `--lora LABEL=PATH` [...] `--reference-dir refs\` |
| `05_validate_identity_score.cmd` | `--render LABEL=PATH` [...] `--reference-dir refs\` |

Every script prints usage when run without arguments; validation wrappers forward all
flags to the underlying Python script (full flag reference:
`course_lora_kit/scripts/` docstrings and the course repo's assets README).
