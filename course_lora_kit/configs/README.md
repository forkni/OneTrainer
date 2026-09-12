# course_lora_kit configs

These are **sparse overlays**, the same convention as OneTrainer's own shipped presets:
each JSON contains only the fields it overrides. The `03_train_*.cmd` scripts apply them
with `--preset-path` (the shipped preset) + `--config-path` (the overlay), so anything not
listed here comes from the preset, and anything not in the preset comes from OneTrainer's
built-in defaults. One-off tweaks don't need an edit — forward
`--config-value KEY=VALUE` through the `.cmd` script (note: `lora_rank` and `lora_alpha`
are flat top-level keys, not nested under a `lora.` prefix).

## `character_sdxl.json` — the measured character recipe

The exact dials of the course's Round 1 winning arm (**A2**, picked over four competing
recipes by measurement — the full case study is part of the course materials):

| Field | Value | Why |
| --- | --- | --- |
| `lora_rank` / `lora_alpha` | 16 / 16.0 | alpha = rank (scale 1.0), the LoRA paper's own default |
| `learning_rate` | 1e-4 | pairs with scale 1.0; the preset's 3e-4 pairs with its alpha 1.0 — never mix the two conventions |
| `learning_rate_warmup_steps` | 30.0 | explicit literal step count; the unset default is 200 *steps*, which silently eats most of a short run |
| `loss_weight_fn` / `loss_weight_strength` | `MIN_SNR_GAMMA` / 5.0 | the measured fix for a colour-channel collapse (with offset noise below) |
| `offset_noise_weight` | 0.03 | the other half of that fix |
| `train_dtype` | `BFLOAT_16` | wider dynamic range than the preset's fp16 at the same memory cost; LoRA weights stay fp32 |
| `layer_filter_preset` | `attn-mlp` | do **not** widen to `full` — a full-filter kohya LoRA is unloadable by diffusers-based runtimes (StreamDiffusionTD included) |
| `epochs` / `batch_size` | 44 / 4 | 1188 steps on the measured Round 1 set: 55 curated images × 2 concepts = 110 items → 27 steps/epoch at batch 4 (drop-last) × 44 epochs; retune epochs to your dataset by views-per-image: `(steps × batch) ÷ images`, aim for ~40–60 |
| `save_every` / unit | 100 / `STEP` | intermediate checkpoints are OFF by default; without them there is nothing to screen in phase 4 |

## `style_sdxl.json` — preset + course overrides

The shipped preset's recipe (rank 16 / alpha 1.0 / LR 3e-4, `attn-mlp`, batch 4) is
left intact — that exact config is what the course's style-LoRA numbers were measured
on. The overlay is the Round 2 **S0b** recipe: it adds what the preset leaves unset and
overrides three inherited defaults that measured badly.

| Field | Value | Why |
| --- | --- | --- |
| `epochs` / `save_every` | 50 / 100 `STEP` | a checkpoint budget, not a training target — Round 2 measured the knee near epoch 20 on a 48-image set, and Round 3a (2026-09-12) kept improving to epoch 45 on the curated 50-image set with the same recipe; the norm screen and the sweep find yours |
| `learning_rate_warmup_steps` | 30.0 | the inherited default is 200 *steps*. Round 2 measured that default as an ablation arm (S1): 200 of 1,200 steps spent ramping suppressed early learning by 56–66% at matched checkpoints and the deficit never closed |
| `loss_weight_fn` / `loss_weight_strength` / `offset_noise_weight` | `MIN_SNR_GAMMA` / 5.0 / 0.03 | the inherited `CONSTANT` / 0.0 drifted magenta per seed; S0b cut the step-1099 render's mean CIELAB a\* from 31.0 to 12.0 at the same gating (`scripts\color_stats.py` is the metric) |
| `train_dtype` / `fallback_train_dtype` | `BFLOAT_16` / `BFLOAT_16` | neither the preset nor OneTrainer's default sets bf16 (`train_dtype` defaults to `FLOAT_16`, `TrainConfig.py:1074`; the fallback already defaults to `BFLOAT_16`, `:1075`). The measured S0b run trained bf16, so the overlay pins both — unpinned, the same overlay trained fp16 for a while and nobody noticed. `cmd\00b_print_recipe.cmd` shows the merged result |
| `backup_after_unit` | `NEVER` | no resumable backups; the 100-step saves are the artefact |
| `layer_filter` | `attentions` | the field the trainer actually reads. `ModuleFilter.create` builds the filter from `layer_filter` only (`modules/util/ModuleFilter.py:39-43`); `layer_filter_preset` is a UI label that fills this field when clicked and does nothing on its own. The shipped preset carries only the label, so with an empty `layer_filter` every layer trains (Round 3a first launch, 2026-09-12: 794 selected / 0 deselected, step-100 save 198 MB, aborted at step 112). `attentions` is what the `attn-mlp` label maps to on SDXL (`BaseStableDiffusionXLSetup.py:40`) and what the measured S0b config carried (722 / 72) |

See `references/round2-style-case-study.md` in the `lora-training-pipeline` skill for
the six-arm ablation these come from.

## `concepts_contrastive_template*.json` — copy, don't edit in place

Two templates, identical except for one augmentation switch (below):
`concepts_contrastive_template.json` for the character track,
`concepts_contrastive_template_style.json` for the style track. `cmd\00_verify_setup.cmd`
does the copy for you: on its first run it asks once where your LoRA work lives
(`LORA_ROOT`, remembered in the gitignored `course_lora_kit\local_paths.cmd`), creates
`%LORA_ROOT%\dataset\trigger` and `\notrigger`, and writes both files into
`training_concepts/` (gitignored, i.e. yours) under the names the training scripts expect
— `character_concepts.json` and `style_concepts.json` — with their two `"path"` fields
pointing at those folders (`scripts\init_concepts.py` does the writing; it never overwrites
an existing file). Edit the paths only if your dataset lives elsewhere, or do the copy by
hand from the template.

Each defines **two concepts over the same images**:

1. `my_subject` (`my_style`) — `type: STANDARD`, your captioned dataset, trigger word
   leading every caption.
2. `my_subject_notrigger_prior` (`my_style_notrigger_prior`) — `type: PRIOR_PREDICTION`
   (OneTrainer's prior-preservation concept type), pointing at a **copy** of the same
   images whose captions never mention the trigger.

The second concept is what teaches the trigger to *gate*: training sees the subject both
with the trigger (learn it) and without (learn what "no trigger" looks like). Measured
effect on the course's style dataset: net-of-null gating ratio 0.30 → 1.67. Skipping it —
or putting the trigger in a caption prefix so it appears in 100% of captions — produces a
LoRA that applies on every prompt whether the trigger is present or not. The full
measurement method is covered in the course materials.

The `keep_tags_count: 1` setting is deliberate: it pins the trigger at tag position 0 if
you ever enable tag shuffling/dropout on the STANDARD concept.

Augmentation is off in both templates except one switch: the style template sets
`enable_random_flip: true` on **both** concepts. Every Round 2 style arm trained with it
— a mirrored frame is free augmentation for a look that has no handedness. The character
template keeps it `false`: a face, a logo, or lettering flipped is a different subject.
(The Round 1 character run enabled only random-rotate/brightness *toggles* at 0.0
strength, which is equivalent to off.) Whichever template you copy, the flip applies to
the prior concept as well, so the gating measurement stays paired.
