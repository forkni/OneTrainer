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
| `epochs` / `batch_size` | 44 / 4 | 1188 steps on the measured ~48+48-image two-concept dataset; retune epochs to your dataset by views-per-image: `(steps × batch) ÷ images`, aim for ~40–60 |
| `save_every` / unit | 100 / `STEP` | intermediate checkpoints are OFF by default; without them there is nothing to screen in phase 4 |

## `style_sdxl.json` / `style_sd15.json` — preset + course overrides

The shipped preset's recipe (rank 16 / alpha 1.0 / LR 3e-4) is left intact — that exact
config is what the course's style-LoRA numbers were measured on. The overlay only adds
what the preset leaves dangerously unset: `epochs` 50 (start of the measured ~40–60
views-per-image window; retune to your dataset), `save_every` 100 steps, and
`learning_rate_warmup_steps` pinned to the same 200 the measurements ran with (explicit
instead of inherited, so changing it is a visible decision).

## `concepts_contrastive_template.json` — copy, don't edit in place

Copy it to `training_concepts/` (gitignored, i.e. yours) under the name the training
script expects — `character_concepts.json` or `style_concepts.json` — then edit the two
`"path"` fields.

It defines **two concepts over the same images**:

1. `my_subject` — `type: STANDARD`, your captioned dataset, trigger word leading every
   caption.
2. `my_subject_notrigger_prior` — `type: PRIOR_PREDICTION` (OneTrainer's
   prior-preservation concept type), pointing at a **copy** of the same images whose
   captions never mention the trigger.

The second concept is what teaches the trigger to *gate*: training sees the subject both
with the trigger (learn it) and without (learn what "no trigger" looks like). Measured
effect on the course's style dataset: net-of-null gating ratio 0.30 → 1.67. Skipping it —
or putting the trigger in a caption prefix so it appears in 100% of captions — produces a
LoRA that applies on every prompt whether the trigger is present or not. The full
measurement method is covered in the course materials.

The `keep_tags_count: 1` setting is deliberate: it pins the trigger at tag position 0 if
you ever enable tag shuffling/dropout on the STANDARD concept. The template ships with
augmentation off; the measured run enabled only random-rotate/brightness *toggles* at 0.0
strength, which is equivalent.
