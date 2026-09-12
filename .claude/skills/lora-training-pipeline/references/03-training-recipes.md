# Phase 3 — Training recipes

Scripts: `course_lora_kit\cmd\03_train_character_sdxl.cmd`,
`03_train_style_sdxl.cmd`. Each layers OneTrainer's shipped SDXL preset under a sparse
course overlay (`course_lora_kit/configs/`) and forwards extra args, so one-off tweaks
are `--config-value KEY=VALUE`, not config edits. The kit trains SDXL only — no SD 1.5
track exists (removed 2026-09-12; it was never measured).

All code citations below are verified at OneTrainer commit `ee1ec47`.

## The character recipe (`character_sdxl.json`) — measured, not folklore

This is the exact config of the course's Round 1 winning arm (**A2**), which beat four
competing recipes on every measured axis — identity, cross-seed consistency,
flexibility, colour neutrality, and gating stability (see `round1-case-study.md`):

| Dial | Value | Notes |
| --- | --- | --- |
| `lora_rank` / `lora_alpha` | 16 / 16.0 (scale 1.0) | alpha = rank is the LoRA paper's own default (Hu et al. 2021 §4.1: "tuning α is roughly the same as tuning the learning rate … we simply set α to the first r we try") |
| `learning_rate` | 1e-4, AdamW | pairs with scale 1.0. See the alpha ÷ rank section below before changing either |
| `learning_rate_scheduler` | `CONSTANT` | |
| `learning_rate_warmup_steps` | 30.0 (literal steps) | explicit — the unset default is 200 steps (see "Two dials" below) |
| `loss_weight_fn` / `loss_weight_strength` | `MIN_SNR_GAMMA` / 5.0 | with offset noise below, the measured fix for a colour-channel collapse — arms without these two dials went magenta, arms with them stayed neutral (`round1-case-study.md`) |
| `offset_noise_weight` | 0.03 | |
| `train_dtype` | `BFLOAT_16` (LoRA weights fp32) | wider dynamic range than fp16, same memory, on RTX 30-series and up. Must be pinned: OneTrainer's default is `FLOAT_16` (`TrainConfig.py:1074`) and no shipped preset sets it |
| `layer_filter_preset` | `attn-mlp` | verified mapping: `attn-mlp` → `["attentions"]`, `attn-only` → `["attn"]`, `full` → `[]` = no restriction (`modules/modelSetup/BaseStableDiffusionXLSetup.py:40-42`). Do **not** use `full` for a LoRA that must load in a diffusers-based runtime — see the A3 arm in the case study. **The label alone does nothing**: the trainer builds its filter from `layer_filter` only (`modules/util/ModuleFilter.py:39-43`), and an empty `layer_filter` matches every layer. Always set `layer_filter: "attentions"` explicitly in the overlay (the kit's style overlay does since 2026-09-12; the first Round 3a launch trained 794 / 0 layers before this was caught) and confirm `Selected layers: 722 / Deselected layers: 72` in the first screen of the log |
| Text encoders | frozen (preset) | SDXL's dual TEs overfit fast on a small character set; gating comes from the contrastive concept, not TE training |
| `epochs` / `batch_size` | 44 / 4 → 1188 steps | on the measured Round 1 set: 55 curated images × 2 concepts = 110 items → 27 steps/epoch (drop-last) × 44; retune epochs by views-per-image (below) for a different dataset size |
| `save_every` / unit | 100 / `STEP` | mandatory for phases 4–5 — the default is no intermediate saves at all |
| Concepts | 2: `STANDARD` + `PRIOR_PREDICTION` | `contrastive-concept-gating.md` |

On the measured run, the pick was **not the final checkpoint**: step 1099 of 1188 won on
every axis; the finals were never even scored under readable names at first. Never plan
to ship "the last checkpoint" — plan to ship the checkpoint that screens best (phase 4/5).

## The style path (`style_sdxl.json`)

Keeps the shipped preset's core recipe intact — rank 16, **alpha 1.0** (scale 0.0625),
LR 3e-4, `attn-mlp` — because that exact config is what the course's style-LoRA numbers
(views-per-image window, ‖ΔW‖_F bands) were measured on. The overlay adds `epochs`
50 and `save_every` 100 `STEP`, pins `learning_rate_warmup_steps` at **30**,
overrides the preset's loss weighting: `loss_weight_fn` **`MIN_SNR_GAMMA`** (strength
5.0) + `offset_noise_weight` **0.03** — the same values the character recipe pins, now
measured on the style track too — and pins `train_dtype` / `fallback_train_dtype` to
**`BFLOAT_16`**, which the measured S0b run used and which nothing else in the chain
sets (OneTrainer's default is `FLOAT_16`; the fallback already defaults to bf16). Run
`00b_print_recipe.cmd` to see the merged result before training. The style concepts
template (`concepts_contrastive_template_style.json`) differs from the character one by
one switch: `enable_random_flip: true` on both concepts, as every Round 2 arm trained.
Both loss/warmup overrides carry measured provenance:

Warmup provenance: the original style measurements ran with the inherited default of
200 folded in, and the overlays initially pinned that 200 for fidelity. Round 2 then
measured the default directly as ablation arm S1 (same dataset, one delta): 200 of
1,200 steps spent ramping suppressed early weight movement by 56–66% at matched
checkpoints, doubled the growth knee, and the deficit never closed — the endpoint sat
where the control had been ~50 steps earlier. On a short style run the 200 is pure
waste, so the overlays now pin 30 (see `round2-style-case-study.md`).

Loss-weighting provenance: the original measurements also ran with the preset's
`CONSTANT` / 0.0 folded in, and the overlays initially preserved that for fidelity. It
failed in live batches: Round 2's control (S0) passed every weight-space instrument,
then drifted **magenta stochastically per seed** — one fixed-prompt batch of 4 seeds
gave two clean, one drifting, one fully pink. A corrected re-run (S0b) changing only
these two dials halved the mean-CIELAB-a* drift at every matched checkpoint (worst
case 31.0 → 12.0), never entered the saturated-pink regime, and rendered 4/4 coherent
in the same batch test. The two dials were changed together in both rounds — the split
between min-SNR and offset noise is unmeasured (see `round2-style-case-study.md`).

## The preset, decoded (with kohya equivalents)

A preset file is a sparse overlay, not a complete recipe — `rank`, `alpha`, and `AdamW`
aren't in the preset JSON at all; they're OneTrainer's defaults
(`TrainConfig.default_values()`) that the preset simply doesn't override.

| OneTrainer field (tab) | Preset value | kohya equivalent | Note |
| --- | --- | --- | --- |
| rank (LoRA) | 16 | `network_dim` | 16–32 is the working range for styles; bigger is not better. The paper's own ablation (§7.2, Table 6) shows accuracy essentially flat r=1→64; its H.2 optimal range is 4–16 |
| alpha (LoRA) | 1.0 | `network_alpha` | effective strength scales with alpha ÷ rank — see below |
| Learning Rate | 3e-4 | `learning_rate` | deliberately high *because* alpha is 1.0 — the two travel together |
| Epochs | 100 | `max_train_epochs` | a baseline default, not tuned — set by views-per-image (below) |
| Optimizer | AdamW | `optimizer_type` | Prodigy alternative below |
| Resolution | 1024 | `resolution` | aspect bucketing handles non-square |
| Batch Size | 4 | `train_batch_size` | first dial to lower when VRAM runs out; 1 is fine, just slower |
| Loss Weight Function | Constant | ≈ `--min_snr_gamma` | the preset leaves min-SNR off; **both** course recipes turn it on for measured reasons (Round 1 character, Round 2 S0b style — the preset's Constant let colour drift magenta in live batches) |
| Offset Noise Weight | 0.0 | `--noise_offset` | off in the preset; 0.03 in both course recipes (measured — see the style overlay's loss-weighting provenance above) |
| Train Data Type | float16 | `--mixed_precision` | bf16 is the safer choice on RTX 30-series+; **unpinned, this preset trains fp16** — both course overlays now pin `BFLOAT_16` |
| LoRA Weight Data Type | float32 | — | OneTrainer's own default; adapter weights stay fp32 regardless of train dtype |
| EMA | Off | — | turning it on tends to *reduce* output diversity — against a single-concept LoRA's goal |
| LoRA Decompose / DoRA | Off | LyCORIS `--dora_wd` | a DoRA checkpoint carries an extra `dora_scale` key per module — `checkpoint_norm_analyzer.py` detects and calls it out |
| Text encoders | frozen | `network_train_unet_only` | set by the SDXL preset; OneTrainer's own default would train the TE, capped at 30 epochs |
| Output Format | Kohya | — | single `.safetensors`; diffusers-based components load it |

## The alpha ÷ rank rule (read before copying any community recipe)

OneTrainer and kohya share the same convention: the learned change is scaled by
**alpha ÷ rank** — `modules/module/LoRAModule.py:581` scales by `alpha / self.rank` on
every forward pass (LoHa/LoKr identically at `:289`/`:478`). Any claim that OneTrainer's
alpha is a "raw multiplier" unrelated to rank is wrong at this commit; trust the code.

Two internally-consistent conventions exist — **never mix them**:

- Preset/style: alpha 1.0 (scale 1/16 at rank 16) + **high** LR (3e-4).
- Community/character: alpha = rank (scale 1.0) + **low** LR (~1e-4).

Copying a community alpha = rank into a config that keeps LR 3e-4 adjusts one variable
twice in opposite directions. Round 1's A1 arm did exactly this in reverse (alpha 1.0 at
3e-4 vs clean arms' equivalents) and produced a badly *rotated* low-magnitude update
(endpoint cosine 0.045 vs the clean arms' 0.14–0.25). Fixing alpha/LR together is
necessary — but the case study shows it wasn't sufficient for every symptom; read
`round1-case-study.md` before diagnosing any collapse as "alpha/LR."

## Prodigy, in full

Learning Rate 1.0 is necessary but not the whole recipe: β1 0.9 / β2 0.99 (0.99 is the
more commonly cited β2 for Prodigy), weight decay 0.01, `decouple` +
`use_bias_correction` + `safeguard_warmup` all on, `d_coef` ≈ 0.6 (Prodigy's confidence
multiplier on its self-estimated rate — lower is more conservative), a cosine LR
schedule, and `learning_rate_warmup_steps` 0 (Prodigy already ramps gently on its own).
Round 1's Prodigy arm (A4) trained healthy but lost to A2 — and its own *earlier*
checkpoint beat its later one on render metrics, so sweep and screen rather than
assuming more steps help.

## Two dials the preset doesn't set (both silently break later phases)

- **`learning_rate_warmup_steps` defaults to 200 — a literal step count, not a
  fraction.** OneTrainer reads a value ≤ 1 as a *fraction* of total steps and > 1 as
  literal steps (`create.py:1127`); the default is 200 (`TrainConfig.py:1062`) and
  neither shipped preset overrides it. On a 300-step style run that's two thirds of the
  run spent ramping — a fact folded invisibly into the original measured views-per-image
  numbers, and later priced directly by Round 2's S1 arm (56–66% early-learning
  suppression that never recovers — `round2-style-case-study.md`). Set it explicitly:
  30–60 for a 1000–1500-step run (all kit overlays now pin 30). The ≤1-fraction form is
  a reasonable adaptation when total steps vary a lot — but it is *not* what the
  measured runs used, so treat it as an adaptation, not the recipe.
- **`save_every`/`save_every_unit` default to `0`/`NEVER`** (`TrainConfig.py:1276-1277`)
  — intermediate checkpoints are **off** by default, and every screening/sweep technique
  in phases 4–5 depends on having more than one checkpoint. The kit overlays set
  100/`STEP`. (Backups, on the backup tab, are for *resuming a run* — they are neither
  your deliverable nor your sweep.)

## Set epochs by views per image, not a steps target

A steps total means nothing without dataset size. What the model experiences is
**views per image** = `(steps × batch size) ÷ images`. Measured across three runs on the
course's 21-image style dataset (batch 4): the healthy window is **≈40–60 views/image**
— clean at 38 and 57 views, fully memorized and prompt-independent at 362. The collapsed
run changed *nothing* but epochs (380 vs 60) against the clean one. Epoch count tracks
views/image close to 1:1 (an epoch is one pass over every image), so target ~50
views/image ⇒ start near epoch 50 regardless of image count or batch size.

**Scoping:** the 40–60 window was measured at rank 16 / alpha 1.0 / `attn-mlp` /
LR 3e-4 / warmup 200, 21-image *style* dataset. The transferable claim is the *shape* — a
knee somewhere around 40–60ish views, far below the several-hundred-view collapse — not
the exact numbers. The character recipe's 44 epochs × its dataset landed in the same
spirit (measured directly rather than derived).

**Round 2 correction — and its own correction.** Re-run on a 48-image contrastive
style set, the usable window first *appeared* to end near **epoch 20** (step ~500) —
but that reading didn't survive the round: the late-checkpoint "washout" was the
`CONSTANT`/0.0 colour drift (loss-weighting provenance above), not overtraining. Under
the corrected loss weighting the working pick landed at **step 1099 = epoch 45 ≈ 46
views/image — inside the original 40–60 window after all.** What does survive: ‖ΔW‖_F
grows with **optimizer steps**, so a bigger dataset takes more steps per epoch and
absolute norm bands are local to the step counts they were measured at — and to the
loss weighting (the corrected run's healthy pick reads ‖ΔW‖_F **8.1**, across the old
"collapsed" ≥6.70 band). Practical rule unchanged: plan epochs as a checkpoint budget
(~50 is fine), keep `save_every` 100, and let the phase-4 norm screen and render sweep
find where *your* dataset's window ends (`round2-style-case-study.md`).

## Wall-clock

Measured: ~1.3–1.4 s/step at batch 4, 1024px, no OOM on a high-VRAM card (Round 2's
48-image contrastive runs measured 1.62 s/step — 32.5 min per 1,200-step arm). A
300-step style run ≈ six-seven minutes; the 1188-step character run under half an hour.
Mid-range hardware should plan for longer — but it's an "over lunch" job, not overnight.

## CLI layering (what the `.cmd` scripts actually run)

```
venv\Scripts\python.exe scripts\train.py ^
    --preset-path "training_presets\SDXL\#sdxl 1.0 LoRA.json" ^
    --config-path course_lora_kit\configs\character_sdxl.json ^
    [--config-value KEY=VALUE ...]
```

Preset first, overlay second, `--config-value` overrides last. Flag syntax and gotchas:
`cli-cheatsheet.md`.
