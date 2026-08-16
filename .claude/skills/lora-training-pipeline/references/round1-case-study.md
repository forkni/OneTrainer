# Round 1 case study — where the character recipe comes from

Four competing character-LoRA recipes ("arms") trained on the same ~48-image anime
character dataset (plus its contrastive copy), 1188 steps each, checkpoints every 100
steps, scored with the exact scripts this kit ships. Two follow-up isolation retrains
turned a collapse into a root cause. This file is the "why" behind every dial in
`configs/character_sdxl.json` — load it when a user asks why the recipe is what it is,
hits a colour cast, or wants to deviate.

## The arms

| Arm | Recipe | Fate |
| --- | --- | --- |
| A1 | style-preset baseline: rank 16 / **alpha 1.0** (scale 0.0625), **LR 3e-4**, `CONSTANT` loss weight, **no offset noise**, warmup 200 | collapsed — dropped |
| A2 | kohya-conventional: rank 16 / alpha 16 (scale 1.0), LR 1e-4, MIN_SNR_GAMMA 5.0, offset noise 0.03, warmup 30 | **winner — the shipped recipe** |
| A3 | rank 32, `layer_filter=""` (full — no restriction) | unloadable in diffusers — dropped |
| A4 | Prodigy (LR 1.0, cosine, full Prodigy config) | healthy, lost to A2 |

## A1's collapse — and the two-cause misdiagnosis

A1 collapsed visibly: a magenta cast growing dose-response with ‖ΔW‖_F (R−G +4.6 →
+70.2 → +108.2 → +134.3 across steps 199→1099; green is the channel that collapses),
cross-seed consistency cratering 0.738 → 0.336 by step 1099, norm sweep 0.115 → 9.877
with endpoint cosine **0.045** — a badly rotated update. No A1 checkpoint was ever both
neutral and identity-bearing. Note DINOv2 identity scored the cast renders as good
likeness (0.451/0.468) — the identity metric is colour-tolerant; only the per-channel
colour data and the norm analyzer's "between reference bands" call on A1_499 caught it.

A1's recipe had two problems at once: alpha 1.0 at rank 16 (16× below the LoRA paper's
alpha = rank default) *and* a 3× higher LR than the clean arms — two knobs the paper
says are "roughly the same" knob, pushed in opposite directions. The obvious fix was to
correct both. Two retrains tested that hypothesis properly:

- **A1b** (alpha 16, LR 1e-4 — only the alpha/LR confound fixed): came back **clean on
  every weight-only diagnostic** — smooth decelerating ‖ΔW‖_F (0.899 → 17.403, tracking
  A2's almost exactly), single early knee, endpoint cosine 0.143 vs A1's 0.045. Judged
  on the rendering-free screen alone, the fix worked. **Rendered, it did not**: the same
  magenta cast, slightly worse (R−G +9.7 → +149.7 → +132.5 → +150.7). It also ran the
  hottest gating in the round (net-of-null 2.32 out-of-domain, 3.49 in-domain). The norm
  analyzer is blind to colour collapse — a metric confirming one hypothesis is not the
  same as ruling out a second one.
- **A1c** (A1b + only `loss_weight_fn CONSTANT → MIN_SNR_GAMMA` and
  `offset_noise_weight 0.0 → 0.03` flipped; warmup deliberately left at 200, not A2's
  30): **neutral** — R−G +2.6 → +0.7 → −23.0 → −4.0, with A1c_799's −23.0 landing
  within 0.4 of A2_799's −23.41. Gating converged onto the clean band too (net-of-null
  1.01 vs A2's 1.07, A4's 0.99). Since A1c still differs from A2 on warmup and came back
  clean anyway, this **isolates the cause to `loss_weight_fn`/`offset_noise_weight`** —
  warmup 200-vs-30 confirmed irrelevant to every signal checked.

Mechanistically: MIN_SNR_GAMMA down-weights loss at extreme-noise timesteps, where the
model can cheaply reduce loss by biasing toward an easy colour/luminance shortcut;
CONSTANT weighting doesn't discourage that, and offset noise attacks the same
mid-gray/luminance-bias class of problem. That's why both dials are in the shipped
recipe, and why the alpha/LR rule is taught as *necessary but not sufficient*.

## A3 — the loader trap

A3's `layer_filter=""` includes conv/resnet and SGM-only modules. Its checkpoints
**load fine in ComfyUI's native `LoraLoader`** but cannot be loaded by diffusers'
`_maybe_map_sgm_blocks_to_diffusers()` (chokes on keys like `lora_unet_label_emb_0_0`),
which means every diffusers-based validation script fails on it — and a diffusers-based
real-time component logs an `ERROR` and then **silently continues on the plain base
model**, with nothing in the render to flag it. It was only ever scorable by the
(conv-aware) norm analyzer, and even those numbers weren't comparable: at rank 32 it
sits on bands calibrated at rank 16, and the LoRA paper's Table 7 shows ‖ΔW‖_F falls as
rank rises. Dropped. This is why the shipped config pins `attn-mlp` and warns against
`full`. (On rank itself: the paper's §7.2 shows accuracy essentially flat r=1→64, and
its §7.1 says adapting more weight-matrix *types* beats raising rank — capacity was
never A3's promise anyway.)

## A4 — Prodigy: healthy, but a step-selection lesson

A4 trained clean (fast convergence — 13× norm growth by step 199, near-flat after; no
collapse). It lost to A2 on the final table, and its own *earlier* checkpoint beat its
later one on render metrics (cross-seed 0.651 → 0.608, flexibility 0.655 → 0.589 from
499 → 1099) — with self-tuning optimizers especially, more steps are not more better.

## The final table and the pick

Per-arm 499 → 1099, one render sitting (bold = better step):

| | A2 | A1c | A4 |
| --- | --- | --- | --- |
| identity | 0.341 → **0.359** | **0.365** → 0.354 | 0.337 → **0.343** |
| cross-seed | 0.641 → **0.732** | 0.675 → **0.756** | **0.651** → 0.608 |
| flexibility | 0.710 → **0.718** | 0.611 → **0.738** | **0.655** → 0.589 |
| B−G (0 = neutral) | +39.9 → **+21.8** | +40.7 → **+19.7** | +40.5 → **+26.8** |
| gating net-of-null OOD | 1.070 → 1.126 (+5%) | 0.990 → 1.327 (**+34%**) | 0.978 → 1.031 |

**A2 at step 1099 wins every axis measured — render and gating both** — and is what
shipped (`lain_A2_s1099.safetensors`). A1c ties or edges A2 on some render metrics
(best-in-table cross-seed 0.756) but is the arm whose gating degrades most at 1099; if
ever staged, A1c_499 is the safer step despite weaker render numbers. The general
lessons the recipe encodes:

1. The winning checkpoint was **not the final one** (1099 of 1188) — sweep and screen,
   never ship "the last checkpoint" by default.
2. An initial step pick made for cross-arm comparability (499) was **revisited per-arm**
   once broken arms were dropped — and the revisit changed the answer for A2 and would
   have changed it wrongly for A1c without the gating re-measure.
3. Weight-only diagnostics, render metrics, colour checks, and gating are **four
   independent axes**; each caught a failure the others missed (A1b clean-on-norms but
   magenta; A1 cast scored as likeness by DINOv2; A1c render-best but gating-worst at
   1099).

## Measurement hygiene notes (why these numbers are trustworthy)

- All cross-arm comparisons come from single-sitting batches; cross-session comparisons
  were only trusted after an anchor label reproduced (A2_499 gating 1.070/1.658 vs the
  prior session's 1.07/1.64). A1's colour collapse also reproduced almost exactly across
  environments (+4.57 → +134.27 vs +4.6 → +134.3) — the colour signal is robust even
  when absolute identity scores drift.
- The no-LoRA control scored identity 0.141, below every trained checkpoint at every
  step — the harness measures the LoRA, not itself.
