# Phase 4 — Screening a checkpoint without rendering it

Script: `course_lora_kit\cmd\04_checkpoint_screen.cmd <checkpoint_dir_or_file> [...]`
(wraps `course_lora_kit/scripts/checkpoint_norm_analyzer.py`). No GPU, no base model —
reads the `.safetensors` files directly; a full sweep screens in under a minute. The
script prints its own `Column key:` (`‖dW‖_F`, `rank/a/scale`, `cos_prev`, `growth`,
flags `KNEE` / `HOT-START` / `CONVENTION`; `band` only with `--bands`) at the end of every run; the course's Appendix H mirrors the same
definitions in *What the columns mean*, grouped with the other scripts' columns.

## What it measures

**‖ΔW‖_F** — the aggregate Frobenius norm of the weight delta the LoRA adds — computed
straight from the `lora_down`/`lora_up`/`alpha` factors via a trace identity (never
materializing the full delta matrix), plus the **cosine similarity** between adjacent
checkpoints' deltas and a **knee flag** (a jump in the norm's own growth ratio above
1.5× the sweep's median). Each checkpoint's line also prints its own rank/alpha/scale.
Conv2d LoRA modules (wide layer filters) are handled by an exact reshape; DoRA
checkpoints and mixed-rank sweeps are detected and called out rather than silently
mis-measured. Pass several directories to compare arms — each is analyzed as its own
independent group, not merged into one sweep.

## The measured reference bands — and their exact scope

**At rank 16, alpha 1.0, `attn-mlp` (scale 0.0625) only** — the style config:

- Every checkpoint that stayed clean and prompt-responsive under the component's live
  `Weight` slider measured **‖ΔW‖_F ≤ 3.26**.
- Every checkpoint that had collapsed into a memorized, prompt-independent stamp
  measured **≥ 6.70**. Between the bands = suspect (Round 1's A1_499 read 3.55,
  "between reference bands" — and it was indeed already cast-ridden).
- Adjacent checkpoints in a sweep run ~**0.85** cosine; first-vs-last of the collapsed
  sweep ran **0.19** — past the healthy window, the edit *rotates* toward a degenerate
  direction rather than just growing.
- Growth is nonlinear: roughly steps^2.5 early, then closer to linear — a checkpoint at
  twice the steps is not "twice the LoRA."

**These absolute numbers do not transfer to a different rank/alpha.** An alpha = rank
config (the character recipe) lands roughly **16× higher** on raw ‖ΔW‖_F at the same
effective strength, purely from the scale term — A2's healthy sweep runs 2.46 → 17.05
and never collapses. With `--bands` the script projects each norm onto the reference scale
(`norm × (0.0625 ÷ scale)`) before applying the bands, documented as a first-order
approximation unverified at other scales; since 2026-09-12 the bands are off by default
(Round 3's clean runs all crossed 6.70 with no collapsed checkpoint to re-measure against).
**Use the flags as the config-independent signal**: `KNEE`, `CONVENTION` (file rank/alpha vs
the recipe's, `--recipe <merged recipe json>`, default 16 / 1) and `HOT-START` (first save ≥ 4×
the recipe's reference first save 0.35 on the *raw* norm, growth never accelerating after —
the projection divides a hot start back out, which is why the flag reads raw). Treat the
projected band as a secondary hint. (It doesn't correct for *rank*
either — the LoRA paper's Table 7 shows ‖ΔW‖_F falls as rank rises at comparable task
performance, so a rank-32 arm's numbers aren't comparable to rank-16 bands on two axes.)

**They are step-count- and loss-weighting-relative too.** Round 2's control arm S0 ended
perfectly healthy at **10.71** after 1,200 steps, and the S0b working pick (min-SNR +
offset noise) reads **8.1** — both across the old ≥ 6.70 "collapsed" line. Quote a band
only with the step count *and recipe* it was measured at; the knee flag is the signal that
survives. Full finding: `round2-style-case-study.md`.

## `Weight` can't fix an overcooked checkpoint

The obvious question with a live `Weight` dial: if a checkpoint overcooked, just turn it
down? Measured answer: **no.** The 0.19 first-vs-last cosine above is why — dialing
`Weight` down on an overcooked checkpoint scales a vector that's already pointed
somewhere else; the result is a *faint* version of the degenerate style, not a
restrained version of the good one. Past the window, the fix is a different checkpoint,
not a lower `Weight`.

## Collapse has a visible signature — the magenta cast

On Round 1's A1 arm (the exact config the bands are calibrated on), collapse also showed
as a per-channel colour shift growing in lockstep with the norm — a dose-response curve,
not a sudden artifact:

| checkpoint | ‖ΔW‖_F | R−G | B−G | look |
| --- | --- | --- | --- | --- |
| no LoRA | — | +4.4 | +2.5 | neutral |
| step 199 | 0.48 | +4.6 | +1.1 | neutral |
| step 499 | 3.55 | +70.2 | +71.4 | mauve cast, identity still readable |
| step 799 | 6.53 | +108.2 | +103.0 | flat magenta |
| step 1099 | 9.88 | +134.3 | +119.2 | flat fuchsia, banding |

The channel that collapses is **green** — not a red/blue channel swap (a swap is a fixed
permutation that would leave green untouched; this tracks training step). Two other arms
in the same run and harness show no drift, so the cause is the arm's weights, not the
render pipeline. Mechanically, an `attentions`-filtered LoRA never touches the VAE or
conv path — it can't shift colour directly, but it pushes the UNet's latents far enough
out of distribution that whatever the VAE decodes reads as a cast. Past the clean band,
expect the cast to get *worse*, not subtler — the same "rotated, not just stronger"
behavior the cosine numbers describe.

Bonus tell: DINOv2 identity cosine is largely colour-tolerant — it scored A1's cast
renders as good likeness (0.451/0.468). A per-channel R−G/B−G check over a handful of
renders (against a no-LoRA baseline of roughly +4/+2) catches what the identity metric
hides.

## What this screen cannot see (the A1b lesson)

The norm analyzer detects **rotation and magnitude** pathologies — it is **blind to
colour-channel collapse.** Round 1's A1b retrain came back clean on *every* weight-only
diagnostic (smooth decelerating ‖ΔW‖_F, single early knee, endpoint cosine 0.143 vs
A1's 0.045) yet rendered the same magenta cast, slightly worse (R−G +9.7 → +149.7 →
+132.5 → +150.7). The cause was two dials the weight metrics don't respond to
(`loss_weight_fn`, `offset_noise_weight`) — full story in `round1-case-study.md`.

**A rendering-free pass is necessary, not sufficient.** Run it first — it's free and
catches rotation/overcooking before you burn GPU time — then confirm the survivors with
the phase-5 render checks (`05-validation-scripts.md`), including a colour check if
your renders look even slightly tinted.
