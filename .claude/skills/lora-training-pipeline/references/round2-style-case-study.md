# Round 2 case study — the style ablation (what each rule protects you from)

Six style-LoRA arms designed on one 48-image animated-series style dataset (plus its
contrastive copy), all sharing the kit's style recipe — rank 16 / alpha 1.0 /
LR 3e-4 / `attn-mlp` / warmup 30 / epochs 50 / batch 4, two concepts, one seed — with
each violation arm changing **exactly one** dial (config-diff verified). Five arms
trained and measured in a single sitting (2026-08-21); the overtrain arm is designed
but not yet run; a live failure then forced an unplanned seventh training — **S0b**,
the corrected control (2026-08-21/22, its own section below). Every number below
traces to the round's scoresheet
(`round2_style_scoresheet.md` on the measurement machine). Load this file when a user
asks what a style-track rule is protecting them from, why the style overlays pin
warmup 30, or why the norm bands can't be quoted as absolutes.

## The arms

| Arm | The one delta | Rule it violates | Fate |
| --- | --- | --- | --- |
| S0 control | — | none | clean and gated; the anchor every arm is diffed against |
| S1 warmup-200 | `learning_rate_warmup_steps` 30 → 200 (OneTrainer's unset default, shipped by the kit's style overlays until this round) | set warmup explicitly | undertrained at every matched checkpoint; deficit never closed |
| S2 dataset | pixels only: same captions, every image padded with symmetric pillarbox bars | crop letterbox/pillarbox | bars trained into the style; only the pre-flight profiler caught it cheaply |
| S3 lr-alpha | `lora_alpha` 1.0 → 16.0, LR left at 3e-4 | never mix the alpha/LR conventions | destroyed by step 199; broke the norm screen's selection rule too |
| S4 overtrain | `epochs` 50 → 360 | the views window | **designed, not yet run** — collapse end anchored by the original 362-view collapse. Its staged config predates S0b (still `CONSTANT` / 0.0); rebuild it on the S0b base before running so the one delta stays *epochs* |
| S5 caption-notrigger | the `PRIOR_PREDICTION` concept deleted (trigger in 100% of captions, no counterweight) | contrastive gating design | trigger went **inert** — prediction confirmed in number, inverted in mechanism |
| S0b corrected control | `loss_weight_fn` CONSTANT → `MIN_SNR_GAMMA` (5.0) + `offset_noise_weight` 0.0 → 0.03 (the character recipe's pair — two dials, changed together as a unit) | none — it *fixes* one: the preset's loss weighting | drift halved, batch 4/4 clean; the working pick of the round (step 1099) |

Deliberate non-delta: S5 runs ~half the steps/epoch of S0 (12 vs 24 — one concept
instead of two). The controlled variable is epochs = views-per-image, not step count.

## Headline finding — ‖ΔW‖_F tracks optimizer steps, not views

The reference bands (clean ≤ 3.26 / collapsed ≥ 6.70, measured on a 21-image set) did
not transfer as absolutes: S0's perfectly healthy endpoint hit **10.71** at 1,200 steps
with a clean adjacent-cosine signature throughout, and the sweep showed the late
checkpoints washing out. At 48 images × 2 concepts, 50 epochs is 1,200 steps — and the
norm grows with **steps**. The clean band first appeared to end near **step 500 =
epoch 20** on this set — a reading S0b later overturned (the "washout" was colour
drift, not overtraining; see the S0b section). Consequences, in kit terms:

- The absolute bands are valid near the step counts they were measured at; the **knee
  flag** and the screen-then-sweep workflow are what transfer.
- Plan epochs as a checkpoint budget, not a training target; the phase-4 screen plus
  the render sweep find the real window.
- A big terminal norm alone cannot diagnose overtraining (S0's healthy endpoint crosses
  the "collapsed" number) — S4's decisive instruments, when it runs, are the render
  sweep and gating (and `color_stats.py`, so drift is not misread as washout again).

## S1 — the shipped warmup default, priced

200 of 1,200 steps spent ramping: early weight movement suppressed **56–66%** vs S0 at
matched checkpoints (step 99: ‖ΔW‖_F 0.124 vs 0.366), the growth knee doubled (flags at
steps 199 *and* 299), the update direction settled later (adjacent cosine 0.545/0.664
vs S0's 0.642/0.875 at the same steps), and the run never caught up — endpoint 10.21 vs
10.71, i.e. sitting where S0 had been ~50 steps earlier, with gating slightly under
S0's (2.75 vs 3.01 net-of-null). This measurement is why the style overlays now pin
warmup **30** (the character overlay always did). Contrast with Round 1's A1c finding —
warmup 200-vs-30 was irrelevant to the *colour-collapse* signals there; this round
prices what it does cost: early progress on a short run.

## S2 — bars in, bars out (tool asymmetry)

Norms blind (trajectory statistically identical to S0: 3.10 vs 3.24 at the shortlist
step), gating blind (3.02 vs 3.01) — but `dataset_hygiene_profiler.py` flagged the
spoiled folder **48/48 for pillarbox before a single training step**, and the renders
confirmed it: symmetric colour-banded bars at both margins from ~step 700, a vertical
striping rhythm through the whole composition, saturation running hot (triggered-render
saturation 207–235 vs S0's ~160). One rule, three instruments, and only the cheapest
pre-flight one catches it before GPU time is spent.

## S3 — mixed conventions break the training *and* the instruments

Alpha 16 with LR still 3e-4 is a 16× effective step. The first checkpoint (99 steps)
had already moved 68% of S0's total 1,200-step distance; steps 199–299 rendered as
blown-out solid magenta; the run oscillated and never stabilized; even the
least-damaged save gated at half of S0 (1.63). **Bonus casualty: the norm screen's
selection rule.** The analyzer's alpha-convention projection labeled S3's trajectory
"clean-range" while the renders showed destruction — the first-order α/r projection
assumes dynamics scale linearly with alpha, and a 16×-effective-step run is not linear.
The "last clean-band checkpoint" rule is only valid *within* the recipe's own alpha
convention; S3's shortlist had to come from the render sweep instead.

## S5 — prediction inverted, rule confirmed

Pre-registered: with no contrastive concept, the style fires with the trigger absent
(leak — the historical prefix-trap shape, net-of-null ~0.30 on the original dataset).
Observed: the opposite mask. S5's *triggered* render looks like S0's *untriggered*
control — the trigger stopped summoning the style — while S5's weights moved ~1.9×
faster per step than S0's (no null concept spending gradient on holding the prior). The
style bound itself to the shared caption vocabulary instead of concentrating on the
trigger, so on out-of-domain prompts the trigger carries nothing. The gating instrument
caught it identically: **0.28 net-of-null vs the historical 0.30 floor**, naive ratio
0.97 ≈ "the trigger adds token noise, nothing more." Teach the mechanism honestly: the
`PRIOR_PREDICTION` null doesn't just *prevent leakage* — it **concentrates** the style
onto the trigger. Leak and dilution are the same failure read from opposite ends, and
the same measurement catches both.

## S0b — the corrected control (the drift no weight-space instrument could see)

S0 passed every instrument in the round, then failed in live batches: **stochastic
magenta drift** — per-seed, not per-checkpoint. One fixed-prompt batch of 4 seeds: two
clean, one drifting, one fully pink. Mechanism: per-pixel i.i.d. training noise has
near-zero power in an image's global mean (the DC component), so the model never
learns to *correct* whole-image colour shifts — a fine-tune that disturbs that channel
drifts freely, seed by seed. Offset noise (per-channel constant noise) makes the
global mean denoisable; min-SNR-γ caps the loss weight at min(SNR, γ=5), shifting
gradient share toward the high-noise timesteps where global colour and composition are
decided — timesteps `CONSTANT` under-trains.

S0b re-ran the control with exactly the character recipe's two loss-weighting dials —
`loss_weight_fn` CONSTANT → **`MIN_SNR_GAMMA`** (strength 5.0), `offset_noise_weight`
0.0 → **0.03** — and nothing else. Measured with `course_lora_kit/scripts/color_stats.py`
(image-mean CIELAB a* per sweep render, magenta > 0; no-LoRA controls sit at 0.6–1.5):

- **Drift halved at every matched checkpoint** — step 499: 28.0 → 12.9; 999:
  30.1 → 11.7; 1099: 31.0 → 12.0 — and the saturated-pink regime (a* ≥ 28) was never
  entered: S0b's ceiling is ~13 across all 13 sweep renders.
- **The same 4-seed batch test: 4/4 coherent, 0 pink** (S0: two clean, one drifting,
  one fully pink) — the decisive metric, since the drift is per-seed.
- **Everything else held:** endpoint direction unchanged (cos-vs-final 0.111 vs S0's
  0.110), magnitude ~16% slower, gating net-of-null 2.14 (s499) / 1.73 (s1099) vs the
  0.28 failure floor.
- **Honesty note:** the two dials were changed together in both rounds — the split
  between min-SNR and offset noise is not measured. Treat them as a pair.

The round's working pick is **S0b step 1099 = epoch 45 ≈ 46 views/image**. Two
upstream corrections follow: (1) the "usable window ends ~epoch 20" reading was drift
masquerading as overtraining washout — the corrected pick lands *inside the original
40–60 views window after all*; (2) the norm bands gain a third relativity axis: the
healthy pick reads ‖ΔW‖_F **8.1**, across the old "collapsed" ≥6.70 band, so bands
don't survive a loss-weighting change either. This measurement is why the style
overlays now pin `MIN_SNR_GAMMA`/0.03 (the character overlay always did). Meta-lesson:
norms, cosines, and gating were all colour-blind to the failure — **screen with
weights; validate with renders, plural, across seeds.**

## Scoping (read before quoting any number)

- Norm bands are **step-count-relative and loss-weighting-relative** (this round's
  findings — S0's healthy 10.71 and S0b's healthy 8.1 both cross the old ≥6.70 band);
  quote them only with the step counts *and recipe* they were measured at.
- The clean gating value is **dataset- and recipe-relative**: ~1.6–1.7 on the original
  21-image set, **3.01** on this one under CONSTANT, **2.14 / 1.73** under the
  corrected loss weighting. What transfers is the reading, not the value — and the
  ~0.30 floor ("trigger inert / non-gating") reproduced across both datasets.
- Wall-clock: 1.62 s/step at batch 4, 1024-class buckets → 32.5 min per 1,200-step arm.
- All render measurements: SDXL 1.0 base, Euler-a 25 steps, CFG 7.0, fixed seed, LoRA
  weight 1.0, untriggered control per render.
