# The contrastive concept — teaching the trigger to gate

Template: `course_lora_kit/configs/concepts_contrastive_template.json` (copy into
`training_concepts/`, edit paths). This is the method behind the course's biggest single
measured improvement, and it applies identically to style and character LoRAs.

## Why a trigger fails by default

A trigger word only means something if training also saw the same concept *without* it.
If the trigger appears in 100% of captions (the classic mistake: putting it in the
captioner's `Prefix` field), training has no negative examples to contrast against — the
cheapest loss is to shift the whole style unconditionally, trigger or not. Measured on
the course's 21-image style dataset: net-of-null gating ratio **0.30** — the LoRA
applied almost as strongly with the trigger absent as present.

## The method

Add a **second concept** in the concepts file pointing at a *copy* of the same images
whose captions never mention the trigger — plain description only. In the measured
config the second concept has **`type: PRIOR_PREDICTION`** (OneTrainer's
prior-preservation concept type — the first concept stays `STANDARD`), same seed as the
first, same everything else. Training now sees the concept both with the trigger (learn
it) and without (learn what "no trigger" looks like), and the loss has something to gate
against. The template ships this exact two-concept shape, taken from the winning A2 run
(`keep_tags_count: 1` on both — see pitfall below — and seed 424242 on both).

## Measured result

Net-of-null gating ratio went **0.30 → 1.67** — more than fivefold — and `d_leak` (how
much the style leaks into prompts that never mention the trigger) fell **66–68%** on two
of three out-of-domain test prompts, while `d_gate` (strength *with* the trigger) held
or grew on the same prompts. Those two moving in opposite directions is the tell: a LoRA
that had simply gotten weaker would drag both down together.

Honest caveat: gating is prompt-dependent. The third out-of-domain prompt (`ramen_bowl`)
stayed near-inert before *and* after — some prompts just don't invite the style
regardless of the trigger. Two prompts improving is real; the one that didn't is too.

## The net-of-null rule (how to measure honestly)

Inserting *any* token into a caption shifts SDXL's cross-attention somewhat, even at
LoRA scale 0 — a tokenizer effect, not the LoRA. `test_lora_gating_measure.py` renders a
LoRA-off-with-trigger cell to measure this floor (`d_null`) and subtracts it. The naive
ratio on the bad dataset read **1.02** (looks like healthy gating); the honest one read
**0.30**. Report net-of-null or you're measuring the tokenizer.

## Levers that looked promising and weren't

- **Tag dropout** doesn't help — it *protects* the trigger rather than diluting it,
  because `keep_tags_count: 1` pins the trigger at `tags[0]` and drops from the rest of
  the caption instead. (That same setting is *why* the template keeps
  `keep_tags_count: 1` — it guarantees the trigger survives any dropout.)
- **Text-encoder caption dropout** makes gating *worse* — more unconditional training,
  the opposite of the goal.
- **OneTrainer's `concept_stats` panel** is a stale UI cache — it reflects whatever was
  last computed, not your current captions. Don't read it as a diagnostic.

## Re-measure gating at the checkpoint you ship

A checkpoint with better identity/consistency numbers is **not** automatically the one
with the better gating profile. On Round 1's three surviving arms, every render-side
metric improved from step 499 to 1099 — the obvious read was "later is strictly better."
Re-running gating at both steps (all six labels in one sitting):

| Label | net-of-null out-of-domain | net-of-null in-domain |
| --- | --- | --- |
| A2_499 → A2_1099 | 1.070 → 1.126 (**+5%**) | 1.658 → 1.367 |
| A1c_499 → A1c_1099 | 0.990 → 1.327 (**+34%**, highest in table) | 2.452 → 1.867 |
| A4_499 → A4_1099 | 0.978 → 1.031 | 1.689 → 2.083 (improves) |

Out-of-domain leak rose at 1099 for every arm, but the size differed sharply: A2's +5%
was a mild trade against a clean sweep of render wins (shipped); A1c — the arm whose
*render* case for 1099 was strongest — leaked +34% with the sharpest in-domain drop, so
if A1c were ever staged, A1c_499 is the safer pick despite weaker render numbers.
Nothing in the render metrics predicted which arm would show this. Gating and
identity/consistency are independent measurements of different things; a win on one
doesn't transfer to the other, in either direction.
