# Phase 2 — Captioning

Script: `course_lora_kit\cmd\02_caption_auto.cmd <image_folder> [MODEL]` (wraps
OneTrainer's `scripts/generate_captions.py`; writes a `.txt` next to every image). In the
GUI it's the **tools** tab → `Dataset Tools` → `Generate Captions`.

## Captioner choice

- **WD14_VIT_2** (the `.cmd` default) — booru-style tags; the right choice for anime and
  illustration sources.
- **BLIP / BLIP2** — natural-language captions; better for photographic sources.

## The trigger word

A made-up token that will summon the concept at prompt time (`zxqstyle`,
`lainiwakura`-style — not a real word). Two mechanics worth knowing:

- The trigger tokenizes like anything else — `sepiagraph` splits into four BPE tokens
  (`se`, `pi`, `agra`, `ph`), not one. That's fine; it still trains and still gates.
- At inference the trigger goes in the component's prompt — loading the file applies
  nothing by itself; typing the trigger is what asks for the concept it gates.

## The prefix trap (measured, and the single biggest mistake)

**Do not put the trigger word in the captioner's `Prefix` field.** A trigger present in
100% of captions gives training nothing to contrast against, so the cheapest loss is to
shift the *whole* style unconditionally — the LoRA applies at full strength on any
prompt and the trigger does nothing. Measured: net-of-null gating ratio **0.30** (the
LoRA applies almost as strongly with the trigger absent as present). The fix is the
**contrastive concept** — a second, trigger-free concept sharing the same images — which
took the same dataset to **1.67**. Full method: `contrastive-concept-gating.md`.

Two smaller caption-text rules:

- The `Prefix` field does raw string concatenation with **no separator** — true of all
  three captioners (`BlipModel.py:34`, `Blip2Model.py:34`, `WDModel.py:74`).
  `sepiagraph` with no trailing space produced `sepiagrapha close up of…`. End any prefix
  you do use with `, `.
- Auto-caption first, then a **by-hand pass** in a text editor: trigger placement, delete
  noise, add what the machine missed. The by-hand pass is not optional — it's where half
  the quality comes from.

## Style vs character captions — opposite rules

- **Style LoRA:** lead every caption with the trigger word, then a plain description.
- **Character LoRA:** **caption the variable traits, delete the permanent ones.** An
  auto-tagger tags everything it sees, including traits true in every image (hair colour,
  hair length, eye colour). If those tags stay in every caption they get learned as
  conditional on the caption text, not bound to the trigger — a prompt that omits them
  can render a different hair colour entirely. Delete the permanent traits after
  auto-tagging; keep and edit the variable ones (outfit, pose, expression, framing,
  background). What you *don't* caption is what binds to the trigger. This is the inverse
  of ordinary captioning advice, which is exactly why it's easy to skip by habit.

## Output of this phase

Two folders, ready for the concepts file (phase 3):

1. The dataset with trigger-bearing captions (the `STANDARD` concept).
2. A **copy** of the same images with captions that never mention the trigger (the
   `PRIOR_PREDICTION` contrastive concept).
