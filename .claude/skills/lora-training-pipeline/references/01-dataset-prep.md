# Phase 1 — Dataset prep

Script: `course_lora_kit\cmd\01_dataset_hygiene.cmd <image_folder>` (wraps
`course_lora_kit/scripts/dataset_hygiene_profiler.py`).

## How many images, and which

- **Style LoRA:** 30–150 images that genuinely share the aesthetic. Consistency of style
  matters more than count.
- **Character LoRA:** 40–60 images curated for angle and shot-type spread, not raw
  volume — past roughly 60, extra images mostly add redundant near-duplicates. A workable
  split for ~48 images:

| Group | Count | Why |
| --- | --- | --- |
| Face / bust close-ups | ~18 | Front, ¾-left, ¾-right, profile, slight up/down — angle spread is what stops the LoRA from memorizing one photo instead of a face |
| Medium / upper body | ~16 | |
| Full body | ~10 | Silhouette and proportions, not just the face |
| Expression / lighting outliers | ~4 | Deliberately atypical frames so the LoRA doesn't treat "neutral studio lighting" as part of the identity |

- **Hold back 8–12 images the LoRA never trains on.** They are the only honest way to
  measure identity after training — the `--reference-dir` every identity script expects.
  Scoring against training images measures memorization, not identity.

## Resolution and framing rules

- At least **1024px on the short side** for SDXL (512px for SD1.5) — aspect bucketing
  handles non-square images, not undersized ones.
- **Don't upscale** low-resolution source to hit the floor — upscaling inflates apparent
  sharpness the model then learns as texture; drop the image instead.
- **Crop letterbox/pillarbox bars off** (common with 4:3 video screencaps of anime
  sources) — a uniform matching-colour border at both edges reads to the model as "this
  is always part of the frame," and bucketing has no idea it's not content.

## Balance rules

- **Subfolders/motifs:** a dataset split, say, 16:5 between two motifs trains the
  majority motif's texture into the *whole* style, not just its own share of the images —
  that imbalance is exactly what dominated one measured collapse. Keep group counts close.
- **Characters — outfits:** no single outfit over roughly half the set, and caption the
  outfit on every image (see `02-captioning.md`) so it stays something the trigger can
  vary rather than part of "what this character looks like."

## What the profiler checks

Per-image greyscale standard deviation (tonal contrast), edge energy (sharpness/detail
proxy), and mean HSV saturation, reported as z-scores against the folder's own average —
no fixed pass/fail thresholds; an outlier is relative to *that* folder. It tells you
where to look, not what to do. It also flags: undersized short sides (`--min-side`,
opt-in — the `.cmd` passes 1024 by default; use `--min-side 512` for SD1.5), symmetric
letterbox/pillarbox borders (always checked), and with `--recursive` (the `.cmd` default)
per-subfolder counts with an `--imbalance-threshold` warning (default 0.5).

Healthy signal: no image beyond the z-threshold you'd struggle to explain, no letterbox
flags, subfolder counts close. Investigate every flag before captioning — one wrong-toned
photo quietly drags the whole average.
