# Phase 1 — Dataset prep

Scripts: `course_lora_kit\cmd\01_dataset_hygiene.cmd <image_folder>` (wraps
`course_lora_kit/scripts/dataset_hygiene_profiler.py`), then
`course_lora_kit\cmd\01b_palette_screen.cmd <image_folder> [dropped_folder]` (wraps
`course_lora_kit/scripts/palette_ab_stats.py`, the palette screen below).

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

- At least **1024px on the short side** (the kit trains SDXL only) — aspect bucketing
  handles non-square images, not undersized ones. The *effective* floor is the aspect
  bucket, not the nominal resolution: at resolution 1024 a 4:3 image trains in the
  1152×896 bucket, so a uniform 1280×960 set is a pure downscale and passes
  `--min-side 896`; the wrapper's default 1024 would flag every image of such a set.
- **Don't upscale** low-resolution source to hit the floor — upscaling inflates apparent
  sharpness the model then learns as texture; drop the image instead. Mixing upscaled and
  native images is what the rule forbids. One measured exception: when the *whole* source
  is one low native size (the course's style set, 640×480 DVD screencaps), a **uniform**
  upscale is acceptable under four conditions (Appendix H, *the uniform-upscale
  amendment*):
  1. *Curate first, upscale last* — every curation pass at native size, upscale only the
     final set.
  2. *One model, one pass, every image* — the course set went 640×480 → 1280×960 through
     `RealESRGAN_x2` at its native 2×, no second resample. The texture it adds is trained
     in with everything else and is invisible to a folder-relative screen; whether the
     renders carry it is **unmeasured** (no native-resolution control exists at the 1024
     bucket, and the bucket downscale to 1152×896 attenuates the contribution without
     removing it).
  3. *Run the hygiene screen on the upscaled folder* — its job is to catch a frame the
     upscaler mangled, not what the DVD looked like.
  4. *View three or four frames at 100% beside their natives* — haloed line art, smeared
     gradients. No script in the pipeline sees a texture every training image shares.
  Do not "fix" the texture by putting a tag such as "AI upscaled" into every caption: a
  word present in every caption on both concept sides co-occurs perfectly with the trigger
  and with every image, so the loss has nothing to contrast it against — it is the prefix
  trap and arm S5 wearing a second word (see `02-captioning.md`).
- **Crop letterbox/pillarbox bars off** (common with 4:3 video screencaps of anime
  sources) — a uniform matching-colour border at both edges reads to the model as "this
  is always part of the frame," and bucketing has no idea it's not content.

## Balance rules

- **Subfolders/motifs:** sort the curation folder into one subfolder per group and let the
  screen count them — `--recursive` is the `.cmd` default; it prints `Per-subfolder counts:`
  with a percent per folder and `<<< imbalanced` on any folder above
  `--imbalance-threshold` (0.5, strictly greater). The number to read is *largest group ÷
  total*. The course's original 21-image style set held 16 close-up circular-ornament
  drawings against 5 city / clock-tower drawings: 16 ÷ 21 = 76%, far over the line, and
  its memorised run rendered the ornament texture whatever the prompt asked for — the
  majority motif's texture trained into the *whole* style, not just its share of the
  images. That set was flat on disk; the folders are what this rule asks for now.
- **Style sets — which axis to fold on:** subject (`subject_main\`, `subject_other\`,
  `no_subject\`). Shot type (close / medium / wide) stays a by-eye tally, because the
  profiler counts folders, not framings. A style set may legitimately trip the 50% line on
  `no_subject\` — that subject-free majority is what keeps it a style rather than a
  character (Round 3a Aeon Flux: 29 / 14 / 5 = 60% / 29% / 10%, warning printed and
  accepted). Read the warning as a question — is the majority folder one motif or many? —
  and answer it by eye before captioning; it is a stop only when the answer is "one".
- **Characters — outfits:** no single outfit over roughly half the set, and caption the
  outfit on every image (see `02-captioning.md`) so it stays something the trigger can
  vary rather than part of "what this character looks like."

## What the profiler checks

Per-image greyscale standard deviation (tonal contrast), edge energy (sharpness/detail
proxy), and mean HSV saturation, reported as z-scores against the folder's own average —
no fixed pass/fail thresholds; an outlier is relative to *that* folder. It tells you
where to look, not what to do. It also flags: undersized short sides (`--min-side`,
opt-in — the `.cmd` passes 1024 by default; use the bucket's short side for a
non-square set, e.g. `--min-side 896` for 4:3 at 1024), symmetric
letterbox/pillarbox borders (always checked), and with `--recursive` (the `.cmd` default)
per-subfolder counts with an `--imbalance-threshold` warning (default 0.5).

Healthy signal: no image beyond the z-threshold you'd struggle to explain, no letterbox
flags, subfolder counts close. Investigate every flag before captioning — one wrong-toned
photo quietly drags the whole average.

## The palette screen (`01b_palette_screen.cmd`)

The hygiene profiler reads tone, sharpness and saturation; it does not read *which* colour.
`palette_ab_stats.py` applies the statistic the seed-batch validator applies to renders
(`color_stats.py`) to the dataset instead: per image the mean CIELAB a\* (red > 0, green < 0)
and b\* (yellow > 0, blue < 0), z-scored against the folder's own mean, with a coarse hue read
(orange / red / magenta / purple / blue / cyan / green / yellow, `neutral` at low chroma) and,
with `--recursive` (the `.cmd` default), a per-subfolder mean block. Pass a second folder and
it prints the delta of mean a\*/b\* against the first — the way to see what a cut removed
(Round 3a Aeon Flux: on 2026-09-11 the five morning drops read mean a\* +16.8 against
the then-kept 48's +5.95, i.e. the cut took out the red-leaning frames; the final 50-image
set reads a\* +5.21 (sd 9.15), b\* +5.95 (sd 16.62) with 5 folder-relative flags at the
two ends of the palette — orange skies, blue nights, one red corridor — all kept after
viewing).

Read it like the profiler: the flags are folder-relative, not pass/fail, and they describe
the dataset only — nothing here predicts what a trained LoRA renders. A style set whose frames
fall on one side of an axis is a palette, not a defect; a single frame far from the rest on
an axis the style does not use is the thing to look at.
