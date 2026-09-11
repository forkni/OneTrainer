# Phase 5 — Validation scripts

Scripts live in `course_lora_kit/scripts/` (byte-identical mirrors of the course repo's
`course_v3/appendices/assets/`, minus the two StreamDiffusion-side checks that need a
StreamDiffusion checkout; that folder's README is the canonical deep-dive). Each has a
`.cmd` wrapper in `course_lora_kit/cmd/` that runs it in OneTrainer's venv:

| `.cmd` | Script | Question it answers |
| --- | --- | --- |
| `05_validate_grid.cmd` | `test_lora_grid_sdxl_base.py` | Did training take, does the trigger gate, does the style leak — one 2×2 grid, first look |
| `05_validate_gating.cmd` | `test_lora_gating_measure.py` | The full multi-prompt gating measurement (net-of-null ratio, `d_leak`/`d_gate`) |
| `05_validate_sweep.cmd` | `test_lora_checkpoint_sweep.py` | Where in the sweep the style takes vs collapses — rendered contact sheet |
| `05_validate_batch.cmd` | `test_lora_seed_batch.py` | Is *one* checkpoint stable across seeds — N seeds LoRA on + the same N LoRA off, mean CIELAB a\*/b\* per render, drift count against a provisional a\* line |
| `06_stage_pick.cmd` | `stage_pick.py` | Hand-off of the pick: recompute ‖ΔW‖_F + module count, check rank/alpha (and `--expected-norm` against the screen), copy — never move — to the loras folder, SHA-256 both sides |
| `05_validate_identity_consistency.cmd` | `test_lora_identity_consistency.py` | Is it *the same subject* across prompts and seeds (renders its own grid) |
| `05_validate_identity_score.cmd` | `test_lora_identity_score.py` | Same identity metric for renders you already have (ComfyUI, component captures) |
| (no wrapper — run in the venv) | `color_stats.py --sweep-dir <sweep> [--baseline-dir <other sweep>]` | Colour drift as a number: mean CIELAB a\* per sweep render, with a Δa\* column against another arm — the metric behind Round 2's 31.0 → 12.0 |

All render scripts use plain `diffusers`/`torch` — none import `streamdiffusion`. Final
judgment still happens in the real-time component at real step counts (phase 6, in the
course's Appendix D) — these scripts validate the file, not the deployment.

## Conventions shared by all of them

- **`--reference-dir` = your held-out images** (the 8–12 you never trained on, from
  phase 1). Scoring against training images measures memorization, not identity. Use
  the *same* reference dir across scripts/sessions or scores aren't comparable.
- **HF_HOME:** the scripts call `os.environ.setdefault("HF_HOME", ...)` and load with
  `local_files_only=True` — they use the cache but never populate it. The `.cmd`
  wrappers set `HF_HOME` to `workspace\hf_cache` (where a training run puts the base
  models); DINOv2 (`facebook/dinov2-base`) and, for the consistency script, CLIP
  (`openai/clip-vit-base-patch32`) must be seeded into the same cache once (a plain
  `transformers` load without `local_files_only`) before first use.
- **Output dirs:** most default to `outputs/<name>/` next to the script (gitignored),
  but not uniformly — `dataset_hygiene_profiler.py` defaults to `.`,
  `test_lora_grid_sdxl_base.py` to `outputs/lora_sdxl_base_grid`, and
  `test_lora_identity_score.py` keeps its `test_` prefix. Pass `--output-dir` when it
  matters. `checkpoint_norm_analyzer.py` has no output dir at all (console + `--json`).

## Per-script notes

- **Grid** (`test_lora_grid_sdxl_base.py`): 2×2 = LoRA on/off × trigger present/absent,
  at normal step counts on SDXL base. Cells: top-left no LoRA / no trigger, top-right no
  LoRA / trigger, bottom-left LoRA / no trigger, bottom-right LoRA / trigger. Bottom-left is
  the discriminating cell: it should look like the top row, not like bottom-right.
  `--prompts-file` (JSON list of `[label, prompt]`) runs the matrix once per prompt in one
  invocation.
- **Seed batch** (`test_lora_seed_batch.py`): the grid and the sweep render one seed each;
  Round 2's S0 arm looked fine at one seed and split 2 clean / 1 drifting / 1 pink across
  four in a live ComfyUI batch judged by eye (S0b: 4/4 coherent in the same batch; prompt
  and seeds were not recorded). The batch renders `--seeds` N seeds of one prompt with the
  LoRA on and the same N with it off, prints mean a\*/b\* for every render, and counts
  renders over `--a-threshold` (default 20 — **provisional**, and a flag means *look*, not
  *reject*). Where 20 comes from: the seed-42 sweep renders measured by `color_stats.py` —
  controls 0.6–1.5, S0b up to ~13 across 13 checkpoints, S0's pink renders 28–31. The
  first per-seed a\* on the S0b pick itself (default prompt, seeds 42–45, 2026-09-11)
  read 12.1 / 9.7 / 28.5 / 25.3 against controls 0.3–0.6 — two over the line, both
  structurally intact: seed 44 a purple-orange sunset sky over a cream tower (palette,
  neutrals held), seed 45 the tower itself salmon-pink (tint reached the neutrals). Mean
  a\* measures tint, not collapse, and cannot separate a warm palette from drift; the
  by-eye tell is whether the neutrals stayed neutral. Re-derive the line on your own set
  from a checkpoint you can see is clean and a render you can see has drifted.
- **Gating** (`test_lora_gating_measure.py`): four cells per prompt — LoRA off,
  off+trigger, on, on+trigger — yielding `d_leak`, `d_gate`, and a token-insertion
  noise floor `d_null`. **Always report the net-of-null ratio**: inserting *any* word
  shifts cross-attention even at LoRA scale 0; the naive ratio on the measured dataset
  read 1.02 (looks healthy) where the honest one read 0.30. Uses out-of-domain prompts
  by design — an in-domain prompt can't distinguish "the trigger gates nothing" from
  "this prompt summons the concept on its own." `--prompts-file` takes
  `[label, prompt, in_domain]` triples.
- **Sweep** (`test_lora_checkpoint_sweep.py`): renders every saved checkpoint at fixed
  prompt/seed/weight, plus an untriggered control per checkpoint (`d_control` — watch
  for it trending toward 0: gating drifting even as the style improves). Get
  `--final-step` from your run's log or config — never guess it from a filename; an
  earlier hardcoded label was wrong the one time it got reused.
- **Identity consistency** (`test_lora_identity_consistency.py`): DINOv2-embeds its own
  renders across a prompt × seed grid; reports **identity** (cosine vs reference dir),
  **cross-seed consistency**, **flexibility** (1 − pairwise cosine across *different*
  prompts — the anti-memorization check: a collapsed checkpoint can score high identity
  while flexibility craters), and a CLIP prompt-adherence score. `--lora LABEL=PATH`
  repeatable; `--lora-scale LABEL=SCALE` is a separate flag (a suffix would collide
  with Windows drive-letter colons). Cross-check a low-flexibility arm against a knee
  from the phase-4 screen — same failure seen from the weight side.
- **Identity score** (`test_lora_identity_score.py`): same DINOv2 identity for existing
  images — `--render LABEL=PATH` (file, folder, or quoted glob; repeatable, same label
  to hand-pick from a mixed folder). Also reports `render_consistency` (pairwise cosine
  within a label — catches one drifted render a healthy mean hides; this is what
  settled a Round 1 tiebreak that mean identity called even). Under ~3 images per
  label is directional, not conclusive. Lightest script here: no diffusers, no GPU
  required.

## Two measured pitfalls (both silently invalidated real batches)

1. **Copy the trigger from a caption file — never retype it.** A render batch prompted
   `lainwakura` while every caption said `lainiwakura`; the one-letter mismatch
   invalidated the entire batch with no error anywhere. The prompt renders fine, the
   scores just measure the wrong thing.
2. **Never compare a fresh measurement against numbers from an earlier session.**
   Library/model-environment differences shift absolute scores between sessions (a
   `none` control once moved 0.141 → 0.173 across batches). Re-measure one anchor label
   in every new batch as a control — when an anchor reproduced (A2_499's gating came
   back 1.070/1.658 vs a prior session's 1.07/1.64), the sessions were provably on the
   same footing and *only then* were cross-session comparisons trusted. Within-batch
   comparisons are always valid; cross-batch ones must earn it.

## When the grid and the ratio disagree

The ratio decides. The grid is one seed of one prompt read by eye; the net-of-null ratio
is three out-of-domain prompts with the tokenizer's own shift subtracted. Measured anchors
(all net-of-null, `ratio_net_d_gate_over_d_leak`, Round 2 unless noted): 0.28 S5 (naive
0.97 — the trigger inert, not leaking), 0.30 the original round's prefix-trap floor, 1.63
S3 (rank/alpha damaged), 1.67 the original round after the contrastive concept, 1.73 the
S0b pick, 2.75 S1, 3.01 S0, 3.02 S2. Nothing has been measured between 0.30 and 1.63, so
there is no cutoff there — a ratio in that gap is unexplored, not "mild". Two things the
ratio cannot see and the grid can: colour drift (S0b vs S0) and baked-in bars (S2 gated at
3.02 with 214 px pillarbox bars in every render). So: ratio for gating, grid + seed batch
for what the pixels do, hygiene before training for what the pixels *were*.

`d_null` is the base model's own response to the inserted token, measured with the LoRA
off in the same run and subtracted before the ratio (`test_lora_gating_measure.py:232`),
so there is no pre-training version of this measurement — without a LoRA `d_leak` is
zero and the script prints the ratio as undefined. The anchors above are this course's
two datasets' values (1.67 on 21 images, 3.0 on 48, same recipe), not targets for a
third; a value off them is read from the raw `d_leak` / `d_gate` / `d_null` the script
prints.

## Which checkpoint to ship

Run identity/consistency across your sweep's survivors *and* re-run gating at the
specific checkpoint you're about to ship — the render-side winner is not automatically
the gating winner (`contrastive-concept-gating.md` has the measured case). Run the seed
batch on that checkpoint, then stage it with `06_stage_pick.cmd` (pass `--expected-norm`
from the screen; the copy is refused if the recomputed norm is off by more than 2%) so
the file in the loras folder is provably the row in the norm table. Then judge the pick
in the real component at real step counts before calling it done.
