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
  at normal step counts on SDXL base. `--prompts-file` (JSON list of
  `[label, prompt]`) runs the matrix once per prompt in one invocation.
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

## Which checkpoint to ship

Run identity/consistency across your sweep's survivors *and* re-run gating at the
specific checkpoint you're about to ship — the render-side winner is not automatically
the gating winner (`contrastive-concept-gating.md` has the measured case). Then judge
the pick in the real component at real step counts before calling it done.
