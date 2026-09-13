# Round 3 case study — the same arms on the curated 50-image set (curation is the lever)

Round 2 measured the style rules on a 48-image set. Round 3 (2026-09-12) re-cut that
dataset by eye to **50 frames**, retrained the S0b recipe on it unchanged (Round 3a, the
control), and then re-ran five of the Round 2 violation arms on the *same* curated set
(Round 3b), each arm named after the one dial it moves. Load this file when a user asks
whether the Round 2 numbers transfer to their own dataset, what "curate the dataset" is
worth in measured terms, or why the arm run names in `arms.json` look the way they do.
Every number traces to `F:\LORA_AEONFLUX_R3\` on the measurement machine (the control's
Phase 8/9 logs at the root, the arms under `arms\`); the course's Appendix H carries the
same figures with a ledger row each.

## What changed between Round 2 and Round 3 — the dataset, nothing else

| | Round 2 (S0b) | Round 3a control | Source |
| --- | --- | --- | --- |
| Frames | 48, upscaled 640×480 → 1280×960 | **50**, same upscale path (parity 48/48 byte-identical on the carried frames) | mapping file; `upscale_x2_comfy_parity.py` |
| Re-cut | — | one frame dropped, two returned, nine added (three warm face close-ups, three action frames, one blue daylight cityscape, two blue night character frames) | walkthrough delta block 2026-09-12 |
| Shot-type split | warning printed (one folder over 50 %) | 22 / 9 / 19 = 44 % / 18 % / 38 %, no warning | `01_dataset_hygiene.cmd` |
| Palette screen | mean a\* +5.95 on the kept 48; the morning drops read +16.8 | mean a\* **+5.21** (sd 9.15), b\* +5.95; 5 flags at the palette ends, all viewed, all kept | `01b_palette_screen.cmd` |
| Captions | WD14 fill + by-hand pass | eleven re-captioned by eye (11 edited / 39 unchanged); trigger leads all 50 `trigger\` captions, absent from all 50 `notrigger\` | `caption_byhand_pass.py`; `check_trigger_word.py` |
| Recipe | rank 16 / alpha 1.0 / LR 3e-4 / `attentions` / warmup 30 / epochs 50 / batch 4 / MIN_SNR γ 5 / offset 0.03 | **identical** — `00b_print_recipe.cmd style` prints the same merged values | `configs\style_sdxl.json` |
| Steps | 48 × 2 ÷ 4 = 24 steps/epoch → 1,200 | 50 × 2 ÷ 4 = **25 steps/epoch → 1,250**; saves at 99 … 1199, final at 1,250 | save file names `save-99-3-24` … `save-1199-47-24` |
| Trainable modules | 722 selected / 72 deselected | 722 / 72 (the first Round 3a launch trained all 794 because the overlay lacked `layer_filter`; aborted and relaunched; Appendix H's layer-filter row records both counts) | run log lines 12–13 |

Deliberate non-delta: the trigger (`qzlv`), the two-concept contrastive design, the seed
(42), the prompt set and every kit threshold are unchanged, so any movement in the
instruments below is the dataset's.

## Round 3a — the control, measured against S0b

Same recipe, curated set, two more frames. Every instrument moved in the same direction:

| Instrument | S0b pick (step 1099, 48 images) | R3a matched step 1099 | R3a pick (step 1199) | Read |
| --- | --- | --- | --- | --- |
| Gating net-of-null (seed 42) | **1.73** (re-measured this round as the anchor: 1.7259) | 1.55 | **2.47** (naive 4.12) | the trigger does more gating work on the curated set; the S0b anchor was re-run on the same day with the same script so the comparison is like for like |
| Seed batch, a\* over 20 (seeds 42–45) | 4/4 clean by eye; a\* ≤ ~13 | — | **0/4 flagged**; a\* LoRA −0.84 / −2.45 / +7.96 / −0.89 against controls 0.30–0.59 | colour-neutral at every seed — the Round 2 drift is gone, not merely halved |
| ‖ΔW‖_F at the pick | 8.1 (step 1099) | 8.33 | **9.62** (step 1199) | past every "collapsed" band from Round 2 and the 21-image round, with a clean render — the bands are step-count- and recipe-local (Round 2 finding 1, confirmed a third time) |
| Knee flags | steps 199, 299 | — | steps **199, 299** (norm 0.74 → 1.48, growth ratio 2.1 → 2.0) | the knee stays where the optimizer puts it; the usable window runs to the last save |
| Endpoint cosine | 0.111 | — | **0.108** | trajectory still turning, no saturation |
| Staging recompute | 8.104 vs screen 8.109 (0.06 %) | — | **9.616 vs 9.623** (0.07 %), 722 modules, SHA-256 match both sides | `06_stage_pick.cmd`, staged as `aeonflux_R3a_s1199` |

Pick: **step 1199 = epoch 48, ≈ 48 views/image**, inside the 40–60 views window. It is
the round's working style LoRA and the control every Round 3b arm is diffed against.
(In the Round 3b arm table it is also staged under the self-describing name
`aeonflux50_control_minsnr5_offset003_s1199`; same file, second name.)

What this buys the student: the recipe did not change and the numbers did, so the
lever that moved them is the dataset — the re-cut, the palette read, the by-hand
captions. "Curate first" is the cheapest intervention on this page and the one with
the largest measured effect.

## Round 3b — the arms, renamed after their one delta

Round 2's labels (S1, S2 …) meant nothing without the table. Round 3 names every run
after the dial it moves and the value it takes, so the run folder, the log, the norm
JSON group and the staged file all read the same:

| Run name | The one delta vs the control | Round 2 twin | Steps |
| --- | --- | --- | --- |
| `aeonflux50_control_minsnr5_offset003` | — (Round 3a, reused, not retrained) | S0b | 1,250 |
| `aeonflux50_constant_loss_no_offset_noise` | `loss_weight_fn` MIN_SNR_GAMMA → `CONSTANT`, `offset_noise_weight` 0.03 → 0.0 (the preset's pair, changed as a unit) | S0 | 1,250 |
| `aeonflux50_warmup200_preset_default` | `learning_rate_warmup_steps` 30 → 200 | S1 | 1,250 |
| `aeonflux50_alpha16_with_lr3e-4` | `lora_alpha` 1.0 → 16.0, LR left at 3e-4 | S3 | 1,250 |
| `aeonflux50_single_concept_no_notrigger` | the `PRIOR_PREDICTION` concept removed (`style_concepts_single.json`) | S5 | 600 (12 steps/epoch, 6 saves, last at 599) (50 items ÷ 4 → 12 or 13 steps/epoch; the log decides) |
| `aeonflux50_pillarbox_214px_bars` | pixels only: every frame padded with symmetric 214 px bars (1280×960 → 1708×960), captions identical (`style_concepts_pillarbox.json`) | S2 | 1,250 |

Not re-run: S4 overtrain (still designed, not run — 360 epochs is a day of GPU for a
collapse end already anchored by the 362-view measurement).

Each arm is the kit's own `03_train_style_sdxl.cmd` plus `--config-value` overrides;
the workspace, cache and output paths are also overrides so the six runs never share a
folder. **Pre-flight is mandatory**: `BaseConfig.from_dict` swallows a value it cannot
coerce and prints `Could not set <key> as <value>` instead of failing, so every arm's
merged config was printed with `print_effective_config.py --all --json` and deep-diffed
against the control's before launch — exactly the intended keys differed, no
`Could not set` lines (`arms\check_arm_recipes.py`, 5 PASS). Every arm log carries
`Selected layers: 722` / `Deselected layers: 72`.

Wall-clock on this machine: about 2.9-3.6 min per 100 steps with both concepts cached
(37-47 min per 1,250-step arm; Round 2 measured 1.62 s/step on the same card, so
the difference is the save cadence and the caching pass, not the recipe).

### Results at the matched step (step 1199; step 599 for the single-concept arm)

Screen: `04_checkpoint_screen.cmd` on all six save folders at once → `arms\R3b_arms_norms.json`.
Validation per arm: `05_validate_sweep` (all saves + final), `05_validate_gating` and
`05_validate_batch` (seeds 42–45) at the matched step and at the arm's last knee step,
`05_validate_grid` for the two dataset arms. Same prompt and seed as the control.

| Run | ‖ΔW‖_F matched | knee steps | gating net (naive) | a\* flags /4 | Verdict vs Round 2 twin |
| --- | --- | --- | --- | --- | --- |
| control | 9.62 | 199, 299 | 2.47 (4.12) | 0/4 | anchor |
| constant_loss_no_offset_noise | 10.79 | 199 | 2.95 (4.08) | 4/4 | Drift is back and worse: 4/4 seeds flagged, a* +29 to +58 against controls 0.3-0.6, every render pink and blurred; the 2.95 gating figure is the colour cast counting as trigger work. Curation did not remove it; the MIN_SNR 5 + offset 0.03 pair is still the fix (control 0/4 on the same set) |
| warmup200_preset_default | 9.04 | 199, 299 | 1.58 (2.76) | 0/4 | Under-trains again, smaller gap: norm 9.04 vs 9.62 (6 % low), gating 1.58 vs 2.47 (36 % low), lowest endpoint cosine of the six (0.049); 0/4 flags, renders clean. The deficit narrowed but had not closed by the last save |
| alpha16_with_lr3e-4 | 57.65 | none | 1.34 (2.27) | 0/4 | Not destroyed this time: renders coherent and heavily stylised from the first save (norm 7.30 at step 99, where the control needs about 1,000 steps); no knee flagged, so the screen's selection rule breaks again; gating 1.34, about half the control; 0/4 flags. 'Screen blind, gating halved' transferred; 'destroyed by step 199' did not |
| single_concept_no_notrigger | 8.19 | 199 | 0.38 (0.90) | 0/4 | Trigger inert again: net 0.38 at step 599 and 0.22 at step 199 against the ~0.30 floor; the 2x2 grid is the same base-model drawing with or without the trigger or the LoRA; 0/4 flags. Round 2's 0.28 reproduced on the curated set |
| pillarbox_214px_bars | 9.33 | 199 | 1.06 (2.36) | 0/4 | Weights blind again: norm 9.33, knee 199, 0/4 flags, but gating 1.06 (less than half the control) and a* +3 to +11 with a warm/pink cast from about step 900. At 1024 px the bars do not show as bars in this prompt's renders; the sweep drifts to tight crops instead. The hygiene profiler (48/48 in Round 2) is still the only instrument that names the cause |

Staged picks (`06_stage_pick.cmd`, norm within 2 % and SHA-256 match on both sides):
each arm at its matched step and at its last knee step, named `<run name>_s<step>`, so a
student can load any of them next to the control in ComfyUI and see the rule's cost
directly.

## What transferred from Round 2, and what did not

Four of the five Round 2 verdicts reproduced on the curated set; one softened. In the order the questions were asked:

- **Constant loss drifted again, harder.** 4/4 seeds flagged at step 1199 (a\* +29.11 / +57.74 / +41.34 / +36.96 against controls 0.30-0.59; Round 2 had 2/4). At the knee step 199 it is still clean (0/4, a\* within 1.4 of zero), so the drift is a late-training effect the recipe pair prevents and curation does not. The loss-weighting fix stays in the recipe.
- **Warmup 200 still under-trained at every matched step**, but by less: norm 9.04 vs 9.62 at step 1199, gating net 1.58 vs 2.47, endpoint cosine 0.049 (the flattest trajectory of the six). Round 2's 56-66 % suppression became a 6 % norm gap and a 36 % gating gap. Direction transferred, size did not.
- **Alpha 16 did not destroy the run.** The sweep is coherent and saturated in the style from step 99 (norm 7.30, reached by the control only around step 999); by step 999 it drifts to tight crops. What did transfer: no knee flagged (scale 1.0 instead of 1/16 keeps the growth ratio under the screen's threshold, so the selection rule is blind), gating 1.34 (roughly half), norm 57.65 at step 1199. Stage it with `--alpha 16` or the recompute fails.
- **The single-concept trigger went inert again.** Net-of-null 0.38 at step 599 and 0.22 at 199 against the ~0.30 floor (Round 2: 0.28); the 2x2 grid shows the same base-model drawing in all four cells. 12 steps/epoch (drop-last on 50 items / 4), 600 steps, 6 saves. Curation cannot supply the contrast the missing `notrigger\` copy provided.
- **Pillarbox: weights still blind, renders less obviously so.** Norm 9.33, knee 199, 0/4 flags: nothing in the screen or the seed batch names the bars. Gating 1.06 at step 1199 is below the knee-step value (1.42) and less than half the control, and a\* runs +3 to +11 with a warm/pink cast from about step 900. In this prompt's 1024 px renders the bars do not appear as bars (Round 2 saw them from about step 700 on a different prompt); the sweep drifts to tight crops instead. The 20-second hygiene profiler remains the only instrument that reports the cause before the GPU is spent.

What this adds to the Round 3a lesson: curation moved every instrument on the control, and it moved the arms too (the warmup deficit shrank, the alpha-16 renders survived), but it did not repair a single rule violation. The recipe pair, the two-concept design, the alpha/LR coupling and the dataset hygiene pass each still cost what Round 2 said they cost.

## Scoping (read before quoting any number)

- One dataset, one seed, one prompt set, one machine. The arms measure *direction and
  rough size* against a matched control, not effect sizes that transfer to other sets.
- The control was trained once (Round 3a) and reused; the five arms were trained
  afterwards in one sitting. Nothing else ran on the card during the arms.
- The a\* threshold (20) is provisional in the script and the gating ~0.30 floor comes
  from two earlier datasets; both are read as "look at this render", not as pass/fail.
- The Round 2 twin column is a design correspondence, not a numeric one: the twins ran
  on a different dataset with 24 steps/epoch, so compare verdicts, not norms.
