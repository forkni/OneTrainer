# Changes in this fork

This repository is a modified version of [OneTrainer](https://github.com/Nerogar/OneTrainer)
by Nerogar and the OneTrainer contributors, licensed under the GNU Affero General Public
License v3.0 (see `LICENSE.txt`). This file is the notice required by AGPL-3.0 §5(a).

Modified by Alexander Korneev since 2026-08-16 (last updated 2026-10-22). One commit,
"Adapt LoRA course skill for Codex" (2026-09-10, merged as PR #1), is by Stanislav Glazov.

## Added

- `course_lora_kit/`: a LoRA training course kit for SDXL. It holds numbered `.cmd`
  wrappers for each pipeline phase (`cmd/`), training and concept configs (`configs/`), and
  analysis scripts (`scripts/`: dataset hygiene, trigger-word checks, CIELAB colour stats,
  checkpoint sweeps, identity scoring, stage picking).
- `.claude/skills/lora-training-pipeline/` and `.agents/skills/lora-training-pipeline/`: an
  agent skill that walks through the course kit's pipeline.
- `scripts/git/`, `.githooks/` and `cgw.conf.example`: git workflow tooling
  ([claude-git-workflow](https://github.com/forkni/claude-git-workflow)).
- `.markdownlint.json` and `.markdownlint-cli2.jsonc`: markdown lint configuration.
- `CHANGES.md` (this file) and a fork notice at the top of `README.md`.

## Modified

- `.gitignore`: ignores the install log, smoke-test scratch, `logs/`, the local CGW config,
  and everything under `.claude/` except the `lora-training-pipeline` skill.

No upstream source file under `modules/`, `scripts/` (other than `scripts/git/`) or the UI has
been modified. The full record is the git history; compare with upstream with
`git log upstream/master..HEAD`.
