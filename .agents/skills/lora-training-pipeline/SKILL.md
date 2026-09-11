---
name: lora-training-pipeline
description: Guide a StreamDiffusionTD student through the course_lora_kit workflow for preparing, captioning, training, screening, and validating a OneTrainer style or character LoRA (StreamDiffusionTD course, Appendix D). Use when the user wants to prepare a dataset, caption it, train a style or character LoRA, screen checkpoints, or measure identity/gating. Use only for this course kit, not generic diffusion-training advice.
metadata:
  author: StreamDiffusionTD course
  version: 1.4.2
---

# LoRA training pipeline (Codex entry point)

This folder is the Codex discovery point only. The skill body, the phase table, the
operating boundary and all nine `references/*.md` files are tracked once, at
`.claude/skills/lora-training-pipeline/` (repo root), so that Claude Code and Codex read
the same text and the same measured numbers.

Read `.claude/skills/lora-training-pipeline/SKILL.md` now and follow it in full. Resolve
every `references/...` path it names against that folder, not this one. Do not copy or
summarise its contents into this file — a second copy would drift from the measured runs
the references are backed by.
