"""
Print the recipe OneTrainer will actually train with, after the same merge
scripts/train.py performs: built-in defaults -> --preset-path -> --config-path
-> each --config-value KEY=VALUE.

Overlay files only list the keys they change, so a dial nobody set silently
takes OneTrainer's default (train_dtype resolved to FLOAT_16 in the course
style overlay for a while and nobody saw it). This script makes the merged
result visible before the GPU spins up. It imports OneTrainer's TrainConfig
but never loads a model, so it runs in a second and needs no GPU.

Usage (from the fork root, same flags as train.py):
  python course_lora_kit\\scripts\\print_effective_config.py ^
      --preset-path "training_presets\\SDXL\\#sdxl 1.0 LoRA.json" ^
      --config-path course_lora_kit\\configs\\style_sdxl.json ^
      [--config-value epochs=30 ...] [--all] [--json out.json]

--all dumps every top-level key instead of the curated recipe list.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# course_lora_kit/scripts/ -> fork root, so `modules.*` resolves like train.py's shim.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.util.config.TrainConfig import TrainConfig  # noqa: E402

# (dotted key, one-line meaning). Order = how the course appendices discuss them.
RECIPE_KEYS = [
    ("model_type", "base architecture"),
    ("base_model_name", "HF id or path of the base model"),
    ("training_method", "LORA expected"),
    ("layer_filter_preset", "which layers get adapters (attn-mlp = 722 on SDXL)"),
    ("layer_filter", "custom layer filter (empty = use the preset)"),
    ("lora_rank", "r"),
    ("lora_alpha", "alpha; alpha/r is the effective scale"),
    ("lora_weight_dtype", "adapter parameter dtype"),
    ("learning_rate", "peak LR"),
    ("learning_rate_scheduler", "CONSTANT for the course recipes"),
    ("learning_rate_warmup_steps", "warmup steps (S1 arm priced 200 vs 30)"),
    ("optimizer.optimizer", "optimizer"),
    ("epochs", "epochs"),
    ("batch_size", "images per step"),
    ("gradient_accumulation_steps", "effective batch = batch_size x this"),
    ("resolution", "training resolution (bucketed by aspect)"),
    ("aspect_ratio_bucketing", "buckets on = non-square images keep their aspect"),
    ("loss_weight_fn", "MIN_SNR_GAMMA in the corrected style recipe"),
    ("loss_weight_strength", "gamma for Min-SNR"),
    ("offset_noise_weight", "offset noise (0.03 measured; 0 = off)"),
    ("train_dtype", "mixed-precision dtype; default FLOAT_16 unless pinned"),
    ("fallback_train_dtype", "used where train_dtype is unsupported"),
    ("unet.weight_dtype", "frozen U-Net weights"),
    ("text_encoder.train", "text encoder 1 trained?"),
    ("text_encoder_2.train", "text encoder 2 trained?"),
    ("output_model_format", "KOHYA_LORA expected"),
    ("output_dtype", "dtype of the saved adapter"),
    ("output_model_destination", "final LoRA path"),
    ("workspace_dir", "run folder (save/, cache/)"),
    ("cache_dir", "latent/text cache"),
    ("clear_cache_before_training", "wipes cache_dir (never save/)"),
    ("save_every", "intermediate save interval"),
    ("save_every_unit", "STEP / EPOCH / NEVER"),
    ("backup_after_unit", "NEVER = no resumable backups"),
    ("concept_file_name", "concepts JSON (ignored when 'concepts' is inline)"),
    ("concepts", "inline concepts (None = concept_file_name is used)"),
]


def resolve(config, dotted):
    target = config
    for part in dotted.split("."):
        target = getattr(target, part)
    return target


def render(value):
    if value is None:
        return "None"
    if hasattr(value, "name") and hasattr(value, "value"):  # Enum
        return value.name
    if isinstance(value, list):
        return f"<list of {len(value)}>"
    if hasattr(value, "to_dict"):
        return "<config object>"
    return repr(value) if isinstance(value, str) else str(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset-path", dest="preset_path", help="built-in preset, applied before --config-path")
    ap.add_argument("--config-path", dest="config_path", required=True, help="overlay config")
    ap.add_argument("--config-value", dest="config_values", action="append",
                    help="KEY=VALUE override, dot notation for nested keys; repeatable")
    ap.add_argument("--all", action="store_true", help="dump every top-level key, not just the recipe list")
    ap.add_argument("--json", help="write the printed keys to this JSON file")
    args = ap.parse_args()

    for p in (args.preset_path, args.config_path):
        if p and not os.path.isfile(p):
            ap.error(f"file not found: {p}")

    # --- identical merge to scripts/train.py ---
    config = TrainConfig.default_values()
    if args.preset_path is not None:
        with open(args.preset_path, "r") as f:
            config.from_dict(json.load(f), migrate=False)
    with open(args.config_path, "r") as f:
        config.from_dict(json.load(f), migrate=args.preset_path is None)
    for config_value in args.config_values or []:
        key, _, value = config_value.partition("=")
        *parent_keys, leaf_key = key.split(".")
        target = config
        for parent_key in parent_keys:
            target = getattr(target, parent_key)
        if target.types[leaf_key] is bool:
            value = value.lower() in ("true", "1", "yes")
        target.from_dict({leaf_key: value}, migrate=False)

    # Which file last touched each top-level key, so a surprising value has a source.
    sources = {}
    if args.preset_path:
        with open(args.preset_path, "r") as f:
            for k in json.load(f):
                sources[k] = "preset"
    with open(args.config_path, "r") as f:
        for k in json.load(f):
            sources[k] = "config"
    for cv in args.config_values or []:
        sources[cv.partition("=")[0].split(".")[0]] = "--config-value"

    print("Effective OneTrainer config (defaults -> preset -> config -> --config-value)")
    print(f"  preset : {args.preset_path or '(none)'}")
    print(f"  config : {args.config_path}")
    if args.config_values:
        print(f"  values : {' '.join(args.config_values)}")
    print()

    rows = []
    if args.all:
        for k in sorted(config.to_dict()):
            rows.append((k, resolve(config, k), ""))
    else:
        for k, note in RECIPE_KEYS:
            try:
                rows.append((k, resolve(config, k), note))
            except AttributeError:
                rows.append((k, "<not in this OneTrainer version>", note))

    kw = max(len(k) for k, _, _ in rows)
    vw = max(len(render(v)) for _, v, _ in rows)
    for k, v, note in rows:
        src = sources.get(k.split(".")[0], "default")
        line = f"  {k:<{kw}}  {render(v):<{vw}}  [{src}]"
        if note:
            line += f"  {note}"
        print(line)

    if getattr(config, "train_dtype", None) is not None and config.train_dtype.name == "FLOAT_16":
        print()
        print("  NOTE: train_dtype is FLOAT_16 (OneTrainer's default). The measured course")
        print("        style recipe ran BFLOAT_16; the shipped style overlays pin it.")

    if args.json:
        out = {k: render(v) for k, v, _ in rows}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
