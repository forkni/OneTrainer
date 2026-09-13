"""
SDXL base 1.0 LoRA checkpoint sweep -- plain diffusers, no StreamDiffusion wrapper.

Same prompt/seed/weight across every intermediate checkpoint OneTrainer saved during
training, to find the step count where the style takes vs. where it collapses into
overfit/memorized noise. Companion to test_lora_grid_sdxl_base.py and
checkpoint_norm_analyzer.py -- this script renders the sweep; the norm analyzer
screens it without rendering.

By default also renders an UNTRIGGERED control at every checkpoint (same seed,
prompt with the trigger word omitted) and reports the mean-absolute-pixel diff
between the triggered and untriggered render as a "d_control" column. A single
triggered-only sweep can't tell "gating held across training" from "gating drifted
apart (or leaked) as training went on" -- the trigger's gating strength at step 60
and step 900 can differ even when both renders look fine in isolation. Pass
--no-control to render triggered-only, at half the wall-clock, if you've already
confirmed gating separately (e.g. via test_lora_gating_measure.py on the final
checkpoint) and only want the style/collapse sweep.

Needs only plain diffusers -- no streamdiffusion import, no StreamDiffusion checkout.

Usage:
    python test_lora_checkpoint_sweep.py --ckpt-dir path/to/save/ --final path/to/final_lora.safetensors --trigger mytrigger
    python test_lora_checkpoint_sweep.py --ckpt-dir path/to/save/ --final path/to/save/last.safetensors --trigger mytrigger
        # --final already inside --ckpt-dir: rendered once, not twice (see discover_checkpoints)
    python test_lora_checkpoint_sweep.py --ckpt-dir path/to/save/ --final final.safetensors --trigger mytrigger --no-control
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

from diffusers import AutoencoderTiny, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_PROMPT = "a tall art deco tower, architectural drawing"
DEFAULT_NEG_PROMPT = "blurry, low quality"

def resolve_local_snapshot(model: str) -> str:
    """Return the cached snapshot folder for a Hub repo id when one exists, else `model` unchanged.

    huggingface_hub >= 1.22 stores a listing of the repo's full file tree next to the cache and,
    under local_files_only=True, refuses a snapshot that lacks any file in that listing -- even
    files diffusers never asked for (fp32 duplicates, ONNX/Flax weights, README images). The kit
    only ever cached what SDXL base needs, so this points diffusers at the snapshot folder itself,
    which loads the same files without the listing check. Found 2026-09-11 when every rendering
    script in the kit started failing with IncompleteSnapshotError on a cache that had rendered
    all of Round 2; verified by re-rendering a Round 2 grid byte for byte after the change.
    """
    if os.path.isdir(model):
        return model
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "hub"
    )
    repo_dir = Path(hub) / ("models--" + model.replace("/", "--"))
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        snap = repo_dir / "snapshots" / ref.read_text().strip()
        if (snap / "model_index.json").is_file():
            return str(snap)
    return model


# Matches OneTrainer's own intermediate-checkpoint naming convention, e.g.
# "...-save-300-60-0.safetensors" -> step 300. This is OneTrainer's naming, not
# machine-specific -- safe to keep as-is for any OneTrainer run.
STEP_RE = re.compile(r"-save-(\d+)-(\d+)-\d+\.safetensors$")


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    xa = np.asarray(a.convert("RGB"), dtype=np.float64)
    xb = np.asarray(b.convert("RGB"), dtype=np.float64)
    return float(np.abs(xa - xb).mean())


def discover_checkpoints(ckpt_dir: Path, final_path: Path) -> list[tuple[int, Path, bool]]:
    """Discover intermediate checkpoints plus the final checkpoint, de-duplicated.

    If --final resolves to a path already found inside --ckpt-dir (a common
    OneTrainer layout -- the last "-save-N-..." checkpoint is often also the one
    passed as --final), it's flagged via the third tuple element (is_final)
    rather than appended a second time as a redundant sentinel entry, so it
    gets rendered once, not twice.
    """
    found: list[list] = []
    for p in sorted(ckpt_dir.glob("*.safetensors")):
        m = STEP_RE.search(p.name)
        if m:
            found.append([int(m.group(1)), p, False])
    found.sort(key=lambda t: t[0])

    final_resolved = final_path.resolve() if final_path.exists() else None
    dup_index = None
    if final_resolved is not None:
        for i, (_, p, _) in enumerate(found):
            if p.resolve() == final_resolved:
                dup_index = i
                break

    if final_resolved is not None:
        if dup_index is not None:
            found[dup_index][2] = True
        else:
            found.append([-1, final_path, True])  # -1 = sort-last sentinel, label overridden below

    return [(step, path, is_final) for step, path, is_final in found]


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA checkpoint sweep across training steps")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ckpt-dir", required=True, help="Directory of OneTrainer intermediate checkpoints")
    parser.add_argument("--final", required=True, help="Path to the final/last-saved LoRA checkpoint")
    parser.add_argument(
        "--final-label",
        default=None,
        help="Label for the --final checkpoint in output filenames and the contact sheet "
        "(default: 'final(step{N})' using --final-step, or 'final' if --final-step is omitted). "
        "Get this from your training tool's own step count -- don't guess.",
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=None,
        help="Training step the --final checkpoint corresponds to, used to build the default "
        "--final-label. Check your training run's own log/config for the true value.",
    )
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--trigger", required=True, help="Trigger word/phrase")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Skip the untriggered control render at each checkpoint (halves wall-clock, "
        "but loses the d_control gating-drift column -- see module docstring).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "lora_checkpoint_sweep"),
    )
    args = parser.parse_args()
    args.model = resolve_local_snapshot(args.model)

    final_label = args.final_label
    if final_label is None:
        final_label = f"final(step{args.final_step})" if args.final_step is not None else "final"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(Path(args.ckpt_dir), Path(args.final))
    if not checkpoints:
        print("No checkpoints found.")
        return 1

    print(f"Loading {args.model} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True
    )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    triggered_prompt = f"{args.trigger}, {args.prompt}"
    cells = []
    rows = []
    for train_step, ckpt_path, is_final in checkpoints:
        if is_final:
            label = final_label if train_step == -1 else f"step{train_step}({final_label})"
        else:
            label = f"step{train_step}"
        print(f"--- {label} --- {ckpt_path.name}")
        pipe.load_lora_weights(str(ckpt_path))

        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        img = pipe(
            prompt=triggered_prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            cross_attention_kwargs={"scale": args.weight},
        ).images[0]

        d_control = None
        if not args.no_control:
            gen_ctrl = torch.Generator(device="cuda").manual_seed(args.seed)
            control_img = pipe(
                prompt=args.prompt,  # same prompt, trigger word omitted
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen_ctrl,
                cross_attention_kwargs={"scale": args.weight},
            ).images[0]
            d_control = mean_abs_diff(img, control_img)
            control_img.save(out_dir / f"{label}__control.png")

        pipe.unload_lora_weights()

        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 200, 24], fill=(0, 0, 0))
        control_text = f"  d_ctrl={d_control:.1f}" if d_control is not None else ""
        draw.text((4, 4), f"{label}{control_text}", fill=(255, 255, 255))

        path = out_dir / f"{label}.png"
        img.save(path)
        cells.append(img)
        rows.append(
            {
                "label": label,
                "train_step": train_step,
                "checkpoint_file": ckpt_path.name,
                "is_final": is_final,
                "d_control_untriggered_vs_triggered": round(d_control, 3) if d_control is not None else None,
            }
        )
        print(f"Saved: {path}" + (f"  (d_control={d_control:.3f})" if d_control is not None else ""))

    cols = 5
    grid_rows = (len(cells) + cols - 1) // cols
    w, h = cells[0].size
    contact = Image.new("RGB", (w * cols, h * grid_rows), (32, 32, 32))
    for i, img in enumerate(cells):
        r, c = divmod(i, cols)
        contact.paste(img, (c * w, r * h))
    contact_path = out_dir / "contact_sheet.png"
    contact.save(contact_path)

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "lora_dir": args.ckpt_dir,
                "final": args.final,
                "trigger": args.trigger,
                "weight": args.weight,
                "seed": args.seed,
                "control_rendered": not args.no_control,
                "checkpoints": rows,
            },
            indent=2,
        )
    )

    print("=" * 60)
    print(f"Saved contact sheet: {contact_path}")
    print(f"Saved summary: {summary_path}")
    if not args.no_control:
        print("Column key: d_control (printed on each tile as d_ctrl=) is the mean per-pixel-per-")
        print("channel |RGB delta|, 0-255 scale, between the triggered render and the SAME seed")
        print("with the trigger word omitted (both LoRA on) -- watch for it trending toward 0")
        print("across the sweep (gating leaking) rather than staying stable.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
