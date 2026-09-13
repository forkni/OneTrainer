"""
Seed batch for one LoRA checkpoint: is the pick stable across seeds, and does its
colour drift? Plain diffusers on SDXL base 1.0 -- no StreamDiffusion wrapper.

The grid (test_lora_grid_sdxl_base.py) and the sweep (test_lora_checkpoint_sweep.py)
each render ONE seed. Round 2 of the course's style track found the failure that a
single seed hides: the unweighted control recipe (S0) produced 2 clean / 1 drifting /
1 pink render across four seeds at the same checkpoint, while the corrected recipe
(S0b, Min-SNR gamma 5 + offset noise 0.03) produced 4 coherent renders. Per-seed
colour was the tell: image-mean CIELAB a* (green < 0, magenta > 0) sat at 0.6-1.5 on
the LoRA-off controls, peaked around 13 on S0b renders, and spiked to 28-31 on the S0
renders that had gone pink.

What this script does, for one --lora:
  * renders --seeds N seeds of the same triggered prompt with the LoRA on (weight
    --weight), and the same N seeds with the LoRA OFF (the base model's own colour);
  * measures mean a* / b* of every render (the color_stats.py metric, same formula);
  * writes seed_batch/<label>__sNN.png, <label>__sNN__control.png, a contact sheet
    (top row LoRA on, bottom row LoRA off) and seed_batch_summary.json;
  * counts renders whose a* exceeds --a-threshold and prints the verdict.

The threshold is PROVISIONAL, and a flag marks a render to LOOK AT, not to reject.
20 sits between the a* clusters Round 2 measured at seed 42 (color_stats.py over the
sweep renders: controls 0.6-1.5, S0b renders up to ~13 across 13 checkpoints, S0's pink
renders 28-31). Run on the course's S0b pick (step 1099) with this script's default
prompt and seeds 42-45 (2026-09-11), the pick itself read a* 12.1 / 9.7 / 28.5 / 25.3
against controls 0.3-0.6: two of four over the line, both structurally intact -- seed 44
a purple-orange sunset sky over a cream tower (a palette; the neutrals held), seed 45
the tower itself salmon-pink (the tint reached the neutrals). Mean a* measures tint, not
collapse, and cannot tell a warm palette from drift; the by-eye tell is whether the
neutrals stayed neutral. Round 2's "4/4 coherent" verdict on S0b was a live ComfyUI
batch judged by eye, prompt and seeds not recorded; this script is the first per-seed
a* on the pick. Re-derive the line for your own set: run this on a checkpoint you can
see is clean and put the line between its LoRA-on readings and a render you can see has
drifted. The script prints the numbers; you set the line.

Usage:
    python test_lora_seed_batch.py --lora path/to/lora.safetensors --trigger mytrigger
    python test_lora_seed_batch.py --lora path/to/lora.safetensors --trigger mytrigger --seeds 6 --a-threshold 15
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

from diffusers import AutoencoderTiny, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_PROMPT = "a tall art deco tower, architectural drawing"
DEFAULT_NEG_PROMPT = "blurry, low quality"
DEFAULT_A_THRESHOLD = 20.0  # provisional -- see module docstring

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



def mean_lab(img: Image.Image) -> tuple[float, float]:
    """(mean a*, mean b*) of an image: sRGB -> linear -> XYZ (D65) -> CIELAB. Same as color_stats.py."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])  # D65 white
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return float(a.mean()), float(b.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed batch + colour-drift count for one LoRA checkpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lora", required=True, help="Path to the LoRA .safetensors to test")
    parser.add_argument("--trigger", required=True, help="Trigger word/phrase (copy it from a caption file)")
    parser.add_argument("--label", default=None, help="Name used in output files (default: the LoRA file stem)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--seeds", type=int, default=4, help="How many seeds to render (default 4)")
    parser.add_argument("--first-seed", type=int, default=42, help="Seeds are first-seed, first-seed+1, ...")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument(
        "--a-threshold",
        type=float,
        default=DEFAULT_A_THRESHOLD,
        help=f"Flag a render whose mean a* exceeds this (default {DEFAULT_A_THRESHOLD}, provisional -- see docstring)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "seed_batch"),
    )
    args = parser.parse_args()
    args.model = resolve_local_snapshot(args.model)

    label = args.label or Path(args.lora).stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.first_seed + i for i in range(args.seeds)]

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

    def render(prompt: str, seed: int, scale: float) -> Image.Image:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=gen,
            cross_attention_kwargs={"scale": scale},
        ).images[0]

    # LoRA-off controls first (base model only), then LoRA-on -- one load, one unload.
    controls: dict[int, Image.Image] = {}
    for seed in seeds:
        controls[seed] = render(triggered_prompt, seed, 0.0)

    print(f"Loading LoRA weights from {args.lora} ...")
    pipe.load_lora_weights(args.lora)
    renders: dict[int, Image.Image] = {}
    for seed in seeds:
        renders[seed] = render(triggered_prompt, seed, args.weight)
    pipe.unload_lora_weights()

    rows = []
    flagged = 0
    print("=" * 72)
    print(f"{'seed':>6}  {'a*  LoRA':>9}  {'b*  LoRA':>9}  {'a* ctrl':>8}  {'b* ctrl':>8}  flag")
    for seed in seeds:
        a_on, b_on = mean_lab(renders[seed])
        a_off, b_off = mean_lab(controls[seed])
        flag = a_on > args.a_threshold
        flagged += int(flag)
        print(f"{seed:>6}  {a_on:>9.2f}  {b_on:>9.2f}  {a_off:>8.2f}  {b_off:>8.2f}  {'<<< a* over threshold' if flag else ''}")
        for tag, img in (("", renders[seed]), ("__control", controls[seed])):
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 260, 24], fill=(0, 0, 0))
            a_val = a_on if tag == "" else a_off
            draw.text((4, 4), f"{label} s{seed}{' ctrl' if tag else ''}  a*={a_val:.1f}", fill=(255, 255, 255))
            img.save(out_dir / f"{label}__s{seed}{tag}.png")
        rows.append(
            {
                "seed": seed,
                "a_star_lora": round(a_on, 3),
                "b_star_lora": round(b_on, 3),
                "a_star_control": round(a_off, 3),
                "b_star_control": round(b_off, 3),
                "flagged": flag,
            }
        )

    w, h = renders[seeds[0]].size
    contact = Image.new("RGB", (w * len(seeds), h * 2), (32, 32, 32))
    for i, seed in enumerate(seeds):
        contact.paste(renders[seed], (i * w, 0))
        contact.paste(controls[seed], (i * w, h))
    contact_path = out_dir / f"{label}__contact_sheet.png"
    contact.save(contact_path)

    summary = {
        "lora": args.lora,
        "label": label,
        "trigger": args.trigger,
        "prompt": triggered_prompt,
        "weight": args.weight,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seeds": seeds,
        "a_threshold": args.a_threshold,
        "a_threshold_note": "provisional; sits between the seed-42 sweep clusters (controls 0.6-1.5, S0b <= ~13, S0 pink 28-31); the S0b pick's own 4-seed batch read 12.1/9.7/28.5/25.3 -- a flag means look, not reject",
        "flagged_count": flagged,
        "renders": rows,
    }
    summary_path = out_dir / f"{label}__seed_batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 72)
    print(f"Saved contact sheet: {contact_path}  (top row LoRA on, bottom row LoRA off)")
    print(f"Saved summary: {summary_path}")
    print(f"a* over {args.a_threshold:g} (provisional line): {flagged}/{len(seeds)} renders")
    print("Column key: a*/b* LoRA are this render's mean CIELAB (LoRA on, at --weight); a*/b* ctrl")
    print("are the SAME seed and prompt with the LoRA at scale 0.0. flag = a* LoRA strictly above")
    print("--a-threshold (LoRA-on only; the control's own a* is not compared to the line).")
    print("Controls give the base model's own colour at these seeds; read the LoRA-on")
    print("column against them, not against zero. A flag marks a render to look at: did")
    print("the tint reach the neutrals (drift), or only the palette (a sunset is not a")
    print("failure)? Round 2's S0 read 2 clean / 1 drifting / 1 pink by eye at one")
    print("checkpoint; the S0b pick read 2/4 over this line on the default prompt with")
    print("both renders intact. Several drifted seeds at one checkpoint: fix the recipe,")
    print("not the seed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
