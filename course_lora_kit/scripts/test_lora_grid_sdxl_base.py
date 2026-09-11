"""
SDXL base 1.0 LoRA validation grid -- plain diffusers, no StreamDiffusion wrapper.

Runs a 2x2 matrix at a normal (non-turbo) step count: LoRA on/off x trigger word
present/absent, same seed. Settles three things in one pass: did training take,
does the trigger word gate the style, does the style leak without the trigger.

SDXL base 1.0 is generally not in a TouchDesigner StreamDiffusion component's model
menu (which is usually hardcoded to a short turbo/lightning list), so this grid has
to run outside the component via plain diffusers.

By default runs a single 2x2 grid for --prompt. Pass --prompts-file to run the same
2x2 matrix across an entire prompt battery (e.g. the verification ladder's
identity/flexibility/override/class-hijack/out-of-domain prompts) in one invocation --
one grid_2x2__{label}.png per prompt, plus a combined summary. The file is JSON: a
list of either [label, prompt] pairs or {"label": ..., "prompt": ...} objects.

Needs only plain diffusers -- no streamdiffusion import, no StreamDiffusion checkout.

Usage:
    python test_lora_grid_sdxl_base.py --lora path/to/lora.safetensors --trigger mytrigger
    python test_lora_grid_sdxl_base.py --lora path/to/lora.safetensors --trigger mytrigger --prompts-file battery.json

battery.json example:
    [
        ["identity", "portrait, looking at viewer, plain background"],
        {"label": "flexibility", "prompt": "standing in a snowy forest at night, wide shot"}
    ]
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

# Set HF_HOME (or export it in your shell) before this script imports diffusers if
# you want a non-default Hugging Face cache location. Reads the env var as-is if
# already set; only touches it if unset.
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



def load_prompt_battery(prompts_file: str | None, fallback_prompt: str) -> list[tuple[str, str]]:
    """Load a [(label, prompt), ...] battery from --prompts-file, or a single
    ("default", fallback_prompt) entry when no file is given. See the module
    docstring for the JSON format.
    """
    if prompts_file is None:
        return [("default", fallback_prompt)]
    data = json.loads(Path(prompts_file).read_text())
    battery = []
    for entry in data:
        if isinstance(entry, dict):
            battery.append((entry["label"], entry["prompt"]))
        else:
            label, prompt = entry
            battery.append((label, prompt))
    if not battery:
        raise ValueError(f"--prompts-file {prompts_file} contained no prompts")
    return battery


def render_grid(pipe, trigger: str, prompt: str, args) -> dict[str, Image.Image]:
    cells = {}
    for lora_on in (False, True):
        scale = args.weight if lora_on else 0.0
        for trigger_on in (False, True):
            cell_prompt = f"{trigger}, {prompt}" if trigger_on else prompt
            label = f"lora_{lora_on}_trigger_{trigger_on}"
            print(f"--- {label} --- prompt={cell_prompt!r} scale={scale}")
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            img = pipe(
                prompt=cell_prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
                cross_attention_kwargs={"scale": scale},
            ).images[0]
            cells[label] = img
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description="SDXL base 1.0 LoRA 2x2 validation grid")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lora", required=True, help="HF repo id or local .safetensors path")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--trigger", required=True, help="Trigger word/phrase to test gating for")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompts-file",
        default=None,
        metavar="FILE",
        help="JSON file of [label, prompt] entries to run the 2x2 grid across a whole battery "
        "in one invocation instead of just --prompt -- see module docstring for the format.",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "lora_sdxl_base_grid"),
    )
    args = parser.parse_args()
    args.model = resolve_local_snapshot(args.model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    battery = load_prompt_battery(args.prompts_file, args.prompt)

    print(f"Loading {args.model} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    # A fork of diffusers that adds kvo_cache to its Attention/AttnProcessor2_0 API
    # (varshith15/diffusers, used by some StreamDiffusion builds) can leave
    # AutoencoderKL's UNetMidBlock2D.forward unaware of its own new tuple-returning
    # attention contract -- the full VAE's self-attention mid-block then passes that
    # tuple straight into the next resnet and crashes with "AttributeError: 'tuple'
    # object has no attribute 'dim'". TinyVAE has no attention blocks, so it never
    # touches the broken path -- and it's what most real-time components use by
    # default anyway. If you're on stock diffusers this substitution is unnecessary
    # but harmless.
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True
    )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"Loading LoRA weights from {args.lora} ...")
    pipe.load_lora_weights(args.lora)

    grid_paths = []
    for prompt_label, prompt in battery:
        print(f"=== battery entry: {prompt_label} ===")
        cells = render_grid(pipe, args.trigger, prompt, args)

        suffix = "" if prompt_label == "default" and args.prompts_file is None else f"__{prompt_label}"
        for label, img in cells.items():
            path = out_dir / f"{label}{suffix}.png"
            img.save(path)
            print(f"Saved: {path}")

        w, h = next(iter(cells.values())).size
        grid = Image.new("RGB", (w * 2, h * 2))
        grid.paste(cells["lora_False_trigger_False"], (0, 0))
        grid.paste(cells["lora_False_trigger_True"], (w, 0))
        grid.paste(cells["lora_True_trigger_False"], (0, h))
        grid.paste(cells["lora_True_trigger_True"], (w, h))
        grid_name = "grid_2x2.png" if suffix == "" else f"grid_2x2{suffix}.png"
        grid_path = out_dir / grid_name
        grid.save(grid_path)
        grid_paths.append((prompt_label, prompt, grid_path))
        print(f"Saved grid: {grid_path}")

    print("=" * 60)
    for prompt_label, prompt, grid_path in grid_paths:
        print(f"[{prompt_label}] prompt={prompt!r} -> {grid_path}")
    print("Layout: TL=no LoRA/no trigger  TR=no LoRA/trigger  BL=LoRA/no trigger  BR=LoRA/trigger")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
