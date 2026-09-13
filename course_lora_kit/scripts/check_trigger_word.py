"""
Trigger-word check -- is a candidate trigger free of loaded meaning?

Two stages, run before you caption a dataset with the trigger:

1. Tokenizer split (instant, no GPU): prints how both of SDXL's text encoders
   break the trigger into BPE tokens. A split into generic multi-token fragments
   is what you want; a single dedicated token means the model already has an
   embedding for the word -- pick a more made-up one.

2. Render check (needs the base model in the local HF cache): renders the bare
   trigger as the ENTIRE prompt at several seeds, next to empty-prompt controls
   at the same seeds, and saves one labeled contact sheet.

Reading the sheet: if the trigger-only row shows consistent, specific imagery
that the empty-prompt row doesn't -- the same face, product, or art style, seed
after seed -- the token already carries meaning in the base model. Pick another
trigger and re-run. If the trigger row is as unrelated image-to-image as the
control row, the token is free ground.

The render stage loads with local_files_only=True like the other validation
scripts -- it uses the cache but won't download the multi-gigabyte base model
(your first training run populates the cache). The tokenizer stage allows a
tiny one-off download of the two tokenizer files if they aren't cached yet, so
stage 1 works even before your first training run.

Needs only plain diffusers -- no streamdiffusion import, no StreamDiffusion checkout.

Cache resolution order: --hf-home flag > HF_HOME environment variable > Hugging
Face's own default (~/.cache/huggingface). The cmd wrapper sets HF_HOME to the
repo's workspace cache only when the variable isn't already set, so an exported
HF_HOME always wins over the repo default.

Usage:
    python check_trigger_word.py --trigger mytrigger
    python check_trigger_word.py --trigger mytrigger --seeds 6 --steps 25
    python check_trigger_word.py --trigger mytrigger --hf-home X:\\hf_cache
"""

import argparse
import os
from pathlib import Path

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def print_token_split(model: str, trigger: str) -> None:
    """Stage 1: print the BPE split from both SDXL text-encoder tokenizers."""
    from transformers import CLIPTokenizer, CLIPTokenizerFast

    print("=" * 60)
    print(f"Tokenizer split for trigger: {trigger!r}  (model: {model})")
    for subfolder, cls in (("tokenizer", CLIPTokenizer), ("tokenizer_2", CLIPTokenizerFast)):
        try:
            tok = cls.from_pretrained(model, subfolder=subfolder, local_files_only=True)
        except OSError:
            # Tokenizer files are a couple of megabytes -- unlike the render
            # stage, a one-off download here is cheap and lets the split print
            # before the first training run has populated the cache.
            tok = cls.from_pretrained(model, subfolder=subfolder)
        pieces = tok.tokenize(trigger)
        print(f"  {subfolder}: {len(pieces)} token(s) -> {pieces}")
    print(
        "  Reading: several generic fragments = good (the token trains fine and\n"
        "  gates fine). ONE dedicated token = the model already has an embedding\n"
        "  for this exact word -- pick a more made-up trigger."
    )
    print("=" * 60)


def label_cell(img, text: str):
    """Stamp a small label strip onto the top of a cell (in place)."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1: no size argument
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 40], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)


def resolve_local_snapshot(model: str):
    """Return the cached snapshot folder for a hub id, or None if not cached.

    Loading by folder path matters: offline, huggingface_hub refuses a cached
    repo unless EVERY repo file is present -- including READMEs and the
    multi-gigabyte fp32/single-file checkpoints a variant-only download
    deliberately skips. A directory load only checks the files the pipeline
    actually needs.
    """
    if Path(model).exists():
        return Path(model)
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(model, "model_index.json")
    if isinstance(hit, str):
        return Path(hit).parent
    return None


def render_contact_sheet(args) -> int:
    """Stage 2: bare-trigger renders vs empty-prompt controls, same seeds."""
    import torch
    from PIL import Image

    try:
        from diffusers import AutoencoderTiny, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

        model_src = resolve_local_snapshot(args.model) or args.model
        print(f"Loading {args.model} ...")
        # Prefer the fp16-variant weights (half the download/disk of fp32);
        # fall back to the non-variant files for caches populated with fp32.
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_src,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                local_files_only=True,
            )
        except OSError:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_src,
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=True,
            )
        # Same TinyVAE substitution as the other validation scripts: a fork of
        # diffusers used by some StreamDiffusion builds (varshith15/diffusers)
        # breaks the full VAE's attention mid-block; TinyVAE has no attention
        # blocks and sidesteps it. On stock diffusers this is unnecessary but
        # harmless.
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True
        )
    except OSError as exc:
        print("-" * 60)
        print("Render stage SKIPPED: the base model is not in the local HF cache.")
        print(f"  ({exc})")
        print("The tokenizer split above still stands. Run this script again after")
        print("your first training run has populated the cache (or pre-download the")
        print("model), and it will render the contact sheet.")
        print("-" * 60)
        return 0

    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.base_seed + i * 1000 for i in range(args.seeds)]
    rows = {}
    for row_label, prompt in (("trigger", args.trigger), ("control", "")):
        cells = []
        for seed in seeds:
            print(f"--- {row_label} seed={seed} --- prompt={prompt!r}")
            gen = torch.Generator(device="cuda").manual_seed(seed)
            img = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
            ).images[0]
            label_cell(img, f"{row_label} | seed {seed}" + (f" | {args.trigger!r}" if row_label == "trigger" else " | (empty prompt)"))
            path = out_dir / f"{row_label}_seed{seed}.png"
            img.save(path)
            print(f"Saved: {path}")
            cells.append(img)
        rows[row_label] = cells

    w, h = rows["trigger"][0].size
    sheet = Image.new("RGB", (w * len(seeds), h * 2))
    for col, img in enumerate(rows["trigger"]):
        sheet.paste(img, (col * w, 0))
    for col, img in enumerate(rows["control"]):
        sheet.paste(img, (col * w, h))
    sheet_path = out_dir / "contact_sheet.png"
    sheet.save(sheet_path)

    print("=" * 60)
    print(f"Saved contact sheet: {sheet_path}")
    print("Layout: top row = bare trigger as the whole prompt, bottom row = empty")
    print("prompt, same seed per column.")
    print("Reading: consistent SPECIFIC imagery in the top row that the bottom row")
    print("doesn't show (same face / product / art style, seed after seed) = the")
    print("token already carries meaning -- pick another trigger and re-run.")
    print("Top row as unrelated image-to-image as the bottom row = free ground.")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a candidate LoRA trigger word for loaded meaning")
    parser.add_argument("--trigger", required=True, help="Candidate trigger word/phrase to check")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", type=int, default=4, help="Number of seeds (columns) to render")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "check_trigger_word"),
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Hugging Face cache folder for this run (sets HF_HOME; overrides the environment)",
    )
    args = parser.parse_args()

    # Must happen before the lazy transformers/diffusers imports inside the
    # stage functions -- they read HF_HOME at import time.
    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser().resolve())
    if args.hf_home:
        source = "--hf-home"
    elif os.environ.get("HF_HOME"):
        source = "HF_HOME env"
    else:
        source = "HF default"
    hub_cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    )
    print(f"HF cache in use: {hub_cache}  ({source})")

    print_token_split(args.model, args.trigger)
    return render_contact_sheet(args)


if __name__ == "__main__":
    raise SystemExit(main())
