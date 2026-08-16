"""
LoRA identity & consistency measurement -- DINOv2 embeddings, plain diffusers,
no StreamDiffusion wrapper.

Every other script in this folder measures a LoRA by mean-absolute PIXEL
difference: does the image change when the LoRA/trigger is on vs off. That
answers "did something happen", not "is it still the same face" -- for a
character LoRA (as opposed to the style LoRA these scripts were originally
built and measured against), identity surviving prompt changes and seed
changes IS the deliverable, and pixel diff cannot see it: two renders of two
different-looking people can have a smaller pixel diff than two renders of the
same person in different lighting.

This script instead embeds every image with DINOv2 (facebook/dinov2-base by
default), a self-supervised vision model whose embedding cosine tracks
perceptual/identity similarity far better than raw pixels or even CLIP's image
tower (CLIP is trained to align with TEXT, which makes it good at "does this
image match this prompt" -- used here for exactly that -- but a weaker judge of
"is this the same subject" than a model with no text supervision at all).

Per LoRA arm (a label pointing at a checkpoint, NOT just a bare file path --
see --lora), across a prompt battery x seed grid, it computes:

    identity                Mean DINOv2 cosine of every generated image against
                             every image in --reference-dir (held-out images
                             NEVER trained on). High = the arm reliably
                             reproduces the trained subject's identity.

    cross_seed_consistency  Mean pairwise DINOv2 cosine among images from the
                             SAME prompt at different seeds. High = the "same"
                             prompt draws the same person every time, not a
                             different one per noise draw.

    flexibility             1 - mean pairwise DINOv2 cosine among images from
                             DIFFERENT prompts (one representative seed each).
                             This is the check identity alone can't provide: a
                             memorized/collapsed checkpoint can still score
                             high identity (it always renders "the trained
                             photo") while flexibility collapses toward 0,
                             because every prompt produces the same output
                             regardless of what it asked for.

    clip_score               Mean CLIP image-text cosine between each generated
                             image and the prompt text that produced it (via
                             --clip-model, default openai/clip-vit-base-patch32)
                             -- a second, independent read on the same
                             memorization failure mode: has the arm stopped
                             listening to the prompt at all.

A healthy arm: high identity, high cross-seed consistency, high flexibility,
non-collapsing CLIP score. A memorized/overcooked arm: identity can still look
good, but flexibility and CLIP score drop as the checkpoint stops responding to
prompt text -- the same failure checkpoint_norm_analyzer.py flags via a
||dW||_F knee and adjacent-checkpoint cosine drop, seen here from the rendered
side instead of the weight-delta side.

--lora is repeatable and takes LABEL=PATH, not just PATH -- this script compares
ARMS (different training configs/checkpoints), not one file's checkpoints in
isolation. Two built-in conveniences for the smoke test this course's fact-check
process uses before trusting any new script:
  - LABEL=none renders on the bare base model with no LoRA loaded (scale forced
    to 0 regardless of --weight/--lora-scale) -- a genuine no-LoRA baseline.
  - --lora-scale LABEL=SCALE overrides --weight per label, so a same-file
    on/off-scale control is one extra flag, not a second full run:
        --lora on=lora.safetensors --lora off=lora.safetensors --lora-scale off=0.0
    The "off" (scale 0) arm must score near the base model's own baseline
    identity against your reference set, not the trained arm's -- if it doesn't,
    something in the harness (not the LoRA) is producing the identity signal.

Needs: numpy, Pillow, torch, diffusers, safetensors (via diffusers' own LoRA
loader), and transformers (AutoModel/AutoImageProcessor for DINOv2,
CLIPModel/CLIPProcessor for the CLIP score) -- one more dependency than the
other render scripts in this folder.

Usage:
    # Smoke test: on vs scale-0 control, one existing LoRA
    python test_lora_identity_consistency.py \\
        --lora on=lora.safetensors --lora off=lora.safetensors --lora-scale off=0.0 \\
        --reference-dir path/to/held_out_refs/

    # Round 1: compare training arms
    python test_lora_identity_consistency.py \\
        --lora A1=arm1/final.safetensors --lora A2=arm2/final.safetensors \\
        --lora A3=arm3/final.safetensors --lora A4=arm4/final.safetensors \\
        --reference-dir path/to/held_out_refs/ --prompts-file battery.json

battery.json example (same [label, prompt] / {"label":..,"prompt":..} shape as
test_lora_grid_sdxl_base.py's --prompts-file):
    [
        ["identity", "lainiwakura, portrait, looking at viewer, plain background"],
        {"label": "flexibility", "prompt": "lainiwakura, standing in a snowy forest at night, wide shot"}
    ]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

from diffusers import AutoencoderTiny, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEG_PROMPT = "blurry, low quality"
DEFAULT_DINO_MODEL = "facebook/dinov2-base"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_SEEDS = [42, 1234, 7777]
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# A minimal 3-prompt battery covering identity, flexibility, and a described
# override -- swap via --prompts-file for the full verification-ladder battery
# (which also adds class-hijack and out-of-domain prompts -- those belong in
# test_lora_gating_measure.py instead, since gating, not identity, is what they
# test).
DEFAULT_PROMPTS = [
    ("identity", "portrait, looking at viewer, plain background"),
    ("flexibility", "standing in a snowy forest at night, wide shot"),
    ("override", "wearing a red raincoat, city street"),
]


def parse_lora_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--lora expects LABEL=PATH, got: {raw!r}")
    label, path = raw.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--lora expects LABEL=PATH, got: {raw!r}")
    return label, path


def parse_scale_arg(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--lora-scale expects LABEL=SCALE, got: {raw!r}")
    label, scale = raw.split("=", 1)
    return label.strip(), float(scale.strip())


def load_prompts(prompts_file: str | None) -> list[tuple[str, str]]:
    if prompts_file is None:
        return list(DEFAULT_PROMPTS)
    data = json.loads(Path(prompts_file).read_text())
    prompts = []
    for entry in data:
        if isinstance(entry, dict):
            prompts.append((entry["label"], entry["prompt"]))
        else:
            label, prompt = entry
            prompts.append((label, prompt))
    if not prompts:
        raise ValueError(f"--prompts-file {prompts_file} contained no prompts")
    return prompts


def discover_reference_images(reference_dir: Path) -> list[Path]:
    return sorted(p for p in reference_dir.rglob("*") if p.is_file() and p.suffix.lower() in DEFAULT_EXTENSIONS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


class DinoEmbedder:
    """Wraps DINOv2 for L2-normalized whole-image embeddings.

    Uses `pooler_output` if the model provides one, else the first
    (CLS-equivalent) token of `last_hidden_state` -- the standard choice for
    whole-image similarity with DINOv2.
    """

    def __init__(self, model_name: str, device: str):
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        vec = pooled[0].float().cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec


class ClipScorer:
    """Wraps CLIP for image-text cosine similarity (prompt adherence)."""

    def __init__(self, model_name: str, device: str):
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(self, img: Image.Image, text: str) -> float:
        inputs = self.processor(
            text=[text], images=[img.convert("RGB")], return_tensors="pt", padding=True
        ).to(self.device)
        out = self.model(**inputs)
        image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((image_embeds @ text_embeds.T)[0, 0].cpu())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DINOv2-based identity/consistency/flexibility measurement across LoRA arms."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--lora",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable. LABEL=PATH to a kohya-format .safetensors LoRA, or LABEL=none for a "
        "no-LoRA baseline (scale forced to 0). Pass one entry per training arm/checkpoint you "
        "want compared in the same contact sheet and JSON table -- see module docstring.",
    )
    parser.add_argument(
        "--lora-scale",
        action="append",
        default=[],
        metavar="LABEL=SCALE",
        help="Repeatable. Per-label LoRA scale override (default: --weight). E.g. "
        "--lora-scale off=0.0 for an on/off-scale smoke test -- see module docstring.",
    )
    parser.add_argument("--weight", type=float, default=1.0, help="Default LoRA scale for labels without --lora-scale.")
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory of held-out reference images (never trained on) establishing ground-truth identity.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        metavar="FILE",
        help="JSON list of [label, prompt] pairs or {'label':..,'prompt':..} objects. Defaults to "
        "a 3-prompt identity/flexibility/override battery -- see module docstring.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Seeds rendered per (label, prompt), for cross-seed consistency (default: {DEFAULT_SEEDS}).",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "lora_identity_consistency"),
    )
    parser.add_argument("--json", metavar="FILE", help="Also write the results table to this JSON path (in addition to output-dir's copy).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loras = [parse_lora_arg(raw) for raw in args.lora]
    scale_overrides = dict(parse_scale_arg(raw) for raw in args.lora_scale)
    prompts = load_prompts(args.prompts_file)
    prompt_text_by_label = {pl: pt for pl, pt in prompts}
    prompt_labels_all = [pl for pl, _ in prompts]

    reference_dir = Path(args.reference_dir)
    reference_paths = discover_reference_images(reference_dir)
    if not reference_paths:
        print(f"No reference images found in {reference_dir}", file=sys.stderr)
        return 1

    device = "cuda"

    print(f"Loading DINOv2 ({args.dino_model}) ...")
    dino = DinoEmbedder(args.dino_model, device)
    print(f"Loading CLIP ({args.clip_model}) ...")
    clip = ClipScorer(args.clip_model, device)

    print(f"Embedding {len(reference_paths)} reference image(s) ...")
    reference_embeds = [dino.embed(Image.open(p)) for p in reference_paths]

    print(f"Loading {args.model} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
    )
    # See test_lora_grid_sdxl_base.py for why TinyVAE is substituted here.
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True
    )
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    label_results = {}
    contact_rows = []  # (label, {prompt_label: representative_image})

    for label, lora_path in loras:
        scale = scale_overrides.get(label, args.weight)
        is_none = lora_path.strip().lower() == "none"
        if is_none:
            scale = 0.0
            print(f"=== {label}: no-LoRA baseline (scale forced 0) ===")
        else:
            print(f"=== {label}: loading {lora_path} @ scale {scale} ===")
            pipe.load_lora_weights(lora_path)

        images: dict[str, dict[int, Image.Image]] = {}
        for prompt_label, prompt_text in prompts:
            images[prompt_label] = {}
            for seed in args.seeds:
                print(f"    {label}/{prompt_label}/seed{seed}")
                gen = torch.Generator(device="cuda").manual_seed(seed)
                img = pipe(
                    prompt=prompt_text,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                    cross_attention_kwargs={"scale": scale},
                ).images[0]
                images[prompt_label][seed] = img
                img.save(out_dir / f"{label}__{prompt_label}__seed{seed}.png")

        if not is_none:
            pipe.unload_lora_weights()

        # Identity: mean DINOv2 cosine of every generated image vs. every
        # reference image, averaged across all prompts/seeds for this label
        # (identity_mean), and broken out per prompt (per_prompt_identity) --
        # the pooled mean hides cases where reference framing matches some
        # prompts (e.g. a portrait prompt against head/face references) much
        # better than others (e.g. a wide shot), which is exactly the gap
        # between the "identity" and "flexibility" prompts in our battery.
        identity_scores = []
        per_prompt_identity = {}
        for prompt_label, by_seed in images.items():
            prompt_scores = []
            for img in by_seed.values():
                emb = dino.embed(img)
                score = float(np.mean([cosine(emb, ref) for ref in reference_embeds]))
                identity_scores.append(score)
                prompt_scores.append(score)
            per_prompt_identity[prompt_label] = float(np.mean(prompt_scores)) if prompt_scores else None
        identity_mean = float(np.mean(identity_scores))
        identity_std = float(np.std(identity_scores))

        # Cross-seed consistency: same prompt, different seeds -- mean pairwise
        # cosine, averaged over prompts.
        per_prompt_consistency = {}
        for prompt_label, by_seed in images.items():
            embs = [dino.embed(img) for img in by_seed.values()]
            pairs = [cosine(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
            per_prompt_consistency[prompt_label] = float(np.mean(pairs)) if pairs else None
        valid_consistency = [v for v in per_prompt_consistency.values() if v is not None]
        cross_seed_consistency = float(np.mean(valid_consistency)) if valid_consistency else None

        # Flexibility: pairwise DINOv2 DISTANCE (1 - cosine) between DIFFERENT
        # prompts' representative (first-seed) images. See module docstring for
        # why this, not identity, is what catches memorization.
        representative = {pl: images[pl][args.seeds[0]] for pl in prompt_labels_all}
        rep_embeds = {pl: dino.embed(img) for pl, img in representative.items()}
        flex_pairs = [
            1.0 - cosine(rep_embeds[prompt_labels_all[i]], rep_embeds[prompt_labels_all[j]])
            for i in range(len(prompt_labels_all))
            for j in range(i + 1, len(prompt_labels_all))
        ]
        flexibility = float(np.mean(flex_pairs)) if flex_pairs else None

        # CLIP prompt-adherence score, per generated image, averaged.
        clip_scores = []
        for prompt_label, by_seed in images.items():
            prompt_text = prompt_text_by_label[prompt_label]
            for img in by_seed.values():
                clip_scores.append(clip.score(img, prompt_text))
        clip_score_mean = float(np.mean(clip_scores)) if clip_scores else None

        label_results[label] = {
            "lora_path": lora_path,
            "scale": scale,
            "identity_cosine_mean": round(identity_mean, 4),
            "identity_cosine_std": round(identity_std, 4),
            "identity_cosine_per_prompt": {
                k: (round(v, 4) if v is not None else None) for k, v in per_prompt_identity.items()
            },
            "cross_seed_consistency": round(cross_seed_consistency, 4) if cross_seed_consistency is not None else None,
            "cross_seed_consistency_per_prompt": {
                k: (round(v, 4) if v is not None else None) for k, v in per_prompt_consistency.items()
            },
            "flexibility_dino_distance": round(flexibility, 4) if flexibility is not None else None,
            "clip_score_mean": round(clip_score_mean, 4) if clip_score_mean is not None else None,
        }
        contact_rows.append((label, representative))
        per_prompt_str = "  ".join(f"{k}={v:.4f}" if v is not None else f"{k}=None" for k, v in per_prompt_identity.items())
        print(
            f"    identity={identity_mean:.4f} ({per_prompt_str})  cross_seed={cross_seed_consistency}  "
            f"flexibility={flexibility}  clip={clip_score_mean}"
        )

    # Contact sheet: rows = labels, cols = prompts (representative/first-seed image)
    w, h = next(iter(contact_rows[0][1].values())).size
    cols = len(prompt_labels_all)
    grid_rows = len(contact_rows)
    contact = Image.new("RGB", (w * cols, h * grid_rows), (32, 32, 32))
    for r, (label, thumbs) in enumerate(contact_rows):
        for c, prompt_label in enumerate(prompt_labels_all):
            tile = thumbs[prompt_label].copy()
            draw = ImageDraw.Draw(tile)
            draw.rectangle([0, 0, 220, 24], fill=(0, 0, 0))
            draw.text((4, 4), f"{label}/{prompt_label}", fill=(255, 255, 255))
            contact.paste(tile, (c * w, r * h))
    contact_path = out_dir / "contact_sheet.png"
    contact.save(contact_path)

    summary = {
        "reference_dir": str(reference_dir),
        "reference_count": len(reference_paths),
        "seeds": args.seeds,
        "prompts_file": args.prompts_file,
        "prompts": prompts,
        "dino_model": args.dino_model,
        "clip_model": args.clip_model,
        "labels": label_results,
    }
    summary_path = out_dir / "identity_consistency.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))

    print("=" * 100)
    print(f"{'label':<16} {'identity':>9} {'cross_seed':>11} {'flexibility':>12} {'clip_score':>11}")
    print("-" * 100)
    for label, r in label_results.items():
        cs = f"{r['cross_seed_consistency']:.4f}" if r["cross_seed_consistency"] is not None else "  --"
        fx = f"{r['flexibility_dino_distance']:.4f}" if r["flexibility_dino_distance"] is not None else "  --"
        cl = f"{r['clip_score_mean']:.4f}" if r["clip_score_mean"] is not None else "  --"
        print(f"{label:<16} {r['identity_cosine_mean']:>9.4f} {cs:>11} {fx:>12} {cl:>11}")
    print("-" * 100)
    print(f"Saved contact sheet: {contact_path}")
    print(f"Saved summary: {summary_path}")
    print("Healthy arm: high identity, high cross_seed, high flexibility, non-collapsing clip_score.")
    print("A memorized/overcooked arm keeps high identity but flexibility and clip_score drop --")
    print("cross-check against a ||dW||_F knee from checkpoint_norm_analyzer.py on the same checkpoint.")
    print("A LABEL=none or --lora-scale LABEL=0.0 control should score near the base model's own")
    print("baseline identity against your reference set, not the trained arm's -- if it doesn't,")
    print("something in the harness, not the LoRA, is producing the identity signal.")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
