"""
LoRA identity scoring for ALREADY-RENDERED images -- DINOv2 embeddings, no
diffusers pipeline, no GPU render step. Companion to
test_lora_identity_consistency.py, not a replacement for it.

That script renders its own images through a plain diffusers SDXL pipeline and
scores them -- useful when you want a controlled prompt/seed grid, but no help
when the images you actually care about came from somewhere else entirely: a
ComfyUI workflow, StreamDiffusionTD, a different machine, a different sampler.
This script skips the render step and scores whatever PNGs/JPGs you already
have against the same held-out reference set, using the same metric
(--reference-dir, DINOv2 cosine, facebook/dinov2-base by default) -- so a
number produced here is directly comparable to test_lora_identity_consistency.py's
"identity" column, and to Appendix D's own reference numbers, as long as the
--reference-dir is the same one.

Per labelled group of renders (--render LABEL=PATH, repeatable -- see below),
it reports:

    identity              Mean DINOv2 cosine of every image in the group
                           against every image in --reference-dir (held-out
                           images never trained on). High = the group reliably
                           reproduces the trained subject's identity.

    render_consistency     Mean pairwise DINOv2 cosine among the group's OWN
                           images (only computed when a label has 2+ images).
                           High = the renders you're comparing actually look
                           like each other, not just like the reference set
                           individually -- a label with high mean identity but
                           low render_consistency means at least one render in
                           the group drifted, which the mean alone would hide.

This deliberately reports fewer metrics than test_lora_identity_consistency.py
(no cross_seed_consistency, no flexibility, no clip_score) because those all
depend on knowing the prompt/seed grid that produced each image -- information
this script doesn't have and doesn't need for the one question it answers:
"does this arm's already-rendered output actually look like the reference
set, and does it look like itself across the renders I gave it."

--render LABEL=PATH is repeatable and PATH may be:
  - a single image file
  - a directory (every image file inside, non-recursive extension match)
  - a glob pattern (quote it so your shell doesn't expand it first), e.g.
    --render A2="C:\\path\\to\\output\\LAIN_A2_*.png"
Repeat --render with the SAME label to accumulate hand-picked files from a
folder that mixes several arms together (exactly the ComfyUI output-folder
case this script was written for) instead of one file/dir/glob per label.

Needs: numpy, Pillow, torch, transformers (AutoModel/AutoImageProcessor for
DINOv2). No diffusers, no CUDA required (falls back to CPU automatically),
much lighter than the render-and-score scripts in this folder.

Usage:
    # Score two ComfyUI renders per arm against the held-out reference set
    python test_lora_identity_score.py \\
        --render A1="C:\\...\\output\\LAIN_A1_s1099_*.png" \\
        --render A2="C:\\...\\output\\LAIN_A2_s1099_*.png" \\
        --render A4="C:\\...\\output\\LAIN_A4_s1099_*.png" \\
        --reference-dir path/to/held_out_refs/

    # Accumulate individual files under one label
    python test_lora_identity_score.py \\
        --render A2=render1.png --render A2=render2.png --render A2=render3.png \\
        --reference-dir path/to/held_out_refs/ --json a2_score.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

from transformers import AutoImageProcessor, AutoModel

DEFAULT_DINO_MODEL = "facebook/dinov2-base"
EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_render_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--render expects LABEL=PATH, got: {raw!r}")
    label, path = raw.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(f"--render expects LABEL=PATH, got: {raw!r}")
    return label, path


def resolve_paths(pattern: str) -> list[Path]:
    p = Path(pattern)
    if any(ch in pattern for ch in "*?["):
        matches = sorted(p.parent.glob(p.name))
        return [m for m in matches if m.is_file() and m.suffix.lower() in EXTENSIONS]
    if p.is_dir():
        return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in EXTENSIONS)
    if p.is_file():
        return [p]
    raise FileNotFoundError(f"--render path not found: {pattern}")


def discover_reference_images(reference_dir: Path) -> list[Path]:
    return sorted(p for p in reference_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


class DinoEmbedder:
    """Wraps DINOv2 for L2-normalized whole-image embeddings.

    Uses `pooler_output` if the model provides one, else the first
    (CLS-equivalent) token of `last_hidden_state` -- the same convention
    test_lora_identity_consistency.py uses, so scores from the two scripts
    are directly comparable.
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DINOv2 identity cosine for already-rendered images against a held-out reference set."
    )
    parser.add_argument(
        "--render",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeatable. LABEL=PATH to a single image, a directory of images, or a quoted glob "
        "pattern. Repeat with the same label to accumulate hand-picked files -- see module docstring.",
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Directory of held-out reference images (never trained on) establishing ground-truth identity.",
    )
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "test_lora_identity_score"),
    )
    parser.add_argument("--json", metavar="FILE", help="Also write the results table to this JSON path.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    renders: dict[str, list[Path]] = {}
    for raw in args.render:
        label, pattern = parse_render_arg(raw)
        paths = resolve_paths(pattern)
        if not paths:
            print(f"WARNING: --render {label}={pattern!r} matched no image files", file=sys.stderr)
        renders.setdefault(label, []).extend(paths)

    reference_dir = Path(args.reference_dir)
    reference_paths = discover_reference_images(reference_dir)
    if not reference_paths:
        print(f"No reference images found in {reference_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 ({args.dino_model}) on {device} ...")
    dino = DinoEmbedder(args.dino_model, device)

    print(f"Embedding {len(reference_paths)} reference image(s) from {reference_dir} ...")
    reference_embeds = [dino.embed(Image.open(p)) for p in reference_paths]

    label_results = {}
    print("=" * 88)
    print(f"{'label':<14} {'file':<32} {'identity cos':>13}")
    print("-" * 88)
    for label, paths in renders.items():
        per_image = []
        embeds = []
        for path in paths:
            emb = dino.embed(Image.open(path))
            embeds.append(emb)
            score = float(np.mean([cosine(emb, ref) for ref in reference_embeds]))
            per_image.append({"file": str(path), "identity_cosine": round(score, 4)})
            print(f"{label:<14} {path.name:<32} {score:>13.4f}")

        identity_scores = [r["identity_cosine"] for r in per_image]
        identity_mean = float(np.mean(identity_scores)) if identity_scores else None
        identity_std = float(np.std(identity_scores)) if identity_scores else None

        render_consistency = None
        if len(embeds) >= 2:
            pairs = [cosine(embeds[i], embeds[j]) for i in range(len(embeds)) for j in range(i + 1, len(embeds))]
            render_consistency = float(np.mean(pairs))

        label_results[label] = {
            "n": len(paths),
            "identity_cosine_mean": round(identity_mean, 4) if identity_mean is not None else None,
            "identity_cosine_std": round(identity_std, 4) if identity_std is not None else None,
            "render_consistency": round(render_consistency, 4) if render_consistency is not None else None,
            "per_image": per_image,
        }

    print("-" * 88)
    print(f"{'label':<14} {'n':>3} {'mean identity':>14} {'std':>8} {'render_consistency':>19}")
    print("-" * 88)
    for label, r in label_results.items():
        mean = f"{r['identity_cosine_mean']:.4f}" if r["identity_cosine_mean"] is not None else "  --"
        std = f"{r['identity_cosine_std']:.4f}" if r["identity_cosine_std"] is not None else "  --"
        rc = f"{r['render_consistency']:.4f}" if r["render_consistency"] is not None else "n<2"
        print(f"{label:<14} {r['n']:>3} {mean:>14} {std:>8} {rc:>19}")
    print("=" * 88)
    print("Column key (DINOv2 cosine, 0-1 scale, higher = more similar):")
    print("  identity cos       one render vs one reference image (per-row table above)")
    print("  mean identity      mean of that label's identity-cos column")
    print("  std                sd of that label's identity-cos column (n<2 -> n<2, not a number)")
    print("  render_consistency mean pairwise cosine among that label's OWN renders (n<2 -> n<2)")
    print("A label with high mean identity but low render_consistency has at least one image that")
    print("drifted off-identity -- the mean alone hides this. Treat n<3-per-label numbers as directional,")
    print("not conclusive; one odd render swings a small mean a lot.")

    summary = {
        "reference_dir": str(reference_dir),
        "reference_count": len(reference_paths),
        "dino_model": args.dino_model,
        "labels": label_results,
    }
    summary_path = out_dir / "identity_score.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
