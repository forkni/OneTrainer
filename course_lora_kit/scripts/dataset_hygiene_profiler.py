"""
Dataset hygiene profiler -- a cheap pre-flight screen for a LoRA training folder.

"Consistency matters" is a feeling until it's a number. This script computes three
per-image statistics across a training-image folder and flags the images that sit
far from the folder's own average on any of them:

    greyscale_std   Overall tonal contrast (std of the L channel). A flat, hazy,
                    or badly exposed frame reads low against a folder of punchy
                    ones (or vice versa).
    edge_energy     Mean gradient magnitude of the greyscale image -- a cheap
                    proxy for detail/sharpness. An off-crop or motion-blurred
                    frame stands out here.
    saturation      Mean HSV saturation. Surfaces the one wrong-toned photo in an
                    otherwise consistently-graded set (or the reverse: a stray
                    vivid frame in a desaturated set).

This does not replace looking at the images -- it tells you where to look. A
single flagged outlier quietly drags the training average toward itself; in a
21-image style set that's ~5% of the gradient signal coming from one bad frame.

No fixed pass/fail thresholds are prescribed for the three metrics above (unlike
checkpoint_norm_analyzer.py's ||dW||_F bands) -- what counts as an outlier is
relative to the rest of THIS folder. The script reports a z-score per image per
metric and flags anything beyond --z-threshold (default 2.0) standard deviations
from the folder mean.

Two more checks ARE absolute, not relative, because they're pass/fail regardless
of what the rest of the folder looks like:

    --min-side       Flags any image whose shorter side is below this many
                      pixels. Anime screencaps in particular run well under
                      1024 on the short side (SEL's source is a 4:3 TV master);
                      an undersized frame that passes the z-score screen still
                      forces either upscaling (soft, no new detail) or a smaller
                      bucket than the rest of the set at train time.
    letterbox/pillarbox   Flags a frame with uniform-color bars along top/bottom
                      or left/right -- the un-cropped remainder of a 4:3 source
                      composited into a 16:9 (or other) frame. These bars are
                      dead weight the model has no reason to reproduce, and OK
                      to crop before training even though they don't affect any
                      of the three relative metrics above.

With --recursive, also reports a per-subfolder image count -- useful when a
dataset is organized into shot-type or outfit subfolders (e.g. face/, medium/,
full_body/) and you want to confirm the mix matches your plan before training,
not after.

Needs: numpy, Pillow. matplotlib is optional (only for --plot).

Usage:
    python dataset_hygiene_profiler.py path/to/training_images/
    python dataset_hygiene_profiler.py path/to/training_images/ --min-side 1024
    python dataset_hygiene_profiler.py path/to/training_images/ --recursive --min-side 1024
    python dataset_hygiene_profiler.py path/to/training_images/ --json hygiene.json
    python dataset_hygiene_profiler.py path/to/training_images/ --plot --output-dir out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def discover_images(folder: Path, extensions: tuple[str, ...], recursive: bool) -> list[Path]:
    pattern_iter = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in pattern_iter if p.is_file() and p.suffix.lower() in extensions)


def greyscale_std(gray: np.ndarray) -> float:
    return float(gray.std())


def edge_energy(gray: np.ndarray) -> float:
    """Mean gradient magnitude -- a cheap, dependency-free sharpness/detail proxy."""
    gy, gx = np.gradient(gray.astype(np.float64))
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


def mean_saturation(img: Image.Image) -> float:
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    return float(hsv[..., 1].mean())  # 0-255 scale


def detect_letterbox(
    gray: np.ndarray, threshold_std: float = 4.0, min_border_frac: float = 0.10, mean_tolerance: float = 12.0
) -> dict:
    """Detect uniform-color letterbox/pillarbox bars -- video-master padding,
    not an incidental plain background or a narrow art-print margin.

    Scans inward from each of the four edges; a row/column counts as "border"
    while its own std stays below threshold_std (a solid-color line, not image
    content). A REAL letterbox/pillarbox bar is symmetric: the same matte
    color on BOTH opposing edges (the un-cropped remainder of a 4:3 source
    composited into a wider frame), AND substantial -- a genuine aspect-ratio
    mismatch (e.g. 4:3 video letterboxed into 16:9) eats double-digit percent
    of the frame, not a few pixels. So this only flags letterboxed/pillarboxed
    when both the top and bottom (or left and right) show a border of at least
    min_border_frac / 2 each, of matching mean brightness (within
    mean_tolerance, 0-255 scale). Both conditions matter: requiring both sides
    to match keeps a plain photographic background touching just ONE edge
    (common in portrait/character sets) from tripping this; requiring a
    substantial fraction keeps a narrow, symmetric pale-paper margin from
    tripping it too -- measured on this course's own sepiagraph set (21
    architectural line drawings on parchment-toned paper, borders on both
    edges by the art style itself, not by a video source): the two-sided-only
    version of this check (matching color required, no size floor) still
    flagged 12/21 of those images at a 1%-per-side floor, all of them well
    under 5% border per side -- i.e. a plain-paper margin, not a bar. The
    default 10% combined (5% per side) floor clears the entire sepiagraph set
    while still catching a real 4:3-in-16:9 letterbox, whose bars run ~12.5%
    per side.
    """
    h, w = gray.shape

    def scan(get_line, count: int) -> tuple[int, float | None]:
        n = 0
        total = 0.0
        for i in range(count):
            line = get_line(i)
            if float(line.std()) < threshold_std:
                n += 1
                total += float(line.mean())
            else:
                break
        return n, (total / n if n else None)

    top, top_mean = scan(lambda i: gray[i, :], h)
    bottom, bottom_mean = scan(lambda i: gray[h - 1 - i, :], h)
    left, left_mean = scan(lambda i: gray[:, i], w)
    right, right_mean = scan(lambda i: gray[:, w - 1 - i], w)

    top_frac = top / h if h else 0.0
    bottom_frac = bottom / h if h else 0.0
    left_frac = left / w if w else 0.0
    right_frac = right / w if w else 0.0
    half_min = min_border_frac / 2

    letterboxed = (
        top_frac >= half_min
        and bottom_frac >= half_min
        and top_mean is not None
        and bottom_mean is not None
        and abs(top_mean - bottom_mean) <= mean_tolerance
    )
    pillarboxed = (
        left_frac >= half_min
        and right_frac >= half_min
        and left_mean is not None
        and right_mean is not None
        and abs(left_mean - right_mean) <= mean_tolerance
    )

    return {
        "border_top_frac": round(top_frac, 4),
        "border_bottom_frac": round(bottom_frac, 4),
        "border_left_frac": round(left_frac, 4),
        "border_right_frac": round(right_frac, 4),
        "letterboxed": bool(letterboxed),
        "pillarboxed": bool(pillarboxed),
    }


def profile_image(path: Path, min_side: int | None) -> dict:
    img = Image.open(path).convert("RGB")
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    short_side = min(img.width, img.height)
    border = detect_letterbox(gray)
    row = {
        "file": path.name,
        "width": img.width,
        "height": img.height,
        "short_side": short_side,
        "undersized": bool(min_side is not None and short_side < min_side),
        "greyscale_std": greyscale_std(gray),
        "edge_energy": edge_energy(gray),
        "saturation": mean_saturation(img),
    }
    row.update(border)
    return row


def zscores(rows: list[dict], metric: str) -> list[float]:
    values = np.array([r[metric] for r in rows], dtype=np.float64)
    mean = values.mean()
    std = values.std()
    if std < 1e-9:
        return [0.0] * len(values)
    return list((values - mean) / std)


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-image dataset hygiene screen: contrast, edges, saturation.")
    parser.add_argument("folder", help="Directory of training images.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories too.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        metavar="EXT",
        help=f"File extensions to include (default: {' '.join(DEFAULT_EXTENSIONS)}).",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.0,
        help="Flag an image if any of the three relative metrics' |z-score| exceeds this (default: 2.0).",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=None,
        metavar="PX",
        help="Flag any image whose shorter side is below this many pixels (absolute check, "
        "not relative to the folder). Omit to skip the check.",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="With --recursive, warn if any one subfolder holds more than this fraction of "
        "all images (default: 0.5 -- i.e. no single shot-type/outfit subfolder over half the set).",
    )
    parser.add_argument("--json", metavar="FILE", help="Write the full per-image table to this JSON file.")
    parser.add_argument("--plot", action="store_true", help="Save a scatter plot (requires matplotlib).")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for --plot output (default: current directory).",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    extensions = tuple(e if e.startswith(".") else f".{e}" for e in args.extensions)
    images = discover_images(folder, extensions, args.recursive)
    if not images:
        print(f"No images found in {folder} (extensions: {extensions})", file=sys.stderr)
        return 1

    rows = []
    for path in images:
        print(f"Profiling {path.name} ...", file=sys.stderr, flush=True)
        try:
            rows.append(profile_image(path, args.min_side))
        except Exception as e:
            print(f"SKIP {path.name}: {e}", file=sys.stderr)

    if len(rows) < 2:
        print("Need at least 2 images to compute meaningful z-scores.", file=sys.stderr)
        return 1

    metrics = ("greyscale_std", "edge_energy", "saturation")
    z_by_metric = {m: zscores(rows, m) for m in metrics}
    for i, row in enumerate(rows):
        row_z = {m: z_by_metric[m][i] for m in metrics}
        row["z"] = {m: round(v, 3) for m, v in row_z.items()}
        hygiene_issue = row["undersized"] or row["letterboxed"] or row["pillarboxed"]
        row["hygiene_issue"] = hygiene_issue
        row["flagged"] = any(abs(v) > args.z_threshold for v in row_z.values()) or hygiene_issue

    flagged = [r for r in rows if r["flagged"]]
    undersized = [r for r in rows if r["undersized"]]
    bordered = [r for r in rows if r["letterboxed"] or r["pillarboxed"]]

    print("=" * 110)
    print(
        f"{'file':<28} {'grey_std':>9} {'edge_e':>9} {'sat':>7}   {'z_std':>6} {'z_edge':>7} "
        f"{'z_sat':>6}  {'short':>6}  border  flag"
    )
    print("-" * 110)
    for row in rows:
        z = row["z"]
        mark = "  <<<" if row["flagged"] else ""
        under = "U" if row["undersized"] else " "
        lb = "L" if row["letterboxed"] else " "
        pb = "P" if row["pillarboxed"] else " "
        print(
            f"{row['file']:<28} {row['greyscale_std']:>9.2f} {row['edge_energy']:>9.3f} "
            f"{row['saturation']:>7.2f}   {z['greyscale_std']:>6.2f} {z['edge_energy']:>7.2f} "
            f"{z['saturation']:>6.2f}  {row['short_side']:>6}  {under}{lb}{pb}{mark}"
        )
    print("-" * 110)
    if flagged:
        print(f"{len(flagged)}/{len(rows)} image(s) flagged (z-outlier and/or hygiene issue):")
        for row in flagged:
            reasons = []
            if any(abs(v) > args.z_threshold for v in row["z"].values()):
                reasons.append("z-outlier")
            if row["undersized"]:
                reasons.append(f"undersized (short side {row['short_side']} < {args.min_side})")
            if row["letterboxed"]:
                reasons.append("letterboxed")
            if row["pillarboxed"]:
                reasons.append("pillarboxed")
            print(f"  {row['file']}: {', '.join(reasons)}")
    else:
        print(f"No images exceeded |z| > {args.z_threshold} or tripped a hygiene check.")
    if args.min_side is not None:
        print(f"Undersized (short side < {args.min_side}px): {len(undersized)}/{len(rows)}")
    print(f"Letterboxed/pillarboxed (symmetric matching-color border, >=5% per side): {len(bordered)}/{len(rows)}")
    print("=" * 110)
    print("Column key: U=undersized L=letterboxed P=pillarboxed. z-flags are relative to this")
    print("folder's own average, not a universal pass/fail cutoff -- look at what got flagged")
    print("before deciding whether to drop, recrop, upscale, or recaption it. Undersized/")
    print("bordered flags are absolute (not relative to the folder).")

    if args.recursive:
        by_subfolder: dict[str, int] = {}
        for path in images:
            rel_parent = path.relative_to(folder).parent
            key = str(rel_parent) if str(rel_parent) != "." else "(root)"
            by_subfolder[key] = by_subfolder.get(key, 0) + 1
        total = len(images)
        print("-" * 110)
        print("Per-subfolder counts:")
        for key in sorted(by_subfolder):
            count = by_subfolder[key]
            frac = count / total if total else 0.0
            warn = "  <<< imbalanced" if frac > args.imbalance_threshold else ""
            print(f"  {key:<30} {count:>4}  ({frac:.0%}){warn}")
        imbalanced = [k for k, c in by_subfolder.items() if total and c / total > args.imbalance_threshold]
        if imbalanced:
            print(
                f"WARNING: {', '.join(imbalanced)} exceed(s) {args.imbalance_threshold:.0%} of the dataset -- "
                "a single shot-type/outfit subfolder this dominant risks binding that trait to the trigger."
            )
        print("=" * 110)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"z_threshold": args.z_threshold, "min_side": args.min_side, "images": rows}, indent=2))
        print(f"Wrote {args.json}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("--plot requires matplotlib (pip install matplotlib); skipping plot.", file=sys.stderr)
            return 0

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, metric in zip(axes, metrics):
            values = [r[metric] for r in rows]
            colors = ["tab:red" if r["flagged"] else "tab:blue" for r in rows]
            ax.bar(range(len(rows)), values, color=colors)
            ax.set_title(metric)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([r["file"] for r in rows], rotation=90, fontsize=6)
        fig.tight_layout()
        plot_path = out_dir / "dataset_hygiene.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Wrote {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
