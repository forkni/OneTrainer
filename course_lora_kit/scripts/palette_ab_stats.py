"""Palette screen for a training folder: mean CIELAB L*/a*/b* per image, folder-relative.

Same colour statistic the kit's seed-batch validator applies to renders (mean a* = the
red-green axis, positive = red/magenta, negative = green; mean b* = the yellow-blue axis,
positive = yellow/orange, negative = blue), applied to the dataset instead. It says what
the DATASET looks like; it says nothing about what a trained LoRA will render.

What it prints, per folder given:
  - one line per image: L*, a*, b*, z-scores of a* and b* against THIS folder's own mean,
    a hue read (orange / red / magenta / purple / blue / cyan / green / yellow, or neutral
    when chroma is low), and a flag when |z| crosses --z-threshold on either axis;
  - a summary: n, mean/sd of a* and b*, and the images that pull hardest on each axis;
  - with --recursive, a per-subfolder mean a*/b* block.
With several folders it also prints a comparison line per folder against the first one
(delta of mean a* / b*), so "kept set vs dropped set" is just two folder arguments.

z-flags are relative to the folder, not a pass/fail cutoff: look at what got flagged and
decide by eye whether it is a palette outlier or the palette itself.

Usage:
  python palette_ab_stats.py <folder> [<folder> ...] [--recursive] [--z-threshold 2.0]
                             [--sort a|b|chroma|name] [--top N] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def lab_mean(path: Path) -> tuple[float, float, float]:
    """Mean L*, a*, b* of an image (sRGB -> linear -> XYZ D65 -> CIELAB)."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = (lin.reshape(-1, 3) @ _M.T) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return float(L.mean()), float(a.mean()), float(b.mean())


def hue_read(a: float, b: float, neutral_chroma: float = 8.0) -> str:
    """Coarse hue sector of the mean colour; 'neutral' when the mean chroma is small."""
    if math.hypot(a, b) < neutral_chroma:
        return "neutral"
    h = math.degrees(math.atan2(b, a)) % 360.0
    sectors = [
        (0, 30, "red"), (30, 75, "orange"), (75, 105, "yellow"), (105, 165, "yellow-green"),
        (165, 200, "green"), (200, 240, "cyan"), (240, 285, "blue"), (285, 320, "purple"),
        (320, 345, "magenta"), (345, 360, "red"),
    ]
    for lo, hi, name in sectors:
        if lo <= h < hi:
            return name
    return "red"


def scan(folder: Path, recursive: bool) -> list[dict]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    rows = []
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in EXTS:
            try:
                L, a, b = lab_mean(p)
            except Exception as exc:  # unreadable file: report, keep going
                print(f"  ! skipped {p.relative_to(folder)}: {exc}")
                continue
            rows.append({
                "file": str(p.relative_to(folder)),
                "subfolder": str(p.parent.relative_to(folder)) if p.parent != folder else ".",
                "L": L, "a": a, "b": b, "chroma": math.hypot(a, b),
                "hue": hue_read(a, b),
            })
    return rows


def add_z(rows: list[dict]) -> None:
    for key in ("a", "b"):
        v = np.array([r[key] for r in rows])
        sd = v.std() if len(v) > 1 else 0.0
        for r, x in zip(rows, v):
            r["z_" + key] = float((x - v.mean()) / sd) if sd > 0 else 0.0


def report(folder: Path, rows: list[dict], args) -> dict:
    n = len(rows)
    print("=" * 100)
    print(f"{folder}  ({n} image{'s' if n != 1 else ''}{', recursive' if args.recursive else ''})")
    print("-" * 100)
    if n == 0:
        print("  no images found")
        return {"folder": str(folder), "n": 0}
    add_z(rows)
    key = {"a": lambda r: -r["a"], "b": lambda r: -r["b"],
           "chroma": lambda r: -r["chroma"], "name": lambda r: r["file"]}[args.sort]
    ordered = sorted(rows, key=key)
    shown = ordered if args.top <= 0 else ordered[:args.top]
    w = max(len(r["file"]) for r in rows)
    w = min(max(w, 4), 48)
    print(f"{'file':<{w}}  {'L*':>6} {'a*':>7} {'b*':>7}  {'z_a':>5} {'z_b':>5}  {'hue':<12} flag")
    flagged = []
    for r in shown:
        flag = ""
        if abs(r["z_a"]) >= args.z_threshold or abs(r["z_b"]) >= args.z_threshold:
            flag = "<<<"
            flagged.append(r)
        name = r["file"] if len(r["file"]) <= w else "..." + r["file"][-(w - 3):]
        print(f"{name:<{w}}  {r['L']:6.1f} {r['a']:+7.2f} {r['b']:+7.2f}  "
              f"{r['z_a']:+5.2f} {r['z_b']:+5.2f}  {r['hue']:<12} {flag}")
    if args.top > 0 and n > args.top:
        print(f"  ... {n - args.top} more (use --top 0 for all)")
        flagged = [r for r in rows if abs(r["z_a"]) >= args.z_threshold or abs(r["z_b"]) >= args.z_threshold]

    a = np.array([r["a"] for r in rows]); b = np.array([r["b"] for r in rows])
    L = np.array([r["L"] for r in rows])
    print("-" * 100)
    print(f"mean a* {a.mean():+6.2f} (sd {a.std():5.2f}, min {a.min():+6.2f}, max {a.max():+6.2f})   "
          f"mean b* {b.mean():+6.2f} (sd {b.std():5.2f}, min {b.min():+6.2f}, max {b.max():+6.2f})   "
          f"mean L* {L.mean():5.1f}")
    hues = {}
    for r in rows:
        hues[r["hue"]] = hues.get(r["hue"], 0) + 1
    print("hue reads: " + ", ".join(f"{k} {v}" for k, v in sorted(hues.items(), key=lambda kv: -kv[1])))
    print(f"{len(flagged)}/{n} image(s) beyond |z| >= {args.z_threshold} on a* or b* "
          f"(relative to this folder's own mean; look before dropping):")
    for r in sorted(flagged, key=lambda r: -(abs(r["z_a"]) + abs(r["z_b"]))):
        print(f"  {r['file']}: a* {r['a']:+.2f} (z {r['z_a']:+.2f}), b* {r['b']:+.2f} (z {r['z_b']:+.2f}), {r['hue']}")

    subs = {}
    for r in rows:
        subs.setdefault(r["subfolder"], []).append(r)
    if args.recursive and len(subs) > 1:
        print("-" * 100)
        print("Per-subfolder means:")
        for name, rs in sorted(subs.items()):
            sa = np.array([r["a"] for r in rs]); sb = np.array([r["b"] for r in rs])
            print(f"  {name:<30} n={len(rs):3d}  a* {sa.mean():+6.2f} (sd {sa.std():5.2f})  "
                  f"b* {sb.mean():+6.2f} (sd {sb.std():5.2f})")
    return {"folder": str(folder), "n": n, "mean_a": float(a.mean()), "sd_a": float(a.std()),
            "mean_b": float(b.mean()), "sd_b": float(b.std()), "mean_L": float(L.mean()),
            "flagged": [r["file"] for r in flagged], "images": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folders", nargs="+", type=Path, help="image folder(s); several = compare against the first")
    ap.add_argument("--recursive", action="store_true", help="include subfolders (and print per-subfolder means)")
    ap.add_argument("--z-threshold", type=float, default=2.0, help="|z| on a* or b* that flags an image (default 2.0)")
    ap.add_argument("--sort", choices=["a", "b", "chroma", "name"], default="a",
                    help="row order: a (most red first, default), b (most yellow first), chroma, name")
    ap.add_argument("--top", type=int, default=0, help="show only the first N rows after sorting (0 = all)")
    ap.add_argument("--json", type=Path, help="write every number to this JSON file")
    args = ap.parse_args(argv)

    results = []
    for folder in args.folders:
        if not folder.is_dir():
            print(f"not a folder: {folder}")
            return 2
        results.append(report(folder, scan(folder, args.recursive), args))

    if len(results) > 1:
        print("=" * 100)
        base = results[0]
        print(f"Comparison against the first folder ({base['folder']}, mean a* {base.get('mean_a', float('nan')):+.2f}, "
              f"mean b* {base.get('mean_b', float('nan')):+.2f}):")
        for r in results[1:]:
            if r["n"] == 0 or base["n"] == 0:
                continue
            print(f"  {r['folder']}: n={r['n']}, mean a* {r['mean_a']:+.2f} ({r['mean_a'] - base['mean_a']:+.2f}), "
                  f"mean b* {r['mean_b']:+.2f} ({r['mean_b'] - base['mean_b']:+.2f})")
        allrows = [row for r in results if r["n"] for row in r["images"]]
        a = np.array([x["a"] for x in allrows]); b = np.array([x["b"] for x in allrows])
        print(f"  all folders pooled: n={len(allrows)}, mean a* {a.mean():+.2f} (sd {a.std():.2f}), "
              f"mean b* {b.mean():+.2f} (sd {b.std():.2f})")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    print("=" * 100)
    print("Column key: a* +red/-green, b* +yellow/-blue, z = this folder's own mean and sd.\n"
          "Dataset-side statistic only -- it does not predict what a trained LoRA renders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
