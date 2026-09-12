"""
Colour-drift metric for a checkpoint sweep: image-mean CIELAB a* and b*.

For every render in a sweep folder (the stepNNN.png / final(...).png files
that 05_validate_sweep.cmd writes), compute the mean CIELAB a* (green < 0,
magenta > 0) and b* (blue < 0, yellow > 0) over the whole image. The a* axis
is exactly where the per-seed magenta drift of an unweighted recipe lives,
so a sweep-wide a* table makes "the colour went pink at step N" a number
instead of an impression.

With --baseline-dir, every render that exists in both folders also gets a
delta a* column (sweep minus baseline) -- the Round 2 S0b-vs-S0 comparison
(mean a* 12.0 vs 31.0) was produced this way.

Only the *.png files are read; contact_sheet.png is skipped, and the
"__control" renders (same seed, LoRA off) are reported like any other file so
you can see the untouched base model's colour on the same table. The label
bar the sweep script paints is dark and near-neutral, so it barely moves the
mean; no crop needed.

Needs only numpy and Pillow (both in the OneTrainer venv).

Usage:
  python color_stats.py --sweep-dir <sweep_folder>
  python color_stats.py --sweep-dir <sweep_folder> --baseline-dir <other_sweep_folder>
  python color_stats.py --sweep-dir <sweep_folder> --json out.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image


def mean_lab(path):
    """Return (mean a*, mean b*) of an image, sRGB -> linear -> XYZ (D65) -> CIELAB."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
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


def step_key(name):
    """Sort key: numeric step order, 'final*' last, control renders next to their step."""
    if name.startswith("final"):
        base = 10**6
    else:
        mnum = re.search(r"(\d+)", name)
        base = int(mnum.group(1)) if mnum else -1
    return (base, "__control" in name)


def scan(folder):
    rows = {}
    for p in sorted(glob.glob(os.path.join(folder, "*.png"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name == "contact_sheet":
            continue
        rows[name] = mean_lab(p)
    return rows


def summarize(rows, control):
    """Mean a*/b* over the LoRA-on renders (or the controls) in a scan."""
    sel = [v for k, v in rows.items() if ("__control" in k) == control]
    if not sel:
        return None
    arr = np.asarray(sel)
    return {"count": len(sel), "mean_a": float(arr[:, 0].mean()), "mean_b": float(arr[:, 1].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, help="folder of sweep renders (stepNNN.png, final*.png)")
    ap.add_argument("--baseline-dir", help="second sweep folder; adds a delta a* column (sweep - baseline)")
    ap.add_argument("--json", help="also write the per-render table and the means to this file")
    args = ap.parse_args()

    if not os.path.isdir(args.sweep_dir):
        ap.error(f"--sweep-dir not found: {args.sweep_dir}")
    if args.baseline_dir and not os.path.isdir(args.baseline_dir):
        ap.error(f"--baseline-dir not found: {args.baseline_dir}")

    sweep = scan(args.sweep_dir)
    base = scan(args.baseline_dir) if args.baseline_dir else {}
    if not sweep:
        ap.error(f"no .png renders in {args.sweep_dir}")

    names = sorted(set(sweep) | set(base), key=step_key)
    w = max(len(n) for n in names)
    if base:
        print(f"{'render':<{w}} {'base a*':>8} {'base b*':>8}   {'sweep a*':>8} {'sweep b*':>8}   {'delta a*':>8}")
    else:
        print(f"{'render':<{w}} {'a*':>8} {'b*':>8}")
    for n in names:
        cs = sweep.get(n)
        cb = base.get(n)
        fs = f"{cs[0]:8.2f} {cs[1]:8.2f}" if cs else f"{'-':>8} {'-':>8}"
        if base:
            fb = f"{cb[0]:8.2f} {cb[1]:8.2f}" if cb else f"{'-':>8} {'-':>8}"
            da = f"{cs[0] - cb[0]:8.2f}" if cs and cb else f"{'-':>8}"
            print(f"{n:<{w}} {fb}   {fs}   {da}")
        else:
            print(f"{n:<{w}} {fs}")

    print()
    summary = {"sweep": {"dir": args.sweep_dir,
                         "lora_on": summarize(sweep, control=False),
                         "controls": summarize(sweep, control=True)}}
    if base:
        summary["baseline"] = {"dir": args.baseline_dir,
                               "lora_on": summarize(base, control=False),
                               "controls": summarize(base, control=True)}
    for label, block in summary.items():
        for kind in ("lora_on", "controls"):
            s = block[kind]
            if s:
                print(f"{label:<8} {kind:<8} n={s['count']:<3} mean a* {s['mean_a']:6.2f}   mean b* {s['mean_b']:6.2f}")
    print("reading: a* > 0 is magenta, < 0 green; a LoRA-on mean far above the controls is colour drift")
    print("Column key: a*/b* are CIELAB, averaged over every pixel of one render (b* > 0 is yellow,")
    print("< 0 is blue). base/sweep are the same render from --baseline-dir and --sweep-dir; delta a*")
    print("is sweep minus baseline. __control rows are the same seed with the LoRA at scale 0.0.")

    if args.json:
        out = {"renders": {"sweep": {k: {"a": v[0], "b": v[1]} for k, v in sweep.items()},
                           "baseline": {k: {"a": v[0], "b": v[1]} for k, v in base.items()}},
               "summary": summary}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
