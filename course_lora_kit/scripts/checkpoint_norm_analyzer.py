"""
Checkpoint norm analyzer -- rendering-free LoRA collapse detector.

Reads kohya-format LoRA .safetensors files directly (the format OneTrainer, the
training tool this course uses, writes) and computes, per checkpoint, without
loading a base model, without a GPU, and without rendering a single image:

    ||dW||_F   Aggregate Frobenius norm of the full weight delta the LoRA adds,
               summed across every lora_down/lora_up/alpha module triple.
               dW_module = lora_up @ lora_down * (alpha / rank)

    cosine     Cosine similarity between the flattened dW vectors of two
               checkpoints -- how similar in DIRECTION two checkpoints' edits
               are, independent of their magnitude.

    growth     ||dW||_F of this checkpoint over the previous one, and a KNEE
               flag when that ratio jumps well above the sweep's own median --
               a scale-independent signal (see "What transfers" below).

Both ||dW||_F and cosine are computed WITHOUT ever forming the full
[out_features, in_features] dW matrix. For a real SDXL UNet LoRA that matrix,
concatenated across all ~722 modules, runs to a couple billion elements (tens
of GB, even at float32) -- materializing it is what makes a naive version of
this script crash on a real checkpoint. Instead, dot(dW_a, dW_b) is computed
straight from the low-rank factors via the trace identity
    trace(dW_a.T @ dW_b) = scale_a * scale_b * sum(down_a * (up_a.T @ up_b @ down_b))
which only ever materializes rank x rank and rank x in_features intermediates
(rank is 16-64 for a typical LoRA) -- the same number, computed cheaply. Setting
a == b gives ||dW||_F^2 for free from the same routine.

Conv2d LoRA modules (present when OneTrainer trains with layer_filter "" / a
filter wide enough to include the UNet's conv layers, e.g. a "full" preset)
are supported too. kohya's Conv2d LoRA writes lora_down as a real
[rank, in_ch, kh, kw] conv kernel and lora_up as a pointwise (1x1) conv
[out_ch, rank, 1, 1] -- the 1x1 shape is structural (OneTrainer's
create_layer() hardcodes the up-projection kernel size to (1, 1)), so
flattening down to [rank, in_ch*kh*kw] and up to [out_ch, rank] before handing
off to the same trace-identity math above is an EXACT reshape of the same
computation, not an approximation: dW[o,i,kh,kw] = sum_r up[o,r,0,0] *
down[r,i,kh,kw] is exactly up_2d @ down_flat reshaped back. Same Frobenius
norm and cosine either way (see load_lora_modules()).

Why this matters (see Appendix D, "Screening a checkpoint without rendering it"):
a small-dataset LoRA trained past its views-per-image window doesn't just get
"more" of the same style -- ||dW||_F grows steeply (~step^2.5 early, then closer
to linear) and the edit direction rotates rather than scales. Two checkpoints
with high cosine similarity are doing "more of the same thing"; low cosine means
the later one has drifted onto a different, usually memorized, solution. That
answers "can I just lower Weight on an overcooked checkpoint?" -- no: a weaker
dial on a rotated delta still points the wrong way.

Reference values from this course's own measured sweep (rank 16, alpha 1.0 ->
scale 0.0625, layer_filter_preset "attn-mlp", ~722 modules -- see Appendix D
and _reviews/FACT_CHECK_Appendix_D_Training_Your_Own_LoRA.md #32-33 and #35):
    ||dW||_F <= 3.26   every checkpoint that stayed prompt-responsive
    ||dW||_F >= 6.70   every checkpoint that had collapsed into a memorized,
                       prompt-independent style
    cosine ~0.85       adjacent checkpoints in a healthy sweep
    cosine  0.19       first vs. last checkpoint of an overcooked sweep

(REF_CLEAN_MAX is 3.26, not the round 3.25 an earlier version of this script
used -- the measured ceiling checkpoint actually reads 3.2503, which the old
3.25 cutoff mis-banded as "between reference bands" instead of "clean-range".
See _reviews/FACT_CHECK_Appendix_D_Training_Your_Own_LoRA.md #42.)

These are NOT universal thresholds -- they're specific to that rank/alpha/layer
config (scale 0.0625). A checkpoint trained at a different scale (e.g. a
kohya-conventional rank 16 / alpha 16, scale 1.0 -- 16x the reference) is
projected onto the reference scale before banding, on the first-order
assumption that ||dW||_F scales roughly linearly with a fixed alpha/rank
multiplier (see classify()). That projection is NOT independently measured at
other scales -- treat a non-reference-scale band as a rough steer, not a
verified cutoff. What transfers with no projection needed, at any scale, is the
SHAPE: a knee in ||dW||_F growth (flagged in the growth column) and a drop in
adjacent-checkpoint cosine.

Needs: numpy, safetensors. No torch, no GPU, no base model.

Usage:
    python checkpoint_norm_analyzer.py lora.safetensors
    python checkpoint_norm_analyzer.py path/to/checkpoint/dir/
    python checkpoint_norm_analyzer.py ckpt_dir/ --json norms.json
    python checkpoint_norm_analyzer.py ckpt_dir_A/ ckpt_dir_B/ --json norms.json

Each directory argument is its own group: checkpoints are discovered, sorted,
and have their cosine/growth/knee/endpoint analysis run independently within
that group, so passing several arms in one invocation does not interleave
their checkpoints into a single cross-arm trace. Bare file arguments are
collected into one additional "files" group. With a single directory (or
single file) argument -- the common case -- output and --json shape are
exactly what earlier versions of this script produced; multiple groups print
one table per group and, with --json, nest each group's results under its
own key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

# Reference values from this course's measured run -- see the module docstring.
REF_CLEAN_MAX = 3.26
REF_COLLAPSED_MIN = 6.70
REF_ADJACENT_COSINE = 0.85
REF_ENDPOINT_COSINE = 0.19

# Scale (alpha / rank) the bands above were measured at: rank 16, alpha 1.0,
# attn-mlp layer filter. classify() projects a checkpoint's own scale onto this
# reference scale before banding -- see its docstring for the caveat.
REF_SCALE = 1.0 / 16.0

# Matches OneTrainer's intermediate-checkpoint naming, e.g.
# "lora_sepiagraph-save-300-1-0.safetensors" -> step 300. Falls back to plain
# digit extraction for files that don't follow this convention (see natural_sort_key).
_ONETRAINER_STEP_RE = re.compile(r"-save-(\d+)-\d+-\d+\.safetensors$")


def natural_sort_key(path: Path):
    """Sort checkpoints by embedded step number where possible, filename otherwise.

    OneTrainer's own save-step naming is checked first; if that doesn't match,
    falls back to the last run of digits in the filename (so "step60.safetensors"
    sorts before "step300.safetensors" rather than alphabetically).
    """
    m = _ONETRAINER_STEP_RE.search(path.name)
    if m:
        return (0, int(m.group(1)), path.name)
    digit_groups = re.findall(r"\d+", path.stem)
    if digit_groups:
        return (1, int(digit_groups[-1]), path.name)
    return (2, 0, path.name)


def discover_checkpoints(inputs: list[str]) -> list[tuple[str, list[Path]]]:
    """Resolve a mix of file and directory args into named, independent groups.

    Each directory argument becomes its own group (its own glob, naturally
    sorted) -- this is what prevents checkpoints from two different arm
    directories being merged into one sorted list and having cosine/growth
    computed across the arm boundary. Bare file arguments are collected
    together into one additional "files" group, matching the old flat
    behavior for that case. Returns a list of (group_name, checkpoints)
    pairs; a group with zero checkpoints is still included so a bad path is
    still reported by the caller.
    """
    groups: list[tuple[str, list[Path]]] = []
    loose_files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files = sorted(set(p.glob("*.safetensors")), key=natural_sort_key)
            groups.append((str(p), files))
        elif p.is_file():
            loose_files.append(p)
        else:
            print(f"WARNING: path not found, skipping: {p}", file=sys.stderr)
    if loose_files:
        groups.append(("files", sorted(set(loose_files), key=natural_sort_key)))
    return groups


# A module's low-rank factors: down [rank, in_features], up [out_features, rank],
# scale = alpha / rank.
LoraModule = tuple[np.ndarray, np.ndarray, float]


def load_lora_modules(path: Path) -> tuple[dict[str, LoraModule], bool]:
    """Load a kohya-format LoRA's per-module low-rank factors, keyed by module name.

    Expects "<module>.lora_down.weight" [rank, in_features],
    "<module>.lora_up.weight" [out_features, rank], and an optional
    "<module>.alpha" scalar (defaults to rank, matching kohya's own convention,
    when a module has no alpha key). Deliberately does NOT expand these into the
    full [out_features, in_features] delta -- see the module docstring.

    Returns (modules, has_dora). has_dora is True if the file also carries
    ".dora_scale" keys (DoRA's magnitude-decomposition term). This script's
    ||dW||_F does NOT fold that term in -- see the DoRA note in classify() and
    the printed warning in main() when has_dora is True.

    Conv2d modules (down is 4D: [rank, in_ch, kh, kw], up is [out_ch, rank, 1,
    1]) are flattened to 2D before storage -- see the module docstring for why
    this is an exact reshape, not an approximation. Raises ValueError if a
    conv module's up-projection is not the expected 1x1 kernel, since the
    reshape is only exact in that case.

    Raises ValueError with a specific message for two known non-kohya shapes:
    PEFT/diffusers-format files (".lora_A.weight"/".lora_B.weight" keys, e.g. a
    LoRA trained or re-exported outside OneTrainer's Kohya output format) and
    files with no recognizable LoRA keys at all.
    """
    tensors = load_file(str(path))
    down_keys = sorted(k for k in tensors if k.endswith(".lora_down.weight"))
    if not down_keys:
        peft_keys = [k for k in tensors if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight")]
        if peft_keys:
            raise ValueError(
                f"{path.name}: found PEFT/diffusers-style '.lora_A.weight'/'.lora_B.weight' keys, "
                "not kohya's '.lora_down.weight'/'.lora_up.weight'. This script only reads "
                "kohya-format files (OneTrainer's default Output Format on the model tab). "
                "Re-export with Output Format = Kohya, or convert the file, before running this script."
            )
        raise ValueError(
            f"{path.name}: no '.lora_down.weight' keys found -- this doesn't look "
            "like a kohya-format LoRA (the format OneTrainer writes)."
        )

    has_dora = any(k.endswith(".dora_scale") for k in tensors)

    modules: dict[str, LoraModule] = {}
    for down_key in down_keys:
        base = down_key[: -len(".lora_down.weight")]
        up_key = base + ".lora_up.weight"
        alpha_key = base + ".alpha"
        if up_key not in tensors:
            raise ValueError(f"{path.name}: {down_key} has no matching {up_key}")

        down = tensors[down_key].astype(np.float64)
        up = tensors[up_key].astype(np.float64)
        rank = down.shape[0]
        if down.ndim == 4:  # Conv2d module: down [rank, in_ch, kh, kw], up [out_ch, rank, 1, 1]
            if up.shape[2] != 1 or up.shape[3] != 1:
                raise ValueError(
                    f"{path.name}: {up_key} has a non-1x1 up-projection kernel {up.shape} -- "
                    "the flatten-to-2D reshape this script relies on is only exact for a 1x1 "
                    "kohya Conv2d up-projection (see load_lora_modules() docstring)."
                )
            down = down.reshape(rank, -1)
            up = up.reshape(up.shape[0], rank)
        alpha = float(tensors[alpha_key]) if alpha_key in tensors else float(rank)
        scale = alpha / rank if rank else 0.0
        modules[base] = (down, up, scale)

    return modules, has_dora


def module_dot(a: LoraModule, b: LoraModule) -> float:
    """dot(dW_a, dW_b) for one module pair via the trace identity (see module
    docstring) -- never forms the full [out_features, in_features] delta.
    Pass the same module for both a and b to get ||dW||_F^2.
    """
    down_a, up_a, scale_a = a
    down_b, up_b, scale_b = b
    if up_a.shape[0] != up_b.shape[0] or down_a.shape[1] != down_b.shape[1]:
        raise ValueError("module shape mismatch -- not the same base architecture")
    m = up_a.T @ up_b  # [rank_a, rank_b]
    p = m @ down_b  # [rank_a, in_features]
    return float(scale_a * scale_b * np.sum(down_a * p))


def checkpoint_norm(modules: dict[str, LoraModule]) -> float:
    total_sq = sum(module_dot(m, m) for m in modules.values())
    return total_sq**0.5


def checkpoint_cosine(a: dict[str, LoraModule], b: dict[str, LoraModule]) -> float | None:
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    dot = sum(module_dot(a[name], b[name]) for name in shared)
    norm_a = sum(module_dot(a[name], a[name]) for name in shared) ** 0.5
    norm_b = sum(module_dot(b[name], b[name]) for name in shared) ** 0.5
    denom = norm_a * norm_b
    if denom < 1e-12:
        return None
    return dot / denom


def dominant_rank_alpha_scale(modules: dict[str, LoraModule]) -> tuple[int, float, float, bool]:
    """Representative (rank, alpha, scale) for a checkpoint, plus a mixed-config flag.

    Almost every real checkpoint uses one rank/alpha throughout (OneTrainer's
    layer_filter_preset selects WHICH modules train, not a per-module rank/alpha),
    so this just reads the first module and cross-checks the rest agree. `mixed`
    is True on disagreement -- rare, but worth surfacing rather than silently
    picking one value, since it also means the scale-projected bands in
    classify() are unreliable for this checkpoint.
    """
    first = next(iter(modules.values()))
    rank = first[0].shape[0]
    scale = first[2]
    mixed = any(down.shape[0] != rank or abs(s - scale) > 1e-9 for down, _, s in modules.values())
    alpha = scale * rank
    return rank, alpha, scale, mixed


def classify(norm: float, scale: float) -> str:
    """Band a checkpoint's ||dW||_F -- scale-aware.

    The reference bands (REF_CLEAN_MAX/REF_COLLAPSED_MIN) were measured at
    REF_SCALE (rank 16, alpha 1.0 -> 0.0625). This projects the checkpoint's own
    norm onto that reference scale first: norm * (REF_SCALE / scale). The
    assumption behind the projection is that the low-rank factors themselves
    (what training actually adapts) reach a comparable magnitude regardless of
    the fixed alpha/rank multiplier, so ||dW||_F scales roughly linearly with
    that multiplier. This is a first-order approximation, not independently
    measured at other scales -- e.g. it has not been verified that a rank
    16/alpha 16 run (scale 1.0, 16x reference) actually collapses at a
    projected ~3.26-6.70, only that IF the assumption holds, that's where the
    reference bands land after projection. Treat the returned band as a rough
    steer at non-reference scale; the growth-knee and adjacent-cosine signals
    need no such projection and are the ones worth trusting across configs.
    """
    if scale <= 0:
        projected = norm
    else:
        projected = norm * (REF_SCALE / scale)
    if projected <= REF_CLEAN_MAX:
        return "clean-range"
    if projected >= REF_COLLAPSED_MIN:
        return "collapsed-range"
    return "between reference bands"


def analyze_group(checkpoints: list[Path]) -> tuple[list[dict], float | None]:
    """Run the load / norm / cosine / growth / knee pipeline over one sorted
    checkpoint list. Independent per group -- no state carries across groups,
    which is the fix for the cross-arm interleaving discover_checkpoints()
    used to allow.
    """
    rows = []
    loaded: list[dict[str, LoraModule]] = []
    for path in checkpoints:
        print(f"Loading {path.name} ...", file=sys.stderr, flush=True)
        try:
            modules, has_dora = load_lora_modules(path)
        except ValueError as e:
            print(f"SKIP {path.name}: {e}", file=sys.stderr)
            continue
        norm = checkpoint_norm(modules)
        rank, alpha, scale, mixed = dominant_rank_alpha_scale(modules)
        band = "n/a (DoRA)" if has_dora else classify(norm, scale)
        print(
            f"  ||dW||_F = {norm:.4f} ({len(modules)} modules, rank {rank}, alpha {alpha:.4g}, scale {scale:.4f})",
            file=sys.stderr,
            flush=True,
        )
        if has_dora:
            print(
                "  NOTE: also carries '.dora_scale' keys (DoRA magnitude decomposition). "
                "||dW||_F below covers only the low-rank direction term, not DoRA's per-channel "
                "magnitude scaling -- treat it as directional only; the clean/collapsed bands do not apply.",
                file=sys.stderr,
            )
        if mixed:
            print(
                "  NOTE: modules disagree on rank/alpha/scale -- reporting the first module found; "
                "the scale-projected band above is unreliable for this checkpoint.",
                file=sys.stderr,
            )
        rows.append(
            {
                "file": path.name,
                "norm_frobenius": round(norm, 4),
                "modules": len(modules),
                "rank": rank,
                "alpha": round(alpha, 4),
                "scale": round(scale, 6),
                "has_dora": has_dora,
                "mixed_config": mixed,
                "band": band,
            }
        )
        loaded.append(modules)

    # Adjacent-pair cosine and growth ratio, computed only across successfully-loaded
    # checkpoints in sorted order. `loaded` and `rows` stay index-aligned since both
    # only grow on a successful load, in the same loop iteration, in the same order.
    for i in range(1, len(rows)):
        rows[i]["cosine_vs_prev"] = checkpoint_cosine(loaded[i - 1], loaded[i])
        prev_norm = rows[i - 1]["norm_frobenius"]
        rows[i]["norm_growth_ratio"] = round(rows[i]["norm_frobenius"] / prev_norm, 4) if prev_norm > 1e-9 else None

    # Knee flag: a checkpoint whose growth ratio sits well above the sweep's own
    # median growth ratio -- the norm curve accelerating relative to itself. This
    # needs no cross-config projection (unlike classify()'s bands), so it's the
    # more portable of the two signals this script reports.
    growth_ratios = [r["norm_growth_ratio"] for r in rows if r.get("norm_growth_ratio") is not None]
    median_growth = float(np.median(growth_ratios)) if len(growth_ratios) >= 2 else None
    for row in rows:
        gr = row.get("norm_growth_ratio")
        row["knee"] = bool(median_growth and gr is not None and gr >= 1.5 * median_growth)

    endpoint_cosine = checkpoint_cosine(loaded[0], loaded[-1]) if len(loaded) >= 2 else None
    return rows, endpoint_cosine


def print_group_table(name: str, rows: list[dict], endpoint_cosine: float | None, show_name: bool) -> None:
    print("=" * 100)
    if show_name:
        print(name)
        print("-" * 100)
    print(f"{'file':<38} {'||dW||_F':>9} {'rank/a/scale':>15} {'cos_prev':>9} {'growth':>8}  band")
    print("-" * 100)
    for row in rows:
        cos = row.get("cosine_vs_prev")
        cos_str = f"{cos:.3f}" if cos is not None else "  --"
        growth = row.get("norm_growth_ratio")
        growth_str = f"{growth:.2f}x" if growth is not None else "  --"
        knee_mark = " KNEE" if row.get("knee") else ""
        ras = f"{row['rank']}/{row['alpha']:g}/{row['scale']:.3f}"
        print(
            f"{row['file']:<38} {row['norm_frobenius']:>9.4f} {ras:>15} {cos_str:>9} {growth_str:>8}  "
            f"{row['band']}{knee_mark}"
        )
    print("-" * 100)
    print(f"Endpoint cosine (first vs. last): {endpoint_cosine:.3f}" if endpoint_cosine is not None else "Endpoint cosine: n/a (need >=2 checkpoints)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rendering-free LoRA collapse detector: ||dW||_F and cosine drift across checkpoints.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more kohya-format .safetensors files and/or directories to glob for *.safetensors. "
        "Each directory argument is analyzed as its own independent group.",
    )
    parser.add_argument("--json", metavar="FILE", help="Write the full results table to this JSON file.")
    args = parser.parse_args()

    groups = discover_checkpoints(args.paths)
    if not groups or not any(checkpoints for _, checkpoints in groups):
        print("No .safetensors files found.", file=sys.stderr)
        return 1

    multi = len(groups) > 1
    group_results: dict[str, dict] = {}
    for name, checkpoints in groups:
        if not checkpoints:
            print(f"WARNING: no .safetensors files in {name}, skipping.", file=sys.stderr)
            continue
        rows, endpoint_cosine = analyze_group(checkpoints)
        print_group_table(name, rows, endpoint_cosine, show_name=multi)
        group_results[name] = {"checkpoints": rows, "endpoint_cosine": endpoint_cosine}

    print()
    print(f"Reference values (rank 16, alpha 1.0 -> scale {REF_SCALE:.4f}, attn-mlp layer filter):")
    print(f"  clean <= {REF_CLEAN_MAX}, collapsed >= {REF_COLLAPSED_MIN}, adjacent cosine ~{REF_ADJACENT_COSINE},")
    print(f"  overcooked-sweep endpoint cosine {REF_ENDPOINT_COSINE}.")
    print("Bands above are each checkpoint's own norm PROJECTED onto this reference scale via")
    print("its own rank/alpha (see classify() docstring) -- a first-order approximation, not")
    print("separately measured at other scales. What transfers with no projection needed: a")
    print("KNEE in the growth column, and a drop in adjacent-checkpoint cosine.")
    print("=" * 100)

    if args.json:
        reference = {
            "clean_max": REF_CLEAN_MAX,
            "collapsed_min": REF_COLLAPSED_MIN,
            "adjacent_cosine": REF_ADJACENT_COSINE,
            "endpoint_cosine": REF_ENDPOINT_COSINE,
            "scale": REF_SCALE,
        }
        if multi:
            out = {"groups": group_results, "reference": reference}
        else:
            # Single group: keep the flat (pre-grouping) shape unchanged.
            (only_result,) = group_results.values()
            out = {**only_result, "reference": reference}
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
