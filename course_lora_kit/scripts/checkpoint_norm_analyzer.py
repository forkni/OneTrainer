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

    recipe     The rank and alpha the file was actually trained with, checked
               against the recipe (the course style recipe by default, or the
               JSON passed with --recipe). A mismatch is reported as a
               CONVENTION warning: alpha raised with the learning rate
               untouched lands every optimizer step (alpha_file / alpha_recipe)
               times harder on the base model (Hu et al. 2021, section 4.1).

    hot start  A run whose FIRST save already sits far above the recipe's
               reference first save and whose growth never accelerates
               afterwards (the first pair is the steepest, no knee anywhere):
               the acceleration happened before the first save. Round 3's
               alpha-16 arm is the measured example -- first save 7.30 against
               the control's 0.35, growth falling from the first pair, and
               no knee, because the knee rule only fires on a run that speeds
               up relative to its own median.

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

What transfers across recipes, and what does not
------------------------------------------------
The SHAPE signals need no calibration: a KNEE in ||dW||_F growth, a drop in
adjacent-checkpoint cosine, a first save that is already large (hot start), and
a rank/alpha that is not what the recipe says (convention). Read those first.

The absolute bands (REF_CLEAN_MAX / REF_COLLAPSED_MIN) were measured on Round 2
(rank 16, alpha 1.0 -> scale 0.0625, 48 frames) and are printed only with
--bands. On Round 3 (50 curated frames, same recipe) every arm that rendered
clean crossed the "collapsed" number by step 1,000 and the control's own pick
reads 9.62 -- the bands are local to the dataset, the step count and the loss
weighting they were measured on, and Round 3 contains no collapsed checkpoint
to re-measure them against. So they are off by default; the reference first
save (REF_FIRST_SAVE_NORM, Round 3 control at step 99) replaced them as the one
absolute number the default output leans on, and only for the hot-start check.

With --bands, a checkpoint trained at a different scale (e.g. a
kohya-conventional rank 16 / alpha 16, scale 1.0 -- 16x the reference) is
projected onto the reference scale before banding, on the first-order
assumption that ||dW||_F scales roughly linearly with a fixed alpha/rank
multiplier (see classify()). Note what that projection does to a hot start: it
divides the 16x edit back out and reads the alpha-16 arm as "clean-range" for
ten saves. The hot-start check therefore compares the RAW norm -- the edit the
base model actually receives -- never the projected one.

Needs: numpy, safetensors. No torch, no GPU, no base model.

Usage:
    python checkpoint_norm_analyzer.py lora.safetensors
    python checkpoint_norm_analyzer.py path/to/checkpoint/dir/
    python checkpoint_norm_analyzer.py ckpt_dir/ --json norms.json
    python checkpoint_norm_analyzer.py ckpt_dir_A/ ckpt_dir_B/ --json norms.json
    python checkpoint_norm_analyzer.py ckpt_dir/ --recipe recipe.json
    python checkpoint_norm_analyzer.py ckpt_dir/ --bands

--recipe takes any JSON with "lora_rank" and "lora_alpha" keys: an OneTrainer
config, or the merged recipe 00b_print_recipe.cmd writes with --json (overlay
files such as configs\\style_sdxl.json inherit rank/alpha from the preset and
do not carry the keys). Without --recipe the course style recipe (rank 16,
alpha 1.0) is assumed; --no-recipe-check turns the check off.

Each directory argument is its own group: checkpoints are discovered, sorted,
and have their cosine/growth/knee/endpoint analysis run independently within
that group, so passing several arms in one invocation does not interleave
their checkpoints into a single cross-arm trace. Bare file arguments are
collected into one additional "files" group. With a single directory (or
single file) argument -- the common case -- output and --json shape are
what earlier versions of this script produced, plus the recipe_check and
hot_start entries; multiple groups print one table per group and, with
--json, nest each group's results under its own key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

# Round 2 reference bands, printed only with --bands -- see the module docstring.
REF_CLEAN_MAX = 3.26
REF_COLLAPSED_MIN = 6.70
REF_ADJACENT_COSINE = 0.85
REF_ENDPOINT_COSINE = 0.19

# Scale (alpha / rank) the bands above were measured at: rank 16, alpha 1.0,
# attn-mlp layer filter. classify() projects a checkpoint's own scale onto this
# reference scale before banding -- see its docstring for the caveat.
REF_SCALE = 1.0 / 16.0

# The course style recipe: OneTrainer's shipped "#sdxl 1.0 LoRA" preset (rank 16,
# alpha 1.0, LR 3e-4) under course_lora_kit\configs\style_sdxl.json. --recipe
# replaces these with the values read from a config or merged-recipe JSON.
RECIPE_RANK = 16
RECIPE_ALPHA = 1.0

# Round 3 control on that recipe (50 curated frames, save_every 100, first save at
# step 99): ||dW||_F 0.3513. A first save at HOT_START_FACTOR times this or more,
# with growth that never accelerates afterwards, is flagged as a hot start. The
# five Round 3 arms on the recipe's alpha read 0.12-0.57 at their first save
# (0.3x-1.6x); the alpha-16 arm read 7.30 (21x).
REF_FIRST_SAVE_NORM = 0.35
HOT_START_FACTOR = 4.0

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


def load_recipe(path: Path) -> tuple[int, float]:
    """Read (lora_rank, lora_alpha) from an OneTrainer config or merged-recipe JSON.

    Accepts the raw numbers a config carries (16, 1.0) and the rendered strings
    print_effective_config.py writes ("16", "1.0"). Raises ValueError, with the
    fix spelled out, when the keys are absent -- an overlay such as
    configs\\style_sdxl.json only lists the dials it changes.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object with lora_rank / lora_alpha keys")
    missing = [k for k in ("lora_rank", "lora_alpha") if k not in data]
    if missing:
        raise ValueError(
            f"{path}: no {' / '.join(missing)} key. Overlay files only list the dials they change and "
            "inherit rank/alpha from the preset; pass the merged recipe instead "
            "(00b_print_recipe.cmd --json recipe.json, i.e. print_effective_config.py --all --json)."
        )

    def num(v) -> float:
        return float(str(v).strip().strip("'\""))

    rank = num(data["lora_rank"])
    alpha = num(data["lora_alpha"])
    if rank <= 0 or rank != int(rank):
        raise ValueError(f"{path}: lora_rank must be a positive integer, got {data['lora_rank']!r}")
    return int(rank), alpha


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
    """Band a checkpoint's ||dW||_F against the Round 2 reference -- scale-aware,
    and only used with --bands.

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
    steer at non-reference scale; the growth-knee, adjacent-cosine, hot-start
    and convention signals need no such projection and are the ones worth
    trusting across configs.
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


def recipe_check(rows: list[dict], recipe_rank: int, recipe_alpha: float) -> dict:
    """Compare every checkpoint's own rank/alpha with the recipe's.

    A checkpoint that disagrees is a convention mix, not a training outcome:
    the file was trained with a different alpha/rank multiplier than the recipe
    says, so with the learning rate left as the recipe set it, every optimizer
    step landed (scale_file / scale_recipe) times harder on the base model
    (Hu et al. 2021, section 4.1: with Adam, tuning alpha is roughly the same as
    tuning the learning rate). The Round 3 alpha-16 arm is the measured case:
    the same 3e-4, sixteen times the step.
    """
    mismatched = [
        r for r in rows if r["rank"] != recipe_rank or abs(r["alpha"] - recipe_alpha) > 1e-6
    ]
    result = {
        "recipe_rank": recipe_rank,
        "recipe_alpha": recipe_alpha,
        "match": not mismatched,
        "mismatched_files": [r["file"] for r in mismatched],
    }
    if mismatched:
        r0 = mismatched[0]
        recipe_scale = recipe_alpha / recipe_rank if recipe_rank else 0.0
        result["file_rank"] = r0["rank"]
        result["file_alpha"] = r0["alpha"]
        result["effective_step_factor"] = round(r0["scale"] / recipe_scale, 3) if recipe_scale > 0 else None
    return result


def hot_start_check(rows: list[dict], ref_first_save: float) -> dict:
    """Flag a run whose first save is already far above the recipe's reference
    first save and whose growth never accelerates afterwards.

    Three conditions, all on the RAW norm (the edit the base model receives,
    not the scale-projected one -- projection is exactly what hides a hot
    start, see the module docstring):
      1. first save >= HOT_START_FACTOR x ref_first_save;
      2. the first growth pair is the run's steepest (growth ratios fall from
         the first pair, up to save-to-save noise);
      3. no KNEE anywhere -- the acceleration the knee rule looks for happened
         before the first save.
    Needs at least three saves for 2 and 3 to mean anything; with fewer the
    ratio is still reported, the flag stays off.
    """
    if not rows:
        return {"flag": False}
    first = rows[0]["norm_frobenius"]
    ratio = first / ref_first_save if ref_first_save > 0 else None
    growth = [r["norm_growth_ratio"] for r in rows[1:] if r.get("norm_growth_ratio") is not None]
    first_pair_is_steepest = len(growth) >= 2 and max(growth) <= growth[0] + 1e-9
    no_knee = not any(r.get("knee") for r in rows)
    flag = bool(ratio is not None and ratio >= HOT_START_FACTOR and first_pair_is_steepest and no_knee)
    return {
        "flag": flag,
        "first_save_norm": round(first, 4),
        "reference_first_save_norm": ref_first_save,
        "first_save_ratio": round(ratio, 2) if ratio is not None else None,
        "first_pair_is_steepest": first_pair_is_steepest,
        "no_knee": no_knee,
    }


def analyze_group(checkpoints: list[Path], bands: bool = False) -> tuple[list[dict], float | None]:
    """Run the load / norm / cosine / growth / knee pipeline over one sorted
    checkpoint list. Independent per group -- no state carries across groups,
    which is the fix for the cross-arm interleaving discover_checkpoints()
    used to allow. The Round 2 band is attached only when `bands` is set.
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
        print(
            f"  ||dW||_F = {norm:.4f} ({len(modules)} modules, rank {rank}, alpha {alpha:.4g}, scale {scale:.4f})",
            file=sys.stderr,
            flush=True,
        )
        if has_dora:
            print(
                "  NOTE: also carries '.dora_scale' keys (DoRA magnitude decomposition). "
                "||dW||_F below covers only the low-rank direction term, not DoRA's per-channel "
                "magnitude scaling -- treat it as directional only; the Round 2 bands do not apply.",
                file=sys.stderr,
            )
        if mixed:
            print(
                "  NOTE: modules disagree on rank/alpha/scale -- reporting the first module found; "
                "the recipe check and any scale-projected band are unreliable for this checkpoint.",
                file=sys.stderr,
            )
        row = {
            "file": path.name,
            "norm_frobenius": round(norm, 4),
            "modules": len(modules),
            "rank": rank,
            "alpha": round(alpha, 4),
            "scale": round(scale, 6),
            "has_dora": has_dora,
            "mixed_config": mixed,
        }
        if bands:
            row["band"] = "n/a (DoRA)" if has_dora else classify(norm, scale)
        rows.append(row)
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


def print_group_table(
    name: str,
    rows: list[dict],
    endpoint_cosine: float | None,
    show_name: bool,
    recipe: dict | None,
    hot_start: dict,
    bands: bool,
) -> None:
    print("=" * 100)
    if show_name:
        print(name)
        print("-" * 100)
    band_hdr = "  band" if bands else ""
    print(f"{'file':<38} {'||dW||_F':>9} {'rank/a/scale':>15} {'cos_prev':>9} {'growth':>8}  flags{band_hdr}")
    print("-" * 100)
    for i, row in enumerate(rows):
        cos = row.get("cosine_vs_prev")
        cos_str = f"{cos:.3f}" if cos is not None else "  --"
        growth = row.get("norm_growth_ratio")
        growth_str = f"{growth:.2f}x" if growth is not None else "  --"
        flags = []
        if row.get("knee"):
            flags.append("KNEE")
        if i == 0 and hot_start.get("flag"):
            flags.append("HOT-START")
        if recipe and row["file"] in recipe["mismatched_files"]:
            flags.append("CONVENTION")
        flag_str = " ".join(flags) if flags else "--"
        band_str = f"  {row['band']}" if bands else ""
        ras = f"{row['rank']}/{row['alpha']:g}/{row['scale']:.3f}"
        print(
            f"{row['file']:<38} {row['norm_frobenius']:>9.4f} {ras:>15} {cos_str:>9} {growth_str:>8}  "
            f"{flag_str}{band_str}"
        )
    print("-" * 100)
    print(f"Endpoint cosine (first vs. last): {endpoint_cosine:.3f}" if endpoint_cosine is not None else "Endpoint cosine: n/a (need >=2 checkpoints)")

    if recipe:
        rr, ra = recipe["recipe_rank"], recipe["recipe_alpha"]
        if recipe["match"]:
            print(f"Recipe check: rank {rr} / alpha {ra:g} (scale {ra / rr:.4f}) -- every save matches.")
        else:
            fr, fa = recipe["file_rank"], recipe["file_alpha"]
            factor = recipe.get("effective_step_factor")
            n_bad = len(recipe["mismatched_files"])
            which = "every save" if n_bad == len(rows) else f"{n_bad} of {len(rows)} saves"
            print(
                f"Recipe check: CONVENTION MISMATCH -- {which} trained at rank {fr} / alpha {fa:g} "
                f"(scale {fa / fr:.4f}); the recipe says rank {rr} / alpha {ra:g} (scale {ra / rr:.4f})."
            )
            if factor:
                print(
                    f"  With the learning rate left as the recipe set it, every optimizer step landed "
                    f"{factor:g}x harder on the base model (Hu et al. 2021, sec. 4.1: alpha and LR travel"
                )
                print("  together). Check lora_alpha / lora_rank in the run's config before reading anything else.")

    ratio = hot_start.get("first_save_ratio")
    if ratio is not None:
        line = (
            f"Hot start: first save {hot_start['first_save_norm']:.4f} = {ratio:g}x the recipe's reference "
            f"first save ({hot_start['reference_first_save_norm']:g})"
        )
        if hot_start["flag"]:
            print(line + "; first pair is the run's steepest, no KNEE anywhere -> HOT-START.")
            print("  The acceleration the knee looks for happened before the first save; the run began")
            print("  where a healthy run ends its warm-up. A convention mismatch above is the usual cause.")
        else:
            why = []
            if ratio < HOT_START_FACTOR:
                why.append(f"below the {HOT_START_FACTOR:g}x line")
            if not hot_start["first_pair_is_steepest"]:
                why.append("growth still rising after the first pair")
            if not hot_start["no_knee"]:
                why.append("a KNEE fired later")
            print(line + " -- no (" + ", ".join(why) + ").")


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
    parser.add_argument(
        "--recipe",
        metavar="JSON",
        help="OneTrainer config or merged recipe (00b_print_recipe.cmd --json) whose lora_rank / lora_alpha "
        f"every checkpoint is checked against. Default: the course style recipe, rank {RECIPE_RANK} / "
        f"alpha {RECIPE_ALPHA:g}.",
    )
    parser.add_argument("--no-recipe-check", action="store_true", help="Skip the rank/alpha convention check.")
    parser.add_argument(
        "--ref-first-save",
        type=float,
        default=REF_FIRST_SAVE_NORM,
        metavar="NORM",
        help=f"||dW||_F of the recipe's reference run at its first save (default {REF_FIRST_SAVE_NORM}: "
        "Round 3 control, step 99). The hot-start check compares the first save against "
        f"{HOT_START_FACTOR:g}x this.",
    )
    parser.add_argument(
        "--bands",
        action="store_true",
        help="Also print the Round 2 reference bands (clean <= 3.26 / collapsed >= 6.70, scale-projected). "
        "Off by default: they are local to the Round 2 dataset and step count (see the docstring).",
    )
    args = parser.parse_args()

    recipe_rank, recipe_alpha, recipe_source = RECIPE_RANK, RECIPE_ALPHA, "course style recipe (built in)"
    if args.recipe:
        try:
            recipe_rank, recipe_alpha = load_recipe(Path(args.recipe))
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"ERROR reading --recipe: {e}", file=sys.stderr)
            return 2
        recipe_source = str(args.recipe)

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
        rows, endpoint_cosine = analyze_group(checkpoints, bands=args.bands)
        if not rows:
            print(f"WARNING: nothing loadable in {name}, skipping.", file=sys.stderr)
            continue
        recipe = None if args.no_recipe_check else recipe_check(rows, recipe_rank, recipe_alpha)
        hot_start = hot_start_check(rows, args.ref_first_save)
        print_group_table(name, rows, endpoint_cosine, show_name=multi, recipe=recipe, hot_start=hot_start, bands=args.bands)
        group_results[name] = {
            "checkpoints": rows,
            "endpoint_cosine": endpoint_cosine,
            "recipe_check": recipe,
            "hot_start": hot_start,
        }

    print()
    print("Column key:")
    print("  ||dW||_F      Frobenius norm of the whole learned weight delta -- overall size of the edit")
    print("  rank/a/scale  this file's own rank / alpha / (alpha divided by rank)")
    print("  cos_prev      cosine vs the PREVIOUS save's delta: how much the edit ROTATED, not grew")
    print("  growth        this save's ||dW||_F divided by the previous save's")
    print("  KNEE          growth >= 1.5x this sweep's median growth -- the run speeding up against itself")
    print("  HOT-START     first save >= {0:g}x the recipe's reference first save, and the run never".format(HOT_START_FACTOR))
    print("                speeds up afterwards: the acceleration happened before the first save")
    print("  CONVENTION    this file's rank/alpha is not the recipe's -- a mixed convention, not a result")
    if args.bands:
        print("  band          this norm projected onto the Round 2 reference scale, then bucketed")
    print()
    print(f"Recipe: rank {recipe_rank} / alpha {recipe_alpha:g} (scale {recipe_alpha / recipe_rank:.4f}) from {recipe_source}.")
    print(f"Reference first save: {args.ref_first_save:g} (Round 3 control, rank 16 / alpha 1.0, step 99).")
    print("Read the flags and cos_prev first. None of them needs a calibrated threshold beyond the")
    print("reference first save, and that one is only used for HOT-START.")
    if args.bands:
        print()
        print(f"Round 2 reference bands (rank 16, alpha 1.0 -> scale {REF_SCALE:.4f}, attn-mlp layer filter):")
        print(f"  clean <= {REF_CLEAN_MAX}, collapsed >= {REF_COLLAPSED_MIN}, adjacent cosine ~{REF_ADJACENT_COSINE},")
        print(f"  overcooked-sweep endpoint cosine {REF_ENDPOINT_COSINE}.")
        print("Bands are each checkpoint's own norm PROJECTED onto this reference scale via its own")
        print("rank/alpha (see classify() docstring) -- a first-order approximation, not separately")
        print("measured at other scales, and it divides a hot start back out (alpha 16 reads \"clean\").")
        print("They are recipe-local, not universal: on Round 3 every arm that rendered clean crossed")
        print("6.70 by step 1,000 and the control's own pick reads 9.62. Rough steer only.")
    print("=" * 100)

    if args.json:
        reference = {
            "recipe": {"rank": recipe_rank, "alpha": recipe_alpha, "source": recipe_source},
            "first_save_norm": args.ref_first_save,
            "hot_start_factor": HOT_START_FACTOR,
        }
        if args.bands:
            reference.update(
                {
                    "clean_max": REF_CLEAN_MAX,
                    "collapsed_min": REF_COLLAPSED_MIN,
                    "adjacent_cosine": REF_ADJACENT_COSINE,
                    "endpoint_cosine": REF_ENDPOINT_COSINE,
                    "scale": REF_SCALE,
                }
            )
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
