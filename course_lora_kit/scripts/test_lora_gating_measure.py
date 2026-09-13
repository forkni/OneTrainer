"""
Trigger-word gating measurement with OUT-OF-DOMAIN prompts, plus a numeric
d_gate / d_leak ratio -- plain diffusers, no StreamDiffusion wrapper.

Why out-of-domain prompts: an in-domain prompt (one that names the training
subject) cannot distinguish "the trigger gates nothing" from "this prompt summons
the trained concept on its own, trigger or not" -- both explanations predict the
same observed result. This script adds prompts far outside the training
distribution (a dog, a portrait, food) so LoRA leakage without the trigger and
LoRA response with the trigger can be measured separately, plus keeps one
in-domain prompt as a control for context (excluded from the headline number).

Per prompt, four cells per seed:
    off          -- LoRA scale 0 (base model)
    off_trigger  -- LoRA scale 0, trigger word still inserted (the token-insertion
                    noise floor -- see below)
    on_notrigger -- LoRA at --weight, prompt WITHOUT the trigger word
    on_trigger   -- LoRA at --weight, prompt WITH the trigger word prefixed

Metrics (mean absolute pixel difference over uint8 RGB, 0-255 scale):
    d_leak = mean|on_notrigger - off|         -- how much the LoRA changes an
                                                  UNtriggered prompt (should be ~0
                                                  if the trigger is a true gate)
    d_gate = mean|on_trigger - on_notrigger|  -- how much the trigger word ADDS
    d_null = mean|off_trigger - off|          -- token-insertion noise floor: any
                                                  text encoder shifts its whole
                                                  cross-attention pattern when ANY
                                                  word is inserted, even at LoRA
                                                  scale 0. Subtract this before
                                                  trusting a gating ratio.
    ratio        = d_gate / d_leak            -- naive: conflates real gating with
                                                  the token-insertion shift above
    ratio_net    = (d_gate - d_null) / d_leak -- honest about the numerator, but
                   NOT honest about the denominator -- see the floor check below.
                   ~0 means inert trigger (style is unconditionally on); >>1 means
                   the trigger is doing real gating work.

d_leak has its own noise floor, and this script does NOT subtract it (2026-09-12):
    d_null is the noise floor for INSERTING A TOKEN at LoRA scale 0 -- it has no
    connection to what d_leak measures, which is the effect of ENABLING THE
    ADAPTER at a fixed prompt. Computing max(0, d_leak - d_null) would net out a
    confound that was never in d_leak to begin with -- a category error, even
    though the correction points the right direction. What IS true: under the
    default EulerAncestralDiscreteScheduler, two renders that differ only in LoRA
    scale diverge somewhat from injected per-step noise alone, with nothing to do
    with style leaking. A high-magnitude adapter (large |dW|_F) amplifies that
    divergence, inflating d_leak for reasons that have nothing to do with leak,
    which deflates ratio_net for the highest-magnitude runs specifically -- the
    ones most likely to have actually learned something. --sampler deterministic
    and --seeds N (below) target this directly: a non-ancestral scheduler removes
    the injected-noise component, and averaging seeds tightens the estimate.
    leak_over_null and leak_below_floor (below) are a cheap diagnostic for when
    d_leak is small enough that the ratio built on it isn't trustworthy at all --
    they compare d_leak against the UNRELATED d_null floor only as a sanity check
    ("is this distance even bigger than typical measurement noise"), not as a
    correction.

Also reports mean HSV saturation per cell, since a style shift often reads as a
saturation shift and raw pixel diff alone won't distinguish that from unrelated
resampling noise. leak_sat / gate_sat below turn that into a second read on
leak and gate specifically -- still not a style metric (no style metric exists
in this kit), just a proxy that happens to be sensitive to a desaturated target
style the way raw RGB-delta is not.

The built-in prompt battery below (DEFAULT_PROMPTS) is a starting point, not a
fixed list -- pass --prompts-file to swap it out without editing this file. The
file is JSON: a list of either [label, prompt, in_domain] triples or
{"label": ..., "prompt": ..., "in_domain": ...} objects (in_domain defaults to
false when a dict entry omits it). Keep at least one clearly out-of-domain prompt
per unrelated subject so the headline ratio isn't dominated by in-domain leakage.

Needs only plain diffusers -- no streamdiffusion import, no StreamDiffusion checkout.

Usage:
    python test_lora_gating_measure.py --lora path/to/lora.safetensors --trigger mytrigger
    python test_lora_gating_measure.py --lora path/to/lora.safetensors --trigger mytrigger --prompts-file battery.json
    python test_lora_gating_measure.py --lora path/to/lora.safetensors --trigger mytrigger --sampler deterministic --seeds 3

--sampler {ancestral,deterministic} (default ancestral, matches every prior run
    in this kit byte for byte): deterministic swaps in EulerDiscreteScheduler,
    the non-ancestral counterpart of the default EulerAncestralDiscreteScheduler
    -- same solver family, no per-step injected noise, so two renders differing
    only in LoRA scale no longer diverge for reasons unrelated to the adapter.
--seeds N (default 1, matches every prior run byte for byte): renders each cell
    at N seeds starting from --seed and averages the four distances (and the
    saturations) across them before computing the ratios below. Per-seed values
    are kept in "seeds" on each result row for inspection; only the first seed's
    images are written to disk and feed the contact sheet, to keep multi-seed
    runs from multiplying file count and GPU time. Distances at n=1 are a single
    sample of a stochastic quantity; this is the standard fix.

battery.json example:
    [
        ["snowy_forest", "standing in a snowy forest at night, wide shot", false],
        {"label": "class_hijack", "prompt": "a girl with short hair, portrait", "in_domain": true}
    ]
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

from diffusers import (
    AutoencoderTiny,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEG_PROMPT = "blurry, low quality"
SCRIPT_VERSION = "2.0-floor-check"  # 2026-09-12: added leak_over_null/leak_below_floor,
                                     # --sampler, --seeds, leak_sat/gate_sat. All additive;
                                     # defaults (ancestral, seeds=1) reproduce v1 exactly.

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


# (label, prompt, in_domain) -- in_domain prompts are reported separately: they are
# expected to leak somewhat even with a perfectly gated trigger, since the base model's
# own semantics already point at the training subject. Swap these for prompts that make
# sense against your own trigger/subject via --prompts-file (see module docstring);
# keep at least one clearly out-of-domain prompt per unrelated subject so the aggregate
# isn't dominated by in-domain leakage.
DEFAULT_PROMPTS = [
    ("dog_park", "a golden retriever running in a park, photograph", False),
    ("fisherman_portrait", "portrait of an old fisherman, oil painting", False),
    ("ramen_bowl", "a bowl of ramen noodles, food photography", False),
]


def load_prompts(prompts_file: str | None) -> list[tuple[str, str, bool]]:
    """Load the (label, prompt, in_domain) battery from --prompts-file, or fall
    back to DEFAULT_PROMPTS. See the module docstring for the JSON format.
    """
    if prompts_file is None:
        return list(DEFAULT_PROMPTS)
    data = json.loads(Path(prompts_file).read_text())
    prompts = []
    for entry in data:
        if isinstance(entry, dict):
            prompts.append((entry["label"], entry["prompt"], bool(entry.get("in_domain", False))))
        else:
            label, prompt, in_domain = entry
            prompts.append((label, prompt, bool(in_domain)))
    if not prompts:
        raise ValueError(f"--prompts-file {prompts_file} contained no prompts")
    return prompts


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    xa = np.asarray(a.convert("RGB"), dtype=np.float64)
    xb = np.asarray(b.convert("RGB"), dtype=np.float64)
    return float(np.abs(xa - xb).mean())


def mean_saturation(img: Image.Image) -> float:
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    return float(hsv[..., 1].mean())  # 0-255 scale


def render_cells(pipe, base_prompt: str, trigger: str, weight: float, negative_prompt: str,
                  steps: int, guidance_scale: float, seed: int, label: str) -> dict[str, Image.Image]:
    """Render the four gating cells for one prompt at one seed. Same four renders,
    same order, as the original single-seed loop -- pulled into a function so the
    --seeds loop can call it once per seed without duplicating the body.
    """
    cells = {}
    for tag, scale, prompt in (
        ("off", 0.0, base_prompt),
        # Control: LoRA OFF, trigger word still inserted. SDXL's text encoder
        # conditions on token position, so inserting ANY word shifts the whole
        # cross-attention pattern even with zero LoRA scale -- this is the noise
        # floor that a naive on_trigger-vs-on_notrigger diff would conflate with
        # real trigger-gated LoRA behavior.
        ("off_trigger", 0.0, f"{trigger}, {base_prompt}"),
        ("on_notrigger", weight, base_prompt),
        ("on_trigger", weight, f"{trigger}, {base_prompt}"),
    ):
        print(f"--- {label}/{tag} --- prompt={prompt!r} scale={scale} seed={seed}")
        gen = torch.Generator(device="cuda").manual_seed(seed)
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=gen,
            cross_attention_kwargs={"scale": scale},
        ).images[0]
        cells[tag] = img
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description="Out-of-domain trigger-gating measurement")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lora", required=True, help="HF repo id or local .safetensors path")
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--trigger", required=True, help="Trigger word/phrase to test gating for")
    parser.add_argument(
        "--prompts-file",
        default=None,
        metavar="FILE",
        help="JSON file overriding the built-in DEFAULT_PROMPTS battery -- see module docstring "
        "for the format. Lets the prompt battery be swapped per run without editing this file.",
    )
    parser.add_argument(
        "--in-domain-prompt",
        default=None,
        help="Optional prompt that names the training subject directly, reported separately "
        "for context (excluded from the headline gating ratio -- see module docstring).",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampler",
        choices=["ancestral", "deterministic"],
        default="ancestral",
        help="ancestral (default) = EulerAncestralDiscreteScheduler, matches every prior run in "
        "this kit. deterministic = EulerDiscreteScheduler, the non-ancestral counterpart -- no "
        "per-step injected noise, so two renders differing only in LoRA scale no longer diverge "
        "for reasons unrelated to the adapter. See module docstring.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds to average per prompt, starting at --seed and incrementing by 1 "
        "(default 1, matches every prior run in this kit byte for byte). See module docstring.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "lora_gating_measure"),
    )
    args = parser.parse_args()
    args.model = resolve_local_snapshot(args.model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_file)
    if args.in_domain_prompt:
        prompts.append(("in_domain", args.in_domain_prompt, True))

    seed_list = [args.seed + i for i in range(max(1, args.seeds))]

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
    if args.sampler == "deterministic":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print(f"Loading LoRA weights from {args.lora} ...")
    pipe.load_lora_weights(args.lora)

    results = []
    thumbs = []  # (row_label, [off, off_trigger, notrigger, trigger]) -- first seed only
    for label, base_prompt, in_domain in prompts:
        per_seed = []
        first_seed_imgs = None
        for si, seed in enumerate(seed_list):
            cells = render_cells(
                pipe, base_prompt, args.trigger, args.weight, args.negative_prompt,
                args.steps, args.guidance_scale, seed, label,
            )
            if si == 0:
                # Only the first seed's cells are written to disk / feed the contact
                # sheet -- multi-seed runs would otherwise multiply file count for no
                # benefit the JSON doesn't already capture per-seed.
                for tag, img in cells.items():
                    img.save(out_dir / f"{label}__{tag}.png")
                first_seed_imgs = [cells["off"], cells["off_trigger"], cells["on_notrigger"], cells["on_trigger"]]

            d_leak_i = mean_abs_diff(cells["on_notrigger"], cells["off"])
            d_gate_i = mean_abs_diff(cells["on_trigger"], cells["on_notrigger"])
            d_null_i = mean_abs_diff(cells["off_trigger"], cells["off"])
            sat_i = {tag: mean_saturation(img) for tag, img in cells.items()}
            per_seed.append({"seed": seed, "d_leak": d_leak_i, "d_gate": d_gate_i, "d_null": d_null_i, "sat": sat_i})

        n = len(per_seed)
        d_leak = statistics.fmean(x["d_leak"] for x in per_seed)
        d_gate = statistics.fmean(x["d_gate"] for x in per_seed)
        d_null = statistics.fmean(x["d_null"] for x in per_seed)
        d_leak_std = statistics.pstdev(x["d_leak"] for x in per_seed) if n > 1 else 0.0
        d_gate_net = max(0.0, d_gate - d_null)  # gating effect net of the token-shift confound
        ratio = (d_gate / d_leak) if d_leak > 1e-6 else float("inf")
        ratio_net = (d_gate_net / d_leak) if d_leak > 1e-6 else float("inf")
        sat = {tag: statistics.fmean(x["sat"][tag] for x in per_seed) for tag in ("off", "off_trigger", "on_notrigger", "on_trigger")}

        # Floor check, NOT a correction (see module docstring): d_leak and d_null measure
        # different interventions (adapter-on vs token-insertion), so this is a sanity
        # comparison against a noise floor of the same rough scale, not a subtraction.
        leak_over_null = (d_leak / d_null) if d_null > 1e-6 else float("inf")
        leak_below_floor = d_leak < d_null

        # Saturation-based second read on leak/gate: still not a style metric, just a
        # proxy that (for a desaturated target style) is sensitive where raw RGB-delta
        # is blind to composition-preserving colour shifts. See module docstring.
        leak_sat = abs(sat["on_notrigger"] - sat["off"])
        gate_sat = abs(sat["on_trigger"] - sat["on_notrigger"])

        row = {
            "prompt_label": label,
            "in_domain": in_domain,
            "prompt": base_prompt,
            "d_leak": round(d_leak, 3),
            "d_leak_std": round(d_leak_std, 3),
            "d_null_token_insertion_floor": round(d_null, 3),
            "d_gate_net_of_null": round(d_gate_net, 3),
            "ratio_net_d_gate_over_d_leak": round(ratio_net, 4) if ratio_net != float("inf") else None,
            "d_gate": round(d_gate, 3),
            "ratio_d_gate_over_d_leak": round(ratio, 4) if ratio != float("inf") else None,
            "leak_over_null": round(leak_over_null, 4) if leak_over_null != float("inf") else None,
            "leak_below_floor": leak_below_floor,
            "leak_sat": round(leak_sat, 2),
            "gate_sat": round(gate_sat, 2),
            "saturation_off": round(sat["off"], 2),
            "saturation_off_trigger": round(sat["off_trigger"], 2),
            "saturation_on_notrigger": round(sat["on_notrigger"], 2),
            "saturation_on_trigger": round(sat["on_trigger"], 2),
            "seeds": [
                {"seed": x["seed"], "d_leak": round(x["d_leak"], 3), "d_gate": round(x["d_gate"], 3), "d_null": round(x["d_null"], 3)}
                for x in per_seed
            ],
        }
        results.append(row)
        if first_seed_imgs is not None:
            thumbs.append((label, first_seed_imgs))
        print(
            f"    d_leak={d_leak:.3f} (std={d_leak_std:.3f} over {n} seed(s))  d_gate={d_gate:.3f}  d_null={d_null:.3f}  "
            f"d_gate_net={d_gate_net:.3f}  ratio_net={row['ratio_net_d_gate_over_d_leak']}  "
            f"sat(off/offtrig/notrig/trig)={sat['off']:.1f}/{sat['off_trigger']:.1f}/"
            f"{sat['on_notrigger']:.1f}/{sat['on_trigger']:.1f}"
        )
        if leak_below_floor:
            print(
                f"    LEAK-BELOW-FLOOR: d_leak={d_leak:.3f} is under this run's own token-insertion "
                f"floor d_null={d_null:.3f} (leak_over_null={row['leak_over_null']}) -- the ratio "
                f"built on this d_leak is dividing by a distance no bigger than measurement noise."
            )

    # Contact sheet: rows = prompts, cols = off / off_trigger / on_notrigger / on_trigger
    w, h = thumbs[0][1][0].size
    cols = 4
    rows = len(thumbs)
    contact = Image.new("RGB", (w * cols, h * rows), (32, 32, 32))
    col_labels = ["off", "off_trigger", "on_notrigger", "on_trigger"]
    for r, (label, imgs) in enumerate(thumbs):
        for c, img in enumerate(imgs):
            tile = img.copy()
            draw = ImageDraw.Draw(tile)
            draw.rectangle([0, 0, 220, 24], fill=(0, 0, 0))
            draw.text((4, 4), f"{label}/{col_labels[c]}", fill=(255, 255, 255))
            contact.paste(tile, (c * w, r * h))
    contact_path = out_dir / "contact_sheet.png"
    contact.save(contact_path)

    # Aggregate over out-of-domain prompts only (the real gating measurement -- an
    # in-domain prompt is reported for context but excluded from the headline
    # number, since it can leak even under a perfectly gated LoRA). Report both the
    # naive ratio and the null-corrected ratio side by side so the confound is visible
    # rather than silently baked into one number.
    ood = [r for r in results if not r["in_domain"]]
    ood_ratio = [r["ratio_d_gate_over_d_leak"] for r in ood if r["ratio_d_gate_over_d_leak"] is not None]
    ood_ratio_net = [r["ratio_net_d_gate_over_d_leak"] for r in ood if r["ratio_net_d_gate_over_d_leak"] is not None]
    agg_ratio = sum(ood_ratio) / len(ood_ratio) if ood_ratio else None
    agg_ratio_net = sum(ood_ratio_net) / len(ood_ratio_net) if ood_ratio_net else None
    ood_below_floor = sum(1 for r in ood if r["leak_below_floor"])

    summary = {
        "script_version": SCRIPT_VERSION,
        "sampler": args.sampler,
        "seeds": seed_list,
        "lora": args.lora,
        "weight": args.weight,
        "seed": args.seed,
        "prompts_file": args.prompts_file,
        "results": results,
        "mean_ratio_out_of_domain_naive": round(agg_ratio, 4) if agg_ratio is not None else None,
        "mean_ratio_out_of_domain_net_of_null": round(agg_ratio_net, 4) if agg_ratio_net is not None else None,
        "out_of_domain_prompts_below_leak_floor": ood_below_floor,
    }
    summary_path = out_dir / "gating_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 70)
    print(f"Saved contact sheet: {contact_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Sampler: {args.sampler}  Seeds: {seed_list}")
    print(f"Mean d_gate/d_leak, naive:        {summary['mean_ratio_out_of_domain_naive']}")
    print(f"Mean d_gate/d_leak, net of null:  {summary['mean_ratio_out_of_domain_net_of_null']}")
    print(f"Out-of-domain prompts with d_leak below its own d_null floor: {ood_below_floor}/{len(ood)}")
    print("  net-of-null ~0   -> trigger is inert once token-insertion noise is subtracted")
    print("  net-of-null >>1  -> trigger is doing real gating work beyond token-shift noise")
    print("  LEAK-BELOW-FLOOR prompts: the ratio for that row divides by noise -- don't trust it")
    print("Column key: each d_* is a mean |RGB delta| per pixel/channel, 0-255 scale, between")
    print("two renders -- off=LoRA scale 0.0, on=LoRA scale --weight -- averaged over --seeds seed(s).")
    print("  d_leak  on/no-trig vs off/no-trig  -- style present with the trigger absent")
    print("  d_gate  on/trig    vs on/no-trig   -- naive trigger effect (LoRA loaded)")
    print("  d_null  off/trig   vs off/no-trig  -- token-insertion noise floor, LoRA off")
    print("  d_gate_net = max(0, d_gate - d_null); ratio_net = d_gate_net / d_leak")
    print("  leak_over_null = d_leak / d_null  -- sanity check only, NOT a correction to d_leak")
    print("  leak_sat / gate_sat  -- same leak/gate split, read off HSV saturation instead of RGB")
    print("  sat(off/offtrig/notrig/trig)  mean HSV saturation of each render, 0-255")
    print("The two means above average OUT-OF-DOMAIN prompts only -- in-domain rows print")
    print("above but don't feed mean_ratio_*, since style can leak into an in-domain")
    print("prompt even under a perfectly gated LoRA.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
