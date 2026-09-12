"""
Stage a picked checkpoint for deployment -- verify, copy (never move), and prove the copy.

The last step of a training round is the one most often done by hand: a checkpoint
gets renamed and dragged into the component's loras folder, and from then on nothing
says which file it was. This script makes the hand-off a measurement:

  1. loads the checkpoint and recomputes the scaled ||dW||_F from the raw LoRA
     factors -- sum over modules of ||(alpha/rank) * up @ down||_F^2, square-rooted,
     the same quantity checkpoint_norm_analyzer.py reports -- and counts the modules;
  2. checks the rank and alpha every module carries against --rank / --alpha, and
     (with --expected-norm) that the recomputed norm is within --tol of the norm the
     screen reported for this checkpoint;
  3. copies the file to --dest-dir/<label>.safetensors with shutil.copy2 (the source
     stays where the trainer put it) and confirms SHA-256 of both sides match.

Exit code 1 on any failure, and nothing is copied unless the checks pass. The course's
Round 2 pick was staged this way: S0b step 1099, 722 modules, rank 16, alpha 1.0,
||dW||_F 8.109 -- the label in the loras folder and the row in the norm table are the
same file, and the hash proves it.

Usage:
    python stage_pick.py --lora <checkpoint.safetensors> --label my_style_s1099 --dest-dir <loras_folder>
    python stage_pick.py --lora <checkpoint.safetensors> --label my_style_s1099 --dest-dir <loras_folder> --expected-norm 8.109
"""

import argparse
import hashlib
import math
import shutil
from pathlib import Path

from safetensors.torch import load_file


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scaled_norm(sd: dict) -> tuple[float, int, set, set]:
    """Return (||dW||_F over all modules, module count, ranks seen, alphas seen)."""
    modules = sorted(
        {
            k.rsplit(".", 1)[0].removesuffix(".lora_down").removesuffix(".lora_up")
            for k in sd
            if ".lora_down.weight" in k or ".lora_up.weight" in k
        }
    )
    sq_sum = 0.0
    ranks: set = set()
    alphas: set = set()
    for m in modules:
        down = sd[f"{m}.lora_down.weight"].float()
        up = sd[f"{m}.lora_up.weight"].float()
        rank = down.shape[0]
        alpha = float(sd[f"{m}.alpha"]) if f"{m}.alpha" in sd else float(rank)
        ranks.add(rank)
        alphas.add(alpha)
        dw = up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1)
        sq_sum += ((alpha / rank) * dw).norm().item() ** 2
    return math.sqrt(sq_sum), len(modules), ranks, alphas


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and stage a picked LoRA checkpoint")
    parser.add_argument("--lora", required=True, help="Checkpoint to stage (the trainer's own file; it is not moved)")
    parser.add_argument("--label", required=True, help="Name for the staged copy, without extension")
    parser.add_argument("--dest-dir", required=True, help="Folder that receives <label>.safetensors")
    parser.add_argument("--expected-norm", type=float, default=None, help="||dW||_F the norm screen reported for this checkpoint")
    parser.add_argument("--tol", type=float, default=0.02, help="Relative tolerance on --expected-norm (default 0.02)")
    parser.add_argument("--rank", type=int, default=16, help="Rank every module must carry (default 16)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha every module must carry (default 1.0)")
    parser.add_argument("--expected-modules", type=int, default=None, help="Module count to insist on (e.g. 722 for SDXL attn-mlp)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing <label>.safetensors in --dest-dir")
    args = parser.parse_args()

    src = Path(args.lora)
    if not src.is_file():
        print(f"FAIL: not a file: {src}")
        return 1
    dest_dir = Path(args.dest_dir)
    dst = dest_dir / f"{args.label}.safetensors"

    sd = load_file(str(src))
    total, n_modules, ranks, alphas = scaled_norm(sd)
    problems = []
    if ranks != {args.rank}:
        problems.append(f"rank(s) {sorted(ranks)} != {args.rank}")
    if alphas != {args.alpha}:
        problems.append(f"alpha(s) {sorted(alphas)} != {args.alpha}")
    if args.expected_modules is not None and n_modules != args.expected_modules:
        problems.append(f"modules {n_modules} != {args.expected_modules}")
    rel = None
    if args.expected_norm is not None:
        rel = abs(total - args.expected_norm) / args.expected_norm
        if rel > args.tol:
            problems.append(f"norm {total:.3f} vs expected {args.expected_norm} (rel err {rel:.4f} > {args.tol})")
    if dst.exists() and not args.overwrite:
        problems.append(f"{dst} already exists (pass --overwrite to replace it)")

    rel_text = f" rel_err={rel:.4f}" if rel is not None else ""
    print(
        "Column key: modules=count of lora_down/lora_up pairs found; rank/alpha=the values every"
    )
    print(
        "module carries (a set with >1 member means the checkpoint is not uniform); norm=recomputed"
    )
    print(
        "||dW||_F; expected=--expected-norm; rel_err=|norm-expected|/expected, must be <= --tol."
    )
    print(
        f"{args.label}: modules={n_modules} rank={sorted(ranks)} alpha={sorted(alphas)} "
        f"norm={total:.3f}"
        + (f" expected={args.expected_norm}{rel_text}" if args.expected_norm is not None else "")
        + f" -> {'PASS' if not problems else 'FAIL'}"
    )
    for p in problems:
        print(f"  - {p}")
    if problems:
        print("nothing copied")
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    s_src, s_dst = sha256(src), sha256(dst)
    match = s_src == s_dst
    print(f"  staged -> {dst}")
    print(f"  sha256 src={s_src}")
    print(f"  sha256 dst={s_dst} -> {'MATCH' if match else 'MISMATCH'}")
    print(f"  source left in place: {src}")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
