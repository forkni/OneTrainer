"""Write a concepts file for the kit from the shipped template, pointing both concepts at
<root>/dataset/trigger and <root>/dataset/notrigger.

Called by 00_verify_setup.cmd once LORA_ROOT is known; also usable by hand:

    python course_lora_kit/scripts/init_concepts.py --root D:\\my_lora --track style
    python course_lora_kit/scripts/init_concepts.py --root D:\\my_lora --track character --force

Writes training_concepts/<track>_concepts.json (gitignored) with forward-slash paths, which
OneTrainer reads fine on Windows and which survive JSON without escaping. Refuses to overwrite
an existing file unless --force is given, so a hand-edited concepts file is never clobbered.
Edit the two "path" fields afterwards only if your dataset lives somewhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
REPO = KIT.parent
TEMPLATES = {
    "style": KIT / "configs" / "concepts_contrastive_template_style.json",
    "character": KIT / "configs" / "concepts_contrastive_template.json",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="your LoRA root (LORA_ROOT); dataset lives in <root>/dataset")
    ap.add_argument("--track", choices=sorted(TEMPLATES), required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing concepts file")
    ap.add_argument("--out", help="override the output path (default training_concepts/<track>_concepts.json)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out) if args.out else REPO / "training_concepts" / f"{args.track}_concepts.json"
    if out.exists() and not args.force:
        print(f"[init_concepts] kept existing {out} (pass --force to overwrite)")
        return 0

    concepts = json.loads(TEMPLATES[args.track].read_text(encoding="utf-8"))
    if len(concepts) != 2:
        sys.exit(f"[init_concepts] template {TEMPLATES[args.track].name} should hold two concepts, found {len(concepts)}")
    folders = {"STANDARD": "trigger", "PRIOR_PREDICTION": "notrigger"}
    for concept in concepts:
        sub = folders.get(concept.get("type"))
        if sub is None:
            sys.exit(f"[init_concepts] unexpected concept type {concept.get('type')!r} in the template")
        concept["path"] = (root / "dataset" / sub).as_posix()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(concepts, indent=4) + "\n", encoding="utf-8")
    print(f"[init_concepts] wrote {out}")
    for concept in concepts:
        flag = "" if Path(concept["path"]).is_dir() else "   (folder does not exist yet)"
        print(f"    {concept['type']:<16} {concept['path']}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
