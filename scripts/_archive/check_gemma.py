"""Smoke test for an open VLM via the HFVLM backend (e.g. Gemma 4 12B).

Prints environment versions, then -- if a model path/id is given -- loads it
(4-bit by default) and checks that the slice confidence (Eq. 1) is higher for a
query matching the region than for one that doesn't. Builds its own red/blue
test image, so no dataset is needed.

    python scripts/check_gemma.py                       # env check only (no GPU)
    python scripts/check_gemma.py /models/gemma         # full test (4-bit, GPU)
    python scripts/check_gemma.py /models/gemma --bf16  # full precision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="HFVLM/Gemma smoke test.")
    ap.add_argument("model", nargs="?", default=None,
                    help="HF model id or local path; omit for an env-only check")
    ap.add_argument("--grounding", default="bbox", choices=["bbox", "point"])
    ap.add_argument("--bf16", action="store_true", help="load bf16 instead of 4-bit")
    args = ap.parse_args()

    import torch
    import transformers
    print("torch", torch.__version__, "| cuda build", torch.version.cuda,
          "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    try:
        import bitsandbytes as bnb
        print("bitsandbytes", bnb.__version__)
    except Exception as exc:  # noqa: BLE001
        print("bitsandbytes: NOT available:", exc)
    print("transformers", transformers.__version__)
    from transformers import AutoModelForImageTextToText  # noqa: F401
    from slicelens.backends import HFVLM
    print("HFVLM import: OK")

    if not args.model:
        print("\n(no model given -> environment check only)")
        return

    from PIL import Image

    from slicelens.data_types import Region

    w, h = 448, 224
    img = Image.new("RGB", (w, h), (255, 0, 0))
    for x in range(w // 2, w):
        for y in range(h):
            img.putpixel((x, y), (0, 0, 255))
    path = "/tmp/gemma_check.png"
    img.save(path)

    print(f"\nLoading {args.model} ({'bf16' if args.bf16 else '4-bit'}) ...")
    vlm = HFVLM(model=args.model, grounding=args.grounding, load_in_4bit=not args.bf16)
    print("yes-token ids:", vlm._yes_ids, "| no-token ids:", vlm._no_ids)

    red = Region(bbox=(0, 0, w // 2, h))
    print("\ncaption(red region):", vlm.caption_region(path, red, "color")[:160])
    cr = vlm.slice_confidence(path, red, "a red region")
    cb = vlm.slice_confidence(path, red, "a blue region")
    print(f"\nP(yes | 'a red region',  red bbox) = {cr:.3f}")
    print(f"P(yes | 'a blue region', red bbox) = {cb:.3f}")
    if cr > cb:
        print("\n[PASS] slice confidence discriminates the matching query -- HFVLM works.")
    else:
        print("\n[WARN] matching query did not score higher -- check grounding/prompt.")


if __name__ == "__main__":
    main()
