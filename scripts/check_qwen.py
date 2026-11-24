"""Smoke test for the Qwen2.5-VL backend on the GPU server.

Verifies the part that matters most for SliceLens: that the model loads, that
the "yes"/"no" output-token ids resolve, and that the slice confidence (Eq. 1)
is *higher* for a query that matches the region than for one that does not.

It builds its own test image (a red half and a blue half), so no dataset is
needed. Run after `pip install -e ".[qwen]"`::

    python scripts/check_qwen.py
    python scripts/check_qwen.py --model Qwen/Qwen2.5-VL-7B-Instruct --grounding bbox
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens.data_types import Region  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL backend smoke test.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--grounding", choices=["bbox", "point"], default="bbox")
    args = parser.parse_args()

    # 1) Environment.
    try:
        import torch
    except ImportError:
        print("[FAIL] torch not installed. Run: pip install -e \".[qwen]\"")
        sys.exit(1)
    cuda = torch.cuda.is_available()
    print(f"[env ] torch {torch.__version__} | CUDA available: {cuda}")
    if cuda:
        print(f"[env ] device: {torch.cuda.get_device_name(0)}")
    else:
        print("[warn] no CUDA GPU detected -- the 7B model will be very slow on CPU.")

    # 2) Build a test image: left half red, right half blue.
    try:
        from PIL import Image
    except ImportError:
        print("[FAIL] pillow not installed. Run: pip install -e \".[qwen]\"")
        sys.exit(1)
    w, h = 448, 224
    img = Image.new("RGB", (w, h), (255, 0, 0))
    for x in range(w // 2, w):
        for y in range(h):
            img.putpixel((x, y), (0, 0, 255))
    tmp = Path(tempfile.gettempdir()) / "slicelens_qwen_check.png"
    img.save(tmp)
    print(f"[img ] wrote test image -> {tmp}")

    # 3) Load the backend (downloads the model on first run).
    print(f"[load] loading {args.model} ... (first run downloads weights)")
    from slicelens.backends import QwenVLM

    vlm = QwenVLM(model=args.model, grounding=args.grounding)
    print(f"[load] OK | yes-token ids: {vlm._yes_ids} | no-token ids: {vlm._no_ids}")

    red_region = Region(bbox=(0, 0, w // 2, h))  # the red half

    # 4) Grounded caption (Figure 9).
    caption = vlm.caption_region(str(tmp), red_region, attribute="color")
    print(f"\n[cap ] caption of red region: {caption!r}")

    # 5) Slice confidence (Eq. 1): matching vs non-matching query.
    conf_red = vlm.slice_confidence(str(tmp), red_region, "a red region")
    conf_blue = vlm.slice_confidence(str(tmp), red_region, "a blue region")
    print(f"\n[conf] P(yes | 'a red region',  red bbox) = {conf_red:0.3f}")
    print(f"[conf] P(yes | 'a blue region', red bbox) = {conf_blue:0.3f}")

    if conf_red > conf_blue:
        print("\n[PASS] slice confidence discriminates the matching query. "
              "Qwen backend is working.")
    else:
        print("\n[WARN] matching query did not score higher -- check the model / "
              "grounding mode / prompt.")


if __name__ == "__main__":
    main()
