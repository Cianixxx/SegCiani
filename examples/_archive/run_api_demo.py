"""Real Stage-2 demo via the Gemini API -- rank your own images by slice confidence.

The heart of SliceLens with the real VLM, runnable on a laptop with **only a
Gemini key** (no GPU, no detector, no COCO). Point it at a folder of images and
a hypothesis; for each image it computes the slice confidence P(yes | query, r)
(Equation 1, from Gemini logprobs) over the whole image as the region, ranks
them, and shows a grounded caption for the top match.

    pip install google-genai
    export GOOGLE_API_KEY=...        # PowerShell: $env:GOOGLE_API_KEY="..."
    python examples/run_api_demo.py --images-dir my_photos \\
        --query "a person riding a bicycle"

Drop any JPG/PNG photos in the folder. Pure standard library + google-genai
(an image-size sniffer avoids needing Pillow on Python 3.14).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens.data_types import Region  # noqa: E402

_EXTS = {".jpg", ".jpeg", ".png"}


def _png_size(f) -> tuple[int, int] | None:
    f.seek(0)
    if f.read(8) != b"\x89PNG\r\n\x1a\n":
        return None
    f.read(4)  # IHDR length
    if f.read(4) != b"IHDR":
        return None
    w = int.from_bytes(f.read(4), "big")
    h = int.from_bytes(f.read(4), "big")
    return w, h


def _jpeg_size(f) -> tuple[int, int] | None:
    f.seek(0)
    if f.read(2) != b"\xff\xd8":
        return None
    while True:
        b = f.read(1)
        if not b:
            return None
        if b != b"\xff":
            continue
        marker = f.read(1)
        while marker == b"\xff":
            marker = f.read(1)
        if not marker:
            return None
        m = marker[0]
        if 0xD0 <= m <= 0xD9 or m == 0x01:  # standalone markers, no length
            continue
        seg = f.read(2)
        if len(seg) < 2:
            return None
        seg_len = int.from_bytes(seg, "big")
        if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):  # SOF
            f.read(1)  # precision
            h = int.from_bytes(f.read(2), "big")
            w = int.from_bytes(f.read(2), "big")
            return w, h
        f.seek(seg_len - 2, 1)


def image_size(path: str) -> tuple[int, int]:
    """(width, height) for PNG/JPEG via a tiny sniffer; nominal box on failure."""
    try:
        with open(path, "rb") as f:
            return _png_size(f) or (f.seek(0) or _jpeg_size(f)) or (1024, 768)
    except Exception:  # noqa: BLE001
        return (1024, 768)


def main() -> None:
    p = argparse.ArgumentParser(description="Rank images by SliceLens slice confidence (API).")
    p.add_argument("--images-dir", required=True, help="folder with your JPG/PNG images")
    p.add_argument("--query", required=True, help="hypothesis, e.g. 'a bicycle obscured by a person'")
    p.add_argument("--attribute", default=None, help="attribute for the caption (default: the query)")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--limit", type=int, default=0, help="cap number of images (0 = all)")
    args = p.parse_args()

    folder = Path(args.images_dir)
    if not folder.is_dir():
        p.error(f"--images-dir '{folder}' is not a folder")
    paths = sorted(str(q) for q in folder.iterdir() if q.suffix.lower() in _EXTS)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        p.error(f"No JPG/PNG images found in {folder}")

    from slicelens.backends import GeminiVLM

    vlm = GeminiVLM(model=args.model, grounding="bbox")

    print(f"Scoring {len(paths)} images for: {args.query!r}\n")
    scored: list[tuple[float, str]] = []
    for path in paths:
        w, h = image_size(path)
        region = Region(bbox=(0, 0, w, h))
        try:
            conf = vlm.slice_confidence(path, region, args.query)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {Path(path).name}: {exc}")
            continue
        scored.append((conf, path))
        print(f"  {conf:0.3f}  {Path(path).name}")

    if not scored:
        return
    scored.sort(reverse=True)
    print(f"\nTop match: {Path(scored[0][1]).name}  (P={scored[0][0]:0.3f})")

    top_w, top_h = image_size(scored[0][1])
    caption = vlm.caption_region(
        scored[0][1], Region(bbox=(0, 0, top_w, top_h)), args.attribute or args.query
    )
    print(f"Grounded caption: {caption}")
    print("\nThis is the real Eq. 1 slice confidence (Gemini logprobs) on real images.")
    print("The full pipeline adds detector errors + slope analysis (see RUN.md).")


if __name__ == "__main__":
    main()
