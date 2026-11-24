"""Download a small, ready-made image set for the Stage-2 API demo.

Pulls openly-licensed thumbnails from Wikimedia Commons for a few search terms,
so you can run ``examples/run_api_demo.py`` without hunting for images. The
default terms make the "bicycle obscured by a person" demo meaningful: clear
positives, near-misses (bicycles alone), and distractors.

Pure standard library (no API key, no extra deps)::

    python scripts/fetch_demo_images.py --out-dir demo_images
    python examples/run_api_demo.py --images-dir demo_images \\
        --query "a bicycle obscured by a person"
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_UA = "SliceLens-demo/0.1 (thesis project; contact via local use)"
_EXT = {"image/jpeg": ".jpg", "image/png": ".png"}

# Grouped to exercise the occlusion query: positives + near-miss + distractors.
DEFAULT_TERMS = [
    "bicycle person riding",
    "parked bicycle street",
    "motorcycle road",
    "city street pedestrians",
]


def search_images(term: str, limit: int, width: int) -> list[tuple[str, str]]:
    """Return [(thumb_url, mime), ...] for a Commons search term."""
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": term, "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": width,
    }
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    pages = (data.get("query") or {}).get("pages") or {}
    out: list[tuple[str, str]] = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        mime = info.get("mime", "")
        thumb = info.get("thumburl") or info.get("url")
        if thumb and mime in _EXT:
            out.append((thumb, mime))
    return out


def _download(url: str, dest: str, delay: float, retries: int = 3) -> int:
    """Download url -> dest, politely (delay + backoff on HTTP 429). Returns bytes."""
    for attempt in range(retries):
        time.sleep(delay * (attempt + 1))  # be gentle; back off on retry
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
            return len(data)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                continue  # rate-limited: wait longer and retry
            raise
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch demo images from Wikimedia Commons.")
    p.add_argument("--out-dir", default="demo_images")
    p.add_argument("--terms", nargs="*", default=DEFAULT_TERMS,
                   help="search terms (default: a bicycle-occlusion demo set)")
    p.add_argument("--per-term", type=int, default=2, help="images per term")
    p.add_argument("--width", type=int, default=1024, help="thumbnail width (px)")
    p.add_argument("--delay", type=float, default=1.5,
                   help="seconds between requests (avoid rate limiting)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for term in args.terms:
        try:
            time.sleep(args.delay)
            hits = search_images(term, args.per_term, args.width)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] search failed for {term!r}: {exc}")
            continue
        for thumb_url, mime in hits:
            ext = _EXT[mime]
            fname = f"{term.replace(' ', '_')}_{n}{ext}"
            dest = os.path.join(args.out_dir, fname)
            try:
                size = _download(thumb_url, dest, args.delay)
                print(f"  [{n}] {fname}  ({size // 1024} KB)  <- {term}")
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] download failed ({thumb_url}): {exc}")

    print(f"\nDownloaded {n} images to {args.out_dir}/")
    if n:
        print("Next:\n"
              f"  python examples/run_api_demo.py --images-dir {args.out_dir} "
              "--query \"a bicycle obscured by a person\" --model gemini-2.0-flash")


if __name__ == "__main__":
    main()
