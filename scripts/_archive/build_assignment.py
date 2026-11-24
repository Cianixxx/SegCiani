"""Build a per-image -> slice assignment from a run_voc.py report JSON.

Inverts each slice's top-k instances into a per-image view: for every image
(and class) it lists which discovered slice(s) it belongs to and with what
slice confidence. Writes a long-format CSV and a per-image nested JSON.
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="run_voc.py output JSON")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--systematic-only", action="store_true",
                    help="keep only slices flagged systematic by the slope analysis")
    a = ap.parse_args()

    d = json.loads(Path(a.report).read_text(encoding="utf-8"))
    sysset = set(d.get("systematic_error_slices", []))

    rows = []
    per_img: dict = {}
    for c in d["candidates"]:
        issys = c["query"] in sysset
        if a.systematic_only and not issys:
            continue
        for t in c["top_instances"]:
            rows.append({
                "image": t["name"], "label": t["label"], "iou": t["iou"],
                "is_error": int(t["is_error"]), "slice": c["query"],
                "slice_confidence": round(t["slice_confidence"], 4),
                "systematic": int(issys),
            })
            inst = per_img.setdefault(t["name"], {}).setdefault(
                t["label"], {"iou": t["iou"], "is_error": bool(t["is_error"]),
                             "slices": {}})
            inst["slices"][c["query"]] = round(t["slice_confidence"], 4)

    rows.sort(key=lambda r: (r["image"], r["label"], -r["slice_confidence"]))
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image", "label", "iou", "is_error", "slice", "slice_confidence",
            "systematic"])
        w.writeheader()
        w.writerows(rows)
    Path(a.out_json).write_text(
        json.dumps(per_img, indent=2, ensure_ascii=False), encoding="utf-8")

    n_err = sum(1 for v in per_img.values() for i in v.values() if i["is_error"])
    print(f"{len(rows)} (image,slice) rows | {len(per_img)} images | "
          f"{sum(len(v) for v in per_img.values())} instances "
          f"({n_err} errors)")
    print(f"  CSV  -> {a.out_csv}")
    print(f"  JSON -> {a.out_json}")


if __name__ == "__main__":
    main()
