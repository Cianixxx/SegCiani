"""Relaxed cluster gates on copies of the VOC run: what each variant certifies, and whether it holds. CPU only.

For the official run and each variant directory (rescored with SQ_PURITY_ALPHA / SQ_PURITY_OFF / SQ_COH_MIN,
see slice_quality/config.py): the certified cluster slices -- distinct member sets, so a group re-found in a
later peeling round counts once -- with their description, size, error type, the IoU of their class on their
images against the class's other images (slice_image_iou.json) and the split-half replication run under the
same gates (split_half.json); plus the run's coverage.
Run (CPU):  RUNS="official=outputs/slice_quality_inst A=outputs/slice_quality_inst_relaxA ..." \
            python slice_quality/examples/relaxed_report.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "docs")
from _revision_numbers import load_inst_summary  # noqa: E402

RUNS = [r.split("=", 1) for r in os.environ.get(
    "RUNS", "official=outputs/slice_quality_inst A=outputs/slice_quality_inst_relaxA "
            "B=outputs/slice_quality_inst_relaxB").split()]
OUT = os.environ.get("OUT", "outputs/relaxed_cluster_report")
J = lambda p: json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
lines, out = [], {}
say = lambda s="": (lines.append(s), print(s, flush=True))
say("=" * 110)
say("CLUSTER GATES RELAXED ON COPIES OF THE VOC RUN")
say("=" * 110)
for name, d in RUNS:
    idx = J(f"{d}/slices_index.json")
    if not idx:
        say(f"\n{name}: {d} not found")
        continue
    cert = [s for s in idx["slices"] if s.get("in_descriptions") and s["slice_id"].startswith("cluster")]
    seen, distinct = set(), []
    for s in cert:
        key = frozenset(m for ids in s["images"].values() for m in ids)
        if key not in seen:
            seen.add(key)
            distinct.append(s)
    phr = defaultdict(Counter)
    for img, v in (J(f"{d}/image_descriptions.json") or {}).items():
        for p in v.get("patterns", []):
            if p.get("branch") == "cluster":
                phr[p["slice_id"]][p["description"]] += 1
    iou = {r["slice_id"]: r for r in (J(f"{d}/slice_image_iou.json") or {}).get("rows", [])}
    sh = J(f"{d}/split_half.json")
    summ = load_inst_summary(d)
    cov = lambda k: 100.0 * summ[k] / summ["n_err"]
    say(f"\n{name} ({d}): certified cluster slices {len(cert)} ({len(distinct)} distinct), language "
        f"{summ['slices']['language']} | errors in a certified group {cov('cov_union'):.1f}% (cluster {cov('cov_cluster'):.1f}%)"
        + (f" | split-half: {sh['n_replicated']} of {sh['n_discovered']} replicate" if sh else " | split-half: not run"))
    rows = []
    for s in sorted(distinct, key=lambda s: -s["size"]):
        r = iou.get(s["slice_id"], {})
        desc = phr[s["slice_id"]].most_common(1)[0][0] if phr[s["slice_id"]] else "-"
        drop = (100 * (r["cls_iou_out"] - r["cls_iou_in"])) if r.get("cls_iou_in") is not None and r.get("cls_iou_out") is not None else None
        rows.append(dict(slice_id=s["slice_id"], size=s["size"], n_images=s["n_images"], type=s["dominant_error_type"],
                         description=desc, cls_iou_in=r.get("cls_iou_in"), cls_iou_out=r.get("cls_iou_out"), drop=drop))
        say(f"  {s['slice_id']:24s} {s['size']:4d} err {s['n_images']:4d} img {s['dominant_error_type']:9s} | "
            + (f"class IoU {100 * r['cls_iou_in']:5.1f} vs {100 * r['cls_iou_out']:5.1f} ({drop:+5.1f}) | " if drop is not None else "class IoU      -      | ")
            + desc[:70])
    worse = [x for x in rows if x["drop"] is not None and x["drop"] > 0]
    if rows:
        say(f"  class worse on the slice's images: {len(worse)} of {sum(1 for x in rows if x['drop'] is not None)}"
            f"; mean drop {sum(x['drop'] for x in rows if x['drop'] is not None) / max(1, sum(1 for x in rows if x['drop'] is not None)):.1f} points")
    out[name] = dict(dir=d, certified=len(cert), distinct=len(distinct), cov_union=cov("cov_union"),
                     cov_cluster=cov("cov_cluster"), split_half=sh and dict(n_discovered=sh["n_discovered"],
                                                                            n_replicated=sh["n_replicated"]), rows=rows)
json.dump(out, open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("RELAXED_REPORT_DONE", flush=True)
