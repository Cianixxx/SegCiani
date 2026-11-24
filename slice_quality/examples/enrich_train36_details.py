"""Add the geometric detail line to the 36 TRAIN images' descriptions. CPU only.

describe_train36_gpu.py built its pattern rows before describe.py grew
``_detail_clause``, so the 43 `val` cards in the human-comparison page carry the
extra line ("... 3 internal holes ...", "bottom-left") and the 36 `train` cards
do not. This closes that gap WITHOUT another Qwen pass.

Member recovery, and why it is safe to redo it here:
  * cluster rows -- the induction is nearest-centroid within each certified
    slice's 90th-percentile radius: deterministic, and the embeddings are already
    in instances_train36.pkl, so re-running it on CPU reproduces the same sets.
  * language rows -- membership came from Qwen confidences that were never
    cached. They are recovered as "the errors of that class in that image", which
    is only correct when ALL candidates passed the 0.5 threshold.

Both recoveries are CHECKED against the n_instances stored in the existing
descriptions. A row whose recovered set does not match its stored count is left
exactly as it was, without a detail line, and reported -- a wrong detail line is
worse than a missing one.

The existing wording is never recomputed: the pattern dicts are kept as they are
and only gain a "details" key, so this cannot silently reword a description.

Run:  PYTHONPATH=/workspace python slice_quality/examples/enrich_train36_details.py
Env: OUT, PKL, MIN_HOLE_PX (default 20, as in add_geometry_metadata.py).
"""
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import ndimage

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.config import SliceQualityConfig
from slice_quality.describe import _detail_clause
from slice_quality.features import build_features

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/data/voc2012/voc_crops"
OUT = os.environ.get("OUT", "outputs/slice_quality/train36")
PKL = os.environ.get("PKL", "outputs/slice_quality/result_tagged.pkl")
MIN_HOLE_PX = int(os.environ.get("MIN_HOLE_PX", "20"))

cache = pickle.load(open(f"{OUT}/instances_train36.pkl", "rb"))
errors = cache["errors"]
descs = json.load(open(f"{OUT}/descriptions_train36_full.json", encoding="utf-8"))
names = list(descs)
print(f"images: {len(names)} | cached typed errors: {len(errors)}", flush=True)

# ---- 1. holes in the predicted masks (same recipe as add_geometry_metadata) --
samples, _paths = build_voc_samples(DATA, CROPS, names, pred_subdir="mobilenet_pred")
holes = {}
for s in samples:
    for i, p in enumerate(s["preds"]):
        m = np.asarray(p["mask"], dtype=bool)
        area = int(m.sum())
        if area == 0:
            continue
        gaps = ndimage.binary_fill_holes(m) & ~m
        px = int(gaps.sum())
        if px < MIN_HOLE_PX:
            continue
        _lab, n = ndimage.label(gaps)
        holes[f"{s['image_id']}#p{i}"] = {"n_holes": int(n), "hole_px": px,
                                          "hole_frac": round(px / area, 4)}
hit = 0
for e in errors:
    h = holes.get(e.instance_id)
    if h:
        e.metadata.update(h)
        hit += 1
print(f"predicted instances with holes >= {MIN_HOLE_PX}px: {len(holes)} "
      f"| erroneous ones updated: {hit}", flush=True)

# ---- 2. recover cluster membership: re-run the induction on CPU -------------
data = pickle.load(open(PKL, "rb"))
kept = data["result"].kept
cfg = SliceQualityConfig()
old_members, seen = [], set()
for k in kept:
    for m in k.slice_.members:
        if m.instance_id not in seen and m.embedding is not None:
            seen.add(m.instance_id)
            old_members.append(m)
new_members = [e for e in errors if e.embedding is not None]
feats, kept_insts = build_features(old_members + new_members, cfg.feature)
fid = {i.instance_id: f for i, f in zip(kept_insts, feats)}

slice_of = {}
for k in kept:
    X = np.stack([fid[m.instance_id] for m in k.slice_.members if m.instance_id in fid])
    c = X.mean(axis=0)
    slice_of[k.slice_.slice_id] = (
        c, float(np.percentile(np.linalg.norm(X - c, axis=1), 90)),
        k.quality.dominant_error_type)

induced = defaultdict(list)
for inst in new_members:
    best, best_d = None, None
    for sid, (c, radius, subtype) in slice_of.items():
        if inst.seg_subtype != subtype:
            continue
        d = float(np.linalg.norm(fid[inst.instance_id] - c))
        if d <= radius and (best_d is None or d < best_d):
            best, best_d = sid, d
    if best is not None:
        induced[best].append(inst)

cluster_members = defaultdict(list)          # (image, slice_id) -> members
for sid, insts in induced.items():
    for i in insts:
        cluster_members[(i.image_id, sid)].append(i)
print(f"cluster induction reproduced: {sum(len(v) for v in induced.values())} "
      f"instances in {len(induced)} slices", flush=True)

# ---- 3. recover language membership: same-class errors in that image --------
by_img_class = defaultdict(list)
for e in errors:
    cls = e.gt_class or e.pred_class
    if cls:
        by_img_class[(e.image_id, cls)].append(e)

# ---- 4. attach the detail line where the recovery is provably right ---------
added, skipped = 0, []
for name, rec in descs.items():
    for p in rec["patterns"]:
        sid, n = p["slice_id"], p["n_instances"]
        if p["branch"] == "cluster":
            members = cluster_members.get((name, sid), [])
        else:
            members = by_img_class.get((name, re.sub(r"_\d+$", "", sid)), [])
        if len(members) != n:                # recovery not certain -> leave alone
            skipped.append((name, sid, p["branch"], n, len(members)))
            continue
        det = _detail_clause(members)
        if det:
            p["details"] = det
            added += 1

    # re-render the text exactly as describe._render does, detail lines included
    n_total, n_sys, n_iso = rec["n_errors"], rec["n_systematic"], rec["n_isolated"]
    obj = "object" if n_total == 1 else "objects"
    if n_sys == 0:
        lines = [f"Image {name}: {n_total} wrong {obj}, "
                 f"no systematic pattern (isolated errors)."]
    else:
        lines = [f"Image {name}: {n_total} wrong {obj}, {n_sys} in systematic patterns"
                 + (f" and {n_iso} isolated." if n_iso else "."),
                 "Systematic patterns:"]
        for p in sorted(rec["patterns"], key=lambda x: -x["n_instances"]):
            n = p["n_instances"]
            lines.append(f"  - {p['description']} "
                         f"[{p['slice_id']}, {n} obj{'' if n == 1 else 's'}.]")
            if p.get("details"):
                lines.append(f"      · {p['details']}")
    rec["text"] = "\n".join(lines)

rows = sum(len(v["patterns"]) for v in descs.values())
print(f"\ndetail lines added: {added}/{rows} pattern rows", flush=True)
if skipped:
    print(f"rows left without a detail line ({len(skipped)}), recovered count "
          f"!= stored count:", flush=True)
    for name, sid, br, want, got in skipped:
        print(f"  {name} {sid} ({br}): stored {want}, recovered {got}", flush=True)
    print("  (membership uncertain -- no line is better than a wrong line)", flush=True)

with open(f"{OUT}/descriptions_train36_full.json", "w", encoding="utf-8") as f:
    json.dump(descs, f, indent=2, ensure_ascii=False)
with open(f"{OUT}/descriptions_train36_full.txt", "w", encoding="utf-8") as f:
    for name in names:
        f.write(descs[name]["text"] + "\n\n")
with open(f"{OUT}/instances_train36.pkl", "wb") as f:
    pickle.dump(cache, f)                    # keep the hole metadata for later
print(f"saved to {OUT}/", flush=True)
print("ENRICH_TRAIN36_DONE", flush=True)
