"""Describe the 36 hand-annotated TRAIN images with the ALREADY-CERTIFIED slices.

Why this is inference, not discovery: 36 images cannot certify anything (N_min is
2% of the dataset), and re-running discovery on them would invent slices no
statistical test could support. Instead the 14 cluster slices certified on `val`
are INDUCED onto the new instances by nearest-centroid assignment within each
slice's own radius -- the same mechanism validate_split.py uses to test
replication on held-out images, so this is an established path in this codebase,
not one invented for this run.

Caveat to keep attached to any number this produces: these images are in VOC
`train`, which the segmenter was fitted on, so its errors here are NOT
representative of deployment behaviour. The comparison this enables --
"does the pipeline DESCRIBE an image the way a human did" -- stays meaningful;
"how often does the model fail" does not.

Phase 1 (this script, CPU only): Block 1 + CPU DINOv2 + cluster induction.
Attributes are NOT re-elicited here, so cluster rows fall back to the honest
"grouped by appearance only" wording. Phase 2 (GPU) would add Qwen verify-mode
tagging and grounding of the 35 language hypotheses.

Run (CPU):
  PYTHONPATH=/workspace python slice_quality/examples/describe_train36.py
Env: NAMES_FILE (default /tmp/train36.txt), OUT (default outputs/slice_quality/train36)
"""
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.config import SliceQualityConfig
from slice_quality.describe import describe_instances
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.features import build_features
from slice_quality.typed_error import decompose_dataset

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/data/voc2012/voc_crops"
PKL = "outputs/slice_quality/result_tagged.pkl"
NAMES_FILE = os.environ.get("NAMES_FILE", "/tmp/train36.txt")
OUT = os.environ.get("OUT", "outputs/slice_quality/train36")

names = [l.strip() for l in open(NAMES_FILE, encoding="utf-8") if l.strip()]
print(f"images to describe: {len(names)}", flush=True)

# ---- Block 1 on the new images -------------------------------------------
samples, image_paths = build_voc_samples(DATA, CROPS, names, pred_subdir="mobilenet_pred")
errors, reference = decompose_dataset(samples)
print(f"typed errors: {len(errors)} | correct references: {len(reference)}", flush=True)
print("taxonomy:", dict(Counter(e.seg_subtype.value for e in errors).most_common()),
      flush=True)


def make_mask_lookup(samples):
    by_img = {s["image_id"]: s for s in samples}

    def lookup(inst):
        s = by_img.get(inst.image_id)
        if s is None:
            return None
        pool = s["gts"] if inst.metadata.get("from_gt_mask") else s["preds"]
        idx = int(inst.instance_id.rsplit("#", 1)[1][1:])
        return pool[idx]["mask"] if idx < len(pool) else None

    return lookup


# ---- Block 2 on CPU (few images, no GPU needed) ---------------------------
print("computing DINOv2 embeddings on CPU ...", flush=True)
attach_embeddings(errors, DINOv2Extractor(image_paths, device="cpu"),
                  make_mask_lookup(samples))
n_emb = sum(1 for e in errors if e.embedding is not None)
print(f"embeddings: {n_emb}/{len(errors)}", flush=True)

# ---- load the certified slices -------------------------------------------
data = pickle.load(open(PKL, "rb"))
kept = data["result"].kept
print(f"certified cluster slices to induce: {len(kept)}", flush=True)

# Build features for certified members AND new instances in ONE call so the
# representation is identical by construction (l2 is per-row, but this removes
# any doubt about config drift between the two runs).
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
print(f"feature space: {feats.shape}", flush=True)

# ---- induce each certified slice onto the new instances -------------------
slice_of = {}
for k in kept:
    sid = k.slice_.slice_id
    X = np.stack([fid[m.instance_id] for m in k.slice_.members if m.instance_id in fid])
    c = X.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(X - c, axis=1), 90))
    slice_of[sid] = (k, c, radius, k.quality.dominant_error_type)

induced = defaultdict(list)
for inst in new_members:
    best, best_d = None, None
    for sid, (k, c, radius, subtype) in slice_of.items():
        if inst.seg_subtype != subtype:        # same-subtype only, as in split-half
            continue
        d = float(np.linalg.norm(fid[inst.instance_id] - c))
        if d <= radius and (best_d is None or d < best_d):
            best, best_d = sid, d
    if best is not None:
        induced[best].append(inst)

n_assigned = sum(len(v) for v in induced.values())
print(f"instances induced into a certified slice: {n_assigned}/{len(new_members)} "
      f"({len(induced)} distinct slices)", flush=True)

# ---- per-image descriptions ----------------------------------------------
by_img = defaultdict(lambda: defaultdict(list))
for sid, insts in induced.items():
    for i in insts:
        by_img[i.image_id][sid].append(i)

all_by_img = defaultdict(set)
for e in errors:
    all_by_img[e.image_id].add(e.instance_id)

out = {}
for name in names:
    n_total = len(all_by_img.get(name, ()))
    pats, sys_ids = [], set()
    for sid, members in by_img.get(name, {}).items():
        k = slice_of[sid][0]
        pats.append({"slice_id": sid, "branch": "cluster",
                     "error_type": k.quality.dominant_error_type.value,
                     "description": describe_instances(members, k.slice_, k.quality),
                     "n_instances": len(members)})
        sys_ids.update(m.instance_id for m in members)
    n_sys = len(sys_ids)
    n_iso = n_total - n_sys
    obj = "object" if n_total == 1 else "objects"
    if n_sys == 0:
        lines = [f"Image {name}: {n_total} wrong {obj}, "
                 f"no systematic pattern (isolated errors)."]
    else:
        head = f"Image {name}: {n_total} wrong {obj}, {n_sys} in systematic patterns"
        head += f" and {n_iso} isolated." if n_iso else "."
        lines = [head, "Systematic patterns:"]
        for p in sorted(pats, key=lambda x: -x["n_instances"]):
            n = p["n_instances"]
            plural = "" if n == 1 else "s"
            lines.append(f"  - {p['description']} "
                         f"[{p['slice_id']}, {n} obj{plural}.]")
    out[name] = {"n_errors": n_total, "n_systematic": n_sys, "n_isolated": n_iso,
                 "patterns": pats, "text": "\n".join(lines)}

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/descriptions_train36.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
with open(f"{OUT}/descriptions_train36.txt", "w", encoding="utf-8") as f:
    for name in names:
        f.write(out[name]["text"] + "\n\n")

described = sum(1 for v in out.values() if v["n_systematic"])
print(f"\nimages with >=1 certified pattern: {described}/{len(names)}", flush=True)
print(f"saved to {OUT}/", flush=True)
print("TRAIN36_DONE", flush=True)
