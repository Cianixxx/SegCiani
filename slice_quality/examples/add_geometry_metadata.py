"""Add mask-derived geometry to cached instances: internal holes in the prediction.

The hand-written annotations describe a phenomenon the pipeline had no vocabulary
for at all -- "the prediction mask ... has some holes scattered within the
region, especially on the center" (2007_000123). Block 1 types that image's
errors as boundary/extent and says nothing about the holes, because nothing in
the instance record encodes them. A probe over 60 val images found holes in
about 15% of predicted masks, so this is common, not exotic.

What is written into ``inst.metadata`` (predicted-mask instances only -- an FN
has no predicted mask to put a hole in):
    n_holes      number of background components fully enclosed by the mask
    hole_px      their total area in pixels
    hole_frac    that area over the mask's own area

These are deterministic mask facts, on the same footing as the class or the
bbox: asserted in descriptions, never "verified" by a VLM.

Deliberately NOT added: fragmentation (mask split into several components).
Measured 0/230 on the probe, and structurally so -- Block 1 defines an instance
AS a connected component, so an instance cannot be fragmented by construction.
It would only be measurable per class, before decomposition.

CPU only. Rewrites the pickle in place (a .bak copy is kept). Run:
  PYTHONPATH=/workspace python slice_quality/examples/add_geometry_metadata.py
Env: PKL, PRED_SUBDIR, MIN_HOLE_PX (default 20).
"""
import os
import pickle
import shutil
from collections import Counter

import numpy as np
from scipy import ndimage

from slice_quality.adapters.voc_seg_samples import build_voc_samples

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/data/voc2012/voc_crops"
PKL = os.environ.get("PKL", "outputs/slice_quality/result_tagged.pkl")
PRED_SUBDIR = os.environ.get("PRED_SUBDIR", "mobilenet_pred")
MIN_HOLE_PX = int(os.environ.get("MIN_HOLE_PX", "20"))

d = pickle.load(open(PKL, "rb"))
errors, reference = d["errors"], d["reference"]
names = sorted({e.image_id for e in errors} | {r.image_id for r in reference})
print(f"images: {len(names)} | errors: {len(errors)} | references: {len(reference)}",
      flush=True)

samples, _paths = build_voc_samples(DATA, CROPS, names, pred_subdir=PRED_SUBDIR)
by_img = {s["image_id"]: s for s in samples}
print(f"masks loaded for {len(by_img)} images", flush=True)

# hole stats per predicted instance, keyed the way instance_ids are built
holes_by_id = {}
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
        holes_by_id[f"{s['image_id']}#p{i}"] = {
            "n_holes": int(n), "hole_px": px, "hole_frac": round(px / area, 4)}

print(f"predicted instances with holes >= {MIN_HOLE_PX}px: {len(holes_by_id)}", flush=True)

hit = 0
for inst in list(errors) + list(reference):
    h = holes_by_id.get(inst.instance_id)
    if h:                                   # FN ids are "#g<idx>" and never match
        inst.metadata.update(h)
        hit += 1
print(f"instances updated: {hit}", flush=True)

tx = Counter(e.seg_subtype.value for e in errors if e.metadata.get("n_holes"))
print("holes by error type:", dict(tx.most_common()), flush=True)
tot = sum(e.metadata.get("n_holes", 0) for e in errors)
print(f"total holes on erroneous instances: {tot}", flush=True)

if not os.path.exists(PKL + ".bak"):
    shutil.copy(PKL, PKL + ".bak")
    print(f"backup written: {PKL}.bak", flush=True)
with open(PKL, "wb") as f:
    pickle.dump(d, f)
print(f"saved {PKL}", flush=True)
print("GEOMETRY_DONE", flush=True)
