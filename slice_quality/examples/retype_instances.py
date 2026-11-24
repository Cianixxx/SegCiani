"""Re-run BLOCK 1 with GT instances = VOC objects, reusing what is cached.

The reference pickle holds every predicted component with its DINOv2 embedding
and Qwen tag, and every GT component that was a FN with the same. With the
object adapter (SQ_GT_INSTANCES=object) predicted instances do not change --
same components, same ids, everything is reused. GT instances do: an object cut
into parts becomes ONE instance, two touching objects become TWO. So the GT side
is joined by MASK, not by id: a new GT instance whose mask equals an old GT
component takes that component's embedding and tag; the others -- the cut
objects, the touching ones, and objects no old instance ever embedded -- get a
fresh DINOv2 embedding and a fresh verify-mode Qwen tag here, on the GPU, and
only they do.

Run (GPU container, Qwen mounted):
  SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object PKL=outputs/slice_quality/result_tagged.pkl \\
      OUT=outputs/slice_quality_inst/result_tagged.pkl QWEN_4BIT=0 \\
      PYTHONPATH=/workspace python slice_quality/examples/retype_instances.py
Env: PKL, OUT, QWEN_4BIT, SKIP_GPU=1 (CPU test: leaves the fresh ones unembedded).
"""
import os
import pickle
import time
from collections import Counter

import numpy as np

from slice_quality.adapters.voc_seg_samples import build_voc_samples, make_mask_lookup
from slice_quality.config import SliceQualityConfig
from slice_quality.typed_error import decompose_dataset

PKL = os.environ.get("PKL", "outputs/slice_quality/result_tagged.pkl")
OUT = os.environ.get("OUT", "outputs/slice_quality_inst/result_tagged.pkl")
DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/workspace/voc_crops_full"
SKIP_GPU = os.environ.get("SKIP_GPU") == "1"
TAG_KEYS = ("attribute", "attr_scores", "attribute_mode", "attribute_before")
STALE = ("n_gt_covered", "gt_classes", "gt_ids", "primary_gt", "primary_iou", "primary_e_bnd",
         "primary_e_ext", "paired_by", "cov_gt", "cov_pred", "swallowed_by", "covered_frac",
         "gt_id", "gt_parts", "occluded_by", "union_iou", "union_e_bnd", "union_e_ext",
         "boundary_iou", "missed_px", "excess_px", "n_thin_lost", "thin_lost_px", "frame_frac")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
d = pickle.load(open(PKL, "rb"))
cached = {t.instance_id: t for t in list(d["errors"]) + list(d["reference"])}
image_paths = d["image_paths"]
print(f"cached: {len(d['errors'])} errors, {len(d['reference'])} reference", flush=True)

val = [ln.strip() for ln in open(os.path.join(DATA, "ImageSets/Segmentation/val.txt")) if ln.strip()]
if os.environ.get("LIMIT"):                 # smoke test on a few images
    val = val[:int(os.environ["LIMIT"])]
cfg = SliceQualityConfig().typed_error
assert os.environ.get("SQ_GT_INSTANCES") == "object", "set SQ_GT_INSTANCES=object"
samples, _ = build_voc_samples(DATA, CROPS, val, gt_instances="object")
samples_cc, _ = build_voc_samples(DATA, CROPS, val, gt_instances="components")
new_err, new_ref = decompose_dataset(samples, cfg)
print(f"re-decomposed (objects, claim_cov={cfg.claim_cov}): {len(new_err)} errors, "
      f"{len(new_ref)} reference", flush=True)
print("new taxonomy:", dict(Counter(t.seg_subtype.value for t in new_err)), flush=True)

# GT side: old component index -> its mask, per image, to join by mask equality
cc_by_img = {s["image_id"]: s for s in samples_cc}


def _old_gt_id(inst):
    img, _, tail = inst.instance_id.partition("#")
    j = int(tail[1:])
    s_new, s_old = next(s for s in samples if s["image_id"] == img), cc_by_img.get(img)
    if s_old is None:
        return None
    m = np.asarray(s_new["gts"][j]["mask"], dtype=bool)
    for jj, g in enumerate(s_old["gts"]):
        if g["cls"] == s_new["gts"][j]["cls"] and np.array_equal(np.asarray(g["mask"], dtype=bool), m):
            return f"{img}#g{jj}"
    return None


def _carry(new, old):
    """Give the new instance the cached embedding and tag; typing is the new one."""
    new.embedding = old.embedding
    for k in TAG_KEYS:
        if k in old.metadata:
            new.metadata[k] = old.metadata[k]
    for k in old.metadata:                 # older mask facts of unchanged masks
        if k not in new.metadata and k not in STALE and k not in TAG_KEYS:
            new.metadata[k] = old.metadata[k]
    return new


reused_p = reused_g = fresh = 0
fresh_list = []
for t in new_err + new_ref:
    if t.metadata.get("from_gt_mask"):
        oid = _old_gt_id(t)
        old = cached.get(oid) if oid else None
        if old is not None and old.embedding is not None:
            _carry(t, old)
            reused_g += 1
        else:
            fresh += 1
            fresh_list.append(t)
    else:
        old = cached.get(t.instance_id)
        if old is not None and old.embedding is not None:
            _carry(t, old)
            reused_p += 1
        else:
            fresh += 1
            fresh_list.append(t)
print(f"reused: {reused_p} predicted + {reused_g} GT-side | fresh (no cached mask): {fresh}", flush=True)

if fresh_list and not SKIP_GPU:
    from slice_quality.embeddings import attach_embeddings
    from slice_quality.extractors import DINOv2Extractor
    from slice_quality.tagging import tag_instances_verify
    from slicelens.backends.qwen_vlm import QwenVLM
    t0 = time.time()
    attach_embeddings(fresh_list, DINOv2Extractor(image_paths), make_mask_lookup(samples))
    print(f"DINOv2 on {len(fresh_list)} fresh instances in {time.time() - t0:.0f}s: "
          f"{sum(1 for t in fresh_list if t.embedding is not None)} embedded", flush=True)
    t0 = time.time()
    vlm = QwenVLM(model="/models/qwen", max_new_tokens=10,
                  load_in_4bit=os.environ.get("QWEN_4BIT", "0") == "1")
    tag_instances_verify(fresh_list, image_paths, vlm)
    print(f"Qwen verify tags on {len(fresh_list)} in {time.time() - t0:.0f}s: "
          f"{dict(Counter(t.metadata.get('attribute') for t in fresh_list).most_common())}", flush=True)

keep_err = [t for t in new_err if t.embedding is not None]
keep_ref = [t for t in new_ref if t.embedding is not None]
print(f"dropped without embedding: {len(new_err) - len(keep_err)} errors, "
      f"{len(new_ref) - len(keep_ref)} reference", flush=True)
print("final taxonomy:", dict(Counter(t.seg_subtype.value for t in keep_err)), flush=True)
pickle.dump({"result": None, "errors": keep_err, "reference": keep_ref,
             "image_paths": image_paths, "descs": None}, open(OUT, "wb"))
print(f"PICKLE RETYPED (objects) -> {OUT}", flush=True)
