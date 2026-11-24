"""Cluster-branch instances of val-EA: BLOCK 1 as in the inst run, embeddings and tags
reused where the image did not change, computed fresh where it did.

val-EA (build_voc_ea.py) is VOC val with 690 images replaced by their Pascal-EA
material edits. BLOCK 1 runs exactly as for the inst run (SQ_CLAIM_COV=0.5,
SQ_GT_INSTANCES=object) on the whole of val-EA. An instance of an UNTOUCHED image
with the same id, box and type as in the inst pickle takes that instance's DINOv2
embedding and Qwen tag (same pixels, same mask); every instance of an EDITED image
gets a fresh DINOv2 embedding, and the error instances among them a fresh
verify-mode Qwen tag (the purity gate reads the tags of slice members, which are
errors; a member without a tag would make its slice untestable -- the lesson of the
corruption experiment).

Run (GPU container, Qwen mounted):
  SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object python slice_quality/examples/retype_ea.py
Env: PKL (inst pickle), OUT (outputs/slice_quality_ea/result_tagged.pkl), ROOT (val-EA),
     EDITED, QWEN_4BIT, TAG_REFS (0), SKIP_GPU (1 = count only).
"""
import os
import pickle
import time
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples, make_mask_lookup
from slice_quality.config import SliceQualityConfig
from slice_quality.typed_error import decompose_dataset

PKL = os.environ.get("PKL", "outputs/slice_quality_inst/result_tagged.pkl")
OUT = os.environ.get("OUT", "outputs/slice_quality_ea/result_tagged.pkl")
ROOT = os.environ.get("ROOT", "/data/voc2012_ea/VOCdevkit/VOC2012")
EDITED = os.environ.get("EDITED", "/data/voc2012_ea/edited_stems.txt")
CROPS = os.environ.get("CROPS", "/workspace/voc_crops_full_ea")
SKIP_GPU = os.environ.get("SKIP_GPU") == "1"
TAG_REFS = os.environ.get("TAG_REFS", "0") == "1"
TAG_KEYS = ("attribute", "attr_scores", "attribute_mode", "attribute_before")

assert os.environ.get("SQ_GT_INSTANCES") == "object" and os.environ.get("SQ_CLAIM_COV") == "0.5", \
    "val-EA mirrors the inst run: SQ_GT_INSTANCES=object SQ_CLAIM_COV=0.5"
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
edited = {l.strip() for l in open(EDITED) if l.strip()}
d = pickle.load(open(PKL, "rb"))
cached = {t.instance_id: t for t in list(d["errors"]) + list(d["reference"])}
print(f"inst pickle: {len(d['errors'])} errors, {len(d['reference'])} reference; edited images {len(edited)}",
      flush=True)


def sig(t):
    return (tuple(round(float(x), 1) for x in t.bbox) if t.bbox else None, t.seg_subtype.value,
            t.gt_class, t.pred_class)


val = [l.strip() for l in open(os.path.join(ROOT, "ImageSets/Segmentation/val.txt")) if l.strip()]
cfg = SliceQualityConfig().typed_error
samples, image_paths = build_voc_samples(ROOT, CROPS, val, gt_instances="object")
new_err, new_ref = decompose_dataset(samples, cfg)
print(f"val-EA decomposed (objects, claim_cov={cfg.claim_cov}): {len(new_err)} errors, {len(new_ref)} reference",
      flush=True)
print("taxonomy:", dict(Counter(t.seg_subtype.value for t in new_err)), flush=True)

reused = mismatch = 0
fresh = []
for t in new_err + new_ref:
    old = cached.get(t.instance_id)
    if (t.image_id not in edited and old is not None and old.embedding is not None
            and sig(old) == sig(t)):
        t.embedding = old.embedding
        for k in list(old.metadata):
            if k not in t.metadata or k in TAG_KEYS:
                t.metadata[k] = old.metadata[k]
        reused += 1
    else:
        if t.image_id not in edited:
            mismatch += 1
        fresh.append(t)
fresh_err = [t for t in fresh if t.seg_subtype.value != "correct"]
print(f"reused {reused}; fresh {len(fresh)} ({len(fresh_err)} errors), of which in untouched images "
      f"{mismatch} (should be ~0)", flush=True)

if fresh and not SKIP_GPU:
    from slice_quality.embeddings import attach_embeddings
    from slice_quality.extractors import DINOv2Extractor
    from slice_quality.tagging import tag_instances_verify
    from slicelens.backends.qwen_vlm import QwenVLM
    t0 = time.time()
    attach_embeddings(fresh, DINOv2Extractor(image_paths), make_mask_lookup(samples))
    print(f"DINOv2 on {len(fresh)} in {time.time() - t0:.0f}s", flush=True)
    to_tag = fresh if TAG_REFS else fresh_err
    t0 = time.time()
    vlm = QwenVLM(model="/models/qwen", max_new_tokens=10, load_in_4bit=os.environ.get("QWEN_4BIT", "0") == "1")
    tag_instances_verify(to_tag, image_paths, vlm)
    print(f"Qwen verify tags on {len(to_tag)} in {time.time() - t0:.0f}s: "
          f"{dict(Counter(t.metadata.get('attribute') for t in to_tag).most_common())}", flush=True)

keep_err = [t for t in new_err if t.embedding is not None]
keep_ref = [t for t in new_ref if t.embedding is not None]
print(f"dropped without embedding: {len(new_err) - len(keep_err)} errors, {len(new_ref) - len(keep_ref)} reference",
      flush=True)
if not SKIP_GPU:
    pickle.dump({"result": None, "errors": keep_err, "reference": keep_ref, "image_paths": image_paths,
                 "descs": None, "edited": sorted(edited)}, open(OUT, "wb"))
    print(f"PICKLE val-EA -> {OUT}", flush=True)
print("RETYPE_EA_DONE", flush=True)
