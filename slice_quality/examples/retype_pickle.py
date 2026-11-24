"""Re-run BLOCK 1 (split/merge-capable matcher) and RETYPE the cached instances.

CPU only. The connected components -- and therefore the instance ids, the
DINOv2 embeddings and the Qwen tags -- are unchanged; what changed is the TYPE
assignment (fragments FP->SPLIT, merged blobs FP->MERGE, swallowed GTs no longer
double-counted as FNs -- kept as GT-side MERGE instances when SQ_CLAIM_COV is
set -- split primary chosen by IoU). So: decompose the masks again with the
new matcher and join the new typing onto the cached embedded instances by
instance_id. Instances that moved across the error/correct line carry their
cached embedding with them.

Run (in the CPU container):
    PYTHONPATH=/workspace python slice_quality/examples/retype_pickle.py
then re-run rescore_with_language.py.

Same recipe for the containment claim (config.TypedErrorConfig.claim_cov):
    SQ_CLAIM_COV=0.5 OUT=outputs/slice_quality_cov05/result_tagged.pkl         PYTHONPATH=/workspace python slice_quality/examples/retype_pickle.py
OUT keeps the two typings apart -- they are not comparable instance by instance
(a third of the FNs disappear into merges and pairs), so the run must not
overwrite the one every number so far comes from. Block 1 metadata written by
the new typing (n_gt_covered, gt_classes, primary_iou, paired_by) is merged
over the cached one; the Qwen tags survive.
"""
import os
import pickle
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.config import SliceQualityConfig
from slice_quality.typed_error import decompose_dataset

PKL = os.environ.get("PKL", "outputs/slice_quality/result_tagged.pkl")
OUT = os.environ.get("OUT", PKL)
DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/workspace/voc_crops_full"

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
d = pickle.load(open(PKL, "rb"))
errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
print(f"cached: {len(errors)} errors, {len(reference)} reference", flush=True)
print("cached taxonomy:", dict(Counter(t.seg_subtype.value for t in errors)), flush=True)

val = [ln.strip() for ln in
       open(os.path.join(DATA, "ImageSets/Segmentation/val.txt")) if ln.strip()]
samples, _ = build_voc_samples(DATA, CROPS, val)
new_err, new_ref = decompose_dataset(samples, SliceQualityConfig().typed_error)
err_by_id = {t.instance_id: t for t in new_err}
ref_by_id = {t.instance_id: t for t in new_ref}
print(f"re-decomposed: {len(new_err)} errors, {len(new_ref)} reference", flush=True)


def _apply(old, new):
    old.seg_subtype = new.seg_subtype
    old.error_vector = new.error_vector
    old.iou = new.iou
    old.gt_class, old.pred_class = new.gt_class, new.pred_class
    for k in ("n_gt_covered", "gt_classes", "primary_gt", "primary_iou",
              "primary_e_bnd", "primary_e_ext", "paired_by", "cov_gt", "cov_pred",
              "swallowed_by", "covered_frac"):
        old.metadata.pop(k, None)          # stale facts of the previous typing
    old.metadata.update(new.metadata)
    return old


cached = {t.instance_id: t for t in list(errors) + list(reference)}
out_err, out_ref = [], []
retyped = moved = missing_emb = dropped = 0
for iid, new in err_by_id.items():
    old = cached.get(iid)
    if old is None:
        missing_emb += 1          # brand-new instance: no cached embedding
        continue
    if old.seg_subtype is not new.seg_subtype:
        retyped += 1
    out_err.append(_apply(old, new))
for iid, new in ref_by_id.items():
    old = cached.get(iid)
    if old is None:
        missing_emb += 1
        continue
    if old.seg_subtype is not new.seg_subtype:
        moved += 1
    out_ref.append(_apply(old, new))
dropped = len(cached) - len(out_err) - len(out_ref)   # e.g. FNs swallowed by merges

print(f"retyped {retyped}, moved across err/ref {moved}, "
      f"dropped {dropped}, without cached embedding {missing_emb}", flush=True)
print("new taxonomy:", dict(Counter(t.seg_subtype.value for t in out_err)), flush=True)

pickle.dump({"result": d.get("result"), "errors": out_err, "reference": out_ref,
             "image_paths": image_paths, "descs": None}, open(OUT, "wb"))
print(f"PICKLE RETYPED -> {OUT}", flush=True)
