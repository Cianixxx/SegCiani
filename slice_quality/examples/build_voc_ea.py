"""Build "val-EA": VOC 2012 val with the Pascal-EA material edits in place of the originals.

The planted-cause test runs the WHOLE pipeline on a val split in which the 690
images that have a v3 material edit (and a prediction of the segmenter on it,
pascal_ea_compare.py) replace their originals; the other 759 are untouched. The
pipeline is not told which images were edited.

Layout written under ROOT (default /data/voc2012_ea/VOCdevkit/VOC2012), the same
as VOC so every adapter works unchanged by pointing it at ROOT:
  JPEGImages/<stem>.jpg          edited image (JPEG, quality 100) or a relative
                                 symlink to the original
  mobilenet_pred/mask_pr_<stem>.png  the segmenter's prediction on the edited image
                                 (520 crop, R=G=B=class, the thesis format) or a
                                 symlink to the original prediction
  SegmentationClass, SegmentationObject, ImageSets -> symlinks to VOC (the edit
                                 preserves the masks)
  ../../edited_stems.txt         the edited stems (ground truth of the test; the
                                 pipeline never reads it)

Then, CPU only, the two substrates the two branches use, and a check that the
untouched images give EXACTLY the instances of the thesis runs (same ids, boxes,
types) -- which is what allows reusing their embeddings, tags and groundings:
  STEP=substrate  /data/voc2012_ea/substrate_cc.pkl  (components, IoU claims: the
                  July language-branch substrate, rebuilt on val-EA) vs
                  /data/voc2012/substrate_cc.pkl
Run: docker run (no GPU) ... python slice_quality/examples/build_voc_ea.py
Env: EA (/ds/pascal_ea_material), PRED (outputs/pascal_ea/pred_v3), STEP (layout,substrate).
"""
import os
import pickle
from collections import Counter

import numpy as np
from PIL import Image

VOC = "/data/voc2012/VOCdevkit/VOC2012"
EA_ROOT = "/data/voc2012_ea"
ROOT = f"{EA_ROOT}/VOCdevkit/VOC2012"
EA = os.environ.get("EA", "/ds/pascal_ea_material")
PRED = os.environ.get("PRED", "outputs/pascal_ea/pred_v3")
STEPS = os.environ.get("STEP", "layout,substrate").split(",")


def link(src_rel, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src_rel, dst)


def layout():
    os.makedirs(f"{ROOT}/JPEGImages", exist_ok=True)
    os.makedirs(f"{ROOT}/mobilenet_pred", exist_ok=True)
    for d in ("SegmentationClass", "SegmentationObject", "ImageSets"):
        link(f"../../../voc2012/VOCdevkit/VOC2012/{d}", f"{ROOT}/{d}")
    val = [l.strip() for l in open(f"{VOC}/ImageSets/Segmentation/val.txt") if l.strip()]
    edited = []
    for s in val:
        e_img, e_pred = f"{EA}/edited_v3/{s}.png", f"{PRED}/{s}.png"
        dj, dp = f"{ROOT}/JPEGImages/{s}.jpg", f"{ROOT}/mobilenet_pred/mask_pr_{s}.png"
        if os.path.exists(e_img) and os.path.exists(e_pred):
            for p in (dj, dp):
                if os.path.islink(p):
                    os.remove(p)
            if not os.path.exists(dj):
                Image.open(e_img).convert("RGB").save(dj, quality=100)
            if not os.path.exists(dp):
                pr = np.array(Image.open(e_pred))
                Image.fromarray(np.stack([pr] * 3, -1).astype(np.uint8)).save(dp)
            edited.append(s)
        else:
            link(f"../../../../voc2012/VOCdevkit/VOC2012/JPEGImages/{s}.jpg", dj)
            link(f"../../../../voc2012/VOCdevkit/VOC2012/mobilenet_pred/mask_pr_{s}.png", dp)
    open(f"{EA_ROOT}/edited_stems.txt", "w").write("\n".join(edited) + "\n")
    print(f"layout: {len(val)} val images, {len(edited)} edited, {len(val) - len(edited)} linked", flush=True)


def substrate():
    from slice_quality.adapters.voc_seg_samples import build_voc_samples
    from slice_quality.typed_error import decompose_dataset
    assert not os.environ.get("SQ_GT_INSTANCES") and not os.environ.get("SQ_CLAIM_COV"), \
        "the July substrate is components + IoU claims: unset SQ_GT_INSTANCES / SQ_CLAIM_COV"
    edited = {l.strip() for l in open(f"{EA_ROOT}/edited_stems.txt") if l.strip()}
    val = [l.strip() for l in open(f"{VOC}/ImageSets/Segmentation/val.txt") if l.strip()]
    out = f"{EA_ROOT}/substrate_cc.pkl"
    if not os.path.exists(out):
        samples, image_paths = build_voc_samples(ROOT, f"{EA_ROOT}/voc_crops", val)
        errors, reference = decompose_dataset(samples)
        pickle.dump({"errors": errors, "reference": reference, "image_paths": image_paths}, open(out, "wb"))
    d = pickle.load(open(out, "rb"))
    old = pickle.load(open("/data/voc2012/substrate_cc.pkl", "rb"))
    print(f"val-EA substrate: {len(d['errors'])} errors, {len(d['reference'])} reference "
          f"(July: {len(old['errors'])}, {len(old['reference'])})", flush=True)
    sig = lambda t: (tuple(round(float(x), 1) for x in t.bbox), t.seg_subtype.value, t.gt_class, t.pred_class)
    a = {t.instance_id: sig(t) for t in d["errors"] + d["reference"] if t.image_id not in edited}
    b = {t.instance_id: sig(t) for t in old["errors"] + old["reference"] if t.image_id not in edited}
    same = sum(1 for k in a if b.get(k) == a[k])
    print(f"untouched images: {len(a)} instances now, {len(b)} in July; identical {same}; "
          f"only now {len(set(a) - set(b))}; only July {len(set(b) - set(a))}", flush=True)
    new = [t for t in d["errors"] + d["reference"] if t.image_id in edited]
    print(f"instances in edited images (to ground fresh): {len(new)}; by class "
          f"{dict(Counter(t.gt_class or t.pred_class for t in new).most_common())}", flush=True)


if "layout" in STEPS:
    layout()
if "substrate" in STEPS:
    substrate()
print("BUILD_VOC_EA_DONE", flush=True)
