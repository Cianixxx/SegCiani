# -*- coding: utf-8 -*-
"""How often 'instance = connected component' is wrong on VOC val, and what it costs.

VOC ships instance masks (SegmentationObject: one id per object, 255 = void) for
the same images as the class masks. Seen through the pipeline's own substrate
(520 centre crop, components >= 200 px), this script counts, per image:

  cut      an object whose class-mask pixels form >= 2 components (occluded by
           something in front, or split by a thin occluder): the pipeline sees
           two GT "instances" where there is one;
  touching a component that contains >= 2 objects of the same class: the
           pipeline sees one GT "instance" where there are several.

Then it holds the revised run's typing (outputs/slice_quality_cov05b) against the
object ids: a MERGE blob whose swallowed GTs and primary all belong to ONE object
is a false merge (one occluded object, not several); a FN / swallowed part whose
object has another part that was matched is a partial miss of one object.

CPU only, in the container:
  PYTHONPATH=/workspace python docs/measure_instance_ambiguity.py
Env: SQ_OUT (default outputs/slice_quality_cov05b), ANN (annotated image list).
"""
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label

from slicelens.adapters.voc_seg import VOC_CLASSES, _resize_center_crop

DATA = "/data/voc2012/VOCdevkit/VOC2012"
SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality_cov05b")
CROP, MIN_PX, PART_PX = 520, 200, 15

val = [l.strip() for l in open(os.path.join(DATA, "ImageSets/Segmentation/val.txt")) if l.strip()]
ann = set()
for f in ("/data/dataset/voc_patch_matches.json",):
    if os.path.exists(f):
        ann = {v.replace(".jpg", "") for v in json.load(open(f)).values()}

tot = Counter()
per_image = {}
gt_parts = {}          # image -> {component index (as the adapter numbers them) -> object id}
for name in val:
    cls = np.array(_resize_center_crop(Image.open(f"{DATA}/SegmentationClass/{name}.png"), CROP, Image.NEAREST))
    obj = np.array(_resize_center_crop(Image.open(f"{DATA}/SegmentationObject/{name}.png"), CROP, Image.NEAREST))
    rec = Counter()
    comp_obj = {}
    j = 0                                    # adapter order: class by class, components by label order
    for c in range(1, 21):
        m = cls == c
        if not m.any():
            continue
        lab, n = cc_label(m)
        for k in range(1, n + 1):
            comp = lab == k
            if comp.sum() < MIN_PX:
                continue
            ids = Counter(obj[comp].tolist())
            ids.pop(255, None); ids.pop(0, None)
            ids = {i: v for i, v in ids.items() if v >= PART_PX}
            comp_obj[j] = sorted(ids)
            if len(ids) >= 2:
                rec["touching_components"] += 1
                rec["objects_in_touching"] += len(ids)
            j += 1
    rec["components"] = j
    # objects with >= 2 components (among components the adapter keeps)
    obj_comps = defaultdict(int)
    for js, ids in comp_obj.items():
        for i in ids:
            obj_comps[i] += 1
    rec["objects"] = len(obj_comps)
    rec["cut_objects"] = sum(1 for v in obj_comps.values() if v >= 2)
    rec["parts_of_cut"] = sum(v for v in obj_comps.values() if v >= 2)
    per_image[name] = rec
    gt_parts[name] = comp_obj
    tot.update(rec)

n_img = len(val)
img_cut = sum(1 for r in per_image.values() if r["cut_objects"])
img_touch = sum(1 for r in per_image.values() if r["touching_components"])
print(f"val: {n_img} immagini | componenti GT >= {MIN_PX}px: {tot['components']} | oggetti: {tot['objects']}")
print(f"  oggetti TAGLIATI in >= 2 componenti: {tot['cut_objects']} ({tot['parts_of_cut']} parti) in {img_cut} immagini ({100*img_cut/n_img:.0f}%)")
print(f"  componenti con >= 2 oggetti (che si toccano): {tot['touching_components']} ({tot['objects_in_touching']} oggetti) in {img_touch} immagini ({100*img_touch/n_img:.0f}%)")
if ann:
    a = [n for n in val if n in ann]
    print(f"  fra le {len(a)} annotate del val: tagliati in {sum(1 for n in a if per_image[n]['cut_objects'])}, toccanti in {sum(1 for n in a if per_image[n]['touching_components'])}")

# ---- against the revised typing ---------------------------------------
pk = pickle.load(open(f"{SQ_OUT}/result_tagged.pkl", "rb"))
err = pk["errors"]
by_img = defaultdict(list)
for e in err:
    by_img[e.image_id].append(e)


def gidx(iid):
    return int(iid.rsplit("#g", 1)[1])


false_merge = real_merge = unknown = 0
partial_miss = whole_miss = 0
for name, es in by_img.items():
    parts = gt_parts.get(name, {})
    blobs = [e for e in es if e.seg_subtype.value == "merge" and not e.metadata.get("from_gt_mask")]
    swallowed_of = defaultdict(list)          # pickles typed before gt_ids existed
    for e in es:
        if e.seg_subtype.value == "merge" and e.metadata.get("swallowed_by"):
            swallowed_of[e.metadata["swallowed_by"]].append(e.instance_id)
    for b in blobs:
        ids = b.metadata.get("gt_ids") or list(swallowed_of.get(b.instance_id, []))
        if not b.metadata.get("gt_ids") and b.metadata.get("primary_gt"):
            ids = ids + [b.metadata["primary_gt"]]
        objs = set()
        ok = True
        for i in ids:
            o = parts.get(gidx(i))
            if not o:
                ok = False
                break
            objs.update(o)
        if not ok or len(ids) < 2:
            unknown += 1               # swallowed parts without a cached instance
        elif len(objs) == 1:
            false_merge += 1
        else:
            real_merge += 1
    # FN / swallowed parts whose object has another part that was NOT missed
    gt_side = [e for e in es if e.metadata.get("from_gt_mask")]
    missed_idx = {gidx(e.instance_id) for e in gt_side}
    for e in gt_side:
        o = parts.get(gidx(e.instance_id), [])
        siblings = [j for j, ids in parts.items() if j != gidx(e.instance_id) and set(ids) & set(o)]
        if siblings and any(j not in missed_idx for j in siblings):
            partial_miss += 1
        else:
            whole_miss += 1
print(f"\nmerge blob del run {SQ_OUT}: veri (>= 2 oggetti) {real_merge} | FALSI (un solo oggetto occluso) {false_merge} | non decidibili {unknown}")
print(f"istanze lato GT (FN + inglobate): parti di un oggetto di cui un'altra parte e' accoppiata {partial_miss} | oggetti interi {whole_miss}")
json.dump({"per_image": {k: dict(v) for k, v in per_image.items()},
           "totals": dict(tot), "false_merge": false_merge, "real_merge": real_merge,
           "unknown_merge": unknown, "partial_miss": partial_miss, "whole_miss": whole_miss},
          open(f"{SQ_OUT}/instance_ambiguity.json", "w"), indent=1)
print("MEASURE_DONE")
