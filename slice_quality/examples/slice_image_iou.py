"""The segmentation quality of the images of every certified slice, language and cluster. CPU only, reads only.

For each certified slice of a run (in_descriptions), on the images that hold its members:
  miou_pooled   mIoU of those images together: one confusion matrix summed over them, IoU per class, mean
                over the classes present -- the same computation as the dataset's 72.3 (doc_numbers_sbd_ea.py)
  miou_img      mean over the images of each image's own mIoU (classes present in its GT or prediction)
  cls_iou_in    IoU of the slice's class on the slice's images, and cls_iou_out on the OTHER images that
                contain that class: how much worse the class is segmented where the situation holds
  member_iou    mean IoU of the slice's own wrong objects (the typed error instances)
Reference rows: every image of the split; the images with at least one error; the images with an error but
no certified slice. The class of a language slice is in its id; a cluster slice's is its members' majority.
Run (CPU, server):  python -u slice_quality/examples/slice_image_iou.py
Env: SQ_OUT (outputs/slice_quality_inst), ROOT (/data/voc2012/VOCdevkit/VOC2012), OUT (<SQ_OUT>/slice_image_iou)
"""
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from slicelens.adapters.voc_seg import _resize_center_crop

SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality_inst")
ROOT = os.environ.get("ROOT", "/data/voc2012/VOCdevkit/VOC2012")
OUT = os.environ.get("OUT", f"{SQ_OUT}/slice_image_iou")
CLS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
       "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
       "train", "tvmonitor"]

names = [l.strip() for l in open(f"{ROOT}/ImageSets/Segmentation/val.txt") if l.strip()]
CM = {}
for n in names:                                    # one 21x21 confusion matrix per image
    gt = np.array(_resize_center_crop(Image.open(f"{ROOT}/SegmentationClass/{n}.png"), 520, Image.NEAREST))
    pr = np.array(Image.open(f"{ROOT}/mobilenet_pred/mask_pr_{n}.png").convert("RGB"))[:, :, 0]
    m = gt != 255
    CM[n] = np.bincount(gt[m].astype(np.int64) * 21 + pr[m], minlength=441).reshape(21, 21)


def ious(cm):
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    present = union > 0
    return np.where(present, inter / np.maximum(union, 1), np.nan), present


def miou_pooled(imgs):
    iou, present = ious(sum(CM[i] for i in imgs))
    return float(np.nanmean(iou[present]))


def miou_img(imgs):
    vals = []
    for i in imgs:
        iou, present = ious(CM[i])
        vals.append(np.nanmean(iou[present]))
    return float(np.mean(vals))


def cls_iou(imgs, c):
    if not imgs:
        return None
    iou, _ = ious(sum(CM[i] for i in imgs))
    return None if np.isnan(iou[c]) else float(iou[c])


d = pickle.load(open(f"{SQ_OUT}/result_tagged.pkl", "rb"))
err = {e.instance_id: e for e in d["errors"]}
err_imgs = {e.image_id for e in d["errors"]} & set(CM)
has_cls = defaultdict(set)                          # class index -> images whose GT contains it
for n, cm in CM.items():
    for c in np.nonzero(cm.sum(1))[0]:
        has_cls[int(c)].add(n)
idx = json.load(open(f"{SQ_OUT}/slices_index.json", encoding="utf-8"))["slices"]
cert = [s for s in idx if s.get("in_descriptions")]
names_f = f"{SQ_OUT}/cluster_names.json"
cnames = json.load(open(names_f, encoding="utf-8")) if os.path.exists(names_f) else {}
rows, covered = [], set()
for s in cert:
    imgs = sorted(i for i in s["images"] if i in CM)
    covered |= set(imgs)
    members = [err[m] for ids in s["images"].values() for m in ids if m in err]
    if s["slice_id"].startswith("cluster"):
        branch = "cluster"
        cls = Counter((m.gt_class or m.pred_class) for m in members).most_common(1)[0][0]
        name = cnames.get(s["slice_id"], s["slice_id"])
        name = name if isinstance(name, str) else name.get("name", s["slice_id"])
    else:
        branch = "language"
        cls = re.sub(r"_(extra|dd|pend)$", "", s["slice_id"].rsplit("_", 1)[0])   # "<cls>_extra_<n>", "_dd_", "_pend_"
        name = f'{cls} — "{s["hypothesis"]}"'
    c = CLS.index(cls) if cls in CLS else None
    out_imgs = sorted(has_cls[c] - set(imgs)) if c is not None else []
    rows.append(dict(slice_id=s["slice_id"], branch=branch, name=name, cls=cls, n_images=len(imgs),
                     n_members=len(members), miou_pooled=miou_pooled(imgs), miou_img=miou_img(imgs),
                     cls_iou_in=cls_iou(imgs, c) if c is not None else None,
                     cls_iou_out=cls_iou(out_imgs, c) if c is not None else None,
                     member_iou=float(np.mean([m.iou for m in members])) if members else None,
                     error_type=s.get("dominant_error_type")))
ref = {"all images": sorted(CM), "images with an error": sorted(err_imgs),
       "images with an error, in no certified slice": sorted(err_imgs - covered),
       "images in at least one certified slice": sorted(covered)}
refs = {k: dict(n_images=len(v), miou_pooled=miou_pooled(v), miou_img=miou_img(v)) for k, v in ref.items()}

lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))
say("=" * 118)
say(f"SEGMENTATION QUALITY OF THE IMAGES OF EVERY CERTIFIED SLICE ({SQ_OUT})")
say("=" * 118)
for k, v in refs.items():
    say(f"  {k:48s} {v['n_images']:5d} images | mIoU pooled {100 * v['miou_pooled']:5.1f} | mean per image "
        f"{100 * v['miou_img']:5.1f}")
for br in ("language", "cluster"):
    rr = [r for r in rows if r["branch"] == br]
    if not rr:
        continue
    say(f"\n{br.upper()} SLICES ({len(rr)}): mean of the slices' pooled mIoU {100 * np.mean([r['miou_pooled'] for r in rr]):.1f}"
        f" | class IoU in the slice's images {100 * np.nanmean([r['cls_iou_in'] or np.nan for r in rr]):.1f} vs the "
        f"class's other images {100 * np.nanmean([r['cls_iou_out'] or np.nan for r in rr]):.1f}")
    say(f"  {'slice':62s} {'img':>4s} {'obj':>4s} | {'mIoU':>5s} {'per img':>7s} | {'class in':>8s} {'out':>5s} | {'obj IoU':>7s}")
    for r in sorted(rr, key=lambda r: r["miou_pooled"]):
        f = lambda x: f"{100 * x:5.1f}" if x is not None else "    -"
        say(f"  {r['name'][:62]:62s} {r['n_images']:4d} {r['n_members']:4d} | {f(r['miou_pooled'])} {f(r['miou_img']):>7s} | "
            f"{f(r['cls_iou_in']):>8s} {f(r['cls_iou_out'])} | {f(r['member_iou']):>7s}")
json.dump({"refs": refs, "rows": rows}, open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("SLICE_IMAGE_IOU_DONE", flush=True)
