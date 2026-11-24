"""Numbers for the thesis chapters on SBD and on the Pascal-EA planted cause. CPU only, reads only.

  SBD    segmenter mIoU and per-class IoU (SBD vs VOC val, same code as sbd_interim.py), the
         run's headline numbers (population, error types, coverage: docs/_revision_numbers.
         load_inst_summary on the SBD run directory) and the funnel of the language branch
         (hypotheses tried on the sample, slope-systematic there, re-grounded on the class,
         systematic on the class) for VOC val (July + extra runs) and SBD
  EA     mIoU of VOC val and of val-EA (VOC val with the 691 material edits), on all images
         and on the edited ones only

Run: python slice_quality/examples/doc_numbers_sbd_ea.py
Out: outputs/slice_quality_sbd/sbd_summary.json, outputs/slice_quality_ea/ea_summary.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from slicelens.adapters.voc_seg import _resize_center_crop

sys.path.insert(0, "docs")
from _revision_numbers import load_inst_summary  # noqa: E402

CLS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
       "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
       "train", "tvmonitor"]
VOC = "/data/voc2012/VOCdevkit/VOC2012"
SBD = "/data/sbd/VOCdevkit/VOC2012"
EA = "/data/voc2012_ea/VOCdevkit/VOC2012"
EDITED = [l.strip() for l in open("/data/voc2012_ea/edited_stems.txt") if l.strip()]


def miou(root, names=None):
    names = names or [l.strip() for l in open(f"{root}/ImageSets/Segmentation/val.txt") if l.strip()]
    cm = np.zeros((21, 21), np.int64)
    for n in names:
        gt = np.array(_resize_center_crop(Image.open(f"{root}/SegmentationClass/{n}.png"), 520, Image.NEAREST))
        pr = np.array(Image.open(f"{root}/mobilenet_pred/mask_pr_{n}.png").convert("RGB"))[:, :, 0]
        m = gt != 255
        cm += np.bincount(gt[m].astype(np.int64) * 21 + pr[m], minlength=441).reshape(21, 21)
    iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
    return {"miou": float(iou.mean()), "n_images": len(names), "per_class": dict(zip(CLS, map(float, iou)))}


def funnel(dirs):
    """per class and in total: tried on the sample, systematic there, re-grounded, systematic on the class."""
    tot, per = [0, 0, 0, 0], {}
    for c in CLS[1:]:
        row = [0, 0, 0, 0]
        for d in dirs:
            f = f"{d}/voc_paper_{c}_cc.json"
            if os.path.exists(f):
                r = json.load(open(f))
                row[0] += len(r["candidates"])
                row[1] += sum(1 for x in r["candidates"] if x["is_systematic"])
            f = f"{d}/voc_paper_{c}_full.json"
            if os.path.exists(f):
                r = json.load(open(f))
                reg = [x for x in r["candidates"] if len(x["scored"]) == r["num_instances"] and r["num_instances"] > 120]
                row[2] += len(reg)
                row[3] += sum(1 for x in reg if x["is_systematic"])
        per[c] = row
        tot = [a + b for a, b in zip(tot, row)]
    return {"total": tot, "per_class": per}


sbd = {"miou": {"SBD": miou(SBD), "VOC val": miou(VOC)},
       "run": load_inst_summary("outputs/slice_quality_sbd"),
       "funnel": {"VOC val": funnel(["/data/outputs", "/data/outputs_extra"]),
                  "SBD": funnel(["/data/outputs_sbd"])}}
json.dump(sbd, open("outputs/slice_quality_sbd/sbd_summary.json", "w", encoding="utf-8"), indent=1)
print("SBD", round(sbd["miou"]["SBD"]["miou"], 4), round(sbd["miou"]["VOC val"]["miou"], 4),
      sbd["funnel"]["VOC val"]["total"], sbd["funnel"]["SBD"]["total"],
      {k: sbd["run"][k] for k in ("n_err", "n_ref", "images", "with_pattern", "cov_union", "cov_language", "cov_cluster")})

ea = {"miou": {"VOC val": miou(VOC), "val-EA": miou(EA),
               "VOC val, edited images": miou(VOC, EDITED), "val-EA, edited images": miou(EA, EDITED)},
      "n_edited": len(EDITED)}
json.dump(ea, open("outputs/slice_quality_ea/ea_summary.json", "w", encoding="utf-8"), indent=1)
print("EA", {k: round(v["miou"], 4) for k, v in ea["miou"].items()})
print("DOC_NUMBERS_DONE")
