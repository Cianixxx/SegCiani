"""Interim look at the SBD run while the GPUs tag: CPU only, reads what is on disk.

  segmenter   mIoU and per-class IoU on the 9 118 SBD images (520 crop, the run's own
              predictions) beside VOC 2012 val with the same code; VOC ignores its void
              band (255), SBD has none
  BLOCK 1     error taxonomy (inst configuration) per image, SBD vs the inst run
  tags        Qwen attributes of the chunks tagged so far, SBD vs the inst run's errors

Run: docker run (no GPU) ... python slice_quality/examples/sbd_interim.py
Out: outputs/slice_quality_sbd/interim.txt
"""
import glob
import json
import os
import pickle
from collections import Counter

import numpy as np
from PIL import Image

from slicelens.adapters.voc_seg import _resize_center_crop

RUN = "outputs/slice_quality_sbd"
INST = "outputs/slice_quality_inst/result_tagged.pkl"
TREES = {"SBD": "/data/sbd/VOCdevkit/VOC2012", "VOC val": "/data/voc2012/VOCdevkit/VOC2012"}
CLS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
       "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
       "train", "tvmonitor"]
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))


def miou(root):
    names = [l.strip() for l in open(f"{root}/ImageSets/Segmentation/val.txt") if l.strip()]
    cm = np.zeros((21, 21), np.int64)
    for n in names:
        gt = np.array(_resize_center_crop(Image.open(f"{root}/SegmentationClass/{n}.png"), 520, Image.NEAREST))
        pr = np.array(Image.open(f"{root}/mobilenet_pred/mask_pr_{n}.png").convert("RGB"))[:, :, 0]
        m = gt != 255
        cm += np.bincount(gt[m].astype(np.int64) * 21 + pr[m], minlength=441).reshape(21, 21)
    iou = np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1) - np.diag(cm), 1)
    return len(names), iou


say("=" * 78)
say("SBD, RISULTATI INTERMEDI (prima delle slice)")
say("=" * 78)
res = {k: miou(v) for k, v in TREES.items()}
say(f"segmentatore (LR-ASPP MobileNetV3, crop 520): mIoU SBD {100 * res['SBD'][1].mean():.1f} "
    f"su {res['SBD'][0]} immagini | VOC val {100 * res['VOC val'][1].mean():.1f} su {res['VOC val'][0]}")
say("  IoU per classe (SBD / VOC val, punti):")
for c in sorted(range(1, 21), key=lambda c: res["SBD"][1][c] - res["VOC val"][1][c]):
    say(f"    {CLS[c]:12s} {100 * res['SBD'][1][c]:5.1f} / {100 * res['VOC val'][1][c]:5.1f}  "
        f"({100 * (res['SBD'][1][c] - res['VOC val'][1][c]):+.1f})")

sb = pickle.load(open(f"{RUN}/instances_embedded.pkl", "rb"))
vi = pickle.load(open(INST, "rb"))
n_sb, n_vi = len(sb["image_paths"]), len(vi["image_paths"])
ts, tv = Counter(t.seg_subtype.value for t in sb["errors"]), Counter(t.seg_subtype.value for t in vi["errors"])
say()
say(f"Blocco 1 (configurazione inst): SBD {len(sb['errors'])} errori su {n_sb} immagini "
    f"({len(sb['errors']) / n_sb:.2f} per immagine) | VOC val {len(vi['errors'])} su {n_vi} "
    f"({len(vi['errors']) / n_vi:.2f})")
say(f"  {'tipo':11s}{'SBD':>8s}{'%':>7s}{'per img':>9s}{'VOC val':>9s}{'%':>7s}{'per img':>9s}")
for k in sorted(ts, key=lambda k: -ts[k]):
    say(f"  {k:11s}{ts[k]:8d}{100 * ts[k] / len(sb['errors']):6.1f}%{ts[k] / n_sb:9.2f}"
        f"{tv[k]:9d}{100 * tv[k] / len(vi['errors']):6.1f}%{tv[k] / n_vi:9.2f}")

tags = {}
for f in sorted(glob.glob(f"{RUN}/tags/chunk_*.json")):
    tags.update(json.load(open(f)))
a_sb = Counter(v.get("attribute") for v in tags.values())
a_vi = Counter(t.metadata.get("attribute") for t in vi["errors"] if t.metadata.get("attribute_mode") == "verify")
say()
say(f"tag di Qwen finora: {len(tags)} errori su {len(sb['errors'])} ({100 * len(tags) / len(sb['errors']):.0f}%; "
    f"blocchi in ordine di id, quindi soprattutto immagini 2008-2009) | VOC val: {sum(a_vi.values())} errori")
for k in sorted(set(a_sb) | set(a_vi), key=lambda k: -a_sb[k]):
    say(f"  {str(k):11s} SBD {100 * a_sb[k] / max(len(tags), 1):5.1f}%   VOC val {100 * a_vi[k] / max(sum(a_vi.values()), 1):5.1f}%")
open(f"{RUN}/interim.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("INTERIM_DONE", flush=True)
