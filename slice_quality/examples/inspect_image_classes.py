"""What is predicted on each ground-truth object of an image, and what lies under each
predicted component: the raw facts behind a "labelled X instead" sentence. CPU.

Run: python slice_quality/examples/inspect_image_classes.py <root> <stem> [<stem> ...]
     (<root> a VOC-layout tree: JPEGImages, SegmentationClass, [SegmentationObject], mobilenet_pred)
"""
import sys
from collections import Counter

import numpy as np

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.mask_facts import label_map

root, stems = sys.argv[1], sys.argv[2:]
samples, _ = build_voc_samples(root, "/tmp/_inspect_crops", stems, min_pixels=15)
for s in samples:
    shape = np.asarray((s["gts"] or s["preds"])[0]["mask"]).shape
    pr, gt = label_map(s["preds"], shape), label_map(s["gts"], shape)
    print(f"\n=== {s['image_id']} ({s.get('gt_instances')}) | GT {len(s['gts'])} | pred components {len(s['preds'])}")
    for j, g in enumerate(s["gts"]):
        m = np.asarray(g["mask"], bool)
        c = Counter(str(x) or "background" for x in pr[m])
        a = int(m.sum())
        ys, xs = np.where(m)
        print(f"  GT g{j} {g['cls']:12s} {a:6d} px  centre ({xs.mean():.0f},{ys.mean():.0f})  predicted as: "
              + ", ".join(f"{k} {100 * v / a:.0f}%" for k, v in c.most_common(4)))
    for i, p in enumerate(s["preds"]):
        m = np.asarray(p["mask"], bool)
        c = Counter(str(x) or "background" for x in gt[m])
        a = int(m.sum())
        ys, xs = np.where(m)
        print(f"  pred p{i} {p['cls']:12s} {a:6d} px  centre ({xs.mean():.0f},{ys.mean():.0f})  lies on: "
              + ", ".join(f"{k} {100 * v / a:.0f}%" for k, v in c.most_common(3)))
