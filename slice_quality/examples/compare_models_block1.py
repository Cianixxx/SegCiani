"""Stage 1b: run BLOCK 1 on two segmenters and compare their error taxonomies.

The question this answers is not "which model is better" (mIoU already says that)
but "does the KIND of failure change". The typed taxonomy is the right instrument
for it: FN/FP say the object was missed or invented, boundary/extent say it was
found but described badly, merge/split say the instance structure collapsed. A
stronger model that simply makes fewer of the same errors is a different finding
from one that fails in a different way.

CPU only (masks and geometry). Run:
  PYTHONPATH=/workspace python slice_quality/examples/compare_models_block1.py
Env: PRED_DIRS (comma-separated, default "mobilenet_pred,deeplab_pred").
"""
import json
import os
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.data_types import SegErrorType
from slice_quality.typed_error import decompose_dataset

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/data/voc2012/voc_crops"
OUT = "outputs/slice_quality/model_comparison.json"
DIRS = os.environ.get("PRED_DIRS", "mobilenet_pred,deeplab_pred").split(",")

names = [l.strip() for l in
         open(f"{DATA}/ImageSets/Segmentation/val.txt", encoding="utf-8") if l.strip()]

report = {}
for pred_dir in DIRS:
    print(f"\n=== {pred_dir} ===", flush=True)
    samples, _paths = build_voc_samples(DATA, CROPS, names, pred_subdir=pred_dir)
    print(f"images: {len(samples)}", flush=True)
    errors, reference = decompose_dataset(samples)
    tax = Counter(e.seg_subtype.value for e in errors)
    n_gt = sum(len(s["gts"]) for s in samples)
    n_pred = sum(len(s["preds"]) for s in samples)
    imgs_with_err = len({e.image_id for e in errors})
    sev = sorted(1.0 - e.iou for e in errors)
    report[pred_dir] = {
        "images": len(samples), "gt_instances": n_gt, "pred_instances": n_pred,
        "errors": len(errors), "reference_correct": len(reference),
        "images_with_errors": imgs_with_err,
        "errors_per_image": round(len(errors) / max(1, len(samples)), 2),
        "error_rate_vs_gt": round(len(errors) / max(1, n_gt), 3),
        "taxonomy": dict(tax.most_common()),
        "taxonomy_share": {k: round(v / len(errors), 3) for k, v in tax.most_common()},
        "median_severity_1_minus_iou": round(sev[len(sev) // 2], 3) if sev else None,
    }
    for k, v in report[pred_dir].items():
        print(f"  {k}: {v}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("\nsaved", OUT, flush=True)
print("COMPARE_DONE", flush=True)
