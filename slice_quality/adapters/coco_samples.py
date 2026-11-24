"""COCO instance-seg -> slice_quality `samples` (instances are native).

TODO 1 of scripts/run_slice_quality.py, concrete for the user's COCO layout:
  datasets/coco/annotations/instances_val2017.json   (GT)
  datasets/coco/preds/maskrcnn.seg.json              (Mask R-CNN RLE results)
  datasets/coco/val2017/                              (images)

COCO masks are already per-instance, so no connected-components step is needed.
typed_error.py then assigns boundary/extent/segment/FP/FN/split/merge.

Outputs (image_paths + make_mask_lookup) mirror the VOC adapter.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from .common import make_mask_lookup  # re-exported for callers

__all__ = ["build_coco_samples", "make_mask_lookup"]


def _decode(seg, h, w, mask_utils) -> np.ndarray:
    if isinstance(seg, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, h, w)
    else:
        counts = seg["counts"]
        rle = {"size": seg["size"],
               "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    m = mask_utils.decode(rle)
    return (m[:, :, 0] if m.ndim == 3 else m).astype(bool)


def _bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def build_coco_samples(
    gt_json: str,
    pred_json: str,
    image_root: str,
    *,
    categories: list[str] | None = None,   # None = all categories
    score_threshold: float = 0.5,
    min_pixels: int = 100,
    ignore_crowd: bool = True,
    limit_images: int = 0,
) -> tuple[list[dict], dict[str, str]]:
    """Return (samples, image_paths) from COCO GT + an instance-seg results file."""
    from pycocotools import mask as mask_utils

    gt = json.load(open(gt_json, encoding="utf-8"))
    id_to_name = {c["id"]: c["name"] for c in gt.get("categories", [])}
    keep_ids = (None if categories is None
                else {cid for cid, n in id_to_name.items() if n in set(categories)})
    images = {img["id"]: img for img in gt["images"]}

    gts_by_img: dict[int, list] = defaultdict(list)
    for ann in gt.get("annotations", []):
        if keep_ids is not None and ann["category_id"] not in keep_ids:
            continue
        if ignore_crowd and ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("segmentation") is None:
            continue
        gts_by_img[ann["image_id"]].append(ann)

    preds_by_img: dict[int, list] = defaultdict(list)
    for det in json.load(open(pred_json, encoding="utf-8")):
        if keep_ids is not None and det["category_id"] not in keep_ids:
            continue
        if float(det.get("score", 1.0)) < score_threshold:
            continue
        preds_by_img[det["image_id"]].append(det)

    image_ids = sorted(set(gts_by_img) | set(preds_by_img))
    if limit_images:
        image_ids = image_ids[:limit_images]

    samples: list[dict] = []
    image_paths: dict[str, str] = {}
    for image_id in image_ids:
        img = images.get(image_id, {})
        h, w = img.get("height"), img.get("width")
        file_name = img.get("file_name", f"{image_id:012d}.jpg")
        sid = str(image_id)

        gts = []
        for ann in gts_by_img.get(image_id, []):
            m = _decode(ann["segmentation"], h, w, mask_utils)
            if m.sum() >= min_pixels:
                gts.append({"mask": m, "cls": id_to_name[ann["category_id"]], "bbox": _bbox(m)})
        preds = []
        for det in preds_by_img.get(image_id, []):
            m = _decode(det["segmentation"], h, w, mask_utils)
            if m.sum() >= min_pixels:
                preds.append({"mask": m, "cls": id_to_name[det["category_id"]],
                              "bbox": _bbox(m), "score": float(det.get("score", 1.0))})

        if not preds and not gts:
            continue
        samples.append({"image_id": sid, "preds": preds, "gts": gts})
        image_paths[sid] = os.path.join(image_root, file_name)

    return samples, image_paths
