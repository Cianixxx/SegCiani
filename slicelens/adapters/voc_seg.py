"""Pascal-VOC semantic-segmentation -> SliceLens instances.

Builds error instances by comparing a model's predicted mask to the GT mask,
per (image, class). An instance is an *error* when the predicted mask for that
class poorly overlaps the GT region (mask IoU below a threshold = missed /
under-segmented / bad boundary), or when a class is predicted with no GT
(hallucinated = false positive). The region grounded for the VLM is the object's
bounding box, and the VLM sees the original image (resize+center-cropped to the
model's input size so boxes line up).

Dataset layout (the user's `pascal_vlm`):
  SegmentationClass/<name>.png   GT mask  (PIL 'P', class indices; original size)
  <pred_dir>/mask_pr_<name>.png  predicted mask (RGB, R=G=B=class index; 520x520)
  JPEGImages/<name>.jpg          original image
  voc_patch_matches.json         {img_idx: filename}
  answer_gts.jsonl               {img_idx, content}  (the 80 annotated samples)

Requires numpy + pillow (imported lazily).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

from ..data_types import ErrorType, Instance, Region

# Pascal VOC class index -> name (0 = background, 255 = void/ignore).
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def _resize_center_crop(img, size, resample):
    """Resize so the short side == size, then center-crop to size x size."""
    w, h = img.size
    if w < h:
        new_w, new_h = size, int(round(h * size / w))
    else:
        new_h, new_w = size, int(round(w * size / h))
    img = img.resize((new_w, new_h), resample)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _bbox_from_mask(mask):
    import numpy as np

    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def load_voc_seg_instances(
    dataset_root: str,
    crops_dir: str,
    *,
    gt_subdir: str = "SegmentationClass",
    pred_subdir: str = "mobilenet_pred",
    images_subdir: str = "JPEGImages",
    pred_template: str = "mask_pr_{name}.png",
    matches_file: str = "voc_patch_matches.json",
    gt_desc_file: str = "answer_gts.jsonl",
    names_file: str | None = None,
    crop: int = 520,
    iou_threshold: float = 0.5,
    min_pixels: int = 200,
    only_label: str | None = None,
    limit: int = 0,
) -> list[Instance]:
    """Return SliceLens ``Instance``s for the annotated VOC samples.

    Center-cropped original images are written to ``crops_dir`` and used as each
    instance's ``image_path`` (so the VLM sees the same crop the masks align to).
    """
    import numpy as np
    from PIL import Image

    root = dataset_root
    if names_file:
        # Full-dataset mode: process every image listed in a split file (e.g. the
        # VOC2012 Segmentation trainval.txt) -- no answer_gts/matches filtering.
        path = names_file if os.path.isabs(names_file) else os.path.join(root, names_file)
        with open(path, encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        items = [(i, f"{n}.jpg") for i, n in enumerate(names)]
    else:
        matches = json.load(open(os.path.join(root, matches_file), encoding="utf-8"))
        gt_idxs = set()
        for line in open(os.path.join(root, gt_desc_file), encoding="utf-8"):
            line = line.strip()
            if line:
                gt_idxs.add(json.loads(line)["img_idx"])
        # samples = annotated ones, ordered by img_idx
        items = sorted(((int(k), v) for k, v in matches.items() if int(k) in gt_idxs),
                       key=lambda x: x[0])
    if limit:
        items = items[:limit]

    os.makedirs(crops_dir, exist_ok=True)
    name_to_id = {n: i for i, n in enumerate(VOC_CLASSES)}
    target_id = name_to_id.get(only_label) if only_label else None

    instances: list[Instance] = []
    counter = 0
    for img_idx, filename in items:
        stem = os.path.splitext(filename)[0]
        img_p = os.path.join(root, images_subdir, f"{stem}.jpg")
        gt_p = os.path.join(root, gt_subdir, f"{stem}.png")
        pred_p = os.path.join(root, pred_subdir, pred_template.format(name=stem))
        if not (os.path.exists(img_p) and os.path.exists(gt_p) and os.path.exists(pred_p)):
            continue

        # Align everything to crop x crop (NEAREST for masks, LANCZOS for image).
        gt_img = _resize_center_crop(Image.open(gt_p), crop, Image.NEAREST)
        gt = np.array(gt_img)  # class indices (P mode)
        pred = np.array(_resize_center_crop(Image.open(pred_p).convert("RGB"),
                                            crop, Image.NEAREST))[:, :, 0]  # R = class idx
        crop_img_path = os.path.join(crops_dir, f"{stem}.jpg")
        if not os.path.exists(crop_img_path):
            _resize_center_crop(Image.open(img_p).convert("RGB"), crop,
                                Image.LANCZOS).save(crop_img_path, quality=90)

        classes = (set(np.unique(gt)) | set(np.unique(pred))) - {0, 255}
        for c in sorted(classes):
            if c >= len(VOC_CLASSES):
                continue
            if target_id is not None and c != target_id:
                continue
            gt_c = gt == c
            pred_c = pred == c
            gt_px, pred_px = int(gt_c.sum()), int(pred_c.sum())
            if max(gt_px, pred_px) < min_pixels:
                continue
            inter = int((gt_c & pred_c).sum())
            union = int((gt_c | pred_c).sum())
            iou = inter / union if union else 0.0

            if gt_px >= min_pixels:  # a real GT region of this class
                is_err = iou < iou_threshold
                bbox = _bbox_from_mask(gt_c)
                etype = ErrorType.FALSE_NEGATIVE if is_err else ErrorType.CORRECT
            else:  # predicted but (almost) no GT -> hallucinated
                is_err = True
                bbox = _bbox_from_mask(pred_c)
                etype = ErrorType.FALSE_POSITIVE
            if bbox is None:
                continue
            instances.append(Instance(
                instance_id=f"{stem}_{VOC_CLASSES[c]}_{counter}",
                image_path=crop_img_path,
                region=Region(bbox=bbox),
                label=VOC_CLASSES[c],
                is_error=is_err,
                error_type=etype,
                metadata={"img_idx": img_idx, "name": stem, "iou": round(iou, 3),
                          "gt_px": gt_px, "pred_px": pred_px},
            ))
            counter += 1
    return instances


def class_distribution(instances: Sequence[Instance]) -> dict:
    """Per-class (#instances, #errors) summary -- handy to pick a target class."""
    dist: dict[str, list[int]] = {}
    for inst in instances:
        d = dist.setdefault(inst.label, [0, 0])
        d[0] += 1
        d[1] += int(inst.is_error)
    return {k: {"n": v[0], "errors": v[1]} for k, v in
            sorted(dist.items(), key=lambda x: -x[1][0])}
