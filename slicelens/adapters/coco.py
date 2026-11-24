"""COCO adapter -> SliceLens instances.

Loads a COCO ground-truth annotation file plus a COCO-format **detection
results** file and decomposes them into per-instance errors (false negatives /
false positives / correct) for a target class, ready for the SliceLens pipeline.

Formats (standard COCO):

* GT file (e.g. ``instances_val2017.json``)::

    {
      "images":      [{"id": 1, "file_name": "000000001.jpg", ...}, ...],
      "annotations": [{"image_id": 1, "category_id": 2,
                       "bbox": [x, y, w, h], "iscrowd": 0}, ...],
      "categories":  [{"id": 2, "name": "bicycle"}, ...]
    }

* Detection results (what ``pycocotools`` / detectors emit)::

    [{"image_id": 1, "category_id": 2, "bbox": [x, y, w, h], "score": 0.9}, ...]

COCO boxes are ``[x, y, width, height]`` and are converted to
``(xmin, ymin, xmax, ymax)``. Pure standard library -- no pycocotools needed.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Sequence

from ..errors import (
    Detection,
    GroundTruth,
    ImageAnnotation,
    SegDetection,
    SegGroundTruth,
    SegImageAnnotation,
    build_detection_error_instances,
    build_segmentation_error_instances,
)
from ..data_types import Instance


def _xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (float(x), float(y), float(x) + float(w), float(y) + float(h))


def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_coco_images(
    gt_json_path: str,
    results_json_path: str,
    image_root: str,
    target_label: str,
    score_threshold: float = 0.0,
    ignore_crowd: bool = True,
) -> list[ImageAnnotation]:
    """Build :class:`ImageAnnotation` objects for ``target_label`` from COCO files.

    Only images that contain at least one ground-truth object or one detection
    of the target class are returned.
    """
    gt = _load_json(gt_json_path)

    # category name <-> id
    name_to_id = {c["name"]: c["id"] for c in gt.get("categories", [])}
    if target_label not in name_to_id:
        available = ", ".join(sorted(name_to_id)[:20])
        raise ValueError(
            f"Label {target_label!r} not in COCO categories. Available: {available} ..."
        )
    target_id = name_to_id[target_label]

    # image id -> file_name and metadata
    images = {img["id"]: img for img in gt.get("images", [])}

    # GT annotations grouped by image (target class only)
    gts_by_image: dict[int, list[GroundTruth]] = defaultdict(list)
    for ann in gt.get("annotations", []):
        if ann.get("category_id") != target_id:
            continue
        if ignore_crowd and ann.get("iscrowd", 0) == 1:
            continue
        gts_by_image[ann["image_id"]].append(
            GroundTruth(box=_xywh_to_xyxy(ann["bbox"]), label=target_label)
        )

    # Detection results grouped by image (target class only)
    results = _load_json(results_json_path)
    dets_by_image: dict[int, list[Detection]] = defaultdict(list)
    for det in results:
        if det.get("category_id") != target_id:
            continue
        score = float(det.get("score", 1.0))
        if score < score_threshold:
            continue
        dets_by_image[det["image_id"]].append(
            Detection(box=_xywh_to_xyxy(det["bbox"]), label=target_label, score=score)
        )

    image_ids = set(gts_by_image) | set(dets_by_image)
    annotations: list[ImageAnnotation] = []
    for image_id in sorted(image_ids):
        img = images.get(image_id, {})
        file_name = img.get("file_name", f"{image_id}.jpg")
        annotations.append(
            ImageAnnotation(
                image_path=os.path.join(image_root, file_name),
                ground_truths=gts_by_image.get(image_id, []),
                detections=dets_by_image.get(image_id, []),
                metadata={"image_id": image_id, "file_name": file_name},
            )
        )
    return annotations


def load_coco_detection_errors(
    gt_json_path: str,
    results_json_path: str,
    image_root: str,
    target_label: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    include_false_positives: bool = True,
    ignore_crowd: bool = True,
) -> list[Instance]:
    """One-call COCO -> SliceLens ``Instance`` list for ``target_label``.

    Combines :func:`load_coco_images` with the FN/FP decomposition of
    :func:`slicelens.errors.build_detection_error_instances`.
    """
    images = load_coco_images(
        gt_json_path,
        results_json_path,
        image_root,
        target_label,
        score_threshold=score_threshold,
        ignore_crowd=ignore_crowd,
    )
    return build_detection_error_instances(
        images,
        target_label=target_label,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        include_false_positives=include_false_positives,
    )


# ---------------------------------------------------------------------------
# Instance segmentation (uses pycocotools to decode COCO masks)
# ---------------------------------------------------------------------------
def _ann_to_mask(seg, height, width, mask_utils):
    """Decode a COCO ``segmentation`` (polygon / RLE) into a binary mask."""
    import numpy as np

    if isinstance(seg, list):  # polygon(s)
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict):
        if isinstance(seg.get("counts"), list):  # uncompressed RLE
            rle = mask_utils.frPyObjects(seg, height, width)
        else:  # compressed RLE; counts may be str (JSON) or bytes
            counts = seg["counts"]
            rle = {"size": seg["size"],
                   "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
    else:
        return None
    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)


def load_coco_segmentation_errors(
    gt_json_path: str,
    results_json_path: str,
    image_root: str,
    target_label: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    include_false_positives: bool = True,
    ignore_crowd: bool = True,
) -> list[Instance]:
    """COCO instance-segmentation -> SliceLens ``Instance`` list for ``target_label``.

    Reads COCO GT (``segmentation`` polygons/RLE) and a COCO instance-segmentation
    results file (``[{image_id, category_id, segmentation(RLE), score}, ...]``),
    matches predicted vs. GT masks by **mask IoU**, and grounds each instance with
    the mask's bounding box. Requires ``pycocotools``.
    """
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError(
            "pycocotools is required for segmentation. Install the 'segmentation' "
            "extra: pip install -e \".[segmentation]\"."
        ) from exc

    gt = _load_json(gt_json_path)
    name_to_id = {c["name"]: c["id"] for c in gt.get("categories", [])}
    if target_label not in name_to_id:
        available = ", ".join(sorted(name_to_id)[:20])
        raise ValueError(f"Label {target_label!r} not in COCO categories: {available} ...")
    target_id = name_to_id[target_label]
    images = {img["id"]: img for img in gt.get("images", [])}

    gts_by_image: dict[int, list] = defaultdict(list)
    for ann in gt.get("annotations", []):
        if ann.get("category_id") != target_id:
            continue
        if ignore_crowd and ann.get("iscrowd", 0) == 1:
            continue
        if ann.get("segmentation") is None:
            continue
        gts_by_image[ann["image_id"]].append(ann)

    results = _load_json(results_json_path)
    dets_by_image: dict[int, list] = defaultdict(list)
    for det in results:
        if det.get("category_id") != target_id or det.get("segmentation") is None:
            continue
        if float(det.get("score", 1.0)) < score_threshold:
            continue
        dets_by_image[det["image_id"]].append(det)

    image_ids = sorted(set(gts_by_image) | set(dets_by_image))

    def _stream():
        for image_id in image_ids:
            img = images.get(image_id, {})
            h, w = img.get("height"), img.get("width")
            file_name = img.get("file_name", f"{image_id}.jpg")
            seg_gts = []
            for ann in gts_by_image.get(image_id, []):
                m = _ann_to_mask(ann["segmentation"], h, w, mask_utils)
                if m is not None:
                    seg_gts.append(SegGroundTruth(mask=m, label=target_label))
            seg_dets = []
            for det in dets_by_image.get(image_id, []):
                m = _ann_to_mask(det["segmentation"], h, w, mask_utils)
                if m is not None:
                    seg_dets.append(SegDetection(mask=m, label=target_label,
                                                 score=float(det.get("score", 1.0))))
            yield SegImageAnnotation(
                image_path=os.path.join(image_root, file_name),
                ground_truths=seg_gts, detections=seg_dets,
                metadata={"image_id": image_id, "file_name": file_name},
            )

    return build_segmentation_error_instances(
        _stream(),
        target_label=target_label,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        include_false_positives=include_false_positives,
    )
