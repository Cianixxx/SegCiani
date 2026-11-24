"""Model error collection for detection (Section 4.1 / 3.2.1).

SliceLens operates on *error regions* that come from false negatives (missed
ground-truth objects) and false positives (spurious predictions). This module
turns raw per-image detections + ground truth into the :class:`Instance`
population that the pipeline ranks and analyses.

Each instance carries ``is_error`` (used by the slope-trend analysis) and an
:class:`ErrorType`. By default we build instances from the ground-truth objects
of the target class -- a GT object is a false negative if no correct-class
prediction overlaps it above ``iou_threshold`` -- and (optionally) from
unmatched predictions as false positives.

This keeps no third-party dependency: boxes are plain ``(xmin, ymin, xmax,
ymax)`` tuples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .data_types import ErrorType, Instance, Region

Box = tuple[float, float, float, float]


@dataclass
class Detection:
    box: Box
    label: str
    score: float = 1.0


@dataclass
class GroundTruth:
    box: Box
    label: str


@dataclass
class ImageAnnotation:
    image_path: str
    ground_truths: list[GroundTruth] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_detection_error_instances(
    images: Sequence[ImageAnnotation],
    target_label: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    include_false_positives: bool = True,
) -> list[Instance]:
    """Decompose detections vs. GT into error instances for ``target_label``.

    Returns one instance per ground-truth object of the target class (false
    negative if unmatched, otherwise a correct/true-positive instance) and,
    optionally, one instance per spurious prediction (false positive).
    """
    instances: list[Instance] = []
    counter = 0

    for img in images:
        preds = [
            d for d in img.detections
            if d.label == target_label and d.score >= score_threshold
        ]
        gts = [g for g in img.ground_truths if g.label == target_label]
        matched_pred = [False] * len(preds)

        # Ground-truth side: TP if matched by some prediction, else FN.
        for gt in gts:
            best_iou, best_j = 0.0, -1
            for j, pred in enumerate(preds):
                if matched_pred[j]:
                    continue
                ov = iou(gt.box, pred.box)
                if ov > best_iou:
                    best_iou, best_j = ov, j
            is_fn = best_iou < iou_threshold
            if not is_fn and best_j >= 0:
                matched_pred[best_j] = True
            instances.append(
                Instance(
                    instance_id=f"gt_{counter}",
                    image_path=img.image_path,
                    region=Region(bbox=gt.box),
                    label=target_label,
                    is_error=is_fn,
                    error_type=ErrorType.FALSE_NEGATIVE if is_fn else ErrorType.CORRECT,
                    model_score=None if best_j < 0 else preds[best_j].score,
                    metadata=dict(img.metadata),
                )
            )
            counter += 1

        # Prediction side: unmatched predictions are false positives.
        if include_false_positives:
            for j, pred in enumerate(preds):
                if matched_pred[j]:
                    continue
                instances.append(
                    Instance(
                        instance_id=f"pred_{counter}",
                        image_path=img.image_path,
                        region=Region(bbox=pred.box),
                        label=target_label,
                        is_error=True,
                        error_type=ErrorType.FALSE_POSITIVE,
                        model_score=pred.score,
                        metadata=dict(img.metadata),
                    )
                )
                counter += 1

    return instances


# ===========================================================================
# Instance segmentation
# ===========================================================================
# Same FN/FP/TP logic as detection, but matching uses *mask* IoU and the region
# grounded for the VLM is the mask's bounding box (the paper grounds segmentation
# with the bounding box of the mask or a point inside it, Section 3.2.1). The
# heavy mask data is discarded after matching -- only the derived bbox is kept --
# so the rest of the pipeline (verification + slope) is reused unchanged.
# numpy is imported lazily so the detection path stays dependency-free.


@dataclass
class SegGroundTruth:
    mask: Any  # binary HxW array (numpy)
    label: str


@dataclass
class SegDetection:
    mask: Any  # binary HxW array (numpy)
    label: str
    score: float = 1.0


@dataclass
class SegImageAnnotation:
    image_path: str
    ground_truths: list[SegGroundTruth] = field(default_factory=list)
    detections: list[SegDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def mask_iou(a: Any, b: Any) -> float:
    """Intersection-over-union of two binary masks (numpy arrays)."""
    import numpy as np

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def mask_to_bbox(mask: Any) -> tuple[float, float, float, float]:
    """Tight (xmin, ymin, xmax, ymax) bounding box of a binary mask."""
    import numpy as np

    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def build_segmentation_error_instances(
    images: Sequence[SegImageAnnotation],
    target_label: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
    include_false_positives: bool = True,
) -> list[Instance]:
    """Decompose predicted vs. GT masks into error instances for ``target_label``.

    Mirrors :func:`build_detection_error_instances` but with mask IoU. Accepts an
    iterable (can be a generator) so callers may decode masks one image at a time.
    """
    instances: list[Instance] = []
    counter = 0

    for img in images:
        preds = [
            d for d in img.detections
            if d.label == target_label and d.score >= score_threshold
        ]
        gts = [g for g in img.ground_truths if g.label == target_label]
        matched_pred = [False] * len(preds)

        for gt in gts:
            best_iou, best_j = 0.0, -1
            for j, pred in enumerate(preds):
                if matched_pred[j]:
                    continue
                ov = mask_iou(gt.mask, pred.mask)
                if ov > best_iou:
                    best_iou, best_j = ov, j
            is_fn = best_iou < iou_threshold
            if not is_fn and best_j >= 0:
                matched_pred[best_j] = True
            instances.append(
                Instance(
                    instance_id=f"gt_{counter}",
                    image_path=img.image_path,
                    region=Region(bbox=mask_to_bbox(gt.mask)),
                    label=target_label,
                    is_error=is_fn,
                    error_type=ErrorType.FALSE_NEGATIVE if is_fn else ErrorType.CORRECT,
                    model_score=None if best_j < 0 else preds[best_j].score,
                    metadata=dict(img.metadata),
                )
            )
            counter += 1

        if include_false_positives:
            for j, pred in enumerate(preds):
                if matched_pred[j]:
                    continue
                instances.append(
                    Instance(
                        instance_id=f"pred_{counter}",
                        image_path=img.image_path,
                        region=Region(bbox=mask_to_bbox(pred.mask)),
                        label=target_label,
                        is_error=True,
                        error_type=ErrorType.FALSE_POSITIVE,
                        model_score=pred.score,
                        metadata=dict(img.metadata),
                    )
                )
                counter += 1

    return instances
