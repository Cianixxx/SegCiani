"""Tests for instance-segmentation error collection (mask IoU).

Uses small binary numpy masks (no pycocotools needed -- that's only used by the
COCO adapter to decode RLE/polygons). Skips if numpy is unavailable.
"""

import pytest

np = pytest.importorskip("numpy")

from slicelens.data_types import ErrorType
from slicelens.errors import (
    SegDetection,
    SegGroundTruth,
    SegImageAnnotation,
    build_segmentation_error_instances,
    mask_iou,
    mask_to_bbox,
)


def _square(top, left, size, h=64, w=64):
    m = np.zeros((h, w), dtype=bool)
    m[top:top + size, left:left + size] = True
    return m


def test_mask_iou_basic():
    a = _square(0, 0, 10)
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, _square(20, 20, 10)) == 0.0
    # half overlap
    b = _square(0, 5, 10)  # shifted right by 5 -> overlap 10x5=50, union 150
    assert abs(mask_iou(a, b) - 50 / 150) < 1e-9


def test_mask_to_bbox():
    m = _square(8, 12, 10)  # rows 8..17, cols 12..21
    assert mask_to_bbox(m) == (12.0, 8.0, 22.0, 18.0)


def test_segmentation_fn_fp_tp():
    gt_match = _square(0, 0, 20)
    pred_match = _square(1, 1, 20)         # high IoU with gt_match -> TP
    gt_missed = _square(40, 40, 15)        # no overlapping pred -> FN
    pred_spurious = _square(0, 40, 12)     # matches no GT -> FP

    img = SegImageAnnotation(
        image_path="x.jpg",
        ground_truths=[SegGroundTruth(gt_match, "bicycle"),
                       SegGroundTruth(gt_missed, "bicycle")],
        detections=[SegDetection(pred_match, "bicycle", 0.9),
                    SegDetection(pred_spurious, "bicycle", 0.8)],
    )
    inst = build_segmentation_error_instances([img], "bicycle", iou_threshold=0.5)
    by = {}
    for i in inst:
        by.setdefault(i.error_type, []).append(i)
    assert len(by.get(ErrorType.CORRECT, [])) == 1
    assert len(by.get(ErrorType.FALSE_NEGATIVE, [])) == 1
    assert len(by.get(ErrorType.FALSE_POSITIVE, [])) == 1
    assert sum(i.is_error for i in inst) == 2


def test_segmentation_region_is_mask_bbox():
    gt = _square(5, 5, 10)  # rows 5..14, cols 5..14
    img = SegImageAnnotation(
        image_path="x.jpg",
        ground_truths=[SegGroundTruth(gt, "bicycle")],
        detections=[],
    )
    inst = build_segmentation_error_instances([img], "bicycle")
    assert len(inst) == 1
    assert inst[0].region.bbox == (5.0, 5.0, 15.0, 15.0)
    assert inst[0].is_error  # unmatched GT -> FN


def test_other_class_ignored():
    img = SegImageAnnotation(
        image_path="x.jpg",
        ground_truths=[SegGroundTruth(_square(0, 0, 10), "car")],
        detections=[SegDetection(_square(0, 0, 10), "car", 0.9)],
    )
    assert build_segmentation_error_instances([img], "bicycle") == []
