"""Tests for the COCO adapter using a tiny inline dataset."""

import json

from slicelens.adapters import load_coco_detection_errors, load_coco_images
from slicelens.data_types import ErrorType


def _write_coco(tmp_path):
    gt = {
        "images": [
            {"id": 10, "file_name": "a.jpg"},
            {"id": 11, "file_name": "b.jpg"},
        ],
        "categories": [
            {"id": 1, "name": "bicycle"},
            {"id": 2, "name": "car"},
        ],
        "annotations": [
            # img10: two bicycles
            {"image_id": 10, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
            {"image_id": 10, "category_id": 1, "bbox": [100, 100, 10, 10], "iscrowd": 0},
            # img11: one bicycle + one car (car must be ignored for target=bicycle)
            {"image_id": 11, "category_id": 1, "bbox": [50, 50, 20, 20], "iscrowd": 0},
            {"image_id": 11, "category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 0},
        ],
    }
    results = [
        # img10: matches the first bicycle (TP); the second is missed (FN)
        {"image_id": 10, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        # img11: matches the bicycle (TP) + one spurious bicycle (FP)
        {"image_id": 11, "category_id": 1, "bbox": [50, 50, 20, 20], "score": 0.8},
        {"image_id": 11, "category_id": 1, "bbox": [200, 200, 10, 10], "score": 0.7},
    ]
    gt_path = tmp_path / "gt.json"
    res_path = tmp_path / "res.json"
    gt_path.write_text(json.dumps(gt))
    res_path.write_text(json.dumps(results))
    return str(gt_path), str(res_path)


def test_loads_only_target_images(tmp_path):
    gt_path, res_path = _write_coco(tmp_path)
    images = load_coco_images(gt_path, res_path, "/imgs", "bicycle")
    assert len(images) == 2
    # bbox converted from xywh to xyxy
    assert images[0].ground_truths[0].box == (0.0, 0.0, 10.0, 10.0)
    # image path joined with the root
    assert images[0].image_path.replace("\\", "/").endswith("/imgs/a.jpg")


def test_error_decomposition(tmp_path):
    gt_path, res_path = _write_coco(tmp_path)
    instances = load_coco_detection_errors(
        gt_path, res_path, "/imgs", "bicycle", iou_threshold=0.5
    )
    by_type = {}
    for inst in instances:
        by_type.setdefault(inst.error_type, []).append(inst)

    # 2 correct (TP), 1 false negative, 1 false positive.
    assert len(by_type.get(ErrorType.CORRECT, [])) == 2
    assert len(by_type.get(ErrorType.FALSE_NEGATIVE, [])) == 1
    assert len(by_type.get(ErrorType.FALSE_POSITIVE, [])) == 1
    assert sum(i.is_error for i in instances) == 2


def test_car_category_is_ignored(tmp_path):
    gt_path, res_path = _write_coco(tmp_path)
    instances = load_coco_detection_errors(gt_path, res_path, "/imgs", "bicycle")
    # The car GT must not appear among bicycle instances.
    assert all(i.label == "bicycle" for i in instances)
    # 3 bicycle GT + 1 spurious bicycle pred = 4 instances.
    assert len(instances) == 4


def test_unknown_label_raises(tmp_path):
    gt_path, res_path = _write_coco(tmp_path)
    import pytest

    with pytest.raises(ValueError):
        load_coco_detection_errors(gt_path, res_path, "/imgs", "airplane")
