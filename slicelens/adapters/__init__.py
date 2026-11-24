"""Dataset adapters that turn detector outputs into SliceLens ``Instance``s.

Currently provides a COCO adapter (the format the paper builds FeSD on, and the
de-facto standard exportable by Detectron2, mmdetection, YOLO, DETR, ...).
"""

from .coco import (
    load_coco_detection_errors,
    load_coco_images,
    load_coco_segmentation_errors,
)
from .voc_seg import (
    VOC_CLASSES,
    class_distribution,
    load_voc_seg_instances,
)

__all__ = [
    "load_coco_detection_errors",
    "load_coco_segmentation_errors",
    "load_coco_images",
    "load_voc_seg_instances",
    "class_distribution",
    "VOC_CLASSES",
]
