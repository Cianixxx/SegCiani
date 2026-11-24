"""VOC semantic-seg -> slice_quality `samples` (instances = objects, or components).

TODO 1 of scripts/run_slice_quality.py, made concrete for the user's VOC layout
(same paths as slicelens/adapters/voc_seg.py).

VOC is *semantic* segmentation (per-class masks). Predictions are turned into
instance-like units by splitting each per-class binary mask into CONNECTED
COMPONENTS -- exactly Beyond IoU's S(.) operator; a semantic model has nothing
better to offer.

Ground truth has something better: VOC ships ``SegmentationObject``, one id per
object, for every image of the segmentation subset. With ``gt_instances="object"``
(env SQ_GT_INSTANCES=object) a GT instance is an OBJECT: an object cut in two by
an occluder is one instance with a non-connected mask, two touching objects of
one class are two instances. Measured on val through the pipeline's own substrate
(520 crop, >= 200 px): 699 of 3085 objects are cut into >= 2 components (35% of
the images), 118 components hold >= 2 objects; under the component rule 170 of
the 559 merge blobs of the revised run were one occluded object, not several,
and 684 of 2207 "missed" GT instances were parts of an object another part of
which had been matched. Each GT dict then carries ``obj_id`` and ``n_parts``.
Images without an object mask (the synthetic set) fall back to components, so
the option is safe to leave on. typed_error.py works on masks and does not care
whether they are connected.

Outputs
-------
* ``samples``      : list[{"image_id", "preds":[{mask,cls,bbox}], "gts":[...]}].
* ``image_paths``  : {image_id -> cropped image path} (for the DINOv2 extractor).
* ``make_mask_lookup(samples)`` : id -> HxW mask (for BLOCK 2 pooling); parses the
  instance ids "img#pI" / "img#gJ" produced by typed_error.decompose_image.
"""

from __future__ import annotations

import os

import numpy as np

from slicelens.adapters.voc_seg import VOC_CLASSES, _resize_center_crop

from .common import make_mask_lookup  # re-exported for callers

try:
    from scipy.ndimage import label as cc_label
except Exception:  # pragma: no cover
    cc_label = None

__all__ = ["build_voc_samples", "make_mask_lookup"]


def _components(binary: np.ndarray, cls: str, min_pixels: int) -> list[dict]:
    """Split a binary mask into connected components -> instance dicts."""
    if cc_label is None:
        # fallback: treat the whole class mask as one instance
        comps = [(binary, binary.sum())]
    else:
        lab, n = cc_label(binary)
        comps = [(lab == k, int((lab == k).sum())) for k in range(1, n + 1)]
    out = []
    for mask, px in comps:
        if px < min_pixels:
            continue
        ys, xs = np.where(mask)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        out.append({"mask": mask, "cls": cls, "bbox": bbox})
    return out


def _objects(gt: np.ndarray, obj: np.ndarray, min_pixels: int) -> list[dict]:
    """One instance per object id of a VOC SegmentationObject map.

    An object's class is the majority class label under its pixels (the two
    maps share the void boundary, so this is exact up to a few stray pixels);
    the mask is the object's pixels of that class. ``n_parts`` counts its
    connected components: >= 2 means the object is cut -- by an occluder in
    front of it, or by its own thin structure -- which is precisely what the
    component rule mistook for several objects.
    """
    out = []
    for oid in sorted(int(v) for v in np.unique(obj) if v not in (0, 255)):
        pix = obj == oid
        labels = gt[pix]
        labels = labels[(labels != 0) & (labels != 255)]
        if labels.size == 0:
            continue
        c = int(np.bincount(labels).argmax())
        if c >= len(VOC_CLASSES):
            continue
        mask = pix & (gt == c)
        px = int(mask.sum())
        if px < min_pixels:
            continue
        ys, xs = np.where(mask)
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        n_parts = int(cc_label(mask)[1]) if cc_label is not None else 1
        out.append({"mask": mask, "cls": VOC_CLASSES[c], "bbox": bbox,
                    "obj_id": oid, "n_parts": n_parts})
    return out


def build_voc_samples(
    dataset_root: str,
    crops_dir: str,
    names: list[str],
    *,
    gt_subdir: str = "SegmentationClass",
    obj_subdir: str = "SegmentationObject",
    pred_subdir: str = "mobilenet_pred",
    images_subdir: str = "JPEGImages",
    pred_template: str = "mask_pr_{name}.png",
    crop: int = 520,
    min_pixels: int = 200,
    gt_instances: str | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Return (samples, image_paths) for the given image stems.

    ``gt_instances``: "components" (historical) or "object" (VOC object masks,
    falling back to components where the object map is missing). None reads
    SQ_GT_INSTANCES from the environment, default "components".
    """
    from PIL import Image

    if gt_instances is None:
        gt_instances = os.environ.get("SQ_GT_INSTANCES", "components")
    os.makedirs(crops_dir, exist_ok=True)
    samples: list[dict] = []
    image_paths: dict[str, str] = {}
    n_obj = n_cc = 0

    for stem in names:
        img_p = os.path.join(dataset_root, images_subdir, f"{stem}.jpg")
        gt_p = os.path.join(dataset_root, gt_subdir, f"{stem}.png")
        obj_p = os.path.join(dataset_root, obj_subdir, f"{stem}.png")
        pred_p = os.path.join(dataset_root, pred_subdir, pred_template.format(name=stem))
        if not (os.path.exists(img_p) and os.path.exists(gt_p) and os.path.exists(pred_p)):
            continue

        gt = np.array(_resize_center_crop(Image.open(gt_p), crop, Image.NEAREST))
        pred = np.array(_resize_center_crop(Image.open(pred_p).convert("RGB"),
                                            crop, Image.NEAREST))[:, :, 0]
        obj = None
        if gt_instances == "object" and os.path.exists(obj_p):
            obj = np.array(_resize_center_crop(Image.open(obj_p), crop, Image.NEAREST))

        crop_img_path = os.path.join(crops_dir, f"{stem}.jpg")
        if not os.path.exists(crop_img_path):
            _resize_center_crop(Image.open(img_p).convert("RGB"), crop,
                                Image.LANCZOS).save(crop_img_path, quality=90)
        image_paths[stem] = crop_img_path

        classes = (set(np.unique(gt)) | set(np.unique(pred))) - {0, 255}
        preds, gts = [], []
        for c in sorted(classes):
            if c >= len(VOC_CLASSES):
                continue
            name = VOC_CLASSES[c]
            if obj is None:
                gts.extend(_components(gt == c, name, min_pixels))
            preds.extend(_components(pred == c, name, min_pixels))
        if obj is not None:
            gts = _objects(gt, obj, min_pixels)
            n_obj += 1
        else:
            n_cc += 1
        samples.append({"image_id": stem, "preds": preds, "gts": gts,
                        "gt_instances": "object" if obj is not None else "components",
                        # VOC's "don't care" band (255) around every object: a
                        # predicted sliver lying on it is neither right nor wrong
                        "void": gt == 255})

    if gt_instances == "object":
        print(f"[voc_seg_samples] GT instances: objects on {n_obj} images"
              + (f", components on {n_cc} (no object mask)" if n_cc else ""), flush=True)
    return samples, image_paths
