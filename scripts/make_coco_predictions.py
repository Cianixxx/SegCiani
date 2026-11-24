"""Run a pretrained detector over COCO images and write COCO-format predictions.

This produces the ``--results`` file that ``examples/run_coco.py`` (and the
SliceLens COCO adapter) expects -- i.e. the model whose error slices you want to
discover. It uses a torchvision detector (Faster R-CNN by default) pretrained on
COCO, so torchvision's label ids already match COCO ``category_id``.

To keep a first run small and self-consistent it also writes a *filtered* GT
file containing only the images it actually processed (use ``--limit``), so you
don't need all 5000 val images on disk.

Setup (on the AIRLab container the PyTorch base already has torchvision)::

    pip install torch torchvision        # or: pip install -e ".[detector]"

Example::

    python scripts/make_coco_predictions.py \\
        --images-dir datasets/coco/val2017 \\
        --gt         datasets/coco/annotations/instances_val2017.json \\
        --out-results datasets/coco/preds/fasterrcnn.bbox.json \\
        --out-gt      datasets/coco/annotations/instances_subset.json \\
        --limit 200 --score-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrained detector -> COCO results JSON.")
    p.add_argument("--images-dir", required=True, help="folder with COCO images")
    p.add_argument("--gt", required=True, help="COCO instances_*.json (for image list)")
    p.add_argument("--out-results", required=True, help="output COCO results json")
    p.add_argument("--out-gt", default=None,
                   help="optional filtered GT json (only processed images)")
    p.add_argument("--model", default="fasterrcnn_resnet50_fpn",
                   choices=["fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn",
                            "fcos_resnet50_fpn", "maskrcnn_resnet50_fpn"],
                   help="use maskrcnn_resnet50_fpn to also output segmentation masks")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N images that exist (0 = all)")
    p.add_argument("--only-label", default=None,
                   help="only process images whose GT contains this class "
                        "(e.g. 'bicycle') -- gives a class-relevant subset")
    p.add_argument("--score-threshold", type=float, default=0.05,
                   help="drop detections below this score")
    return p.parse_args()


def load_detector(name: str, device: str):
    import torch  # noqa: F401
    from torchvision.models import detection as det

    builders = {
        "fasterrcnn_resnet50_fpn": (det.fasterrcnn_resnet50_fpn,
                                    det.FasterRCNN_ResNet50_FPN_Weights),
        "retinanet_resnet50_fpn": (det.retinanet_resnet50_fpn,
                                   det.RetinaNet_ResNet50_FPN_Weights),
        "fcos_resnet50_fpn": (det.fcos_resnet50_fpn,
                              det.FCOS_ResNet50_FPN_Weights),
        "maskrcnn_resnet50_fpn": (det.maskrcnn_resnet50_fpn,
                                  det.MaskRCNN_ResNet50_FPN_Weights),
    }
    build, weights_enum = builders[name]
    weights = weights_enum.DEFAULT
    model = build(weights=weights).eval().to(device)
    return model, weights.transforms()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from PIL import Image
    except ImportError:
        print("[FAIL] torch/torchvision/pillow required. "
              "Run: pip install torch torchvision pillow", file=sys.stderr)
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env ] device: {device}")

    with open(args.gt, encoding="utf-8") as f:
        gt = json.load(f)
    images = gt.get("images", [])
    print(f"[load] {len(images)} images in GT; model: {args.model}")

    # Optionally restrict to images whose GT contains the target class.
    allowed_ids = None
    if args.only_label:
        name_to_id = {c["name"]: c["id"] for c in gt.get("categories", [])}
        if args.only_label not in name_to_id:
            print(f"[FAIL] label {args.only_label!r} not in categories", file=sys.stderr)
            sys.exit(1)
        tid = name_to_id[args.only_label]
        allowed_ids = {a["image_id"] for a in gt.get("annotations", [])
                       if a.get("category_id") == tid}
        print(f"[load] {len(allowed_ids)} images contain '{args.only_label}'")

    model, preprocess = load_detector(args.model, device)

    # Segmentation models also emit masks -> we encode them as COCO RLE.
    mask_utils = None
    if "maskrcnn" in args.model:
        try:
            from pycocotools import mask as mask_utils  # type: ignore
        except ImportError:
            print("[FAIL] pycocotools needed for masks: pip install pycocotools",
                  file=sys.stderr)
            sys.exit(1)
        import numpy as np

    results: list[dict] = []
    processed_ids: set[int] = set()
    count = 0
    for img in images:
        if args.limit and count >= args.limit:
            break
        if allowed_ids is not None and img["id"] not in allowed_ids:
            continue
        path = os.path.join(args.images_dir, img["file_name"])
        if not os.path.exists(path):
            continue
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] cannot read {path}: {exc}", file=sys.stderr)
            continue

        batch = [preprocess(pil).to(device)]
        with torch.no_grad():
            out = model(batch)[0]

        boxes = out["boxes"].cpu().tolist()
        labels = out["labels"].cpu().tolist()
        scores = out["scores"].cpu().tolist()
        masks = out.get("masks")  # Mask R-CNN: (N, 1, H, W) float in [0, 1]
        for k, ((x0, y0, x1, y1), label, score) in enumerate(zip(boxes, labels, scores)):
            if score < args.score_threshold:
                continue
            entry = {
                "image_id": img["id"],
                "category_id": int(label),  # torchvision label == COCO category_id
                "bbox": [x0, y0, x1 - x0, y1 - y0],  # xywh
                "score": float(score),
            }
            if masks is not None and mask_utils is not None:
                bm = np.asfortranarray((masks[k, 0].cpu().numpy() >= 0.5).astype(np.uint8))
                rle = mask_utils.encode(bm)
                rle["counts"] = rle["counts"].decode("ascii")  # bytes -> str for JSON
                entry["segmentation"] = rle
            results.append(entry)
        processed_ids.add(img["id"])
        count += 1
        if count % 50 == 0:
            print(f"[run ] processed {count} images, {len(results)} detections")

    print(f"[done] {count} images, {len(results)} detections")

    out_results = Path(args.out_results)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_results.write_text(json.dumps(results))
    print(f"[save] results -> {out_results}")

    if args.out_gt:
        filtered = {
            "images": [i for i in images if i["id"] in processed_ids],
            "annotations": [a for a in gt.get("annotations", [])
                            if a["image_id"] in processed_ids],
            "categories": gt.get("categories", []),
        }
        out_gt = Path(args.out_gt)
        out_gt.parent.mkdir(parents=True, exist_ok=True)
        out_gt.write_text(json.dumps(filtered))
        print(f"[save] filtered GT ({len(filtered['images'])} images) -> {out_gt}")


if __name__ == "__main__":
    main()
