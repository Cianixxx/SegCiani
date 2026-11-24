"""Entry point for the slice_quality pipeline -- working COCO example.

Run (from the repo root):

    python scripts/run_slice_quality.py

Uses the local COCO instance-seg data (datasets/coco). Swap ``build_coco_samples``
for ``build_voc_samples`` to run on the server's VOC data instead. The only
model-dependent piece is the backbone extractor (DINOv2 by default).
"""

from __future__ import annotations

import os

from slice_quality import run_pipeline
from slice_quality.adapters.coco_samples import build_coco_samples, make_mask_lookup
from slice_quality.config import SliceQualityConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(REPO, "datasets", "coco", "annotations", "instances_val2017.json")
PRED = os.path.join(REPO, "datasets", "coco", "preds", "maskrcnn.seg.json")
IMG = os.path.join(REPO, "datasets", "coco", "val2017")


def main() -> None:
    cfg = SliceQualityConfig()

    # --- BLOCK 1 input: COCO samples (instances = native COCO masks) --------
    samples, image_paths = build_coco_samples(
        GT, PRED, IMG,
        categories=None,            # or e.g. ["person", "bicycle"]
        score_threshold=0.5,
        limit_images=0,             # 0 = all
    )
    mask_lookup = make_mask_lookup(samples)

    # --- BLOCK 2 backbone: DINOv2 (server-side; needs torch + the images) ---
    # On a machine without torch/GPU/internet, replace with any extractor that
    # implements .feature_map(image_id) -> (C, H, W). See extractors.py.
    from slice_quality.extractors import DINOv2Extractor
    extractor = DINOv2Extractor(image_paths, model_name="dinov2_vits14")

    # --- OPTIONAL: SliceLens language-branch slices -------------------------
    # language_slices = build_language_slices(...)   # see README; runs without it

    result = run_pipeline(
        samples=samples,
        extractor=extractor,
        mask_lookup=mask_lookup,
        language_slices=None,
        cfg=cfg,
    )

    print(f"kept {len(result.kept)} slices, residual error {result.residual_error:.3f}")
    for ks in sorted(result.kept, key=lambda k: -k.advice.priority):
        print(ks.explanation.sentence)
        print("  ->", ks.advice.text)


if __name__ == "__main__":
    main()
