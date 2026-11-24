"""Full pipeline on the server, GPU: real DINOv2 embeddings on real VOC masks.

BLOCK 1 (typed error, masks) + BLOCK 2 (DINOv2 backbone, GPU) + scoring + report.
Run inside segciani:dev with --gpus device=5.
"""
import glob
import os
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples, make_mask_lookup
from slice_quality.typed_error import decompose_dataset
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.report import to_markdown, to_json

DATA = "/data/dataset"
SEG = os.path.join(DATA, "SegmentationClass")
names = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(SEG, "*.png")))
print(f"VOC images: {len(names)}")

samples, image_paths = build_voc_samples(
    DATA, "/tmp/voc_crops", names,
    pred_subdir="mobilenet_pred", pred_template="mask_pr_{name}.png",
    crop=520, min_pixels=200,
)
cfg = SliceQualityConfig()
errors, reference = decompose_dataset(samples, cfg.typed_error)
print(f"errors={len(errors)} reference={len(reference)}")
print("TAXONOMY:", dict(Counter(e.seg_subtype.value for e in errors)))

# --- BLOCK 2: real DINOv2 embeddings on GPU ------------------------------
import torch
print("CUDA available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
extractor = DINOv2Extractor(image_paths, model_name="dinov2_vits14")
mask_lookup = make_mask_lookup(samples)
attach_embeddings(errors + reference, extractor, mask_lookup)
n_emb = sum(1 for e in errors if e.embedding is not None)
print(f"DINOv2 embeddings attached: {n_emb}/{len(errors)} errors "
      f"(dim={errors[0].embedding.shape[0] if n_emb else 0})")

cfg.cluster.n_slices = 10
cfg.def1.n_min_frac = 0.02
cfg.significance.n_boot = 300
res = run_from_instances(errors, reference, cfg=cfg, max_peeling_rounds=4,
                         residual_stop=0.05)
to_json(res, "outputs/slice_quality/voc_dinov2_slices.json")
print(f"\nKEPT={len(res.kept)} residual={res.residual_error:.3f}")
print(to_markdown(res))
print("\nOK (full pipeline: DINOv2 + real VOC masks, GPU)")
