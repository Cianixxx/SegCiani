"""Server smoke test: full mask-based BLOCK 1 on REAL VOC masks (CPU only).

Runs typed_error.decompose_dataset on pascal_vlm masks -> the real 6-case
taxonomy on ALL instances (no language-run cap), then scores + reports with a
cheap geometric embedding (no GPU / DINOv2 needed for this smoke test).
"""
import glob
import os
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.adapters.slicelens_run import attach_geometric_embeddings
from slice_quality.typed_error import decompose_dataset
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.report import to_markdown

DATA = "/data/dataset"                      # pascal_vlm mounted at /data
SEG = os.path.join(DATA, "SegmentationClass")
names = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(SEG, "*.png")))[:60]
print(f"VOC images found: {len(glob.glob(os.path.join(SEG, '*.png')))}, using {len(names)}")

samples, image_paths = build_voc_samples(
    DATA, "/tmp/voc_crops", names,
    pred_subdir="mobilenet_pred",
    pred_template="mask_pr_{name}.png",   # index mask (R=class idx); _rgb is colored viz
    crop=520, min_pixels=200,
)
n_pred = sum(len(s["preds"]) for s in samples)
n_gt = sum(len(s["gts"]) for s in samples)
print(f"samples={len(samples)} pred_instances={n_pred} gt_instances={n_gt}")

cfg = SliceQualityConfig()
errors, reference = decompose_dataset(samples, cfg.typed_error)
print(f"errors={len(errors)} reference(correct)={len(reference)}")
print("TYPED ERROR taxonomy (real VOC masks):",
      dict(Counter(e.seg_subtype.value for e in errors)))

attach_geometric_embeddings(errors + reference)
cfg.cluster.n_slices = 8
cfg.def1.n_min_frac = 0.02
cfg.significance.n_boot = 200
res = run_from_instances(errors, reference, cfg=cfg, max_peeling_rounds=2,
                         residual_stop=0.05)
print(f"\nKEPT={len(res.kept)} residual={res.residual_error:.3f}")
print("\n".join(to_markdown(res).splitlines()[:14]))
print("\nOK (full mask-based BLOCK 1 on real VOC masks, server)")
