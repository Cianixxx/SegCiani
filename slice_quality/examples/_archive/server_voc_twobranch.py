"""COMPLETE two-branch pipeline on the server (GPU).

cluster branch (DINOv2 embeddings) + language branch (a saved SliceLens run,
mapped onto the same real instances). This is the full design with both branches
scored by the same Slice-Quality Vector.
"""
import os
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples, make_mask_lookup
from slice_quality.adapters.slicelens_run import (
    run_image_names, build_language_slices_from_run,
)
from slice_quality.typed_error import decompose_dataset
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.report import to_markdown, to_json

DATA = "/data/dataset"
RUN = "voc_datadriven_cow_VERIFY.json"  # saved SliceLens run in the repo root

# Use exactly the images the saved run covers -> guarantees overlap.
names = run_image_names(RUN)
print(f"language run covers {len(names)} images")

samples, image_paths = build_voc_samples(
    DATA, "/tmp/voc_crops", names,
    pred_subdir="mobilenet_pred", pred_template="mask_pr_{name}.png",
    crop=520, min_pixels=200,
)
cfg = SliceQualityConfig()
errors, reference = decompose_dataset(samples, cfg.typed_error)
print("TAXONOMY:", dict(Counter(e.seg_subtype.value for e in errors)), "errors", len(errors))

# BLOCK 2 -- DINOv2 (cluster branch needs embeddings; language branch reuses them)
extractor = DINOv2Extractor(image_paths, model_name="dinov2_vits14")
attach_embeddings(errors + reference, extractor, make_mask_lookup(samples))
print("DINOv2 embeddings attached")

# language branch: map the saved SliceLens run onto the SAME instances
language = build_language_slices_from_run(RUN, errors, conf_threshold=0.5)
print(f"LANGUAGE slices: {len(language)} "
      f"(sizes: {[s.size for s in language]})")

cfg.cluster.n_slices = 8
cfg.def1.n_min_frac = 0.03
cfg.significance.n_boot = 200
res = run_from_instances(errors, reference, language_slices=language, cfg=cfg,
                         max_peeling_rounds=3, residual_stop=0.05)

# report: show how each branch scored
print("\n--- branch verdicts ---")
for q in res.all_qualities:
    if q.slice_id.startswith("lang"):
        print(f"LANG {q.slice_id:8s} n={q.size:3d} om={q.omega_squared:.2f} "
              f"mc={q.manifold_compactness:.2f} etp={q.error_type_purity:.2f} -> {q.verdict}")
to_json(res, "outputs/slice_quality/voc_twobranch_slices.json")
print(f"\nKEPT={len(res.kept)} (cluster + language) residual={res.residual_error:.3f}")
print("\n".join(to_markdown(res).splitlines()[:16]))
print("\nOK (complete two-branch pipeline: cluster + SliceLens language)")
