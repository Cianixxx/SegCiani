"""Full interpretable pipeline: DINOv2 embeddings + Qwen attribute tags.

Activates slice_purity (#5) and the semantic half of the explanation. Capped to a
few images because Qwen runs one generation per instance. Run on GPU 5 with the
Qwen weights mounted at /models/qwen.
"""
import glob
import os
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples, make_mask_lookup
from slice_quality.typed_error import decompose_dataset
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.tagging import tag_instances
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.report import to_markdown, to_json
from slicelens.backends.qwen_vlm import QwenVLM

DATA = "/data/dataset"
SEG = os.path.join(DATA, "SegmentationClass")
names = sorted(os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(SEG, "*.png")))[:30]   # cap for Qwen
print(f"using {len(names)} images")

samples, image_paths = build_voc_samples(
    DATA, "/tmp/voc_crops", names,
    pred_subdir="mobilenet_pred", pred_template="mask_pr_{name}.png",
    crop=520, min_pixels=200,
)
cfg = SliceQualityConfig()
errors, reference = decompose_dataset(samples, cfg.typed_error)
print("TAXONOMY:", dict(Counter(e.seg_subtype.value for e in errors)), "errors", len(errors))

# BLOCK 2 -- DINOv2 (also disables cuDNN, which Qwen then benefits from)
extractor = DINOv2Extractor(image_paths, model_name="dinov2_vits14")
attach_embeddings(errors + reference, extractor, make_mask_lookup(samples))
print("DINOv2 embeddings attached")

# NEW -- Qwen attribute tags -> slice_purity + semantic explanation
print("loading Qwen2.5-VL from /models/qwen ...", flush=True)
vlm = QwenVLM(model="/models/qwen", max_new_tokens=10)
print(f"tagging {len(errors)} instances ...", flush=True)
tag_instances(errors, image_paths, vlm)
print("ATTRIBUTES:", dict(Counter(e.metadata.get("attribute", "?") for e in errors)))

cfg.cluster.n_slices = 8
cfg.def1.n_min_frac = 0.03
cfg.significance.n_boot = 200
res = run_from_instances(errors, reference, cfg=cfg, max_peeling_rounds=3,
                         residual_stop=0.05)
to_json(res, "outputs/slice_quality/voc_tagged_slices.json")
print(f"\nKEPT={len(res.kept)} residual={res.residual_error:.3f}")
print(to_markdown(res))
print("\nOK (DINOv2 + Qwen tags -> interpretable slices with slice_purity)")
