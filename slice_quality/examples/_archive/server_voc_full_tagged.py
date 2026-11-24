"""TONIGHT job: full VOC val + DINOv2 + Qwen tags -> rich per-image descriptions.

Same as server_voc_full_dataset.py, then tags a SAMPLE of each KEPT slice's
members with Qwen (so attributes are semantic: small / dark / occluded / ...),
recomputes slice_purity, and writes the per-image descriptions + gallery.

GPU: DINOv2 needs <1GB; Qwen-2.5-VL-7B needs ~16GB bf16 OR ~8GB with 4-bit.
  - GPU 3 (1080Ti 11GB): set QWEN_4BIT=1 (default).
  - a free RTX6000 24GB: set QWEN_4BIT=0 for full bf16 quality.
"""
import os
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples, make_mask_lookup
from slice_quality.adapters.slicelens_run import build_language_slices_from_run
from slice_quality.typed_error import decompose_dataset
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.report import to_json
from slice_quality.gallery import build_gallery
from slice_quality.tagging import tag_instances
from slice_quality.metrics.slice_purity import slice_purity
from slice_quality.describe import build_image_descriptions, export_descriptions
from slicelens.backends.qwen_vlm import QwenVLM

DATA = "/data/voc2012/VOCdevkit/VOC2012"
SAMPLE_PER_SLICE = 30                 # instances tagged per kept slice (purity is a proportion)
QWEN_4BIT = os.environ.get("QWEN_4BIT", "1") == "1"

val = [ln.strip() for ln in
       open(os.path.join(DATA, "ImageSets/Segmentation/val.txt")) if ln.strip()]
print(f"VOC val images: {len(val)}", flush=True)

samples, image_paths = build_voc_samples(
    DATA, "/workspace/voc_crops_full", val,   # persistent crops -> offline gallery regen
    pred_subdir="mobilenet_pred", pred_template="mask_pr_{name}.png",
    crop=520, min_pixels=200,
)
cfg = SliceQualityConfig()
cfg.cluster.n_slices = 20
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200
errors, reference = decompose_dataset(samples, cfg.typed_error)
print("TAXONOMY:", dict(Counter(e.seg_subtype.value for e in errors)), flush=True)

# BLOCK 2 -- DINOv2
extractor = DINOv2Extractor(image_paths, model_name="dinov2_vits14")
attach_embeddings(errors + reference, extractor, make_mask_lookup(samples))
print("DINOv2 embeddings attached", flush=True)

# language branch -- SliceLens grounded runs (only cow + person available)
language = []
for f, cls in [("voc_datadriven_cow_VERIFY.json", "cow"),
               ("voc_person_qwen.json", "person")]:
    if os.path.exists(f):
        sl = build_language_slices_from_run(f, errors, conf_threshold=0.5, prefix=cls)
        print(f"  language[{cls}]: {len(sl)} slices", flush=True)
        language += sl
print(f"language slices total: {len(language)}", flush=True)

res = run_from_instances(errors, reference, language_slices=language, cfg=cfg,
                         max_peeling_rounds=5, residual_stop=0.05)
print(f"KEPT={len(res.kept)} residual={res.residual_error:.3f}", flush=True)

# --- Qwen tags on a sample of each KEPT slice's members --------------------
print(f"loading Qwen (4bit={QWEN_4BIT}) ...", flush=True)
vlm = QwenVLM(model="/models/qwen", max_new_tokens=10, load_in_4bit=QWEN_4BIT)
to_tag, seen = [], set()
for ks in res.kept:
    for m in ks.slice_.members[:SAMPLE_PER_SLICE]:
        if m.instance_id not in seen:
            seen.add(m.instance_id)
            to_tag.append(m)
print(f"tagging {len(to_tag)} instances (kept-slice sample) ...", flush=True)
tag_instances(to_tag, image_paths, vlm)
print("ATTRIBUTES:", dict(Counter(m.metadata.get("attribute", "?") for m in to_tag)), flush=True)

# recompute slice_purity for kept slices now that attributes exist
key = cfg.purity_attribute_key
for ks in res.kept:
    ks.quality.slice_purity = slice_purity(
        [m.metadata.get(key) for m in ks.slice_.members])

# --- per-image descriptions (now rich) + report + gallery ------------------
descs = build_image_descriptions(res)
export_descriptions(descs, "outputs/slice_quality/image_descriptions.json",
                    "outputs/slice_quality/image_descriptions.txt")
print(f"per-image descriptions: {len(descs)} images", flush=True)
to_json(res, "outputs/slice_quality/voc_full_val_tagged.json")

# pickle everything -> re-score / regen offline (no GPU). Includes the TAGGED
# instances (with DINOv2 embeddings) so rescore.py can re-cluster on CPU.
import pickle
with open("outputs/slice_quality/result_tagged.pkl", "wb") as f:
    pickle.dump({"result": res, "errors": errors, "reference": reference,
                 "image_paths": image_paths, "descs": descs}, f)

from PIL import Image
from slicelens.adapters.voc_seg import _resize_center_crop
_PAL = Image.open(os.path.join(DATA, "SegmentationClass", val[0] + ".png")).getpalette()


def _panels(name):
    try:
        o = _resize_center_crop(Image.open(f"{DATA}/JPEGImages/{name}.jpg").convert("RGB"), 520, Image.LANCZOS)
        g = _resize_center_crop(Image.open(f"{DATA}/SegmentationClass/{name}.png").convert("RGB"), 520, Image.NEAREST)
        pr = Image.open(f"{DATA}/mobilenet_pred/mask_pr_{name}.png").split()[0]
        pr.putpalette(_PAL)
        p = _resize_center_crop(pr.convert("RGB"), 520, Image.NEAREST)
        return [("original", o), ("ground truth", g), ("prediction", p)]
    except Exception:
        return None


image_panels = {n: pn for n in val[:80] if (pn := _panels(n))}
build_gallery(res, image_paths, "outputs/slice_quality/gallery_full_tagged.html",
              image_panels=image_panels, image_descriptions=descs, max_images=80,
              title="VOC2012 full val - DINOv2 + Qwen tags")
print("DONE (full VOC + tags + per-image descriptions)", flush=True)
