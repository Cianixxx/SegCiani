"""FULL VOC2012 val (1449 images) -- cluster branch with DINOv2.

Predictions already exist at VOCdevkit/VOC2012/mobilenet_pred (mask_pr_<name>.png).
Runs the complete class-agnostic analysis: typed errors + DINOv2 + scoring on ALL
1449 val images, writes the JSON report and a (capped) HTML gallery.

Run on GPU 3 (1080Ti 11GB -- DINOv2 fits with room to spare). CPU-only for the
rest. No Qwen / language branch here (that's the per-class, slower add-on).
"""
import os
from collections import Counter

from slice_quality.adapters.voc_seg_samples import build_voc_samples, make_mask_lookup
from slice_quality.typed_error import decompose_dataset
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.report import to_json
from slice_quality.gallery import build_gallery

DATA = "/data/voc2012/VOCdevkit/VOC2012"
val = [ln.strip() for ln in
       open(os.path.join(DATA, "ImageSets/Segmentation/val.txt")) if ln.strip()]
print(f"VOC val images: {len(val)}", flush=True)

samples, image_paths = build_voc_samples(
    DATA, "/tmp/voc_crops_full", val,
    pred_subdir="mobilenet_pred", pred_template="mask_pr_{name}.png",
    crop=520, min_pixels=200,
)
n_pred = sum(len(s["preds"]) for s in samples)
n_gt = sum(len(s["gts"]) for s in samples)
print(f"samples={len(samples)} pred_instances={n_pred} gt_instances={n_gt}", flush=True)

cfg = SliceQualityConfig()
cfg.cluster.n_slices = 20
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200
errors, reference = decompose_dataset(samples, cfg.typed_error)
print(f"errors={len(errors)} reference={len(reference)}", flush=True)
print("TAXONOMY:", dict(Counter(e.seg_subtype.value for e in errors)), flush=True)

# --- BLOCK 2: DINOv2 on GPU 3 ----------------------------------------------
import torch
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
extractor = DINOv2Extractor(image_paths, model_name="dinov2_vits14")
attach_embeddings(errors + reference, extractor, make_mask_lookup(samples))
print("DINOv2 embeddings attached", flush=True)

res = run_from_instances(errors, reference, cfg=cfg, max_peeling_rounds=5,
                         residual_stop=0.05)
to_json(res, "outputs/slice_quality/voc_full_val.json")
print(f"KEPT={len(res.kept)} residual={res.residual_error:.3f}", flush=True)

# per-image textual descriptions (the deliverable) -- richer once tags are added
from slice_quality.describe import build_image_descriptions, export_descriptions
descs = build_image_descriptions(res)
export_descriptions(descs, "outputs/slice_quality/image_descriptions.json",
                    "outputs/slice_quality/image_descriptions.txt")
print(f"per-image descriptions: {len(descs)} images", flush=True)

# --- gallery (capped): colored GT + palette-colored prediction panels ------
from PIL import Image
from slicelens.adapters.voc_seg import _resize_center_crop

_PAL = Image.open(os.path.join(DATA, "SegmentationClass", val[0] + ".png")).getpalette()


def _panels(name):
    try:
        o = _resize_center_crop(Image.open(f"{DATA}/JPEGImages/{name}.jpg").convert("RGB"), 520, Image.LANCZOS)
        g = _resize_center_crop(Image.open(f"{DATA}/SegmentationClass/{name}.png").convert("RGB"), 520, Image.NEAREST)
        pr = Image.open(f"{DATA}/mobilenet_pred/mask_pr_{name}.png")          # R=G=B=class idx
        idx = pr.split()[0]                                                    # take index channel
        idx.putpalette(_PAL)                                                   # apply VOC palette
        p = _resize_center_crop(idx.convert("RGB"), 520, Image.NEAREST)
        return [("original", o), ("ground truth", g), ("prediction", p)]
    except Exception:
        return None


image_panels = {n: pn for n in val[:80] if (pn := _panels(n))}
out = build_gallery(res, image_paths, "outputs/slice_quality/gallery_full.html",
                    image_panels=image_panels, max_images=80,
                    title="VOC2012 full val (1449) - cluster branch, DINOv2")
print(f"gallery -> {out}", flush=True)
print("DONE (full VOC val)", flush=True)
