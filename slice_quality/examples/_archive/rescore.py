"""Re-score + regenerate from the pickled TAGGED instances -- CPU only, no GPU.

Loads errors+reference (with DINOv2 embeddings + Qwen tags) from result_tagged.pkl,
re-runs the SCORING (so it picks up any pipeline.py / describe.py / gallery.py
edits), and rewrites the descriptions + gallery. Use this instead of a full GPU
re-run whenever you change scoring / descriptions / gallery.

Run:  PYTHONPATH=/workspace python slice_quality/examples/rescore.py
"""
import os
import pickle

from slice_quality import run_from_instances
from slice_quality.config import SliceQualityConfig
from slice_quality.metrics.slice_purity import slice_purity
from slice_quality.describe import build_image_descriptions, export_descriptions
from slice_quality.gallery import build_gallery
from slice_quality.report import to_json

PKL = "outputs/slice_quality/result_tagged.pkl"
DATA = "/data/voc2012/VOCdevkit/VOC2012"

d = pickle.load(open(PKL, "rb"))
errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]

cfg = SliceQualityConfig()
cfg.cluster.n_slices = 20
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200
res = run_from_instances(errors, reference, cfg=cfg, max_peeling_rounds=5,
                         residual_stop=0.05)
key = cfg.purity_attribute_key
for ks in res.kept:
    ks.quality.slice_purity = slice_purity(
        [m.metadata.get(key) for m in ks.slice_.members])
print(f"KEPT={len(res.kept)} residual={res.residual_error:.3f}", flush=True)

descs = build_image_descriptions(res)
export_descriptions(descs, "outputs/slice_quality/image_descriptions.json",
                    "outputs/slice_quality/image_descriptions.txt")
to_json(res, "outputs/slice_quality/voc_full_val_tagged.json")
with open(PKL, "wb") as f:                       # refresh pickle with fixed ids
    pickle.dump({"result": res, "errors": errors, "reference": reference,
                 "image_paths": image_paths, "descs": descs}, f)

from PIL import Image
from slicelens.adapters.voc_seg import _resize_center_crop

val = [ln.strip() for ln in
       open(os.path.join(DATA, "ImageSets/Segmentation/val.txt")) if ln.strip()]
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
print("re-scored + regenerated (CPU, no GPU)", flush=True)
