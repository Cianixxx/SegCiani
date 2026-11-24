"""Visual gallery of both branches -- CPU only (no GPU / DINOv2 needed).

Reuses a saved SliceLens run for the language branch + a cheap geometric embedding
for the cluster branch, so you can EVALUATE results visually any time without
booking a GPU. Writes one self-contained HTML you can scp and open in a browser.

Swap ``attach_geometric_embeddings`` for the DINOv2 extractor (see
server_voc_full.py) when you want the real embeddings.
"""
import glob
import os

from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.adapters.slicelens_run import (
    run_image_names, build_language_slices_from_run, attach_geometric_embeddings,
)
from slice_quality.typed_error import decompose_dataset
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.report import to_json
from slice_quality.gallery import build_gallery

DATA = "/data/dataset"
# Saved SliceLens runs that have grounded per-instance data (bbox) -> language branch.
# Only cow and person are available; aeroplane/chair lack the verify step (GPU-pending).
RUNS = [("voc_datadriven_cow_VERIFY.json", "cow"),
        ("voc_person_qwen.json", "person")]

# cluster branch covers the available VOC images; language maps the runs onto the
# overlap. Include the runs' image names so any that exist locally are used.
SEG = os.path.join(DATA, "SegmentationClass")
glob_names = sorted(os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(SEG, "*.png")))[:40]
run_names = set()
for f, _ in RUNS:
    run_names |= set(run_image_names(f))
names = sorted(set(glob_names) | run_names)
samples, image_paths = build_voc_samples(
    DATA, "/tmp/voc_crops", names,
    pred_subdir="mobilenet_pred", pred_template="mask_pr_{name}.png",
    crop=520, min_pixels=200,
)
cfg = SliceQualityConfig()
cfg.cluster.n_slices = 8
cfg.def1.n_min_frac = 0.03
errors, reference = decompose_dataset(samples, cfg.typed_error)
attach_geometric_embeddings(errors + reference)     # CPU stand-in for DINOv2
language = []
for f, cls in RUNS:
    sl = build_language_slices_from_run(f, errors, conf_threshold=0.5, prefix=cls)
    print(f"  language[{cls}]: {len(sl)} slices (sizes {[s.size for s in sl]})")
    language += sl

res = run_from_instances(errors, reference, language_slices=language, cfg=cfg,
                         max_peeling_rounds=3, residual_stop=0.05)
to_json(res, "outputs/slice_quality/voc_gallery.json")

# "by image" panels: original / GT (SegmentationClass, già a colori) / prediction
# (mask_pr_*_rgb è la viz colorata). Stesso resize/crop delle istanze (520).
from PIL import Image
from slicelens.adapters.voc_seg import _resize_center_crop


def _panels(name):
    try:
        o = _resize_center_crop(Image.open(f"{DATA}/JPEGImages/{name}.jpg").convert("RGB"), 520, Image.LANCZOS)
        g = _resize_center_crop(Image.open(f"{DATA}/SegmentationClass/{name}.png").convert("RGB"), 520, Image.NEAREST)
        p = _resize_center_crop(Image.open(f"{DATA}/mobilenet_pred/mask_pr_{name}_rgb.png").convert("RGB"), 520, Image.NEAREST)
        return [("original", o), ("ground truth", g), ("prediction", p)]
    except Exception:
        return None


image_panels = {n: pn for n in names if (pn := _panels(n))}
out = build_gallery(res, image_paths, "outputs/slice_quality/gallery.html",
                    image_panels=image_panels,
                    title="VOC two-branch (CPU) - language: cow + person")
print(f"scored slices: {len(res.scored)} | kept: {len(res.kept)}")
print(f"gallery -> {out}")
