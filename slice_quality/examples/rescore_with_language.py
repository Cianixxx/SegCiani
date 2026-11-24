"""Re-score from the pickle, refreshing the SliceLens language branch. CPU only.

Reuses the cached DINOv2-embedded + Qwen-tagged instances (result_tagged.pkl), so
adding SliceLens classes needs NO GPU: it maps the voc_paper_<class>.json runs onto
the cached instances, re-clusters (deterministic) + re-scores, and rebuilds the
per-image descriptions + gallery. Skips classes whose run file doesn't exist yet.
"""
import os
import pickle

from slice_quality.adapters.slicelens_run import build_language_slices_from_run
from slice_quality.language_branch import dedup_language_slices
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.describe import (build_image_descriptions,
                                    export_description_index, export_descriptions)
from slice_quality.report import export_slice_index, to_json
from slice_quality.gallery import build_gallery

# SQ_OUT: output directory (a retyped run lives in its own, e.g.
# outputs/slice_quality_cov05, and must not overwrite the reference one)
SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality")
PKL = os.environ.get("PKL", f"{SQ_OUT}/result_tagged.pkl")
# VOC_DATA: the VOC tree whose images the panels show (val-EA has its own)
DATA = os.environ.get("VOC_DATA", "/data/voc2012/VOCdevkit/VOC2012")
# Run preference per class: _full (validated hypotheses re-grounded on ALL
# instances) > _cc (shared substrate, sampled) > legacy (class-level bboxes).
# All 20 VOC classes: classes whose run file does not exist yet are skipped by
# the loop below, so this picks up new grounding runs as they land.
CLASSES = os.environ.get("CLASSES", "").split() or [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]   # CLASSES env: another dataset's labels
# LANG_DIRS: where the SliceLens runs are read from, colon-separated. Each
# directory contributes its own preferred file per class; the default is the
# thesis runs alone, so the default behaviour is unchanged. A second directory
# (e.g. /data/outputs_extra, the extra generic hypotheses) gets its slice ids
# prefixed "<class>_extra" so they cannot collide with "<class>_<n>".
LANG_DIRS = os.environ.get("LANG_DIRS", "/data/outputs").split(":")
# LANG_TAGS: the slice-id tag of each LANG_DIRS entry, colon-separated (":extra:dd" ->
# "<cls>_<n>", "<cls>_extra_<n>", "<cls>_dd_<n>"); default: no tag for the first
# directory and "extra" for the others, as before
_TAGS = (os.environ["LANG_TAGS"].split(":") if os.environ.get("LANG_TAGS")
         else [""] + ["extra"] * (len(LANG_DIRS) - 1))
assert len(_TAGS) == len(LANG_DIRS), "LANG_TAGS needs one tag per LANG_DIRS entry"
RUNS = []
for _k, _dir in enumerate(LANG_DIRS):
    for _cls in CLASSES:
        for _suffix in ("_full", "_cc", ""):
            _f = f"{_dir}/voc_paper_{_cls}{_suffix}.json"
            if os.path.exists(_f) or _suffix == "":
                RUNS.append((_f, f"{_cls}_{_TAGS[_k]}" if _TAGS[_k] else _cls))
                break

d = pickle.load(open(PKL, "rb"))
errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
print(f"loaded {len(errors)} errors, {len(reference)} reference", flush=True)

# LINK_SUBSTRATE (3 November): the substrate pickle the VLM read (errors/reference of the first version of Block 1,
# e.g. /data/voc2012/substrate_cc.pkl). The final typing numbers its objects differently, so a grounded object is
# joined to the final error it IS: same image and class, the largest box IoU >= 0.5. Without it, the join by id.
ID_MAP = None
if os.environ.get("LINK_SUBSTRATE"):
    from collections import defaultdict
    from slice_quality.language_branch import _bbox_iou
    _sub = pickle.load(open(os.environ["LINK_SUBSTRATE"], "rb"))
    _sub = list(_sub["errors"]) + list(_sub["reference"])
    _cls = lambda t: t.gt_class or t.pred_class
    _by_img = defaultdict(list)
    for _t in errors:
        _by_img[_t.image_id].append(_t)
    ID_MAP = {}
    for _a in _sub:
        if _a.bbox is None:
            continue
        _best, _bi = None, 0.5
        for _t in _by_img.get(_a.image_id, []):
            if _t.bbox is None or _cls(_t) != _cls(_a):
                continue
            _v = _bbox_iou(_a.bbox, _t.bbox)
            if _v >= _bi:
                _best, _bi = _t, _v
        if _best is not None:
            ID_MAP[_a.instance_id] = _best.instance_id
    print(f"geometric join ({os.environ['LINK_SUBSTRATE']}): {len(ID_MAP)} of {len(_sub)} substrate objects "
          f"are a final error", flush=True)

language = []
for f, cls in RUNS:
    if os.path.exists(f):
        sl = build_language_slices_from_run(f, errors, conf_threshold=0.5, prefix=cls, id_map=ID_MAP)
        print(f"  language[{cls}]: {len(sl)} slices", flush=True)
        language += sl
    else:
        print(f"  language[{cls}]: run file missing -> skipped", flush=True)
n_before = len(language)
language = dedup_language_slices(language)          # merge paraphrase slices (#7)
print(f"language slices total: {n_before} -> {len(language)} after dedup", flush=True)

cfg = SliceQualityConfig()
cfg.cluster.n_slices = 20
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200
res = run_from_instances(errors, reference, language_slices=language, cfg=cfg,
                         max_peeling_rounds=5, residual_stop=0.05)
pickle.dump({"result": res, "errors": errors, "reference": reference,
             "image_paths": image_paths, "descs": None}, open(PKL, "wb"))

descs = build_image_descriptions(res)
from slice_quality.corroborate import cross_branch_corroboration
lang_ids = {p["slice_id"] for d in descs.values()
            for p in d["patterns"] if p["branch"] == "language"}
cor = cross_branch_corroboration(res, lang_ids)
print(f"corroborated cluster slices: {len(cor)}", flush=True)
export_descriptions(descs, f"{SQ_OUT}/image_descriptions.json",
                    f"{SQ_OUT}/image_descriptions.txt")
export_description_index(descs, f"{SQ_OUT}/descriptions_index.json",
                         f"{SQ_OUT}/descriptions_index.txt")
to_json(res, f"{SQ_OUT}/voc_full_val_tagged.json")
used_ids = {p["slice_id"] for d in descs.values() for p in d["patterns"]}
export_slice_index(res, f"{SQ_OUT}/slices_index.json",
                   f"{SQ_OUT}/slices_index.txt", used_ids=used_ids)
n_sl = sum(1 for k in descs if any("SliceLens" in p["description"]
                                    for p in descs[k]["patterns"]))
print(f"images with SliceLens patterns: {n_sl}", flush=True)
if os.environ.get("GALLERY", "1") == "0":           # a dataset without the VOC layout (GESD)
    print("DONE rescore with language (no gallery)", flush=True)
    raise SystemExit(0)

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


# Panels for EXACTLY the images the gallery shows (sorted error images), not
# val[:N] -- error-free val images shift the window and leave the tail bare.
MAX_IMAGES = 80
shown = sorted(descs)[:MAX_IMAGES]
image_panels = {n: pn for n in shown if (pn := _panels(n))}
build_gallery(res, image_paths, f"{SQ_OUT}/gallery_full_tagged.html",
              image_panels=image_panels, image_descriptions=descs, max_images=MAX_IMAGES,
              title="VOC full val - DINOv2 + Qwen + SliceLens (multi-class)")
print("DONE rescore with language", flush=True)
