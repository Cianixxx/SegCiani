"""The whole pipeline on GESD (the SliceLens / GH-ESD benchmark), instance segmentation part.

GESD (coco_segmentation.json) holds 5 724 objects of 15 classes on 1 657 COCO val2017
images, each with the expert's ground-truth mask and/or the Mask R-CNN mask, and the
expert slice it belongs to (21 slices, "NA" = correct). The run types those predictions
with BLOCK 1 in the inst configuration (claims at 0.5) -- every COCO val2017 object of the
image as ground-truth context, the expert's mask in place of the COCO one it matches
(IoU >= 0.5), exactly as gesd_block1.py (point A) -- and keeps the typed instances that
stand for a GESD object. The same instances feed both branches (Mask R-CNN predicts
instances: there is no separate connected-component substrate as on VOC).
The expert slice names are NOT used by any step here: they are read only by the
comparison that follows the run.

Labels are GESD's category names (teddy_bear, ...); a COCO name with a space gets "_".
GESD does not store the predicted class: a prediction gets its object's category.

Steps (STEP=..., comma-separated, in this order):
  images     the 1 657 GESD images out of val2017.zip -> IMG_DIR                      CPU
  substrate  GDATA/substrate.pkl {errors, reference, image_paths} (the language
             branch's input, read by examples/ground_ea_hypotheses.py and
             examples/gen_datadriven.py) + GDATA/unit_map.json (instance -> GESD objects) CPU
  prep       the same instances + DINOv2 embeddings -> OUT_RUN/instances_embedded.pkl  CPU/GPU
  tag        Qwen verify tags of the errors, queue of chunks (as build_sbd.py)          GPU
  merge      tags onto the instances -> OUT_RUN/result_tagged.pkl                        CPU

Env: GESD, COCO_ANN (annotations_trainval2017.zip), COCO_ZIP (val2017.zip), IMG_DIR,
     GDATA (/data), OUT_RUN (outputs/slice_quality_gesd), CHUNK_IMG (300), CHUNK_TAG (300),
     QWEN_4BIT.
"""
import json
import os
import pickle
import time
import zipfile
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

GESD = os.environ.get("GESD", "/priv/gesd/coco_segmentation.json")
COCO_ANN = os.environ.get("COCO_ANN", "/priv/coco/annotations_trainval2017.zip")
COCO_ZIP = os.environ.get("COCO_ZIP", "/priv/coco/val2017_zip/val2017.zip")
IMG_DIR = os.environ.get("IMG_DIR", "/coco_img")
GDATA = os.environ.get("GDATA", "/data")
OUT_RUN = os.environ.get("OUT_RUN", "outputs/slice_quality_gesd")
CHUNK_IMG = int(os.environ.get("CHUNK_IMG", "300"))
CHUNK_TAG = int(os.environ.get("CHUNK_TAG", "300"))
STEPS = os.environ.get("STEP", "substrate").split(",")
P = lambda v: json.loads(v) if isinstance(v, str) and v else (v or [])
lab = lambda c: c.replace(" ", "_")


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def poly_mask(polys, w, h):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        if len(p) >= 6:
            d.polygon([(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)], fill=1)
    return np.array(m, bool)


def rle_mask(rle, w, h):
    """uncompressed COCO RLE (column-major runs, starting with zeros)."""
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        flat[pos:pos + c] = val
        pos += c
        val = 1 - val
    return flat.reshape((w, h)).T.astype(bool)


def bbox(m):
    """the box the VLM grounds on, as voc_seg_samples.py: (x0, y0, x1, y1), end exclusive."""
    ys, xs = np.nonzero(m)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def load():
    d = json.load(open(GESD))
    cat = {str(c["id"]): c["name"] for c in d["categories"]}
    units = defaultdict(list)
    for a in d["annotations"]:
        units[str(a["image_id"])].append(a)
    with zipfile.ZipFile(COCO_ANN) as z:
        coco = json.loads(z.read("annotations/instances_val2017.json"))
    ccat = {c["id"]: lab(c["name"]) for c in coco["categories"]}
    cann = defaultdict(list)
    for a in coco["annotations"]:
        cann[str(a["image_id"])].append(a)
    imgs = {str(i["id"]): i for i in d["images"] if str(i["id"]) in units}
    return imgs, units, cat, cann, ccat


def build_sample(img, info, us, cat, cann, ccat):
    """{"image_id", "preds", "gts"} of one image + which pred / gt stands for which GESD object
    (gesd_block1.py's construction)."""
    from slice_quality.typed_error import mask_iou
    w, h = int(info["width"]), int(info["height"])
    gts = []
    for a in cann.get(img, []):
        seg = a["segmentation"]
        m = rle_mask(seg, w, h) if isinstance(seg, dict) else poly_mask(seg, w, h)
        if m.any():
            gts.append({"mask": m, "cls": ccat[a["category_id"]], "bbox": bbox(m)})
    preds, pkey, unit_pred, unit_gt = [], {}, {}, {}
    for u in us:
        inf, gt = P(u.get("inference_segmentation")), P(u.get("gt_segmentation"))
        cls = cat.get(str(u["gt_category_id"]), "?")
        if gt:
            gm = poly_mask(gt, w, h)
            if gm.any():
                best = max(range(len(gts)), key=lambda j: mask_iou(gm, gts[j]["mask"]), default=None)
                if best is not None and mask_iou(gm, gts[best]["mask"]) >= 0.5:
                    gts[best] = {"mask": gm, "cls": gts[best]["cls"], "bbox": bbox(gm)}
                    unit_gt[u["id"]] = best
                else:
                    gts.append({"mask": gm, "cls": cls, "bbox": bbox(gm)})
                    unit_gt[u["id"]] = len(gts) - 1
        if inf:
            pm = poly_mask(inf, w, h)
            if pm.any():
                k = (pm.tobytes(), cls)
                if k not in pkey:
                    pkey[k] = len(preds)
                    preds.append({"mask": pm, "cls": cls, "bbox": bbox(pm)})
                unit_pred[u["id"]] = pkey[k]
    return {"image_id": img, "preds": preds, "gts": gts}, unit_pred, unit_gt


def gesd_instances(img, info, us, cat, cann, ccat, cfg):
    """BLOCK 1 on one image; the typed instances standing for GESD objects, labelled with the
    object's GESD category; their links to the objects."""
    from slice_quality.typed_error import decompose_image
    sample, unit_pred, unit_gt = build_sample(img, info, us, cat, cann, ccat)
    errs, refs = decompose_image(img, sample["preds"], sample["gts"], cfg)
    by_id = {t.instance_id: t for t in errs + refs}
    gt_pair = {t.metadata.get("gt_id"): t for t in errs + refs if t.metadata and t.metadata.get("gt_id")}
    in_merge = {g: t for t in errs if t.metadata for g in t.metadata.get("gt_ids", [])}
    keep, links, stats = {}, defaultdict(list), Counter()
    for u in us:
        t = None
        if u["id"] in unit_pred:
            t = by_id.get(f"{img}#p{unit_pred[u['id']]}")
        elif u["id"] in unit_gt:
            gid = f"{img}#g{unit_gt[u['id']]}"
            t = by_id.get(gid) or gt_pair.get(gid) or in_merge.get(gid)
        if t is None:
            stats["object without a typed instance"] += 1
            continue
        c = cat.get(str(u["gt_category_id"]), "?")
        if t.instance_id not in keep:
            if (t.gt_class or t.pred_class) != c:
                t.metadata["coco_class"] = t.gt_class or t.pred_class
                t.gt_class = c
                stats["relabelled to the object's category"] += 1
            keep[t.instance_id] = t
        elif (keep[t.instance_id].gt_class or keep[t.instance_id].pred_class) != c:
            stats["instance shared by objects of two categories (first kept)"] += 1
        links[t.instance_id].append({"unit": str(u["id"]), "slice": u["reason"], "category": c})
    errors = [t for t in errs if t.instance_id in keep]
    reference = [t for t in refs if t.instance_id in keep]
    return sample, errors, reference, links, stats


def images():
    imgs, *_ = load()
    os.makedirs(IMG_DIR, exist_ok=True)
    n = 0
    with zipfile.ZipFile(COCO_ZIP) as z:
        names = set(z.namelist())
        for i in imgs.values():
            fn = i["file_name"]
            assert fn.endswith(".jpg") and "/" not in fn and ".." not in fn, fn
            dst = os.path.join(IMG_DIR, fn)
            if os.path.exists(dst):
                continue
            src = f"val2017/{fn}"
            assert src in names, src
            with z.open(src) as f, open(dst + ".tmp", "wb") as g:
                g.write(f.read())
            os.replace(dst + ".tmp", dst)
            n += 1
    ok = sum(os.path.exists(os.path.join(IMG_DIR, i["file_name"])) for i in imgs.values())
    print(f"images: {n} extracted now, {ok}/{len(imgs)} GESD images in {IMG_DIR}", flush=True)


def _run(with_embeddings):
    from slice_quality.config import TypedErrorConfig
    from slice_quality import typed_error
    assert typed_error.distance_transform_edt is not None, "scipy missing: pip install scipy"
    cfg = TypedErrorConfig(claim_cov=0.5)
    imgs, units, cat, cann, ccat = load()
    paths = {img: os.path.join(IMG_DIR, i["file_name"]) for img, i in imgs.items()}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    assert not missing, f"{len(missing)} images missing (STEP=images first): {missing[:3]}"
    if with_embeddings:
        from slice_quality.adapters.common import make_mask_lookup
        from slice_quality.embeddings import attach_embeddings
        from slice_quality.extractors import DINOv2Extractor
        ext = DINOv2Extractor(paths)
    errors, reference, links, stats = [], [], {}, Counter()
    t0 = time.time()
    for k, ch in enumerate(chunks(sorted(imgs), CHUNK_IMG), 1):
        samples, e_ch, r_ch = [], [], []
        for img in ch:
            s, e, r, ln, st = gesd_instances(img, imgs[img], units[img], cat, cann, ccat, cfg)
            samples.append(s)
            e_ch += e
            r_ch += r
            links.update(ln)
            stats.update(st)
        if with_embeddings:
            attach_embeddings(e_ch + r_ch, ext, make_mask_lookup(samples))
            ext._cache.clear()
            stats["dropped without embedding"] += sum(t.embedding is None for t in e_ch + r_ch)
            e_ch = [t for t in e_ch if t.embedding is not None]
            r_ch = [t for t in r_ch if t.embedding is not None]
        errors += e_ch
        reference += r_ch
        print(f"  chunk {k}: {len(errors)} errors, {len(reference)} reference | {time.time() - t0:.0f}s", flush=True)
    return errors, reference, paths, links, stats


def substrate():
    from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances
    errors, reference, paths, links, stats = _run(False)
    # the VLM grounds on the bbox: an instance without one is invisible to the language branch
    no_box = [t.instance_id for t in errors + reference if t.bbox is None]
    assert not no_box, f"{len(no_box)} instances without bbox, e.g. {no_box[:3]}"
    vis = Counter(i.label for i in typed_to_slicelens_instances(errors, reference, paths))
    print(f"  instances the language branch sees, by class: {dict(vis.most_common())}", flush=True)
    os.makedirs(GDATA, exist_ok=True)
    pickle.dump({"errors": errors, "reference": reference, "image_paths": paths},
                open(f"{GDATA}/substrate.pkl", "wb"))
    json.dump(links, open(f"{GDATA}/unit_map.json", "w"))
    by_cls = Counter(t.gt_class or t.pred_class for t in errors + reference)
    err_cls = Counter(t.gt_class or t.pred_class for t in errors)
    print(f"substrate: {len(errors)} errors, {len(reference)} reference, {sum(len(v) for v in links.values())} "
          f"GESD objects linked; {dict(stats)}\n  by class (instances, errors): "
          f"{ {c: (n, err_cls[c]) for c, n in by_cls.most_common()} }\n  types "
          f"{dict(Counter(t.seg_subtype.value for t in errors).most_common())} -> {GDATA}/substrate.pkl", flush=True)


def prep():
    errors, reference, paths, links, stats = _run(True)
    sub = pickle.load(open(f"{GDATA}/substrate.pkl", "rb"))
    same = {t.instance_id for t in sub["errors"] + sub["reference"]} == {t.instance_id for t in errors + reference}
    os.makedirs(OUT_RUN, exist_ok=True)
    pickle.dump({"errors": errors, "reference": reference, "image_paths": paths},
                open(f"{OUT_RUN}/instances_embedded.pkl", "wb"))
    print(f"prep: {len(errors)} errors, {len(reference)} reference with embeddings ({dict(stats)}); same "
          f"instances as the substrate: {same} -> {OUT_RUN}/instances_embedded.pkl", flush=True)


def _tag_chunks():
    d = pickle.load(open(f"{OUT_RUN}/instances_embedded.pkl", "rb"))
    errs = sorted(d["errors"], key=lambda t: t.instance_id)
    return chunks(errs, CHUNK_TAG), d["image_paths"]


def tag():
    from slice_quality.tagging import tag_instances_verify
    from slicelens.backends.qwen_vlm import QwenVLM
    cks, paths = _tag_chunks()
    tdir = f"{OUT_RUN}/tags"
    os.makedirs(tdir, exist_ok=True)
    print(f"tag queue: {len(cks)} chunks of {CHUNK_TAG} errors", flush=True)
    vlm, n_done = None, 0
    for k, ch in enumerate(cks):
        out = f"{tdir}/chunk_{k:04d}.json"
        if os.path.exists(out):
            continue
        try:
            os.mkdir(f"{tdir}/.claim_{k:04d}")
        except FileExistsError:
            continue
        if vlm is None:
            vlm = QwenVLM(model="/models/qwen", max_new_tokens=10,
                          load_in_4bit=os.environ.get("QWEN_4BIT", "0") == "1")
        t0 = time.time()
        tag_instances_verify(ch, paths, vlm, verbose=False)
        rec = {t.instance_id: {k2: t.metadata[k2] for k2 in ("attribute", "attr_scores", "attribute_mode")
                               if k2 in t.metadata} for t in ch}
        json.dump(rec, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
        n_done += 1
        print(f"  {time.strftime('%m-%d %H:%M')} chunk {k}/{len(cks)}: {len(ch)} errors in "
              f"{time.time() - t0:.0f}s ({(time.time() - t0) / len(ch):.1f} s each)", flush=True)
    left = sum(1 for k in range(len(cks)) if not os.path.exists(f"{tdir}/chunk_{k:04d}.json"))
    print(f"tag worker done ({n_done} chunks here); chunks still missing: {left}", flush=True)


def merge():
    cks, _ = _tag_chunks()
    d = pickle.load(open(f"{OUT_RUN}/instances_embedded.pkl", "rb"))
    tags = {}
    for k in range(len(cks)):
        tags.update(json.load(open(f"{OUT_RUN}/tags/chunk_{k:04d}.json")))   # all must exist
    n = 0
    for t in d["errors"]:
        for k2, v in tags.get(t.instance_id, {}).items():
            t.metadata[k2] = v
        n += t.instance_id in tags
    pickle.dump({"result": None, "errors": d["errors"], "reference": d["reference"],
                 "image_paths": d["image_paths"], "descs": None}, open(f"{OUT_RUN}/result_tagged.pkl", "wb"))
    print(f"merge: tags on {n}/{len(d['errors'])} errors: "
          f"{dict(Counter(t.metadata.get('attribute') for t in d['errors']).most_common())} -> "
          f"{OUT_RUN}/result_tagged.pkl", flush=True)


for st in STEPS:
    {"images": images, "substrate": substrate, "prep": prep, "tag": tag, "merge": merge}[st]()
print(f"BUILD_GESD_DONE {','.join(STEPS)}", flush=True)
