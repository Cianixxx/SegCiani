"""The whole pipeline on SBD: the 9 118 SBD images that VOC 2012 does not annotate.

SBD (Hariharan et al. 2011, benchmark_RELEASE) annotates 11 355 VOC 2011 images with
class and instance masks (.mat). 2 237 of them are also VOC 2012 segmentation images
(train or val); the other 9 118 are new, never seen by the thesis runs, and the
segmenter under test (torchvision LR-ASPP MobileNetV3, trained on COCO) never saw any
of them. The run uses exactly those 9 118, with the inst configuration
(SQ_CLAIM_COV=0.5, SQ_GT_INSTANCES=object) and the cluster branch of the thesis.

One difference from VOC to keep in mind when reading the results: SBD masks have no
"void" band (255) around the objects -- VOC's don't-care border -- and are traced
more coarsely, so the boundary errors are counted on the bare edge.

Steps (STEP=..., comma-separated, in this order):
  layout     ROOT = /data/sbd/VOCdevkit/VOC2012 in the VOC layout: JPEGImages (copied),
             SegmentationClass / SegmentationObject (P-mode PNG, VOC palette, from
             GTcls / GTinst), ImageSets/Segmentation/val.txt = the 9 118 stems (the
             run's image list; every adapter reads "val.txt")                    CPU
  predict    mobilenet_pred/mask_pr_<stem>.png: the segmenter's prediction, short
             side 520 + centre crop, R=G=B=class -- the thesis format and code     GPU
  substrate  /data/sbd/substrate_cc.pkl: components + IoU claims, the language
             branch's substrate (as /data/voc2012/substrate_cc.pkl in July)         CPU
  prep       BLOCK 1 in the inst configuration + DINOv2 embeddings, by chunks of
             images (the feature maps of 9 118 images do not fit in one cache) ->
             outputs/slice_quality_sbd/instances_embedded.pkl                     GPU
  tag        Qwen verify tags of the errors (tag_instances_verify, as every run),
             claimed in chunks from a shared queue (mkdir lock) so that two GPUs
             share the work and a crash loses at most one chunk ->
             outputs/slice_quality_sbd/tags/chunk_<k>.json                       GPU
  merge      tags onto the instances -> outputs/slice_quality_sbd/result_tagged.pkl
             (result None: rescore_with_language.py builds it)                    CPU

Env: SBD (/ds/SBD/benchmark_RELEASE/dataset), ROOT, OUT_RUN (outputs/slice_quality_sbd),
     CHUNK_IMG (400), CHUNK_TAG (300), QWEN_4BIT.
"""
import glob
import json
import os
import pickle
import shutil
import time
from collections import Counter

import numpy as np
from PIL import Image

SBD = os.environ.get("SBD", "/ds/SBD/benchmark_RELEASE/dataset")
VOC = "/data/voc2012/VOCdevkit/VOC2012"
ROOT = os.environ.get("ROOT", "/data/sbd/VOCdevkit/VOC2012")
SBD_DIR = os.path.dirname(os.path.dirname(ROOT))           # /data/sbd
OUT_RUN = os.environ.get("OUT_RUN", "outputs/slice_quality_sbd")
CROPS_CC = f"{SBD_DIR}/voc_crops"                          # language-branch substrate
CROPS_OBJ = os.environ.get("CROPS", "/workspace/voc_crops_sbd")   # cluster branch
CHUNK_IMG = int(os.environ.get("CHUNK_IMG", "400"))
CHUNK_TAG = int(os.environ.get("CHUNK_TAG", "300"))
STEPS = os.environ.get("STEP", "layout").split(",")


def names():
    r = lambda p: {l.strip() for l in open(p) if l.strip()}
    sbd = r(f"{SBD}/train.txt") | r(f"{SBD}/val.txt")
    voc = r(f"{VOC}/ImageSets/Segmentation/train.txt") | r(f"{VOC}/ImageSets/Segmentation/val.txt")
    return sorted(sbd - voc)


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def layout():
    from scipy.io import loadmat
    stems = names()
    pal = Image.open(glob.glob(f"{VOC}/SegmentationClass/*.png")[0]).getpalette()
    for d in ("JPEGImages", "SegmentationClass", "SegmentationObject", "ImageSets/Segmentation"):
        os.makedirs(f"{ROOT}/{d}", exist_ok=True)
    cls_hist, n_obj = Counter(), 0
    for i, s in enumerate(stems, 1):
        dj = f"{ROOT}/JPEGImages/{s}.jpg"
        if not os.path.exists(dj):
            shutil.copyfile(f"{SBD}/img/{s}.jpg", dj)
        dc, do = f"{ROOT}/SegmentationClass/{s}.png", f"{ROOT}/SegmentationObject/{s}.png"
        if not (os.path.exists(dc) and os.path.exists(do)):
            c = loadmat(f"{SBD}/cls/{s}.mat")["GTcls"][0, 0]["Segmentation"].astype(np.uint8)
            o = loadmat(f"{SBD}/inst/{s}.mat")["GTinst"][0, 0]["Segmentation"].astype(np.uint8)
            assert c.max() <= 20 and c.shape == o.shape, s
            for arr, p in ((c, dc), (o, do)):
                im = Image.frombytes("P", (arr.shape[1], arr.shape[0]), np.ascontiguousarray(arr).tobytes())
                im.putpalette(pal)
                im.save(p)
        if i <= 2000:                                      # a cheap look at the labels
            c = np.array(Image.open(dc))
            cls_hist.update(int(v) for v in np.unique(c) if v)
            n_obj += int(np.array(Image.open(do)).max())
        if i % 1000 == 0:
            print(f"  layout {i}/{len(stems)}", flush=True)
    open(f"{ROOT}/ImageSets/Segmentation/val.txt", "w").write("\n".join(stems) + "\n")
    open(f"{SBD_DIR}/README.txt", "w").write(
        "SBD (benchmark_RELEASE) in the VOC layout: the 9118 images of SBD train+val that are\n"
        "neither VOC 2012 segmentation train nor val. Masks from GTcls/GTinst (no void band).\n"
        "ImageSets/Segmentation/val.txt = the run's image list (not SBD's own val split).\n"
        "Built by slice_quality/examples/build_sbd.py.\n")
    print(f"layout: {len(stems)} images -> {ROOT}; first 2000: {n_obj} objects, classes "
          f"{dict(sorted(cls_hist.items()))}", flush=True)


def predict():
    import torch
    import torchvision.models.segmentation as seg
    from torchvision import transforms as T
    torch.backends.cudnn.enabled = False
    S = 520
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = seg.lraspp_mobilenet_v3_large(weights="DEFAULT").eval().to(dev)
    os.makedirs(f"{ROOT}/mobilenet_pred", exist_ok=True)
    stems = [l.strip() for l in open(f"{ROOT}/ImageSets/Segmentation/val.txt") if l.strip()]
    t0 = time.time()
    for i, s in enumerate(stems, 1):
        dp = f"{ROOT}/mobilenet_pred/mask_pr_{s}.png"
        if os.path.exists(dp):
            continue
        img = Image.open(f"{ROOT}/JPEGImages/{s}.jpg").convert("RGB")
        w, h = img.size                                    # gen_voc_predictions' geometry
        r = S / min(w, h)
        W, H = round(w * r), round(h * r)
        l, t = (W - S) // 2, (H - S) // 2
        rgb = img.resize((W, H)).crop((l, t, l + S, t + S))
        with torch.no_grad():
            x = norm(T.ToTensor()(rgb)).unsqueeze(0).to(dev)
            pr = model(x)["out"][0].argmax(0).cpu().numpy().astype(np.uint8)
        Image.fromarray(np.stack([pr] * 3, -1)).save(dp)
        if i % 1000 == 0:
            print(f"  predict {i}/{len(stems)} | {time.time() - t0:.0f}s", flush=True)
    print(f"predictions: {len(stems)} images in {time.time() - t0:.0f}s ({dev})", flush=True)


def substrate():
    from slice_quality.adapters.voc_seg_samples import build_voc_samples
    from slice_quality.typed_error import decompose_dataset
    assert not os.environ.get("SQ_GT_INSTANCES") and not os.environ.get("SQ_CLAIM_COV"), \
        "the July substrate is components + IoU claims: unset SQ_GT_INSTANCES / SQ_CLAIM_COV"
    out = f"{SBD_DIR}/substrate_cc.pkl"
    stems = [l.strip() for l in open(f"{ROOT}/ImageSets/Segmentation/val.txt") if l.strip()]
    errors, reference, paths = [], [], {}
    t0 = time.time()
    for k, ch in enumerate(chunks(stems, CHUNK_IMG), 1):
        samples, ip = build_voc_samples(ROOT, CROPS_CC, ch)
        e, r = decompose_dataset(samples)
        errors += e
        reference += r
        paths.update(ip)
        print(f"  substrate chunk {k}: {len(errors)} errors, {len(reference)} reference | "
              f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump({"errors": errors, "reference": reference, "image_paths": paths}, open(out, "wb"))
    print(f"substrate: {len(errors)} errors, {len(reference)} reference over {len(paths)} images -> {out}; "
          f"by class {dict(Counter(t.gt_class or t.pred_class for t in errors + reference).most_common())}",
          flush=True)


def prep():
    from slice_quality.adapters.common import make_mask_lookup
    from slice_quality.adapters.voc_seg_samples import build_voc_samples
    from slice_quality.config import SliceQualityConfig
    from slice_quality.embeddings import attach_embeddings
    from slice_quality.extractors import DINOv2Extractor
    from slice_quality.typed_error import decompose_dataset
    assert os.environ.get("SQ_GT_INSTANCES") == "object" and os.environ.get("SQ_CLAIM_COV") == "0.5", \
        "the SBD run mirrors the inst run: SQ_GT_INSTANCES=object SQ_CLAIM_COV=0.5"
    os.makedirs(OUT_RUN, exist_ok=True)
    cfg = SliceQualityConfig().typed_error
    stems = [l.strip() for l in open(f"{ROOT}/ImageSets/Segmentation/val.txt") if l.strip()]
    paths = {}
    ext = DINOv2Extractor(paths)                           # reads the dict filled chunk by chunk
    errors, reference = [], []
    t0 = time.time()
    for k, ch in enumerate(chunks(stems, CHUNK_IMG), 1):
        samples, ip = build_voc_samples(ROOT, CROPS_OBJ, ch, gt_instances="object")
        paths.update(ip)
        e, r = decompose_dataset(samples, cfg)
        attach_embeddings(e + r, ext, make_mask_lookup(samples))
        ext._cache.clear()                                 # one chunk of feature maps at a time
        errors += [t for t in e if t.embedding is not None]
        reference += [t for t in r if t.embedding is not None]
        print(f"  prep chunk {k}: {len(errors)} errors, {len(reference)} reference (dropped without "
              f"embedding {len(e) + len(r) - sum(t.embedding is not None for t in e + r)}) | "
              f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump({"errors": errors, "reference": reference, "image_paths": dict(paths)},
                open(f"{OUT_RUN}/instances_embedded.pkl", "wb"))
    print(f"prep: {len(errors)} errors, {len(reference)} reference; taxonomy "
          f"{dict(Counter(t.seg_subtype.value for t in errors))} -> {OUT_RUN}/instances_embedded.pkl",
          flush=True)


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
    cks, paths = _tag_chunks()
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
    {"layout": layout, "predict": predict, "substrate": substrate, "prep": prep,
     "tag": tag, "merge": merge}[st]()
print(f"BUILD_SBD_DONE {','.join(STEPS)}", flush=True)
