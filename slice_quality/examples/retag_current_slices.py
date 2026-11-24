"""Re-tag Qwen attributes on the CURRENT slice structure, then re-score.

Fixes the stale-tag-coverage limitation surfaced by the corruption experiment
(2026-07-18): attribute tags were sampled ONCE, early, on the pre-stratification
k=20 clustering (server_voc_full_tagged.py). Every re-clustering since
(stratification, BIC, retyping, corruption runs) reshuffled instances into a
DIFFERENT, finer set of candidate slices; each inherits only whichever tagged
instances land in it by chance -- often too few (12-19 of 100+ members) to
estimate slice_purity reliably, even when the slice is genuinely coherent.

Also fixes a chicken-and-egg gap: the old script only ever tagged members of
ALREADY-KEPT slices, so a slice blocked ONLY by low purity could never
accumulate the tags needed to prove itself -- it had no path to KEEP. This
script tops up tagging on EVERY candidate cluster slice under-covered
(n_tagged < TARGET_TAGGED), not just the currently-kept ones, then re-scores
from scratch so purity-blocked slices get a fair chance to flip.

GPU: Qwen2.5-VL-7B (bf16 ~16GB on a free RTX6000, or --load-in-4bit ~8GB on an
11GB 1080Ti/TitanX). Run:  QWEN_4BIT=1 python slice_quality/examples/retag_current_slices.py
"""
import os
import pickle
from collections import Counter

from slice_quality.adapters.slicelens_run import build_language_slices_from_run
from slice_quality.language_branch import dedup_language_slices
from slice_quality.config import SliceQualityConfig
from slice_quality import run_from_instances
from slice_quality.tagging import tag_instances, tag_instances_verify
from slice_quality.describe import build_image_descriptions, export_description_index, export_descriptions
from slice_quality.corroborate import cross_branch_corroboration
from slice_quality.report import export_slice_index, to_json
from slice_quality.gallery import build_gallery
from slicelens.backends.qwen_vlm import QwenVLM

PKL = "outputs/slice_quality/result_tagged.pkl"
DATA = "/data/voc2012/VOCdevkit/VOC2012"
TARGET_TAGGED = int(os.environ.get("TARGET_TAGGED", "40"))   # tagged members/slice
MAX_TOTAL_TAGS = int(os.environ.get("MAX_TOTAL_TAGS", "3000"))  # cap on instances
QWEN_4BIT = os.environ.get("QWEN_4BIT", "1") == "1"
# "caption" = one free choice from a closed vocabulary (original path);
# "verify"  = one grounded yes/no question per attribute, argmax (see tagging.py).
TAG_MODE = os.environ.get("TAG_MODE", "caption")
# verify only: "slices" = up to TARGET_TAGGED per candidate slice (paired sample);
# "remaining" = every instance still carrying a captioning tag (finish the switch).
TAG_SCOPE = os.environ.get("TAG_SCOPE", "slices")

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

d = pickle.load(open(PKL, "rb"))
errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
res_old = d["result"]
print(f"loaded {len(errors)} errors, {len(reference)} reference "
      f"| current candidates: {len(res_old.scored)}", flush=True)

# same config used for the eventual re-score, defined early so n_min below
# matches the REAL Def.1 threshold exactly (not a guessed constant).
cfg = SliceQualityConfig()
cfg.cluster.n_slices = 20
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200

# --- identify under-covered CLUSTER candidates (any verdict) ---------------
# Priority: near-misses first (the ACTUAL Def.1 gates all pass except purity,
# see slice_quality_vector.py) so the highest-value slices get topped up if
# the budget runs out; general low-coverage top-up second.
n_min = max(1, int(cfg.def1.n_min_frac * (len(errors) + len(reference))))
seen_ids, near_miss, general = set(), [], []
for s, q in res_old.scored:
    if s.source != "cluster" or q.slice_id in seen_ids:
        continue
    seen_ids.add(q.slice_id)
    n_tagged = q.extras.get("n_tagged", 0)
    if n_tagged >= TARGET_TAGGED:
        continue
    severe = q.extras.get("mean_sev_z", -999) >= 0.0
    near = (severe and q.significant and q.size >= n_min and q.slice_purity < 0.80)
    (near_miss if near else general).append((s, q, n_tagged))
near_miss.sort(key=lambda x: -x[1].size)
general.sort(key=lambda x: -x[1].size)
print(f"under-covered cluster candidates: {len(near_miss)} near-miss (purity-blocked), "
      f"{len(general)} general", flush=True)

to_tag, seen_inst = [], set()
budget = MAX_TOTAL_TAGS
already_tagged = lambda m: m.metadata.get("attribute") not in (None,)  # noqa: E731

verified = lambda m: m.metadata.get("attribute_mode") == "verify"     # noqa: E731

if TAG_MODE == "verify" and TAG_SCOPE == "remaining":
    # Finish the channel switch: every instance still carrying a CAPTIONING tag
    # gets re-elicited, so the tag pool stops being a mixture of two measurement
    # modes (the caveat left open in the report). Slice priority still drives the
    # order, so an interrupted run leaves the most decisive slices done.
    ordered = (near_miss + general
               + [(k.slice_, k.quality, 0) for k in res_old.kept
                  if k.slice_.source == "cluster"])
    for s, _q, _n in ordered:
        for m in s.members:
            if budget <= 0:
                break
            if m.instance_id in seen_inst or verified(m) or not already_tagged(m):
                continue
            seen_inst.add(m.instance_id)
            to_tag.append(m)
            budget -= 1
    for m in errors + reference:                    # anything outside a candidate
        if budget <= 0:
            break
        if m.instance_id in seen_inst or verified(m) or not already_tagged(m):
            continue
        seen_inst.add(m.instance_id)
        to_tag.append(m)
        budget -= 1
elif TAG_MODE == "verify":
    # Re-tagging the SAME instances through the other channel, so the two are
    # paired and comparable: prefer members that already carry a captioning tag
    # (that is the population every current purity verdict rests on), then top up
    # with untagged ones. Slices are visited near-miss-first, as below.
    for s, q, _n in near_miss + general + [(s2, q2, 0) for s2, q2 in
                                           ((k.slice_, k.quality) for k in res_old.kept)
                                           if s2.source == "cluster"]:
        if budget <= 0:
            break
        added = 0
        for m in sorted(s.members, key=lambda x: not already_tagged(x)):
            if added >= TARGET_TAGGED or budget <= 0:
                break
            if m.instance_id in seen_inst:
                continue
            seen_inst.add(m.instance_id)
            to_tag.append(m)
            added += 1
            budget -= 1
else:
    for s, q, n_tagged in near_miss + general:
        if budget <= 0:
            break
        need = max(0, TARGET_TAGGED - n_tagged)
        added = 0
        for m in s.members:
            if added >= need or budget <= 0:
                break
            if m.instance_id in seen_inst or already_tagged(m):
                continue
            seen_inst.add(m.instance_id)
            to_tag.append(m)
            added += 1
            budget -= 1
n_repl = sum(1 for m in to_tag if already_tagged(m))
print(f"instances to tag this run: {len(to_tag)} (mode={TAG_MODE}, "
      f"{n_repl} replacing an existing tag, budget cap {MAX_TOTAL_TAGS}, "
      f"target {TARGET_TAGGED}/slice)", flush=True)

if to_tag:
    print(f"loading Qwen (4bit={QWEN_4BIT}) ...", flush=True)
    vlm = QwenVLM(model="/models/qwen", max_new_tokens=10, load_in_4bit=QWEN_4BIT)
    if TAG_MODE == "verify":
        for m in to_tag:                     # keep the old tag for the paired diff
            if m.metadata.get("attribute"):
                m.metadata["attribute_before"] = m.metadata["attribute"]
        before = Counter(m.metadata.get("attribute", "?") for m in to_tag)
        tag_instances_verify(to_tag, image_paths, vlm)
        after = Counter(m.metadata.get("attribute", "?") for m in to_tag)
        print("ATTRIBUTES before (captioning):", dict(before), flush=True)
        print("ATTRIBUTES after  (verification):", dict(after), flush=True)
        # compare only on the PAIRED subset (instances that already had a tag):
        # untagged ones ('?') are not evidence about the captioning channel.
        paired = [m for m in to_tag if m.metadata.get("attribute_before")]
        inf = lambda a: a not in ("normal", "error", "other", "?", None)  # noqa: E731
        print(f"informative tags overall: "
              f"{sum(1 for m in to_tag if inf(m.metadata.get('attribute')))}/{len(to_tag)}",
              flush=True)
        if paired:
            print(f"paired subset ({len(paired)} instances): captioning "
                  f"{sum(1 for m in paired if inf(m.metadata['attribute_before']))} informative "
                  f"-> verification "
                  f"{sum(1 for m in paired if inf(m.metadata.get('attribute')))} informative",
                  flush=True)
            flips = Counter((m.metadata["attribute_before"], m.metadata.get("attribute"))
                            for m in paired)
            print("top flips (captioning -> verification):",
                  dict(flips.most_common(8)), flush=True)
    else:
        tag_instances(to_tag, image_paths, vlm)
        print("ATTRIBUTES (this batch):",
              dict(Counter(m.metadata.get("attribute", "?") for m in to_tag)), flush=True)
    del vlm
else:
    print("nothing to tag -- coverage already sufficient", flush=True)

if os.environ.get("TAG_PROBE") == "1":
    # Timing/behaviour probe: stop BEFORE the re-score so the pickle (and every
    # certified number resting on it) is left untouched.
    print("TAG_PROBE=1 -> stopping before re-score, pickle untouched", flush=True)
    print("PROBE_DONE", flush=True)
    raise SystemExit(0)

# --- re-run the full closed loop so purity/verdicts reflect fresh tags -----
RUNS = []
for _cls in CLASSES:
    for _suffix in ("_full", "_cc", ""):
        _f = f"/data/outputs/voc_paper_{_cls}{_suffix}.json"
        if os.path.exists(_f) or _suffix == "":
            RUNS.append((_f, _cls))
            break

language = []
for f, cls in RUNS:
    if os.path.exists(f):
        sl = build_language_slices_from_run(f, errors, conf_threshold=0.5, prefix=cls)
        language += sl
language = dedup_language_slices(language)
print(f"language slices (post-dedup): {len(language)}", flush=True)

res = run_from_instances(errors, reference, language_slices=language, cfg=cfg,
                         max_peeling_rounds=5, residual_stop=0.05)
print(f"KEPT (cluster+language) = {len(res.kept)} "
      f"(was {len(res_old.kept)} before re-tagging) | residual={res.residual_error:.3f}",
      flush=True)
kept_cluster_before = sum(1 for k in res_old.kept if k.slice_.source == "cluster")
kept_cluster_after = sum(1 for k in res.kept if k.slice_.source == "cluster")
print(f"KEPT cluster slices: {kept_cluster_before} -> {kept_cluster_after}", flush=True)

pickle.dump({"result": res, "errors": errors, "reference": reference,
             "image_paths": image_paths, "descs": None}, open(PKL, "wb"))

descs = build_image_descriptions(res)
lang_ids = {p["slice_id"] for dd in descs.values()
            for p in dd["patterns"] if p["branch"] == "language"}
cor = cross_branch_corroboration(res, lang_ids)
print(f"corroborated cluster slices: {len(cor)}", flush=True)
export_descriptions(descs, "outputs/slice_quality/image_descriptions.json",
                    "outputs/slice_quality/image_descriptions.txt")
export_description_index(descs, "outputs/slice_quality/descriptions_index.json",
                         "outputs/slice_quality/descriptions_index.txt")
to_json(res, "outputs/slice_quality/voc_full_val_tagged.json")
used_ids = {p["slice_id"] for dd in descs.values() for p in dd["patterns"]}
export_slice_index(res, "outputs/slice_quality/slices_index.json",
                   "outputs/slice_quality/slices_index.txt", used_ids=used_ids)
print(f"images with descriptions: {len(descs)}", flush=True)

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


MAX_IMAGES = 80
shown = sorted(descs)[:MAX_IMAGES]
image_panels = {n: pn for n in shown if (pn := _panels(n))}
build_gallery(res, image_paths, "outputs/slice_quality/gallery_full_tagged.html",
              image_panels=image_panels, image_descriptions=descs, max_images=MAX_IMAGES,
              title="VOC full val - re-tagged current slices")
print("RETAG_DONE", flush=True)
