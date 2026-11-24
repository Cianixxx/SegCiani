"""Phase 2 (GPU): complete the 36 hand-annotated TRAIN images with BOTH branches.

Phase 1 (describe_train36.py, CPU) induced the 14 certified cluster slices onto
these images but left every row as "grouped by appearance only", because the
attribute word comes from a Qwen pass that had not been run. This script adds
the two GPU-dependent halves:

  * verify-mode attribute tagging (6 grounded yes/no questions per instance,
    argmax over P(yes | attribute, region)) -- turns the fallback wording into
    "occluded objects of class 'person' ...";
  * the language branch: the 35 hypotheses ALREADY validated on `val` are
    grounded on this image set's instances of the same class. No slope-trend
    re-verification is attempted -- 36 images cannot certify a hypothesis, and
    re-testing one here would be a weaker test masquerading as a stronger one.
    A certified predicate is APPLIED, exactly as the cluster slices are.

Same caveat as Phase 1: these images are in VOC `train`, which the segmenter was
fitted on. "Does the pipeline describe an image the way a human did" stays a
fair question; "how often does the model fail" does not.

Run (GPU container with Qwen mounted):
  QWEN_4BIT=1 PYTHONPATH=/workspace python slice_quality/examples/describe_train36_gpu.py
Env: NAMES_FILE, OUT, QWEN_4BIT (default 1), MAX_LANG_EVALS (0 = no cap).
"""
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np

from slicelens.data_types import Region
from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.config import SliceQualityConfig
from slice_quality.data_types import Slice
from slice_quality.describe import _detail_clause, describe_instances
from slice_quality.mask_facts import THIN_NOISE, add_mask_facts
from slice_quality.embeddings import attach_embeddings
from slice_quality.extractors import DINOv2Extractor
from slice_quality.features import build_features
from slice_quality.tagging import tag_instances_verify
from slice_quality.typed_error import decompose_dataset

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/data/voc2012/voc_crops"
# SQ_OUT: the run whose certified slices and hypotheses are applied here; a
# retyped run (outputs/slice_quality_cov05b) keeps its own train36 next to it
SQ_OUT = os.environ.get("SQ_OUT", "outputs/slice_quality")
PKL = f"{SQ_OUT}/result_tagged.pkl"
SLICES_INDEX = f"{SQ_OUT}/slices_index.json"
NAMES_FILE = os.environ.get("NAMES_FILE", "/workspace/train36.txt")
OUT = os.environ.get("OUT", f"{SQ_OUT}/train36")
QWEN_4BIT = os.environ.get("QWEN_4BIT", "1") == "1"
MAX_LANG = int(os.environ.get("MAX_LANG_EVALS", "0"))
LANG_CONF_MIN = 0.5          # same membership threshold as build_language_slices_from_run

names = [l.strip() for l in open(NAMES_FILE, encoding="utf-8") if l.strip()]
print(f"images: {len(names)}", flush=True)

# ---- Block 1 --------------------------------------------------------------
samples, image_paths = build_voc_samples(DATA, CROPS, names, pred_subdir="mobilenet_pred")
errors, reference = decompose_dataset(samples)
print(f"typed errors: {len(errors)} | references: {len(reference)}", flush=True)
# mask facts BEFORE the descriptions, so the "details" line is born complete
# (as in describe_synth_gpu.py) instead of being reconstructed afterwards by
# enrich_train36_details.py
fine, _f = build_voc_samples(DATA, CROPS, names, pred_subdir="mobilenet_pred",
                             min_pixels=THIN_NOISE)
add_mask_facts(errors, samples, fine, reference)


def make_mask_lookup(samples):
    by_img = {s["image_id"]: s for s in samples}

    def lookup(inst):
        s = by_img.get(inst.image_id)
        if s is None:
            return None
        pool = s["gts"] if inst.metadata.get("from_gt_mask") else s["preds"]
        idx = int(inst.instance_id.rsplit("#", 1)[1][1:])
        return pool[idx]["mask"] if idx < len(pool) else None

    return lookup


# ---- Block 2 (GPU) --------------------------------------------------------
print("DINOv2 embeddings ...", flush=True)
attach_embeddings(errors, DINOv2Extractor(image_paths), make_mask_lookup(samples))
print(f"embeddings: {sum(1 for e in errors if e.embedding is not None)}/{len(errors)}",
      flush=True)

# ---- Qwen: verify-mode attribute tagging ----------------------------------
from slicelens.backends.qwen_vlm import QwenVLM

t0 = time.time()
print(f"loading Qwen (4bit={QWEN_4BIT}) ...", flush=True)
vlm = QwenVLM(model="/models/qwen", max_new_tokens=10, load_in_4bit=QWEN_4BIT)
print(f"Qwen loaded in {time.time() - t0:.0f}s", flush=True)

print(f"verify-mode tagging on {len(errors)} instances "
      f"({len(errors) * 6} passes) ...", flush=True)
tag_instances_verify(errors, image_paths, vlm)
tags = Counter(e.metadata.get("attribute") for e in errors)
print("attributes:", dict(tags.most_common()), flush=True)

# ---- language branch: apply the ALREADY-validated hypotheses ---------------
idx = json.load(open(SLICES_INDEX, encoding="utf-8"))["slices"]
validated = [s for s in idx if s.get("source") != "cluster" and s.get("in_descriptions")
             and s.get("hypothesis")]
print(f"validated hypotheses to apply: {len(validated)}", flush=True)

err_by_class = defaultdict(list)
for e in errors:
    cls = e.gt_class or e.pred_class
    if cls:
        err_by_class[cls].append(e)

lang_slices, n_eval = [], 0
for s in validated:
    cls = re.sub(r"(_extra|_dd)?_\d+$", "", s["slice_id"])   # "<cls>_<n>", "<cls>_extra_<n>", "<cls>_dd_<n>"
    cands = err_by_class.get(cls, [])
    if not cands:
        continue
    members, membership = [], {}
    for inst in cands:
        if inst.bbox is None or inst.image_id not in image_paths:
            continue
        if MAX_LANG and n_eval >= MAX_LANG:
            break
        conf = float(vlm.slice_confidence(image_paths[inst.image_id],
                                          Region(bbox=inst.bbox), s["hypothesis"]))
        n_eval += 1
        if conf >= LANG_CONF_MIN:
            members.append(inst)
            membership[inst.instance_id] = conf
    if members:
        lang_slices.append((s, Slice(slice_id=s["slice_id"], members=members,
                                     source="language", hypothesis=s["hypothesis"],
                                     membership=membership)))
    if n_eval and n_eval % 100 < len(cands):
        print(f"  language: {n_eval} evals, {len(lang_slices)} slices so far", flush=True)
print(f"language evals: {n_eval} | slices with >=1 member here: {len(lang_slices)}",
      flush=True)

# ---- cluster induction (identical to Phase 1) -----------------------------
data = pickle.load(open(PKL, "rb"))
kept = data["result"].kept
cfg = SliceQualityConfig()
old_members, seen = [], set()
for k in kept:
    for m in k.slice_.members:
        if m.instance_id not in seen and m.embedding is not None:
            seen.add(m.instance_id)
            old_members.append(m)
new_members = [e for e in errors if e.embedding is not None]
feats, kept_insts = build_features(old_members + new_members, cfg.feature)
fid = {i.instance_id: f for i, f in zip(kept_insts, feats)}

slice_of = {}
for k in kept:
    X = np.stack([fid[m.instance_id] for m in k.slice_.members if m.instance_id in fid])
    c = X.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(X - c, axis=1), 90))
    slice_of[k.slice_.slice_id] = (k, c, radius, k.quality.dominant_error_type)

induced = defaultdict(list)
for inst in new_members:
    best, best_d = None, None
    for sid, (k, c, radius, subtype) in slice_of.items():
        if inst.seg_subtype != subtype:
            continue
        d = float(np.linalg.norm(fid[inst.instance_id] - c))
        if d <= radius and (best_d is None or d < best_d):
            best, best_d = sid, d
    if best is not None:
        induced[best].append(inst)
print(f"cluster: {sum(len(v) for v in induced.values())}/{len(new_members)} instances "
      f"in {len(induced)} slices", flush=True)

# ---- per-image descriptions, BOTH branches --------------------------------
by_img = defaultdict(lambda: defaultdict(list))
meta = {}
for sid, insts in induced.items():
    k = slice_of[sid][0]
    meta[sid] = ("cluster", k.slice_, k.quality)
    for i in insts:
        by_img[i.image_id][sid].append(i)


class _Q:                       # minimal quality stand-in for describe_instances
    def __init__(self, et):
        self.dominant_error_type = et


for s, sl in lang_slices:
    et = next((k.quality.dominant_error_type for k in kept
               if k.quality.dominant_error_type.value == s.get("dominant_error_type")), None)
    if et is None:
        et = sl.members[0].seg_subtype
    meta[sl.slice_id] = ("language", sl, _Q(et))
    for i in sl.members:
        by_img[i.image_id][sl.slice_id].append(i)

all_by_img = defaultdict(set)
for e in errors:
    all_by_img[e.image_id].add(e.instance_id)

out = {}
for name in names:
    n_total = len(all_by_img.get(name, ()))
    pats, sys_ids = [], set()
    for sid, members in by_img.get(name, {}).items():
        branch, sl, q = meta[sid]
        pats.append({"slice_id": sid, "branch": branch,
                     "error_type": q.dominant_error_type.value,
                     "description": describe_instances(members, sl, q),
                     "details": _detail_clause(members),
                     "n_instances": len(members)})
        sys_ids.update(m.instance_id for m in members)
    n_sys = len(sys_ids)
    n_iso = n_total - n_sys
    obj = "object" if n_total == 1 else "objects"
    if n_sys == 0:
        lines = [f"Image {name}: {n_total} wrong {obj}, "
                 f"no systematic pattern (isolated errors)."]
    else:
        head = f"Image {name}: {n_total} wrong {obj}, {n_sys} in systematic patterns"
        head += f" and {n_iso} isolated." if n_iso else "."
        lines = [head, "Systematic patterns:"]
        for p in sorted(pats, key=lambda x: -x["n_instances"]):
            n = p["n_instances"]
            plural = "" if n == 1 else "s"
            lines.append(f"  - {p['description']} [{p['slice_id']}, {n} obj{plural}.]")
            if p.get("details"):
                lines.append(f"      \u00b7 {p['details']}")
    out[name] = {"n_errors": n_total, "n_systematic": n_sys, "n_isolated": n_iso,
                 "patterns": pats, "text": "\n".join(lines)}

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/descriptions_train36_full.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
with open(f"{OUT}/descriptions_train36_full.txt", "w", encoding="utf-8") as f:
    for name in names:
        f.write(out[name]["text"] + "\n\n")
with open(f"{OUT}/instances_train36.pkl", "wb") as f:
    pickle.dump({"errors": errors, "reference": reference,
                 "image_paths": image_paths}, f)

described = sum(1 for v in out.values() if v["n_systematic"])
nb = Counter(p["branch"] for v in out.values() for p in v["patterns"])
print(f"\nimages with >=1 pattern: {described}/{len(names)} | pattern rows by branch: "
      f"{dict(nb)}", flush=True)
print(f"saved to {OUT}/", flush=True)
print("TRAIN36_GPU_DONE", flush=True)
