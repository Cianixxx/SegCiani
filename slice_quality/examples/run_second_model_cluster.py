"""Stage 2: the cluster branch, end to end, on the SECOND segmenter.

Stage 1 (compare_models_block1.py) compared error POPULATIONS. This one asks the
question that matters for the thesis: do the same KINDS of situation come back when
the model changes, or were they weaknesses of the first network?

Everything writes under outputs/slice_quality/model2/ and a separate pickle, so the
certified state of the first model is never touched and the two can be compared
side by side.

Phases: Block 1 (CPU) -> DINOv2 embeddings (GPU) -> attribute tagging in
VERIFICATION mode (GPU, the elicitation shown in S6.3 to be the reliable one) ->
scoring and certification (CPU) -> descriptions and indices.

Run (GPU):
  PYTHONPATH=/workspace python slice_quality/examples/run_second_model_cluster.py
Env: PRED_DIR (deeplab_pred), TAG_BUDGET (2000), TARGET_TAGGED (40), OUT_TAG (model2).
"""
import os
import pickle
from collections import Counter

from slice_quality import run_from_instances
from slice_quality.adapters.common import make_mask_lookup
from slice_quality.adapters.voc_seg_samples import build_voc_samples
from slice_quality.config import SliceQualityConfig
from slice_quality.describe import (build_image_descriptions,
                                    export_description_index, export_descriptions)
from slice_quality.embeddings import attach_embeddings
from slice_quality.report import export_slice_index, to_json
from slice_quality.tagging import tag_instances_verify
from slice_quality.typed_error import decompose_dataset

DATA = "/data/voc2012/VOCdevkit/VOC2012"
CROPS = "/data/voc2012/voc_crops"
PRED_DIR = os.environ.get("PRED_DIR", "deeplab_pred")
OUT_TAG = os.environ.get("OUT_TAG", "model2")
# The run the second model is compared with (default the reference run); for the inst run also set
# SQ_CLAIM_COV=0.5 SQ_GT_INSTANCES=object, so Block 1 types model 2's errors the same way.
OUT = f"{os.environ.get('SQ_OUT', 'outputs/slice_quality')}/{OUT_TAG}"
PKL = f"{OUT}/result_tagged.pkl"
TAG_BUDGET = int(os.environ.get("TAG_BUDGET", "2000"))
TARGET_TAGGED = int(os.environ.get("TARGET_TAGGED", "40"))

os.makedirs(OUT, exist_ok=True)
names = [l.strip() for l in
         open(f"{DATA}/ImageSets/Segmentation/val.txt", encoding="utf-8") if l.strip()]

# ---- Block 1 -------------------------------------------------------------
print(f"[1/5] Block 1 on {PRED_DIR} ...", flush=True)
samples, image_paths = build_voc_samples(DATA, CROPS, names, pred_subdir=PRED_DIR)
errors, reference = decompose_dataset(samples)
print(f"      {len(errors)} errors, {len(reference)} reference | "
      f"{dict(Counter(e.seg_subtype.value for e in errors).most_common())}", flush=True)
# model 2's instances alone, so the language branch can be re-grounded on them (reground_validated.py
# --substrate-cache) while the embeddings and the tagging below are still running
pickle.dump({"errors": errors, "reference": reference, "image_paths": image_paths},
            open(f"{OUT}/substrate.pkl", "wb"))
print(f"      substrate -> {OUT}/substrate.pkl", flush=True)

# ---- embeddings ----------------------------------------------------------
print("[2/5] DINOv2 embeddings ...", flush=True)
from slice_quality.extractors import DINOv2Extractor
extractor = DINOv2Extractor(image_paths)
mask_lookup = make_mask_lookup(samples)
attach_embeddings(errors + reference, extractor, mask_lookup)
n_ok = sum(1 for i in errors + reference if i.embedding is not None)
print(f"      embedded {n_ok}/{len(errors) + len(reference)}", flush=True)
del extractor

# ---- first clustering pass, to know WHICH instances are worth tagging ----
cfg = SliceQualityConfig()
cfg.cluster.n_slices = 20
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200

print("[3/5] provisional clustering (untagged) to target the tagging budget ...",
      flush=True)
res0 = run_from_instances(errors, reference, language_slices=None, cfg=cfg)
seen, cands = set(), []
for s, q in res0.scored:
    if s.source != "cluster" or q.slice_id in seen:
        continue
    seen.add(q.slice_id)
    cands.append((s, q))
cands.sort(key=lambda sq: -sq[1].size)
to_tag, chosen = [], set()
for s, _q in cands:
    added = 0
    for m in s.members:
        if len(to_tag) >= TAG_BUDGET or added >= TARGET_TAGGED:
            break
        if m.instance_id in chosen:
            continue
        chosen.add(m.instance_id)
        to_tag.append(m)
        added += 1
    if len(to_tag) >= TAG_BUDGET:
        break
print(f"      {len(cands)} candidate slices | tagging {len(to_tag)} instances", flush=True)

# ---- verification-mode tagging ------------------------------------------
print("[4/5] Qwen attribute tagging (verification mode) ...", flush=True)
from slicelens.backends.qwen_vlm import QwenVLM
vlm = QwenVLM(model=os.environ.get("VLM_MODEL", "/models/qwen"), max_new_tokens=10,
              load_in_4bit=os.environ.get("QWEN_4BIT") == "1")
tag_instances_verify(to_tag, image_paths, vlm)
print("      ATTRIBUTES:",
      dict(Counter(m.metadata.get("attribute", "?") for m in to_tag)), flush=True)
del vlm

# ---- final scoring -------------------------------------------------------
print("[5/5] scoring with the tags in place ...", flush=True)
res = run_from_instances(errors, reference, language_slices=None, cfg=cfg)
kept = res.kept
print(f"KEPT cluster slices = {len(kept)} | residual = {res.residual_error:.3f}",
      flush=True)
if kept:
    ps = [k.quality.slice_purity for k in kept]
    print("  subtypes:", dict(Counter(k.quality.dominant_error_type.value for k in kept)),
          flush=True)
    print("  purity min/med/max: %.2f %.2f %.2f | max p: %.4f" % (
        min(ps), sorted(ps)[len(ps) // 2], max(ps),
        max(k.quality.extras.get("purity_p", 1) for k in kept)), flush=True)
    cov = len({m.instance_id for k in kept for m in k.slice_.members})
    print("  instance coverage: %.1f%%" % (100 * cov / len(errors)), flush=True)
    print("  omega2 range: %.3f .. %.3f" % (
        min(k.quality.omega_squared for k in kept),
        max(k.quality.omega_squared for k in kept)), flush=True)

pickle.dump({"result": res, "errors": errors, "reference": reference,
             "image_paths": image_paths, "descs": None}, open(PKL, "wb"))
descs = build_image_descriptions(res)
export_descriptions(descs, f"{OUT}/image_descriptions.json", f"{OUT}/image_descriptions.txt")
export_description_index(descs, f"{OUT}/descriptions_index.json",
                         f"{OUT}/descriptions_index.txt")
to_json(res, f"{OUT}/voc_val_{OUT_TAG}.json")
used = {p["slice_id"] for d in descs.values() for p in d["patterns"]}
export_slice_index(res, f"{OUT}/slices_index.json", f"{OUT}/slices_index.txt",
                   used_ids=used)
n_sys = sum(1 for v in descs.values() if v["n_systematic"] > 0)
print(f"descriptions: {n_sys} of {len(descs)} error images have a certified pattern",
      flush=True)
print("STAGE2_DONE", flush=True)
