"""Stage 3: combine the language branch (re-grounded on the SECOND model's
instances) with stage 2's cluster branch, producing a real two-branch result
for the stronger segmenter.

The 35 hypotheses are not re-discovered here -- they were already validated on
model 1 (slope-trend verified, re-grounded on its full population). This
script only asks whether the SAME situations hold on model 2's own instances,
via slice_quality/examples/../../examples/reground_validated.py (run first,
separately, against model2's substrate).

Reads:
  outputs/slice_quality/model2/result_tagged.pkl   (errors/reference/image_paths
      + the cluster-only `result` from run_second_model_cluster.py)
  /data/outputs_model2/voc_paper_<cls>_full.json    (written by reground_validated.py
      pointed at the same substrate pickle and at model 1's slices_index.json)

Writes (CPU only):
  outputs/slice_quality/model2/{image_descriptions,descriptions_index,slices_index}.*
  outputs/slice_quality/model2/result_tagged.pkl (overwritten with the combined result)
"""
import os
import pickle

from slice_quality import run_from_instances
from slice_quality.adapters.slicelens_run import build_language_slices_from_run
from slice_quality.config import SliceQualityConfig
from slice_quality.corroborate import cross_branch_corroboration
from slice_quality.describe import (build_image_descriptions,
                                    export_description_index, export_descriptions)
from slice_quality.language_branch import dedup_language_slices
from slice_quality.report import export_slice_index, to_json

OUT = f"{os.environ.get('SQ_OUT', 'outputs/slice_quality')}/{os.environ.get('OUT_TAG', 'model2')}"
PKL = f"{OUT}/result_tagged.pkl"
LANG_DIR = os.environ.get("LANG_DIR", "/data/outputs_model2")

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

d = pickle.load(open(PKL, "rb"))
errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
cluster_only_kept = len(d["result"].kept)
print(f"loaded {len(errors)} errors, {len(reference)} reference | "
      f"cluster-only kept: {cluster_only_kept}", flush=True)

language, missing = [], []
for cls in CLASSES:
    f = f"{LANG_DIR}/voc_paper_{cls}_full.json"
    if os.path.exists(f):
        sl = build_language_slices_from_run(f, errors, conf_threshold=0.5, prefix=cls)
        print(f"  language[{cls}]: {len(sl)} slices", flush=True)
        language += sl
    else:
        missing.append(cls)
if missing:
    print(f"  no re-grounded file (not validated on model 1, or not yet run): "
          f"{missing}", flush=True)
n_before = len(language)
language = dedup_language_slices(language)
print(f"language slices total: {n_before} -> {len(language)} after dedup", flush=True)

cfg = SliceQualityConfig()
cfg.cluster.n_slices = 20
cfg.def1.n_min_frac = 0.005
cfg.significance.n_boot = 200
res = run_from_instances(errors, reference, language_slices=language, cfg=cfg,
                         max_peeling_rounds=5, residual_stop=0.05)
print(f"KEPT (cluster+language) = {len(res.kept)} (was {cluster_only_kept} "
      f"cluster-only) | residual={res.residual_error:.3f}", flush=True)
kept_lang = sum(1 for k in res.kept if k.slice_.source == "language")
kept_clus = sum(1 for k in res.kept if k.slice_.source == "cluster")
print(f"  by branch: cluster {kept_clus}, language {kept_lang}", flush=True)

pickle.dump({"result": res, "errors": errors, "reference": reference,
             "image_paths": image_paths, "descs": None}, open(PKL, "wb"))

descs = build_image_descriptions(res)
lang_ids = {p["slice_id"] for dd in descs.values()
            for p in dd["patterns"] if p["branch"] == "language"}
cor = cross_branch_corroboration(res, lang_ids)
print(f"corroborated cluster slices: {len(cor)}", flush=True)
export_descriptions(descs, f"{OUT}/image_descriptions.json", f"{OUT}/image_descriptions.txt")
export_description_index(descs, f"{OUT}/descriptions_index.json",
                         f"{OUT}/descriptions_index.txt")
to_json(res, f"{OUT}/voc_val_model2_full.json")
used = {p["slice_id"] for dd in descs.values() for p in dd["patterns"]}
export_slice_index(res, f"{OUT}/slices_index.json", f"{OUT}/slices_index.txt",
                   used_ids=used)

n_sys = sum(1 for v in descs.values() if v["n_systematic"] > 0)
cov = len({m.instance_id for k in res.kept for m in k.slice_.members})
print(f"images with a certified pattern: {n_sys} of {len(descs)}", flush=True)
print(f"instance coverage: {100 * cov / len(errors):.1f}%", flush=True)
print("STAGE3_DONE", flush=True)
