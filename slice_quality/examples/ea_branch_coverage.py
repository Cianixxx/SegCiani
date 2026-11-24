"""Planted-cause test, branch by branch: how many Pascal-EA victims would SliceLens alone have found?

ea_planted_eval.py counts the victims (edited subjects the segmenter lost) inside ANY certified slice of the
val-EA run. SliceLens alone is the language branch alone: here the same victims are split by which branch
holds them -- language only, cluster only, both -- with the same linking of a GT instance to the error
instances that concern it. The language slices of val-EA are certified by the slope rule (that run was not
moved to the AUC verdict), which is exactly SliceLens as published.
Run (CPU, after ea_planted_eval.py):  python slice_quality/examples/ea_branch_coverage.py
"""
import json
import os
import pickle
from collections import defaultdict

EA_RUN = os.environ.get("EA_RUN", "outputs/slice_quality_ea")
pl = json.load(open(f"{EA_RUN}/planted_eval.json", encoding="utf-8"))
idx = json.load(open(f"{EA_RUN}/slices_index.json", encoding="utf-8"))["slices"]
member = defaultdict(set)                              # error instance -> branches of its certified slices
for s in idx:
    if s.get("in_descriptions"):
        for ids in s["images"].values():
            for i in ids:
                member[i].add("cluster" if s["slice_id"].startswith("cluster") else "language")
errs = {t.instance_id: t for t in pickle.load(open(f"{EA_RUN}/result_tagged.pkl", "rb"))["errors"]}
link = defaultdict(list)                               # as ea_planted_eval.linked()
for t in errs.values():
    keys = {t.instance_id, t.metadata.get("gt_id"), t.metadata.get("primary_gt")} | set(t.metadata.get("gt_ids") or [])
    for k in keys:
        if k:
            link[k].append(t.instance_id)
res = {"language": 0, "cluster": 0, "both": 0, "none": 0}
for v in pl["victims"]:
    br = set().union(*[member.get(i, set()) for i in link.get(v, [])]) if link.get(v) else set()
    res["both" if len(br) == 2 else (next(iter(br)) if br else "none")] += 1
n = len(pl["victims"])
lang_total, union = res["language"] + res["both"], n - res["none"]
out = {"victims": n, **res, "language_total": lang_total, "union": union, "covered_in_eval": len(pl["covered"])}
lines = [f"PASCAL-EA VICTIMS BY BRANCH ({EA_RUN}): {n} victims",
         f"  in a certified LANGUAGE slice (SliceLens alone, slope rule): {lang_total}  ({100 * lang_total / n:.0f}%)",
         f"  in a certified CLUSTER slice only: {res['cluster']} | in both branches: {res['both']}",
         f"  in any certified slice (the pipeline): {union}  ({100 * union / n:.0f}%)  "
         f"[ea_planted_eval.py: {len(pl['covered'])}]"]
print("\n".join(lines))
json.dump(out, open(f"{EA_RUN}/branch_coverage.json", "w"), indent=1)
open(f"{EA_RUN}/branch_coverage.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("EA_BRANCH_DONE")
