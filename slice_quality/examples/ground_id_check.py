"""Do the groundings of a run name the same objects as the run's own typing? CPU only, reads only.

The official rescore joins a grounded object to a typed error by its id. If the VLM read the objects of another
version of Block 1, the same id can name another object. For every object the VLM read, we look up the same id among
the run's typed objects and compare what both sides record: the class, wrong or correct, and the IoU. On the typing the
groundings were made on, all three agree for every id.
Run (CPU, server):  python -u slice_quality/examples/ground_id_check.py
Env: SQ_OUT (the run), LANG_DIRS (its grounding folders, colon-separated, as in the rescore), TAG (a name for the output)
"""
import json
import os
import pickle
from collections import Counter

SQ_OUT = os.environ["SQ_OUT"]
LANG_DIRS = os.environ["LANG_DIRS"].split(":")
TAG = os.environ.get("TAG", os.path.basename(SQ_OUT))

d = pickle.load(open(os.path.join(SQ_OUT, "result_tagged.pkl"), "rb"))
fin = {t.instance_id: (True, t) for t in d["errors"]}
fin.update({t.instance_id: (False, t) for t in d["reference"]})
seen, subs, files = {}, Counter(), 0
for dd in LANG_DIRS:
    names = sorted(os.listdir(dd))
    classes = sorted({f.split("_paper_")[1].split(".json")[0].replace("_full", "").replace("_cc", "")
                      for f in names if "_paper_" in f and f.endswith(".json")})
    for cls in classes:
        f = next((x for x in names if x.endswith("_paper_%s_full.json" % cls)), None) \
            or next((x for x in names if x.endswith("_paper_%s_cc.json" % cls)), None) \
            or next((x for x in names if x.endswith("_paper_%s.json" % cls)), None)
        if f is None:
            continue
        r = json.load(open(os.path.join(dd, f), encoding="utf-8"))
        files += 1
        subs[r.get("substrate", "?")] += 1
        for c in r.get("candidates", []):
            for s in c.get("scored") or c.get("top_instances") or []:
                if "instance_id" in s:
                    seen[s["instance_id"]] = (r.get("label"), bool(s.get("is_error")), s.get("iou"),
                                              float(s.get("conf", s.get("slice_confidence", 0.0))))
cnt = Counter()
for iid, (cls, err, iou, conf) in seen.items():
    hit = fin.get(iid)
    if hit is None:
        cnt["id absent from the run's typing"] += 1
        continue
    ferr, t = hit
    same_cls = (t.gt_class or t.pred_class) == cls
    same_err = ferr == err
    same_iou = iou is None or abs(float(t.iou) - float(iou)) < 1e-3
    cnt["same id, class %s, label %s, IoU %s" % ("=" if same_cls else "DIFF", "=" if same_err else "DIFF",
                                                 "=" if same_iou else "DIFF")] += 1
print("=" * 90)
print(f"{TAG}: {files} grounding files, substrate {dict(subs)}; {len(seen)} grounded objects, "
      f"{sum(v[1] for v in seen.values())} wrong; run typing: {len(d['errors'])} errors, {len(d['reference'])} correct")
for k, v in cnt.most_common():
    print(f"  {k:52s} {v:6d}")
print("GROUND_ID_CHECK_DONE", flush=True)
