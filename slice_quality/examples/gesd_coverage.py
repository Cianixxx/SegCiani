"""Errors inside a certified group on GESD, as for VOC val and SBD in slicelens_cmp_stats.py (its coverage() unchanged):
SliceLens (slope rule, language branch, thesis gates) against the rank test (language branch) against the pipeline
(both branches), on the two GESD rescores of gesd_cmp.py. CPU only.
  -> outputs/slice_quality_gesd/gesd_coverage.{json,txt}
Env: OUT_SLOPE_RUN (outputs/slice_quality_gesd_cmp_slope), OUT_AUC_RUN (outputs/slice_quality_gesd_cmp_auc), BOOT (5000)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "slicelens_cmp_stats.py"), encoding="utf-8").read()
exec(compile(src.split('say("=" * 110)')[0], "slicelens_cmp_stats_top", "exec"))

RUN_S = os.environ.get("OUT_SLOPE_RUN", "outputs/slice_quality_gesd_cmp_slope")
RUN_A = os.environ.get("OUT_AUC_RUN", "outputs/slice_quality_gesd_cmp_auc")
L = "outputs/slice_quality_gesd"
say("ERRORS INSIDE A CERTIFIED GROUP ON GESD (same errors under the three columns)")
r = coverage("GESD", RUN_S, RUN_A)
say(f"  GESD     rates: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in r["rate"].items())
    + f" ({r['n_err']} errors, {r['n_img']} images)")
# per class: the errors, the share inside a certified group under each column, the certified groups of each run
pk = pickle.load(open(f"{RUN_A}/result_tagged.pkl", "rb"))
cls = {e.instance_id: (e.gt_class or e.pred_class) for e in pk["errors"]}
_, la, _ = membership(RUN_S)
_, lb, cb = membership(RUN_A)
used = lambda d: {p["slice_id"] for v in J(f"{d}/image_descriptions.json").values() for p in v.get("patterns", [])}
nsl = {k: defaultdict(int) for k in ("slope", "auc", "cluster")}
for d, key in ((RUN_S, "slope"), (RUN_A, "auc")):
    pkd, u, seen = pickle.load(open(f"{d}/result_tagged.pkl", "rb")), used(d), set()
    for s, q in pkd["result"].scored:
        if q.slice_id in u and q.slice_id not in seen:
            seen.add(q.slice_id)
            if s.source == "cluster":
                nsl["cluster"]["all"] += 1 if key == "auc" else 0
            else:
                nsl[key][s.slice_id.rsplit("_", 1)[0]] += 1
r["by_class"] = {}
say("\nper class: errors | inside a certified group: SliceLens, + rank test, pipeline | certified phrases: slope, rank")
for c in sorted(set(cls.values())):
    ids = [i for i, k in cls.items() if k == c]
    row = {"n_err": len(ids), "SliceLens": sum(i in la for i in ids) / len(ids),
           "+ rank test": sum(i in lb for i in ids) / len(ids),
           "pipeline": sum(i in lb or i in cb for i in ids) / len(ids),
           "phrases_slope": nsl["slope"].get(c, 0), "phrases_auc": nsl["auc"].get(c, 0)}
    r["by_class"][c] = row
    say(f"  {c:11s} {row['n_err']:4d} | {100 * row['SliceLens']:5.1f}% {100 * row['+ rank test']:5.1f}% "
        f"{100 * row['pipeline']:5.1f}% | {row['phrases_slope']:2d} {row['phrases_auc']:2d}")
say(f"  cluster groups in the pipeline: {nsl['cluster']['all']}")
os.makedirs(L, exist_ok=True)
json.dump(r, open(f"{L}/gesd_coverage.json", "w"), indent=1)
open(f"{L}/gesd_coverage.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("GESD_COVERAGE_DONE", flush=True)
