"""The rank-test verdict (slope_permutation.py PHASE=rank, unchanged statistic) for one directory of
groundings, beside SliceLens's slope rule recorded by the run. CPU only.

Run: RUNS_DIR=/data/outputs_dd_sbd_ground python slice_quality/examples/dd_auc.py
Env: RUNS_DIR, STAGE (cc | full), OUT, BB (500), Q (0.05), RANK_COPY (a directory: copies of the grounding
     files with the AUC verdict in place of the slope rule, as slope_permutation.py writes <dir>_rank),
     MIN_CORRECT (10: on the whole class, a class with fewer correct objects is not tested; 0 = as before 24 October)
"""
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import norm, rankdata

RUNS_DIR = os.environ.get("RUNS_DIR", "/data/outputs_dd_sbd_ground")
STAGE = os.environ.get("STAGE", "cc")
OUT = os.environ.get("OUT", f"{RUNS_DIR}/dd_auc_{STAGE}")
BB, Q = int(os.environ.get("BB", "500")), float(os.environ.get("Q", "0.05"))
MIN_CORRECT = int(os.environ.get("MIN_CORRECT", "10"))
rng = np.random.default_rng(0)


def auc(x, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_test(confs, lab, ids):
    x, y = np.asarray(confs, float), np.asarray(lab, bool)
    a = auc(x, y)
    if a is None:
        return 0.5, 1.0
    groups = defaultdict(list)
    for k, i in enumerate(ids):
        groups[i.split("#")[0]].append(k)
    gl = [np.array(v) for v in groups.values()]
    boots = [v for v in (auc(x[s], y[s]) for s in (np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
                                                     for _ in range(BB))) if v is not None]
    se = float(np.std(boots)) if len(boots) > 10 else 0.0
    return a, (float(norm.sf((a - 0.5) / se)) if se > 0 else (0.0 if a > 0.5 else 1.0))


rows = []
for f in sorted(os.listdir(RUNS_DIR)):
    if not (f.startswith("voc_paper_") and f.endswith(f"_{STAGE}.json")):
        continue
    r = json.load(open(os.path.join(RUNS_DIR, f)))
    for x in r["candidates"]:
        sc = x["scored"]
        if STAGE == "full" and len(sc) != r["num_instances"]:
            continue                              # carried over from the sample, not re-grounded
        a, p = rank_test([s["conf"] for s in sc], [s["is_error"] for s in sc], [s["instance_id"] for s in sc])
        rows.append({"cls": r["label"], "query": x["query"], "n": len(sc), "slope": x["max_positive_slope"],
                     "slope_sys": bool(x["is_systematic"]), "auc": a, "p": p,
                     "n_ok": sum(1 for s in sc if not s["is_error"])})
# on the whole class, a class with fewer than MIN_CORRECT correct objects is not tested (rank_min_correct.py): its
# hypotheses leave the BH family. Every p-value is still computed, so the bootstrap draws of the others do not change.
for r in rows:
    r["untested"] = STAGE == "full" and r["n_ok"] < MIN_CORRECT
fam = [r for r in rows if not r["untested"]]
m = len(fam)
order = sorted(range(m), key=lambda i: fam[i]["p"])
run = 1.0
for rank in range(m, 0, -1):
    i = order[rank - 1]
    run = min(run, fam[i]["p"] * m / rank)
    fam[i]["q"] = run
for r in rows:
    r.setdefault("q", None)
    r["auc_sys"] = r["q"] is not None and r["q"] <= Q
lines = [f"{RUNS_DIR} stage {STAGE}: {len(rows)} hypotheses, {m} tested (minimum {MIN_CORRECT} correct objects) | "
         f"slope rule systematic {sum(r['slope_sys'] for r in rows)} | "
         f"rank test (BH q <= {Q}) {sum(r['auc_sys'] for r in rows)} | both {sum(r['slope_sys'] and r['auc_sys'] for r in rows)} | "
         f"median AUC {np.median([r['auc'] for r in rows]):.3f}"]
for r in sorted(rows, key=lambda r: (r["cls"], -r["auc"])):
    lines.append(f"  {r['cls']:12s} AUC {r['auc']:.3f} q {'  —  ' if r['q'] is None else '%.3f' % r['q']} "
                 f"{'AUC' if r['auc_sys'] else '   '} "
                 f"slope {r['slope']:.2f} {'SYS' if r['slope_sys'] else '   '} | {r['query']}")
print("\n".join(lines), flush=True)
json.dump(rows, open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
if os.environ.get("RANK_COPY"):
    dst = os.environ["RANK_COPY"]
    os.makedirs(dst, exist_ok=True)
    verdict = {(r["cls"], r["query"]): r for r in rows}
    for f in sorted(os.listdir(RUNS_DIR)):
        if not (f.startswith("voc_paper_") and f.endswith(f"_{STAGE}.json")):
            continue
        r = json.load(open(os.path.join(RUNS_DIR, f)))
        for x in r["candidates"]:
            v = verdict.get((r["label"], x["query"]))
            x["is_systematic_slope020"] = x["is_systematic"]
            x["is_systematic"] = bool(v and v["auc_sys"])
            x["auc"], x["auc_p"], x["auc_q"] = (v["auc"], v["p"], v["q"]) if v else (None, None, None)
            if v and v["untested"]:
                x["rank_untested"] = f"fewer than {MIN_CORRECT} correct objects in the class"
        r["systematic_error_slices"] = sorted(x["query"] for x in r["candidates"] if x["is_systematic"])
        json.dump(r, open(os.path.join(dst, f), "w"))
    print(f"rank copies -> {dst}", flush=True)
print("DD_AUC_DONE", flush=True)
