"""Where SliceLens's slope rule and the AUC test disagree on VOC: which situations hold up? CPU only.

The thesis run certified its language situations with the slope rule until 21 October and with the AUC
test since: 31 situations are certified by both, some only by the AUC test, some only by the slope rule.
Two checks that use NEITHER test decide which group is more real:
  precision  Qwen asked again, object by object, whether the phrase is true of 10 members of the
             situation, against same-class objects outside it (eval_description_precision.py, run on the
             AUC run and on the slope run: outputs/slice_quality_inst{,_slope}/eval_precision.json)
  model 2    the same phrase grounded on the SECOND segmenter's objects (same images, a stronger model,
             different errors): are the objects the phrase fits (conf >= 0.5) wrong more often than the
             others of the class? One-sided Fisher exact test, BH over all the situations. The AUC and the
             slope verdict on model 2 are reported beside it, for completeness.
Inputs: the two slice indexes, the two eval_precision.json, model-2 groundings in M2_DIRS.
Run (CPU):  python slice_quality/examples/disagreement_report.py
"""
import glob
import json
import re
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
from scipy.stats import fisher_exact, norm, rankdata

from slicelens.identification import analyze_slope_trend

AUC_RUN = os.environ.get("AUC_RUN", "outputs/slice_quality_inst")
SLOPE_RUN = os.environ.get("SLOPE_RUN", "outputs/slice_quality_inst_slope")
M2_DIRS = os.environ.get("M2_DIRS", "/data/outputs_model2_inst:/data/outputs_model2_slopeonly").split(":")
OUT = os.environ.get("OUT", "outputs/slope_vs_auc_disagreement")
rng = np.random.default_rng(0)


def situations(run):
    out = {}
    for s in json.load(open(f"{run}/slices_index.json", encoding="utf-8"))["slices"]:
        if s["source"] == "language" and s.get("in_descriptions"):
            cls = re.sub(r"_(extra|dd|pend)$", "", s["slice_id"].rsplit("_", 1)[0])
            out[(cls, s["hypothesis"])] = s["slice_id"]
    return out


def precision(run, sits):
    """(cls, hypothesis) -> (precision on members, same-class control rate, n) from eval_precision.json."""
    f = f"{run}/eval_precision.json"
    if not os.path.exists(f):
        return {}
    by_id = {v: k for k, v in sits.items()}
    out = {}
    for g in json.load(open(f, encoding="utf-8"))["groups"]:
        if g["branch"] != "language":
            continue
        for sid in g["slices"]:
            if sid in by_id:
                out[by_id[sid]] = (g["precision"], g.get("control_rate_same_class"), g["n_pos"])
    return out


def auc(x, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_p(x, y, img, bb=300):
    a = auc(x, y)
    if a is None:
        return None, 1.0
    groups = defaultdict(list)
    for k, i in enumerate(img):
        groups[i].append(k)
    gl = [np.array(v) for v in groups.values()]
    boots = [v for v in (auc(x[s], y[s]) for s in (np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
                                                     for _ in range(bb))) if v is not None]
    se = float(np.std(boots)) if len(boots) > 10 else 0.0
    return a, (float(norm.sf((a - 0.5) / se)) if se > 0 else (0.0 if a > 0.5 else 1.0))


def bh(ps, q=0.05):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qv, run = [1.0] * m, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, ps[i] * m / r)
        qv[i] = run
    return qv


A, S = situations(AUC_RUN), situations(SLOPE_RUN)
groups = {"both": sorted(A.keys() & S.keys()), "AUC only": sorted(A.keys() - S.keys()),
          "slope only": sorted(S.keys() - A.keys())}
prec = {**precision(SLOPE_RUN, S), **precision(AUC_RUN, A)}     # the AUC run's value wins for "both"

m2 = {}
for d in M2_DIRS:
    for f in sorted(glob.glob(f"{d}/voc_paper_*_full.json")):
        r = json.load(open(f))
        for x in r["candidates"]:
            sc = x["scored"]
            if len(sc) != r["num_instances"] or (r["label"], x["query"]) in m2:
                continue
            conf = np.array([s["conf"] for s in sc], float)
            err = np.array([bool(s["is_error"]) for s in sc])
            img = [s["instance_id"].split("#")[0] for s in sc]
            mem = conf >= 0.5
            t = [[int((err & mem).sum()), int((~err & mem).sum())], [int((err & ~mem).sum()), int((~err & ~mem).sum())]]
            p_f = fisher_exact(t, alternative="greater")[1] if mem.any() and (~mem).any() else 1.0
            lift = (t[0][0] / max(mem.sum(), 1)) - (t[1][0] / max((~mem).sum(), 1))
            a, p_a = auc_p(conf, err, img)
            cand = SimpleNamespace(scored_instances=[SimpleNamespace(slice_confidence=float(c),
                                                                     instance=SimpleNamespace(is_error=bool(e)))
                                                     for c, e in zip(conf, err)])
            m2[(r["label"], x["query"])] = dict(n=len(sc), members=int(mem.sum()), lift=lift, p_fisher=p_f,
                                                 auc=a, p_auc=p_a,
                                                 slope_sys=bool(analyze_slope_trend(cand).is_systematic))
allk = [k for g in groups.values() for k in g if k in m2]
for k, qf, qa in zip(allk, bh([m2[k]["p_fisher"] for k in allk]), bh([m2[k]["p_auc"] for k in allk])):
    m2[k]["q_fisher"], m2[k]["q_auc"] = qf, qa

lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))
say("=" * 100)
say("WHERE THE TWO TESTS DISAGREE (VOC val, thesis run): which situations hold up on checks neither test uses?")
say("=" * 100)
rows = []
for gname, keys in groups.items():
    pr = [prec[k][0] for k in keys if k in prec]
    pc = [prec[k][1] for k in keys if k in prec and prec[k][1] is not None]
    mm = [m2[k] for k in keys if k in m2]
    rep = sum(1 for v in mm if v["q_fisher"] <= 0.05 and v["lift"] > 0)
    row = dict(group=gname, n=len(keys), n_prec=len(pr), precision=float(np.mean(pr)) if pr else None,
               control=float(np.mean(pc)) if pc else None, n_m2=len(mm), replicated=rep,
               lift=float(np.mean([v["lift"] for v in mm])) if mm else None,
               auc_m2=float(np.mean([v["auc"] for v in mm if v["auc"] is not None])) if mm else None,
               auc_sig_m2=sum(1 for v in mm if v["q_auc"] <= 0.05), slope_sys_m2=sum(v["slope_sys"] for v in mm))
    rows.append(row)
    say(f"\n{gname.upper()}: {len(keys)} situations")
    if pr:
        say(f"  precision (Qwen re-asked on 10 members): {row['precision']:.2f} vs same-class control "
            f"{row['control']:.2f}  [{len(pr)} situations]")
    if mm:
        say(f"  model 2: members wrong more often than the rest (Fisher, BH q <= 0.05): {rep}/{len(mm)} | "
            f"mean lift {row['lift']:+.3f} | mean AUC {row['auc_m2']:.2f} | AUC-significant {row['auc_sig_m2']}/{len(mm)}"
            f" | slope-systematic {row['slope_sys_m2']}/{len(mm)}")
    for k in keys:
        p = prec.get(k)
        v = m2.get(k)
        say(f"    {k[0]:12s} {k[1][:55]:55s} | prec {p[0] if p else float('nan'):.2f} | "
            + (f"m2 lift {v['lift']:+.2f} q {v['q_fisher']:.3f} AUC {v['auc'] if v['auc'] is not None else float('nan'):.2f}"
               if v else "m2 -"))
json.dump({"rows": rows, "per_situation": {f"{k[0]}|{k[1]}": {"group": g, "precision": prec.get(k), "m2": m2.get(k)}
                                           for g, ks in groups.items() for k in ks}},
          open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("DISAGREEMENT_DONE", flush=True)
