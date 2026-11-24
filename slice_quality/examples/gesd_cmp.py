"""SliceLens against the pipeline on GESD, the benchmark published with SliceLens: the same objects, VLM and
hypotheses, only the verdict changes. CPU only.

The three columns of §18.6, here with GESD's own measure:
  SliceLens        the slope rule at both steps: the hypotheses slope-systematic on the 120-object sample go to the
                   whole class, and those still slope-systematic there are certified
  + rank test      the rank test at both steps (one-sided AUC, image-block bootstrap, BH within the step)
  pipeline         the rank test at both steps, the gates of the language branch, and the cluster branch
Each column twice: the verdict alone (a slice = the class's errors with P(yes) >= 0.5, ranked by P(yes)) and after the
language branch's gates of the thesis (the certified slices of a rescore with that verdict). The report then repeats
the measure with the correct objects kept in the language slices (every object of the class with P(yes) >= 0.5): the
paper ranks correct and erroneous units alike and does not say whether its output slice drops the correct ones.

MODE=prep    the hypotheses SliceLens would send to the whole class but never grounded there -> /data/full_slopeonly.json;
             the rank-test copy of the whole-class groundings -> /data/outputs_gesd_cmp_auc (for the rescore)
MODE=slope   after the grounding of /data/full_slopeonly.json: the slope-rule copy -> /data/outputs_gesd_cmp_slope
MODE=report  Precision@10 of each column on the 21 expert slices (paired Wilcoxon over the slices, bootstrap
             interval), the expert situations each column names, and per object the naming of the experts'
             situation against the false naming of correct objects -> outputs/slice_quality_gesd/gesd_cmp.{txt,json}
Env: MODE, OUT_SLOPE_RUN (outputs/slice_quality_gesd_cmp_slope), OUT_AUC_RUN (outputs/slice_quality_gesd_cmp_auc)
"""
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, norm, rankdata, wilcoxon

from slice_quality.describe import _GENERIC_EXTRA

MODE = os.environ.get("MODE", "report")
L, G = "outputs/slice_quality_gesd", "/data"
RUN_S = os.environ.get("OUT_SLOPE_RUN", "outputs/slice_quality_gesd_cmp_slope")
RUN_A = os.environ.get("OUT_AUC_RUN", "outputs/slice_quality_gesd_cmp_auc")
CLASSES = "airplane apple banana bed book carrot clock keyboard orange suitcase surfboard teddy_bear tie umbrella vase".split()
CANON = ["a small object", "an object occluded by another object", "an object that blends into its background",
         "an object cut off at the image border", "an object with thin or elongated parts",
         "several overlapping objects of the same type", "an object that looks similar to a different object category"]
FULL_DIRS = [f"{G}/outputs_gesd", f"{G}/outputs_gesd_naming", f"{G}/outputs_gesd_slopeonly"]
K, Q, BB = 10, 0.05, 500
rng = np.random.default_rng(0)
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))


def auc(x, y):
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_test(confs, lab, ids):
    """slope_permutation.py's verdict, unchanged (as gesd_pointB.py and dd_auc.py)"""
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


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qv, run = [1.0] * m, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, ps[i] * m / r)
        qv[i] = run
    return [x <= Q for x in qv]


# ---- step 1, the sample: the verdicts of gesd_pointB.py STAGE=cc (both rules, all 538 hypotheses)
cc = {(v["cls"], v["q"]): v for v in json.load(open(f"{L}/pointB_cc.json", encoding="utf-8"))["verdicts"]}
SET_S = {k for k, v in cc.items() if v["slope_sys"]}            # SliceLens sends these to the whole class
SET_A = {k for k, v in cc.items() if v["auc_sys"]}              # the rank test sends these


def whole_class():
    """(cls, q) -> the whole-class candidate and the file it comes from, first directory that has it"""
    out = {}
    for d in FULL_DIRS:
        for c in CLASSES:
            f = f"{d}/voc_paper_{c}_full.json"
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            for x in r["candidates"]:
                if len(x["scored"]) == r["num_instances"] and (c, x["query"]) not in out:
                    out[(c, x["query"])] = (x, r)
    return out


def copy_with(dst, verdict):
    """the whole-class groundings with is_systematic = verdict((cls, q)); one file per class, as dd_auc.py RANK_COPY"""
    W = whole_class()
    os.makedirs(dst, exist_ok=True)
    for c in CLASSES:
        base = f"{G}/outputs_gesd/voc_paper_{c}_full.json"
        if not os.path.exists(base):
            continue
        r = json.load(open(base))
        have = {x["query"] for x in r["candidates"] if len(x["scored"]) == r["num_instances"]}
        r["candidates"] += [x for (cc_, q), (x, _) in W.items() if cc_ == c and q not in have]   # other directories
        for x in r["candidates"]:
            full = len(x["scored"]) == r["num_instances"]
            x["is_systematic_recorded"] = x.get("is_systematic_recorded", x["is_systematic"])
            x["is_systematic"] = bool(full and verdict((c, x["query"]), x))
        r["systematic_error_slices"] = sorted(x["query"] for x in r["candidates"] if x["is_systematic"])
        json.dump(r, open(f"{dst}/voc_paper_{c}_full.json", "w"))


def auc_verdict_whole(W):
    """the rank test on the whole class for the hypotheses the rank test sent there; BH within them"""
    keys = sorted(k for k in SET_A if k in W)
    res = {k: rank_test([s["conf"] for s in W[k][0]["scored"]], [s["is_error"] for s in W[k][0]["scored"]],
                        [s["instance_id"] for s in W[k][0]["scored"]]) for k in keys}
    return {k: (res[k][0], ok) for k, ok in zip(keys, bh([res[k][1] for k in keys]))}


if MODE == "prep":
    W = whole_class()
    miss = defaultdict(list)
    for c, q in sorted(SET_S - set(W)):
        miss[c].append(q)
    json.dump(dict(miss), open(f"{G}/full_slopeonly.json", "w"), indent=1)
    say(f"sample: SliceLens sends {len(SET_S)} hypotheses to the whole class, the rank test {len(SET_A)}; already "
        f"grounded on the whole class {len(SET_S & set(W))} and {len(SET_A & set(W))}; to ground for SliceLens: "
        f"{sum(len(v) for v in miss.values())} in {len(miss)} classes -> {G}/full_slopeonly.json")
    V = auc_verdict_whole(W)
    copy_with(f"{G}/outputs_gesd_cmp_auc", lambda k, x: V.get(k, (0, False))[1])
    say(f"rank test on the whole class: {sum(ok for _a, ok in V.values())} of {len(V)} -> {G}/outputs_gesd_cmp_auc")
    print("GESD_CMP_PREP_DONE", flush=True)
    raise SystemExit
if MODE == "slope":
    W = whole_class()
    lost = SET_S - set(W)
    assert not lost, f"still not grounded on the whole class: {sorted(lost)[:5]}"
    copy_with(f"{G}/outputs_gesd_cmp_slope", lambda k, x: k in SET_S and bool(x["is_systematic_recorded"]))
    say(f"slope rule on the whole class: {sum(bool(W[k][0]['is_systematic']) for k in SET_S)} of {len(SET_S)} "
        f"-> {G}/outputs_gesd_cmp_slope")
    print("GESD_CMP_SLOPE_DONE", flush=True)
    raise SystemExit

# ---------------------------------------------------------------- report
kw = json.load(open(f"{L}/pointB_keywords.json", encoding="utf-8"))["slices"]
gesd = json.load(open("/priv/gesd/coco_segmentation.json"))
slice_units, unit_slice = defaultdict(set), {}
for a in gesd["annotations"]:
    unit_slice[str(a["id"])] = a["reason"]
    if a["reason"] != "NA":
        slice_units[a["reason"]].add(str(a["id"]))
umap = json.load(open(f"{G}/unit_map.json"))
inst_units = {i: [u["unit"] for u in v] for i, v in umap.items()}
inst_slices = {i: {u["slice"] for u in v} for i, v in umap.items()}
W = whole_class()
V = auc_verdict_whole(W)


def p_at_k(ranked, s, k=K):
    top = [u for i in ranked for u in inst_units.get(i, [])][:k]
    return sum(unit_slice.get(u) == s for u in top) / min(k, len(slice_units[s]))


def names(cls, q, spec):
    """the phrase names the expert situation: its class, one of the keywords fixed before the run, not a generic phrase"""
    return (cls == spec["class"] and q not in CANON and q not in _GENERIC_EXTRA
            and any(k in q.lower() for k in spec["keywords"]))


def verdict_slices(keys, with_correct=False):
    """per certified hypothesis, the class's errors with P(yes) >= 0.5 ranked by P(yes); with_correct: every object of
    the class with P(yes) >= 0.5, correct ones too (the paper's U_gamma, Sec. 3.4)"""
    out = []
    for c, q in keys:
        sc = sorted(W[(c, q)][0]["scored"], key=lambda s: -s["conf"])
        out.append(("language", c, q, [s["instance_id"] for s in sc if (with_correct or s["is_error"]) and s["conf"] >= 0.5]))
    return out


def gated_slices(run, branches, with_correct=False):
    d = pickle.load(open(f"{run}/result_tagged.pkl", "rb"))
    idx = {x["slice_id"]: x for x in json.load(open(f"{run}/slices_index.json"))["slices"]}
    out, seen = [], set()
    for s, _q in d["result"].scored:
        if s.slice_id in seen or not idx.get(s.slice_id, {}).get("in_descriptions"):
            continue
        seen.add(s.slice_id)
        meta = idx[s.slice_id]
        if meta.get("source") == "language" and "language" in branches:
            c = s.slice_id.rsplit("_", 1)[0]
            h = W.get((c, meta.get("hypothesis")))
            cf = {x["instance_id"]: x["conf"] for x in h[0]["scored"]} if h else {}
            ids = [i for i, p in cf.items() if p >= 0.5] if with_correct and h else [m.instance_id for m in s.members]
            out.append(("language", c, meta.get("hypothesis"), sorted(ids, key=lambda i: -cf.get(i, 0))))
        elif meta.get("source") == "cluster" and "cluster" in branches:
            out.append(("cluster", None, s.slice_id, [m.instance_id for m in s.members]))
    return out


cert_S = sorted(k for k in SET_S if k in W and W[k][0]["is_systematic"])
cert_A = sorted(k for k, (_a, ok) in V.items() if ok)
COLS = {"SliceLens, verdict only": verdict_slices(cert_S), "+ rank test, verdict only": verdict_slices(cert_A),
        "SliceLens, thesis gates": gated_slices(RUN_S, ("language",)),
        "+ rank test, thesis gates": gated_slices(RUN_A, ("language",)),
        "pipeline": gated_slices(RUN_A, ("language", "cluster"))}
EXPERT = sorted(kw, key=lambda s: -len(slice_units[s]))
out = {"n_expert": len(EXPERT), "sample": {"slope": len(SET_S), "auc": len(SET_A)},
       "whole": {"slope": len(cert_S), "auc": len(cert_A)}, "cols": {}}
say("=" * 110)
say(f"GESD: SLICELENS AGAINST THE PIPELINE, SAME OBJECTS, VLM AND HYPOTHESES ({len(EXPERT)} expert slices)")
say("=" * 110)
say(f"sample (120 objects per class): slope rule {len(SET_S)} hypotheses, rank test {len(SET_A)} | whole class: "
    f"slope rule {len(cert_S)}, rank test {len(cert_A)}")
P = {}
for col, sl in COLS.items():
    per, named_best = {}, 0
    for s in EXPERT:
        best = max(sl, key=lambda o: p_at_k(o[3], s), default=None)
        per[s] = p_at_k(best[3], s) if best else 0.0
        named_best += bool(best and best[0] == "language" and names(best[1], best[2], kw[s]))
    named = sum(any(o[0] == "language" and names(o[1], o[2], kw[s]) for o in sl) for s in EXPERT)
    P[col] = per
    out["cols"][col] = {"slices": len(sl), "language": sum(o[0] == "language" for o in sl),
                        "p10": float(np.mean(list(per.values()))), "at_1": sum(v >= 1.0 for v in per.values()),
                        "named": named, "best_named": named_best, "per_slice": per,
                        "named_slices": sorted(s for s in EXPERT if any(o[0] == "language" and names(o[1], o[2], kw[s])
                                                                         for o in sl))}
    o = out["cols"][col]
    say(f"  {col:28s} {o['slices']:3d} slices | Precision@10 {o['p10']:.3f}, {o['at_1']:2d} of {len(EXPERT)} at 1.0 | "
        f"expert situations named by a certified phrase {o['named']:2d} (by the best slice {o['best_named']})")

say("\npaired over the 21 expert slices (Wilcoxon signed-rank, bootstrap 95% interval of the mean difference):")
out["tests"] = {}
for a, b in (("SliceLens, verdict only", "+ rank test, verdict only"), ("SliceLens, thesis gates", "+ rank test, thesis gates"),
             ("SliceLens, thesis gates", "pipeline"), ("SliceLens, verdict only", "pipeline")):
    x, y = np.array([P[a][s] for s in EXPERT]), np.array([P[b][s] for s in EXPERT])
    d = y - x
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    p = float(wilcoxon(y, x).pvalue) if np.any(d != 0) else 1.0
    na = set(out["cols"][a]["named_slices"])
    nb = set(out["cols"][b]["named_slices"])
    pm = binomtest(len(nb - na), len(nb ^ na), 0.5).pvalue if nb ^ na else 1.0
    out["tests"][f"{b} vs {a}"] = {"diff": float(d.mean()), "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                                   "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "p_wilcoxon": p,
                                   "named_only_b": len(nb - na), "named_only_a": len(na - nb), "p_mcnemar_named": float(pm)}
    t = out["tests"][f"{b} vs {a}"]
    say(f"  {b:26s} vs {a:26s}: P@10 {t['diff']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}], better on {t['better']}, "
        f"worse on {t['worse']}, p {p:.2g} | named only by the first {t['named_only_b']}, only by the second "
        f"{t['named_only_a']}, McNemar p {pm:.2g}")

# ---- per object: the description of an object names the experts' situation? (the certified phrases of its class
# with P(yes) >= 0.5, the most confident first; top-3), against the same on the correct objects (false naming)
say("\nper object: the experts' situation named among the first 3 phrases of the object's description")
conf = defaultdict(dict)                                        # instance -> {(cls, q): P(yes)}
for (c, q), (x, _r) in W.items():
    for s in x["scored"]:
        conf[s["instance_id"]][(c, q)] = s["conf"]
out["objects"] = {}
for col, keys in (("SliceLens", cert_S), ("+ rank test", cert_A)):
    ks = set(keys)
    hit, n, fa, nn = 0, 0, 0, 0
    for i, sl in inst_slices.items():
        ph = sorted(((p, k) for k, p in conf.get(i, {}).items() if k in ks and p >= 0.5), reverse=True)[:3]
        exp = [s for s in sl if s in kw]
        if exp:
            n += 1
            hit += any(names(k[0], k[1], kw[exp[0]]) for _p, k in ph)
        elif sl == {"NA"}:
            nn += 1
            fa += any(names(k[0], k[1], kw[s]) for _p, k in ph for s in kw if kw[s]["class"] == k[0])
    out["objects"][col] = {"expert_objects": n, "named": hit, "rate": hit / max(n, 1),
                           "correct_objects": nn, "false": fa, "false_rate": fa / max(nn, 1)}
    o = out["objects"][col]
    say(f"  {col:12s} experts' objects named {o['named']} of {o['expert_objects']} ({100 * o['rate']:.1f}%) | correct "
        f"objects given an expert situation {o['false']} of {o['correct_objects']} ({100 * o['false_rate']:.1f}%)")
# ---- why the columns differ: GESD's measure takes, for each expert slice, the BEST of a method's slices and never
# charges a slice that matches nothing, so it can reward the number of slices rather than what they say
say("\nwhy: what Precision@10 rewards on GESD")
cls_of_slice = {s: kw[s]["class"] for s in EXPERT}
out["why"] = {}
for col in ("SliceLens, verdict only", "+ rank test, verdict only", "pipeline"):
    sl = COLS[col]
    out["why"][col] = {"classes_with_language_slice": len({o[1] for o in sl if o[0] == "language"}),
                       "expert_slices_without_language_slice_in_class":
                           sum(1 for s in EXPERT if not any(o[0] == "language" and o[1] == cls_of_slice[s] for o in sl))}
    w = out["why"][col]
    say(f"  {col:28s} classes with a certified phrase {w['classes_with_language_slice']:2d} of {len(CLASSES)} | expert slices "
        f"whose class has none {w['expert_slices_without_language_slice_in_class']:2d} of {len(EXPERT)}")
# (1) SliceLens cut to the number of slices the rank test certifies, at random
nB = len(COLS["+ rank test, verdict only"])
SL = COLS["SliceLens, verdict only"]
sub = []
for _ in range(500):
    pick = [SL[i] for i in rng.choice(len(SL), size=min(nB, len(SL)), replace=False)]
    sub.append(np.mean([max((p_at_k(o[3], s) for o in pick), default=0.0) for s in EXPERT]))
out["why"]["slicelens_cut_to_rank_count"] = {"n": nB, "p10_mean": float(np.mean(sub)),
                                              "p10_ci": [float(np.quantile(sub, .025)), float(np.quantile(sub, .975))]}
say(f"  SliceLens cut at random to {nB} slices (the rank test's number): Precision@10 {np.mean(sub):.3f} "
    f"[{np.quantile(sub, .025):.3f}, {np.quantile(sub, .975):.3f}] over 500 draws")
# (2) the same certified slices with their errors in random order: does the phrase's ranking matter at all?
for col in ("SliceLens, verdict only", "+ rank test, verdict only"):
    shuf = []
    for _ in range(200):
        shuf.append(np.mean([max((p_at_k(list(rng.permutation(o[3])), s) for o in COLS[col]), default=0.0)
                             for s in EXPERT]))
    out["why"][f"{col}, members shuffled"] = {"p10_mean": float(np.mean(shuf))}
    say(f"  {col:28s} with each slice's members in random order: Precision@10 {np.mean(shuf):.3f} "
        f"(as ranked by the phrase {out['cols'][col]['p10']:.3f})")

# ---- the same with the correct objects kept in the language slices (added 26 Oct). The paper evaluates correct and
# erroneous units alike (Sec. 3.1) and ranks the units with c > gamma = 0.5 by c (Sec. 3.4); it does not say that the
# correct ones are dropped from the slice it outputs. Above, a slice holds the class's errors only; here a language
# slice holds every object of its class with P(yes) >= 0.5, ranked by P(yes), so a correct object among the first k is
# a miss. Cluster slices are groups of errors by construction and stay as they are.
say(f"\nthe same with the correct objects in the language slices (every object of the class with P(yes) >= 0.5, "
    f"the paper's U_gamma; cluster slices unchanged)")
COLS2 = {"SliceLens, verdict only": verdict_slices(cert_S, True), "+ rank test, verdict only": verdict_slices(cert_A, True),
         "SliceLens, thesis gates": gated_slices(RUN_S, ("language",), True),
         "+ rank test, thesis gates": gated_slices(RUN_A, ("language",), True),
         "pipeline": gated_slices(RUN_A, ("language", "cluster"), True)}
is_ok = {i: sl == {"NA"} for i, sl in inst_slices.items()}
P2, out["cols_all"] = {}, {}
for col, sl in COLS2.items():
    per = {s: max((p_at_k(o[3], s) for o in sl), default=0.0) for s in EXPERT}
    P2[col] = per
    top = [i for o in sl if o[0] == "language" for i in o[3][:K]]
    out["cols_all"][col] = {"slices": len(sl), "p10": float(np.mean(list(per.values()))),
                            "at_1": sum(v >= 1.0 for v in per.values()), "per_slice": per,
                            "correct_in_top10": float(np.mean([is_ok.get(i, False) for i in top])) if top else 0.0}
    o = out["cols_all"][col]
    say(f"  {col:28s} {o['slices']:3d} slices | Precision@10 {o['p10']:.3f}, {o['at_1']:2d} of {len(EXPERT)} at 1.0 | "
        f"correct objects among the first {K} of its language slices {100 * o['correct_in_top10']:.0f}%")
out["tests_all"] = {}
for (a, Pa), (b, Pb) in ((("SliceLens, verdict only", P2), ("+ rank test, verdict only", P2)),
                         (("SliceLens, thesis gates", P2), ("+ rank test, thesis gates", P2)),
                         (("SliceLens, thesis gates", P2), ("pipeline", P2)),
                         (("SliceLens, verdict only", P2), ("pipeline", P2)),
                         (("SliceLens, verdict only", P2), ("pipeline, errors only", P))):
    x, y = np.array([Pa[a][s] for s in EXPERT]), np.array([Pb[b.replace(", errors only", "")][s] for s in EXPERT])
    d = y - x
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    p = float(wilcoxon(y, x).pvalue) if np.any(d != 0) else 1.0
    t = out["tests_all"][f"{b} vs {a}"] = {"diff": float(d.mean()), "ci": [float(np.quantile(boots, .025)),
                                                                           float(np.quantile(boots, .975))],
                                           "better": int((d > 0).sum()), "worse": int((d < 0).sum()), "p_wilcoxon": p}
    say(f"  {b:26s} vs {a:26s}: P@10 {t['diff']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}], better on {t['better']}, "
        f"worse on {t['worse']}, p {p:.2g}")
SL2 = COLS2["SliceLens, verdict only"]
sub = [np.mean([max((p_at_k(o[3], s) for o in [SL2[i] for i in rng.choice(len(SL2), size=min(nB, len(SL2)),
                                                                            replace=False)]), default=0.0)
                for s in EXPERT]) for _ in range(500)]
out["cols_all"]["slicelens_cut_to_rank_count"] = {"n": nB, "p10_mean": float(np.mean(sub)),
                                                  "p10_ci": [float(np.quantile(sub, .025)), float(np.quantile(sub, .975))]}
say(f"  SliceLens cut at random to {nB} slices: Precision@10 {np.mean(sub):.3f} "
    f"[{np.quantile(sub, .025):.3f}, {np.quantile(sub, .975):.3f}] over 500 draws")
for col in ("SliceLens, verdict only", "+ rank test, verdict only"):
    shuf = [np.mean([max((p_at_k(list(rng.permutation(o[3])), s) for o in COLS2[col]), default=0.0) for s in EXPERT])
            for _ in range(200)]
    out["cols_all"][col]["p10_members_shuffled"] = float(np.mean(shuf))
    say(f"  {col:28s} with each slice's members in random order: Precision@10 {np.mean(shuf):.3f} "
        f"(as ranked by the phrase {out['cols_all'][col]['p10']:.3f})")

# ---- the classes where the pipeline certifies no phrase (added 26 Oct), and why: no hypothesis passes the rank test
# on the 120-object sample (nothing reaches the whole class), none of those that pass is certified on the whole class,
# or the whole-class rank test certifies some and the gates of the language branch remove them all (the gate of
# describe.build_image_descriptions, recomputed here with its own defaults)
import inspect
from slice_quality.describe import _severity_z, _subtype_severity_stats, build_image_descriptions
GD = {k: v.default for k, v in inspect.signature(build_image_descriptions).parameters.items()
      if v.default is not inspect.Parameter.empty}
rt = pickle.load(open(f"{RUN_A}/result_tagged.pkl", "rb"))["result"]
hyp_of = {x["slice_id"]: x.get("hypothesis") for x in json.load(open(f"{RUN_A}/slices_index.json"))["slices"]}
sev = _subtype_severity_stats(rt)
gated, seen = defaultdict(list), set()
for s, q in rt.scored:
    if s.source != "language" or q.slice_id in seen:
        continue
    seen.add(q.slice_id)
    if not (getattr(s, "metadata", None) or {}).get("is_systematic"):
        continue
    mc = sum(s.membership.values()) / len(s.membership) if s.membership else 0.0
    z = _severity_z(s.members, sev)
    fails = [n for n, ok in (("severity", z >= GD["lang_sev_z_min"]), ("confidence", mc >= GD["lang_conf_min"]),
                             ("size", q.size >= GD["lang_min_size"])) if not ok]
    gated[q.slice_id.rsplit("_", 1)[0]].append({"hypothesis": hyp_of.get(q.slice_id), "size": q.size,
                                                "mean_conf": mc, "sev_z": z, "fails": fails})
with_phrase = {o[1] for o in COLS["pipeline"] if o[0] == "language"}
out["missing"] = {"gate": {k: GD[k] for k in ("lang_sev_z_min", "lang_conf_min", "lang_min_size")}, "classes": {}}
say(f"\nwhy the pipeline certifies no phrase in {len(CLASSES) - len(with_phrase)} classes "
    f"(gate: severity z >= {GD['lang_sev_z_min']}, mean P(yes) >= {GD['lang_conf_min']}, size >= {GD['lang_min_size']})")
for c in CLASSES:
    if c in with_phrase:
        continue
    smp = [v for (c_, _q), v in cc.items() if c_ == c]
    best = min(smp, key=lambda v: v["q_bh"])
    reason = ("gates" if gated.get(c) else "whole class" if any(v["auc_sys"] for v in smp) else "sample")
    exp = [s for s in EXPERT if kw[s]["class"] == c]
    out["missing"]["classes"][c] = {
        "reason": reason, "expert_slices": exp, "n_sample": len(smp), "auc_sys_sample": sum(v["auc_sys"] for v in smp),
        "slope_sys_sample": sum(v["slope_sys"] for v in smp),
        "best_sample": {"q": best["q"], "auc": best["auc"], "q_bh": best["q_bh"]}, "gated": gated.get(c, []),
        "p10_pipeline": {s: P["pipeline"][s] for s in exp}, "p10_rank_verdict": {s: P["+ rank test, verdict only"][s] for s in exp}}
    m = out["missing"]["classes"][c]
    detail = (f"best of {len(smp)} on the sample q {best['q_bh']:.3f} (AUC {best['auc']:.2f}); slope rule sends "
              f"{m['slope_sys_sample']}" if reason == "sample" else
              "; ".join(f"'{g['hypothesis']}' z {g['sev_z']:+.2f} conf {g['mean_conf']:.2f} size {g['size']} -> "
                        f"{','.join(g['fails']) or 'pass'}" for g in m["gated"]))
    say(f"  {c:10s} {reason:11s} {len(exp)} expert slices (pipeline P@10 "
        f"{', '.join('%.2f' % v for v in m['p10_pipeline'].values())}; rank verdict alone "
        f"{', '.join('%.2f' % v for v in m['p10_rank_verdict'].values())}) | {detail}")
json.dump(out, open(f"{L}/gesd_cmp.json", "w"), indent=1)
open(f"{L}/gesd_cmp.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("GESD_CMP_DONE", flush=True)
