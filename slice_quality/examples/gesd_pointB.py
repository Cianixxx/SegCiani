"""Point B: our slices and descriptions on GESD against the 21 expert slices. CPU only.

Rules fixed before the run (outputs/slice_quality_gesd/pointB_keywords.json, 2026-10-17 17:04 UTC):
a description names an expert situation when it belongs to the slice's class and contains one of
its keywords; matches through the generic hypotheses (7 canonical + 8 extra) are kept apart.

  verdicts  per hypothesis: SliceLens's slope rule (recorded by the run) and the AUC verdict of
            slope_permutation.py (same function: one-sided, image-block bootstrap SE, BH q <= 0.05
            over all GESD groundings of the stage)
  level 2   slices: for each expert slice, the best of our slices by GESD's Precision@10 (ids only:
            hits among the first 10 objects / min(10, size)), certified slices of each branch and
            every language candidate (all instances of the class ranked by P(yes)); and whether
            that slice's name names the situation
  level 2b  hypotheses: within the grounded objects of the class, the AUC of each hypothesis for
            "this object is in the expert slice" -- which phrase picks the experts' objects best,
            and does it name their situation
  level 3   objects: description of an object = the hypotheses that pass the AUC verdict and get
            P(yes) >= 0.5 on it, ranked by P; the situation is named at top-1 / top-3 when one phrase
            matches; the same rate on the class's correct (NA) objects is the false-alarm floor
Stage: the 120-instance sample (PHASE files _cc) or, once re-grounded, the class (_full).

Run: python slice_quality/examples/gesd_pointB.py      Env: STAGE (cc | full), OUT
"""
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import norm, rankdata

from slice_quality.describe import _GENERIC_EXTRA

STAGE = os.environ.get("STAGE", "cc")
L = "outputs/slice_quality_gesd"
OUT = os.environ.get("OUT", f"{L}/pointB_{STAGE}")
G = "/data"
CLASSES = "airplane apple banana bed book carrot clock keyboard orange suitcase surfboard teddy_bear tie umbrella vase".split()
CANON = ["a small object", "an object occluded by another object", "an object that blends into its background",
         "an object cut off at the image border", "an object with thin or elongated parts",
         "several overlapping objects of the same type", "an object that looks similar to a different object category"]
K, Q, BB = 10, 0.05, 500
rng = np.random.default_rng(0)
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))

kw = json.load(open(f"{L}/pointB_keywords.json", encoding="utf-8"))["slices"]
gesd = json.load(open("/priv/gesd/coco_segmentation.json"))
slice_units = defaultdict(set)
unit_slice = {}
for a in gesd["annotations"]:
    unit_slice[str(a["id"])] = a["reason"]
    if a["reason"] != "NA":
        slice_units[a["reason"]].add(str(a["id"]))
umap = json.load(open(f"{G}/unit_map.json"))                     # instance -> [{unit, slice, category}]
inst_units = {i: [u["unit"] for u in v] for i, v in umap.items()}
inst_slices = {i: {u["slice"] for u in v} for i, v in umap.items()}


def source(cls, q, dd):
    return "canonical" if q in CANON else "extra" if q in _GENERIC_EXTRA else "data-driven" if q in dd else "gemma"


def matches(spec, q):
    return [k for k in spec["keywords"] if k in q.lower()]


def auc(x, y):
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_test(confs, lab, ids, bb=BB):
    """slope_permutation.py's AUC verdict, unchanged."""
    x, y = np.asarray(confs, float), np.asarray(lab, bool)
    a = auc(x, y)
    if a is None:
        return 0.5, 1.0
    groups = defaultdict(list)
    for k, i in enumerate(ids):
        groups[i.split("#")[0]].append(k)
    gl = [np.array(v) for v in groups.values()]
    boots = []
    for _ in range(bb):
        sel = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
        v = auc(x[sel], y[sel])
        if v is not None:
            boots.append(v)
    se = float(np.std(boots)) if len(boots) > 10 else 0.0
    return a, (float(norm.sf((a - 0.5) / se)) if se > 0 else (0.0 if a > 0.5 else 1.0))


def bh(ps, q=Q):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qv, run = [1.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * m / rank)
        qv[i] = run
    return qv, [x <= q for x in qv]


def p_at_k(ranked_instances, s, k=K):
    units = [u for i in ranked_instances for u in inst_units.get(i, [])]
    top = units[:k]
    return sum(unit_slice.get(u) == s for u in top) / min(k, len(slice_units[s]))


# ---- groundings and verdicts
H = {}                                   # (cls, q) -> dict
# full stage: the protocol re-grounding (outputs_gesd) + the naming phrases re-grounded only to measure
DIRS = [f"{G}/outputs_gesd"] + ([f"{G}/outputs_gesd_naming"] if STAGE == "full" else [])
for c, f in [(c, f"{d}/voc_paper_{c}_{STAGE}.json") for c in CLASSES for d in DIRS]:
    if not os.path.exists(f):
        say(f"missing {f}")
        continue
    dd = {l.strip() for l in open(f"{G}/outputs_dd/dd_{c}.txt", encoding="utf-8") if l.strip() and not l.startswith("#")}
    r = json.load(open(f))
    for x in r["candidates"]:
        if (c, x["query"]) in H:
            continue
        sc = x["scored"]
        if STAGE == "full" and len(sc) != r["num_instances"]:
            continue                                                  # not re-grounded on the class
        H[(c, x["query"])] = {"cls": c, "q": x["query"], "src": source(c, x["query"], dd),
                              "ids": [s["instance_id"] for s in sc], "conf": [s["conf"] for s in sc],
                              "err": [s["is_error"] for s in sc], "slope_sys": bool(x["is_systematic"]),
                              "slope": x["max_positive_slope"]}
for h in H.values():
    h["auc"], h["p"] = rank_test(h["conf"], h["err"], h["ids"])
qv, sig = bh([h["p"] for h in H.values()])
for h, a, b in zip(H.values(), qv, sig):
    h["q_bh"], h["auc_sys"] = a, b

say("=" * 100)
say(f"POINT B ON GESD, stage {STAGE}: our language branch against the 21 expert slices")
say("=" * 100)
say(f"hypotheses grounded: {len(H)} | slope rule systematic {sum(h['slope_sys'] for h in H.values())} | "
    f"AUC verdict {sum(h['auc_sys'] for h in H.values())} | both {sum(h['slope_sys'] and h['auc_sys'] for h in H.values())}")
for s in ("canonical", "gemma", "data-driven", "extra"):
    hs = [h for h in H.values() if h["src"] == s]
    say(f"   {s:11s} {len(hs):3d} hypotheses | slope {sum(h['slope_sys'] for h in hs):3d} | AUC {sum(h['auc_sys'] for h in hs):3d}")

# ---- level 2: slices
SUF = "_cc" if STAGE == "cc" and os.path.exists(f"{L}/slices_index_cc.json") else ""
d = pickle.load(open(f"{L}/result_tagged{SUF}.pkl", "rb"))
idx = {x["slice_id"]: x for x in json.load(open(f"{L}/slices_index{SUF}.json"))["slices"]}
ours = []                                                       # (kind, id, name, ranked instance ids)
seen = set()
for ks in list(d["result"].kept) + list(d["result"].scored):
    sl = ks[0] if isinstance(ks, tuple) else ks.slice_           # scored: (Slice, SliceQuality)
    if sl.slice_id in seen:
        continue
    seen.add(sl.slice_id)
    meta = idx.get(sl.slice_id, {})
    if not meta.get("in_descriptions"):
        continue
    ids = [m.instance_id for m in sl.members]
    if meta.get("source") == "language":
        c = sl.slice_id.rsplit("_", 1)[0]
        h = H.get((c, meta.get("hypothesis")))
        cf = dict(zip(h["ids"], h["conf"])) if h else {}
        ids.sort(key=lambda i: -cf.get(i, 0))
        ours.append(("language", sl.slice_id, meta.get("hypothesis"), ids))
    else:
        ours.append(("cluster", sl.slice_id, meta.get("dominant_error_type"), ids))
cand = [("candidate", f"{h['cls']}:{h['q']}", h["q"], [i for _, i in sorted(zip(h["conf"], h["ids"]), key=lambda t: -t[0])])
        for h in H.values()]
say(f"\nLEVEL 2 -- slices. Our certified slices: {sum(o[0] == 'language' for o in ours)} language, "
    f"{sum(o[0] == 'cluster' for o in ours)} cluster. Precision@{K} of the best slice per expert slice; "
    "name: + names the situation (class-specific phrase), g generic phrase, - no")
res2 = {}
for s, spec in sorted(kw.items(), key=lambda t: -len(slice_units[t[0]])):
    row = {}
    for kind, pool in (("certified", ours), ("language cand.", cand)):
        best = max(pool, key=lambda o: p_at_k(o[3], s), default=None)
        p = p_at_k(best[3], s) if best else 0.0
        nm = best[2] or ""
        mk = ""
        if best and best[0] != "cluster":
            m = matches(spec, nm)
            mk = ("+" if m and nm not in CANON and nm not in _GENERIC_EXTRA else "g" if m else "-")
        row[kind] = {"p": p, "slice": best[1] if best else None, "name": nm, "names": mk}
    res2[s] = row
    a, b = row["certified"], row["language cand."]
    say(f"  {len(slice_units[s]):4d} {s[:50]:50s} certified {a['p']:.1f} [{a['names'] or ' '}] {str(a['name'])[:38]:38s} | "
        f"any hypothesis {b['p']:.1f} [{b['names']}] {b['name'][:38]}")
for kind in ("certified", "language cand."):
    say(f"  mean Precision@{K} over the 21 ({kind}): {np.mean([r[kind]['p'] for r in res2.values()]):.3f}")

# ---- level 2b: which hypothesis picks the experts' objects
say("\nLEVEL 2b -- the phrase that best separates each expert slice's objects from the rest of its class "
    "(AUC on the grounded objects), and the best phrase that names the situation")
res2b = {}
for s, spec in sorted(kw.items(), key=lambda t: -len(slice_units[t[0]])):
    hs = [h for h in H.values() if h["cls"] == spec["class"]]
    if not hs:
        continue
    y = [s in inst_slices.get(i, ()) for i in hs[0]["ids"]]
    n_in = sum(y)
    scored = sorted(((auc(h["conf"], [s in inst_slices.get(i, ()) for i in h["ids"]]) or 0.5, h) for h in hs),
                    key=lambda t: -t[0])
    named = [(a, h) for a, h in scored if matches(spec, h["q"]) and h["src"] in ("gemma", "data-driven")]
    rank = next((k + 1 for k, (a, h) in enumerate(scored) if matches(spec, h["q"]) and h["src"] in ("gemma", "data-driven")), None)
    top_a, top_h = scored[0]
    res2b[s] = {"n_in_grounded": n_in, "top": [top_h["q"], top_a], "named_best": [named[0][1]["q"], named[0][0]] if named else None,
                "named_rank": rank}
    say(f"  {s[:50]:50s} ({n_in:3d} grounded) top: {top_h['q'][:42]:42s} {top_a:.2f} | naming phrase: "
        + (f"#{rank}/{len(hs)} {named[0][1]['q'][:40]} {named[0][0]:.2f}" if named else "none in the vocabulary"))
say(f"  top phrase names the situation (class-specific): {sum(1 for r in res2b.values() if r['named_rank'] == 1)}/21; "
    f"within the top 3: {sum(1 for r in res2b.values() if r['named_rank'] and r['named_rank'] <= 3)}/21")

# ---- level 3: per-object descriptions
say("\nLEVEL 3 -- per object: hypotheses passing the AUC verdict with P(yes) >= 0.5, ranked by P; "
    "situation named at top-1 / top-3 (class-specific phrases only); floor = the class's correct (NA) objects")
res3 = {}
for s, spec in sorted(kw.items(), key=lambda t: -len(slice_units[t[0]])):
    hs = [h for h in H.values() if h["cls"] == spec["class"] and h["auc_sys"]]
    conf = defaultdict(dict)
    for h in hs:
        for i, cf in zip(h["ids"], h["conf"]):
            conf[i][h["q"]] = cf
    allids = [h for h in H.values() if h["cls"] == spec["class"]][0]["ids"] if any(h["cls"] == spec["class"] for h in H.values()) else []

    def named_at(i, k):
        desc = sorted(((cf, q) for q, cf in conf[i].items() if cf >= 0.5), reverse=True)[:k]
        return any(matches(spec, q) and q not in CANON and q not in _GENERIC_EXTRA for _, q in desc)
    inside = [i for i in allids if s in inst_slices.get(i, ())]
    na = [i for i in allids if inst_slices.get(i) == {"NA"}]
    r = {"n": len(inside), "top1": np.mean([named_at(i, 1) for i in inside]) if inside else None,
         "top3": np.mean([named_at(i, 3) for i in inside]) if inside else None,
         "na_n": len(na), "na_top1": np.mean([named_at(i, 1) for i in na]) if na else None,
         "na_top3": np.mean([named_at(i, 3) for i in na]) if na else None,
         "n_phrases": float(np.mean([sum(cf >= 0.5 for cf in conf[i].values()) for i in inside])) if inside else None}
    res3[s] = r
    f = lambda v: "  -  " if v is None else f"{100 * v:4.0f}%"
    say(f"  {s[:50]:50s} n {r['n']:3d}: top-1 {f(r['top1'])} top-3 {f(r['top3'])} | NA n {r['na_n']:3d}: top-1 {f(r['na_top1'])} "
        f"top-3 {f(r['na_top3'])} | phrases per object {r['n_phrases'] or 0:.1f}")
tot = [(r["n"], r["top1"], r["top3"]) for r in res3.values() if r["n"]]
n = sum(t[0] for t in tot)
say(f"  all expert-slice objects grounded: {n}; situation named top-1 {100 * sum(t[0] * t[1] for t in tot) / n:.1f}%, "
    f"top-3 {100 * sum(t[0] * t[2] for t in tot) / n:.1f}%")

# ---- cost of the full re-grounding
pop = Counter()
sub = pickle.load(open(f"{G}/substrate.pkl", "rb"))
for t in sub["errors"] + sub["reference"]:
    pop[t.gt_class or t.pred_class] += 1
cert_h = {(o[1].rsplit("_", 1)[0], o[2]) for o in ours if o[0] == "language"}
auc_h = {k for k, h in H.items() if h["auc_sys"]}
for name, hs in (("certified (thesis protocol)", cert_h), ("certified + AUC verdict", cert_h | auc_h)):
    calls = sum(pop[c] - 120 for c, q in hs if pop[c] > 120)
    say(f"\nfull re-grounding, {name}: {len(hs)} hypotheses, {calls} new groundings, ~{calls * 1.7 / 3600:.1f} GPU-h "
        f"(~{calls * 1.7 / 3600 / 3:.1f} h on 3 GPUs)")

json.dump({"stage": STAGE, "verdicts": [{k: v for k, v in h.items() if k not in ("ids", "conf", "err")} for h in H.values()],
           "level2": res2, "level2b": res2b, "level3": {k: {kk: (float(vv) if vv is not None else None) for kk, vv in v.items()}
                                                         for k, v in res3.items()},
           "ours": [o[:3] + (len(o[3]),) for o in ours]}, open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("GESD_POINTB_DONE", flush=True)
