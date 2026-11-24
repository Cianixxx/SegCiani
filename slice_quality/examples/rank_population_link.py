"""How the groundings of the official VOC run meet the objects of the final typing. CPU only, reads only.

The VLM read the objects of the substrate it was grounded on (``cc-full``: the first typing of Block 1, connected
components with IoU claims; its ids and labels are those of the reference run). The official rescore
(rescore_with_language.py -> build_language_slices_from_run) joins a grounded object to a final error by the id string
alone, and the final typing numbers its objects differently, so the same id can name another object. Here:
  ids        every grounded object: same id in the final typing and same object (same class, box IoU >= 0.5),
             same id but another object, id absent
  members    for every situation of the official run, its members as the id join builds them and as a geometric join
             builds them (the grounded object's box in the reference run -> the final error of the same image and
             class with the largest box IoU >= 0.5): kept, mis-paired, missing, and the gates of the language branch
             (mean confidence >= 0.6, size >= 3, mean z >= -0.25) on the geometric members
  coverage   errors in a certified slice with the situations' members from either join (clusters unchanged)
  verdict    every whole-class hypothesis of the official families tested again with the labels of the final typing,
             on the grounded objects that the geometric join links to a final object: AUC, image bootstrap SE
             (B draws), one-sided p, BH within each family, classes with >= 10 correct objects
Run (local or server):  python slice_quality/examples/rank_population_link.py
Env: GROUND (the four official grounding folders), REF_PKL (reference run), FIN_DIR (official run), B (500), OUT
"""
import json
import os
import pickle
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import norm, rankdata

from slice_quality.language_branch import _bbox_iou

GROUND = os.environ.get("GROUND", "/data")
FAMS = [("", "outputs_rank_m10"), ("extra", "outputs_extra_rank_m10"), ("pend", "outputs_pending_rank_m10"),
        ("dd", "outputs_dd_voc_rank")]
REF_PKL = os.environ.get("REF_PKL", "outputs/slice_quality/result_tagged.pkl")
FIN_DIR = os.environ.get("FIN_DIR", "outputs/slice_quality_inst")
B = int(os.environ.get("B", "500"))
OUT = os.environ.get("OUT", os.path.join(FIN_DIR, "rank_population_link"))
MIN_OK, Q, IOU_MIN = 10, 0.05, 0.5
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))


def objects(path):
    d = pickle.load(open(path, "rb"))
    return d["errors"], d["reference"]


cls_of = lambda t: t.gt_class or t.pred_class
r_err, r_ok = objects(REF_PKL)
f_err, f_ok = objects(os.path.join(FIN_DIR, "result_tagged.pkl"))
ref = {t.instance_id: t for t in r_err + r_ok}
fin_err = {t.instance_id: t for t in f_err}
fin_by_img = defaultdict(list)                       # image -> [(object, is_error)]
for t in f_err:
    fin_by_img[t.image_id].append((t, True))
for t in f_ok:
    fin_by_img[t.image_id].append((t, False))


def geo(iid, errors_only):
    """the final object the grounded object really is: same image and class, largest box IoU >= IOU_MIN"""
    a = ref.get(iid)
    if a is None or a.bbox is None:
        return None
    best, bi = None, IOU_MIN
    for t, e in fin_by_img.get(a.image_id, []):
        if (errors_only and not e) or cls_of(t) != cls_of(a) or t.bbox is None:
            continue
        v = _bbox_iou(a.bbox, t.bbox)
        if v >= bi:
            best, bi = (t, e), v
    return best


# z of the final errors, as the pipeline computes it: 1 - IoU standardised within each type
by_type = defaultdict(list)
for t in f_err:
    by_type[t.seg_subtype].append(1 - t.iou)
mu = {k: float(np.mean(v)) for k, v in by_type.items()}
sd = {k: float(np.std(v)) for k, v in by_type.items()}
zf = {t.instance_id: ((1 - t.iou) - mu[t.seg_subtype]) / sd[t.seg_subtype] if sd[t.seg_subtype] > 0 else 0.0
      for t in f_err}

runs = {}                                            # (tag, class) -> run json
for tag, d in FAMS:
    p = os.path.join(GROUND, d)
    for f in sorted(os.listdir(p)):
        m = re.fullmatch(r"voc_paper_(\w+?)_full\.json", f)
        if m:
            runs[(tag, m.group(1))] = json.load(open(os.path.join(p, f), encoding="utf-8"))

# ---- ids
seen = {}
for r in runs.values():
    for c in r["candidates"]:
        for s in c["scored"]:
            seen[s["instance_id"]] = bool(s["is_error"])
ids = Counter()
for iid, err in seen.items():
    a, b = ref.get(iid), fin_err.get(iid) or next((t for t in f_ok if t.instance_id == iid), None)
    if a is None:
        ids["not in the reference run"] += 1
    elif b is None:
        ids["absent from the final typing"] += 1
    elif cls_of(a) == cls_of(b) and a.bbox and b.bbox and _bbox_iou(a.bbox, b.bbox) >= IOU_MIN:
        ids["same id, same object"] += 1
    else:
        ids["same id, another object"] += 1
say("=" * 100)
say("GROUNDINGS OF THE OFFICIAL VOC RUN AGAINST THE OBJECTS OF THE FINAL TYPING")
say("=" * 100)
say(f"grounded objects: {len(seen)} ({sum(seen.values())} wrong, {len(seen) - sum(seen.values())} correct, labels of "
    f"the reference run: {sum(1 for i, e in seen.items() if i in ref and ref[i] in r_err) == sum(seen.values())})")
for k, v in ids.most_common():
    say(f"  {k:32s} {v:5d}")

# ---- members of the official situations
idx = json.load(open(os.path.join(FIN_DIR, "slices_index.json"), encoding="utf-8"))["slices"]
sits = [x for x in idx if x.get("in_descriptions") and x["source"] == "language"]
clus = [x for x in idx if x.get("in_descriptions") and x["source"] != "language"]
mem_of = lambda x: {m for v in x["images"].values() for m in v}
rows, tot = [], Counter()
U_id, U_geo, U_geo_pass = set(), set(), set()
for x in sits:
    base, ci = x["slice_id"].rsplit("_", 1)
    m = re.fullmatch(r"(.+?)_(extra|dd|pend)", base)
    cls, tag = (m.group(1), m.group(2)) if m else (base, "")
    cand = runs[(tag, cls)]["candidates"][int(ci)]
    assert cand["query"] == x["hypothesis"], (x["slice_id"], cand["query"], x["hypothesis"])
    items = [s for s in cand["scored"] if s["conf"] >= 0.5 and s["is_error"]]
    by_id = {s["instance_id"] for s in items if s["instance_id"] in fin_err}
    gconf = {}
    for s in items:
        g = geo(s["instance_id"], errors_only=True)
        if g is not None:
            gconf.setdefault(g[0].instance_id, s["conf"])
    off = mem_of(x)
    gm = set(gconf)
    ok = len(gm) >= 3 and np.mean(list(gconf.values())) >= 0.6 and np.mean([zf[i] for i in gm]) >= -0.25
    cross = sum(1 for i in off if cls_of(fin_err[i]) != cls)
    rows.append(dict(slice_id=x["slice_id"], cls=cls, q=x["hypothesis"], read=len(items), official=len(off),
                     id_join=len(by_id), geo=len(gm), kept=len(off & gm), mispaired=len(off - gm), missing=len(gm - off),
                     cross_class=cross, gates_geo=bool(ok)))
    tot.update(read=len(items), official=len(off), kept=len(off & gm), mispaired=len(off - gm), missing=len(gm - off),
               cross_class=cross, gates_geo=int(ok))
    U_id |= off
    U_geo |= gm
    if ok:
        U_geo_pass |= gm
C = set().union(*(mem_of(x) for x in clus)) if clus else set()
n = len(f_err)
say(f"\n{len(sits)} situations of the official run: {tot['official']} members (sum over situations), of which "
    f"{tot['kept']} are the object the VLM read, {tot['mispaired']} another object with the same id "
    f"({tot['cross_class']} of another class); {tot['missing']} read objects are missing from them")
say(f"  situations whose geometric members still pass the gates: {tot['gates_geo']} of {len(sits)}")
say(f"  errors in a situation: id join {len(U_id)}, geometric join {len(U_geo)} "
    f"(only situations that still pass the gates: {len(U_geo_pass)})")
say(f"  errors in a certified slice (clusters unchanged, {len(C)} errors): id join {len(U_id | C)} = "
    f"{100 * len(U_id | C) / n:.1f}%, geometric {len(U_geo | C)} = {100 * len(U_geo | C) / n:.1f}%, geometric with "
    f"the gates {len(U_geo_pass | C)} = {100 * len(U_geo_pass | C) / n:.1f}%")
for r in sorted(rows, key=lambda r: -r["mispaired"])[:12]:
    say(f"    {r['slice_id']:22s} {r['q'][:44]:44s} official {r['official']:3d} kept {r['kept']:3d} mis-paired "
        f"{r['mispaired']:3d} (other class {r['cross_class']:2d}) missing {r['missing']:3d} gates {r['gates_geo']}")


# ---- the verdict again with the labels of the final typing
def auc(x, y):
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_p(x, y, img, rng):
    a = auc(x, y)
    if a is None:
        return None, 1.0
    groups = defaultdict(list)
    for k, i in enumerate(img):
        groups[i].append(k)
    gl = [np.array(v) for v in groups.values()]
    boots = []
    for _ in range(B):
        sel = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
        v = auc(x[sel], y[sel])
        if v is not None:
            boots.append(v)
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
    return qv


rng = np.random.default_rng(20261102)
shipped = {(x["slice_id"].rsplit("_", 1)[0], x["hypothesis"]) for x in sits}
ver = []
for tag, d in FAMS:
    fam = []
    for (tg, cls), r in sorted(runs.items()):
        if tg != tag:
            continue
        geo_lab = {}
        for s in r["candidates"][0]["scored"] if r["candidates"] else []:
            g = geo(s["instance_id"], errors_only=False)
            if g is not None:
                geo_lab[s["instance_id"]] = g[1]
        n_ok_fin = sum(1 for v in geo_lab.values() if not v)
        n_ok_ref = sum(1 for s in (r["candidates"][0]["scored"] if r["candidates"] else []) if not s["is_error"])
        for c in r["candidates"]:
            if c.get("auc_q") is None:
                continue        # read on the whole class under the slope protocol, but not admitted by the rank test
            # the same test with the labels of the reference run on every grounded object: it should give back the
            # official verdict, so that a difference below comes from the labels and not from this reimplementation
            x0 = np.array([s["conf"] for s in c["scored"]], float)
            y0 = np.array([bool(s["is_error"]) for s in c["scored"]])
            img0 = [s["instance_id"].split("#")[0] for s in c["scored"]]
            a0, p0 = rank_p(x0, y0, img0, rng) if n_ok_ref >= MIN_OK else (None, None)
            sc = [s for s in c["scored"] if s["instance_id"] in geo_lab]
            if not sc:
                continue
            x = np.array([s["conf"] for s in sc], float)
            y = np.array([geo_lab[s["instance_id"]] for s in sc])
            img = [s["instance_id"].split("#")[0] for s in sc]
            a, p = rank_p(x, y, img, rng) if n_ok_fin >= MIN_OK else (None, None)
            key = (f"{cls}_{tag}" if tag else cls, c["query"])
            fam.append(dict(family=tag or "canonical/specific", cls=cls, q=c["query"], official=bool(c["is_systematic"]),
                            official_q=c.get("auc_q"), shipped=key in shipped, n_ok_ref=n_ok_ref, n_ok_final=n_ok_fin,
                            linked=len(sc), auc_ref=a0, p_ref=p0, auc_final=a, p_final=p))
    for lab in ("ref", "final"):
        tested = [v for v in fam if v["p_" + lab] is not None]
        for v, qv in zip(tested, bh([v["p_" + lab] for v in tested])):
            v["q_" + lab] = qv
            v["certified_" + lab] = qv <= Q and v["auc_" + lab] > 0.5
    ver += fam
off_cert = [v for v in ver if v["official"]]
rep = [v for v in ver if v.get("certified_ref")]
say(f"\nthis reimplementation with the labels of the reference run: {len(rep)} certified, "
    f"{sum(v['official'] for v in rep)} of them official, {sum(not v['official'] for v in rep)} not; "
    f"official ones it misses: {sum(v['official'] and not v.get('certified_ref') for v in ver)}")
nf = [v for v in ver if v.get("certified_final") and not v["official"]]
say(f"  hypotheses certified with the final labels but not officially: {len(nf)}, of which "
    f"{sum(bool(v.get('certified_ref')) for v in nf)} the reimplementation certifies with the reference labels too; by family "
    f"{dict(Counter(v['family'] for v in nf))}, official q of these: "
    f"{dict(Counter('none' if v['official_q'] is None else ('<=0.05' if v['official_q'] <= 0.05 else '>0.05') for v in nf))}")
say(f"\nverdict with the labels of the final typing ({B} image-bootstrap draws, BH within each family, classes with >= "
    f"{MIN_OK} correct objects in the final typing):")
say(f"  hypotheses certified by the official run: {len(off_cert)}; still certified: "
    f"{sum(v.get('certified_final', False) for v in off_cert)}; not testable (class under the minimum): "
    f"{sum(v['p_final'] is None for v in off_cert)}")
sh = [v for v in ver if v["shipped"]]
say(f"  of the {len(sits)} situations: still certified {sum(v.get('certified_final', False) for v in sh)}, "
    f"not testable {sum(v['p_final'] is None for v in sh)}, no longer significant "
    f"{sum(v['p_final'] is not None and not v.get('certified_final', False) for v in sh)}")
say(f"  newly certified with the final labels: {sum(v.get('certified_final', False) and not v['official'] for v in ver)}")
for v in sorted((v for v in sh if v['p_final'] is not None and not v.get('certified_final', False)),
                key=lambda v: v["cls"]):
    say(f"    lost: {v['cls']:12s} {v['q'][:56]:56s} AUC {v['auc_final']:.3f} q {v['q_final']:.3f}")
json.dump({"ids": dict(ids), "members": dict(tot), "situations": rows, "coverage": {
    "n_err": n, "clusters": len(C), "id_join": len(U_id | C), "geo": len(U_geo | C), "geo_gates": len(U_geo_pass | C)},
    "verdict": ver}, open(OUT + ".json", "w", encoding="utf-8"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("RANK_POPULATION_LINK_DONE", flush=True)
