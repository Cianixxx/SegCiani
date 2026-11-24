"""Do the situations added by the pending hypotheses hold up like the official ones? CPU only.

The copy outputs/slice_quality_inst_pending certifies the official 44 situations and the new ones from the
hypotheses grounded on 22 October. The same checks as the official run (§18.5), the new set beside the old:
  precision   Qwen asked again on 10 members of each situation vs same-class objects outside
              (eval_description_precision.py on the copy; groups split by whether their situation is new)
  model 2     the same phrase on the second segmenter's objects: the rank test as dd_auc.py gives it (one-sided,
              image-block bootstrap, BH within each set) and the neutral check (members wrong more often than the
              rest, Fisher, BH) -- the new ones from /data/outputs_model2_pending (verdict <copy>/model2_new_dd_auc_full),
              the official ones from /data/outputs_model2_inst (verdict <official>/model2/dd_auc_full)
  plausible   Gemma's plausibility of the blamed situation on every image with one (judge_cause_val.json on
              the copy), genuine vs control; and on the images whose description gains a new situation
Run (CPU):  python -u slice_quality/examples/pending_validation_report.py
Env: OFFICIAL, COPY, M2_NEW, M2_OFF, OUT
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import fisher_exact, norm, rankdata

OFFICIAL = os.environ.get("OFFICIAL", "outputs/slice_quality_inst")
COPY = os.environ.get("COPY", "outputs/slice_quality_inst_pending")
M2_NEW = os.environ.get("M2_NEW", "/data/outputs_model2_pending")
M2_OFF = os.environ.get("M2_OFF", "/data/outputs_model2_inst")
OUT = os.environ.get("OUT", f"{COPY}/validation_report")
rng = np.random.default_rng(0)
J = lambda p: json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
cls_of = lambda sid: re.sub(r"_(extra|dd|pend)$", "", sid.rsplit("_", 1)[0])


def situations(d):
    return {s["slice_id"]: (cls_of(s["slice_id"]), s["hypothesis"]) for s in J(f"{d}/slices_index.json")["slices"]
            if s["source"] == "language" and s.get("in_descriptions")}


off = set(situations(OFFICIAL).values())
cp = situations(COPY)
new = {k for k in cp.values() if k not in off}


def bh(ps, q=0.05):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qv, run = [1.0] * m, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, ps[i] * m / r)
        qv[i] = run
    return [x <= q for x in qv]


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
    g = defaultdict(list)
    for k, i in enumerate(img):
        g[i].append(k)
    gl = [np.array(v) for v in g.values()]
    boots = [v for v in (auc(x[s], y[s]) for s in (np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
                                                     for _ in range(bb))) if v is not None]
    se = float(np.std(boots)) if len(boots) > 10 else 0.0
    return a, (float(norm.sf((a - 0.5) / se)) if se > 0 else (0.0 if a > 0.5 else 1.0))


def model2(d, want, verdict):
    """The rank-test verdict as dd_auc.py gave it on the directory (BH within its rows), plus the neutral check."""
    rank = {(r["cls"], r["query"]): r for r in (J(verdict) or [])}
    rows = {}
    for f in sorted(glob.glob(f"{d}/voc_paper_*_full.json")):
        r = json.load(open(f))
        for x in r["candidates"]:
            k = (r["label"], x["query"])
            sc = x.get("scored") or []
            if k not in want or k in rows or len(sc) != r["num_instances"]:
                continue
            conf = np.array([s["conf"] for s in sc], float)
            err = np.array([bool(s["is_error"]) for s in sc])
            mem = conf >= 0.5
            t = [[int((err & mem).sum()), int((~err & mem).sum())], [int((err & ~mem).sum()), int((~err & ~mem).sum())]]
            pf = fisher_exact(t, alternative="greater")[1] if mem.any() and (~mem).any() else 1.0
            lift = t[0][0] / max(mem.sum(), 1) - t[1][0] / max((~mem).sum(), 1)
            if k in rank:
                a, sa = rank[k]["auc"], bool(rank[k]["auc_sys"])
            else:
                a, sa = auc_p(conf, err, [s["instance_id"].split("#")[0] for s in sc])[0], None
            rows[k] = dict(auc=a, auc_sys=sa, p_fisher=pf, lift=lift, members=int(mem.sum()))
    keys = list(rows)
    for k, sf in zip(keys, bh([rows[k]["p_fisher"] for k in keys])):
        rows[k]["fisher_sys"] = sf and rows[k]["lift"] > 0
    return rows


lines, out = [], {}
say = lambda s="": (lines.append(s), print(s, flush=True))
say("=" * 100)
say(f"THE {len(new)} NEW SITUATIONS AGAINST THE {len(off)} OFFICIAL ONES, ON THE SAME CHECKS")
say("=" * 100)
# precision
ep = J(f"{COPY}/eval_precision.json")
if ep:
    grp = {"official": [], "new": [], "mixed": []}     # a phrase is one group across classes: it can be both
    for g in ep["groups"]:
        if g["branch"] != "language":
            continue
        ks = {cp.get(sid) for sid in g["slices"]} - {None}
        if not ks:
            continue
        grp["new" if ks <= new else "official" if not ks & new else "mixed"].append(g)
    for name, gs in grp.items():
        if gs:
            pr = np.mean([g["precision"] for g in gs])
            ct = np.mean([g["control_rate_same_class"] for g in gs if g.get("control_rate_same_class") is not None])
            sig = sum(bool(g.get("significant_vs_chance")) for g in gs)
            out[f"precision_{name}"] = dict(groups=len(gs), precision=float(pr), control=float(ct), above_chance=sig)
            say(f"precision, {name:8s}: {pr:.3f} vs same-class control {ct:.3f}; {sig} of {len(gs)} groups above chance")
# model 2
m_off = model2(M2_OFF, off, f"{OFFICIAL}/model2/dd_auc_full.json")
m_new = model2(M2_NEW, new, f"{COPY}/model2_new_dd_auc_full.json")
for name, m, n in (("official", m_off, len(off)), ("new", m_new, len(new))):
    if m:
        out[f"model2_{name}"] = dict(n=n, grounded=len(m), auc_sys=sum(bool(v["auc_sys"]) for v in m.values()),
                                     fisher_sys=sum(v["fisher_sys"] for v in m.values()),
                                     auc_mean=float(np.mean([v["auc"] for v in m.values() if v["auc"] is not None])),
                                     rows=[dict(cls=k[0], query=k[1], **v) for k, v in sorted(m.items())])
        o = out[f"model2_{name}"]
        say(f"model 2, {name:8s}: grounded {o['grounded']} of {n} | rank test significant {o['auc_sys']} | members wrong "
            f"more often (Fisher, BH) {o['fisher_sys']} | mean AUC {o['auc_mean']:.2f}")
for name, m in (("official", m_off), ("new", m_new)):
    if m:
        out[f"model2_{name}"]["members_median"] = float(np.median([v["members"] for v in m.values()]))
# how often the second segmenter is wrong on each class: where it is wrong on nearly every object, no phrase can
# separate its errors from the few correct ones
out["model2_error_rate"] = {}
for f in sorted(glob.glob(f"{M2_NEW}/voc_paper_*_full.json")):
    r = json.load(open(f))
    out["model2_error_rate"][r["label"]] = r["num_errors"] / r["num_instances"]
if m_new:
    say("\nthe new situations on model 2 (AUC, rank test, Fisher):")
    for k, v in sorted(m_new.items(), key=lambda kv: -(kv[1]["auc"] or 0)):
        say(f"  {k[0]:12s} AUC {v['auc'] or 0:.2f} {'SYS' if v['auc_sys'] else '   '} "
            f"{'F' if v['fisher_sys'] else ' '} {v['members']:4d} members | {k[1]}")
# plausibility
jc = J(f"{COPY}/judge_cause_val.json")
if jc:
    ok = [r for r in jc["judgements"] if "score" in r and r["kind"] == "pipeline"]
    gen = {r["image"]: r["score"] for r in ok if r["pairing"] == "genuine"}
    ctl = [r["score"] for r in ok if r["pairing"] == "control"]
    descs = J(f"{COPY}/image_descriptions.json")
    touched = {img for img, v in descs.items() for p in v.get("patterns", [])
               if p.get("branch") == "language" and cp.get(p["slice_id"]) in new}
    gt = [v for k, v in gen.items() if k in touched]
    out["plausibility"] = dict(n=len(gen), genuine=float(np.mean(list(gen.values()))), control=float(np.mean(ctl)),
                               n_new=len(gt), genuine_new=float(np.mean(gt)) if gt else None)
    say(f"plausibility of the blamed situation, every image with one: {out['plausibility']['genuine']:.2f} vs control "
        f"{out['plausibility']['control']:.2f} ({len(gen)} images); images with a new situation: "
        + (f"{np.mean(gt):.2f} ({len(gt)} images)" if gt else "-"))
json.dump(out, open(OUT + ".json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("PENDING_VALIDATION_DONE", flush=True)
