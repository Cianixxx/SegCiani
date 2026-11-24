"""Does the rank test hold when a class has almost no correct objects? CPU only.

The rank test compares P(yes) of a class's wrong objects with that of its correct ones (AUC), and takes its p-value
from the AUC's spread over an image-block bootstrap. With very few correct objects (VOC pottedplant: 1 of 429) the
spread comes almost only from the errors, so it is too small and the p-value too optimistic. A reference that does
not rely on that spread, for every situation the official run certifies:
  exact rank    the probability, if the error labels were assigned to the objects at random, of an AUC at least as
                large (one-sided Mann-Whitney), BH within each family as the official run (one per directory)
and how many correct objects, and images with one, the class has.
Env: DIRS (the official rank directories), EXTRA_DIRS (more families to report, e.g. the data-driven one), OUT
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

DIRS = os.environ.get("DIRS", "/data/outputs_rank:/data/outputs_extra_rank:/data/outputs_pending_rank").split(":")
EXTRA = [d for d in os.environ.get("EXTRA_DIRS", "").split(":") if d]
OUT = os.environ.get("OUT", "outputs/rank_few_correct")
SLICES = os.environ.get("SLICES", "outputs/slice_quality_inst/slices_index.json")   # what the official run ships
Q = 0.05
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))


def auc(x, y):
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qv, run = [1.0] * m, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, ps[i] * m / r)
        qv[i] = run
    return qv


SHIPPED = {(re.sub(r"_(extra|dd|pend)$", "", x["slice_id"].rsplit("_", 1)[0]), x["hypothesis"])
           for x in json.load(open(SLICES))["slices"] if x["source"] == "language" and x.get("in_descriptions")}
rows = []
for fam in DIRS + EXTRA:
    fr = []
    for f in sorted(os.listdir(fam)):
        if not (f.startswith("voc_paper_") and f.endswith("_full.json")):
            continue
        r = json.load(open(os.path.join(fam, f)))
        for c in r["candidates"]:
            sc = c["scored"]
            if len(sc) != r["num_instances"]:
                continue
            x = np.array([s["conf"] for s in sc], float)
            y = np.array([bool(s["is_error"]) for s in sc])
            img = [s["instance_id"].split("#")[0] for s in sc]
            a = auc(x, y)
            if a is None:
                continue
            fr.append({"family": fam, "cls": r["label"], "query": c["query"], "n_err": int(y.sum()),
                       "n_ok": int((~y).sum()), "img_ok": len({i for i, e in zip(img, y) if not e}), "auc": a,
                       "p_boot": c.get("auc_p"), "certified": bool(c["is_systematic"]),
                       "shipped": (r["label"], c["query"]) in SHIPPED,
                       "p_exact": float(mannwhitneyu(x[y], x[~y], alternative="greater").pvalue)})
    for r, q in zip(fr, bh([r["p_exact"] for r in fr])):
        r["q_exact"] = q
    rows += fr
    say(f"{fam}: {len(fr)} hypotheses on whole classes, {sum(r['certified'] for r in fr)} certified")

say("\ncertified situations by the correct objects of their class")
by = defaultdict(list)
for r in rows:
    if r["certified"]:
        by[(r["cls"], r["n_ok"], r["img_ok"], r["n_err"])].append(r)
say(f"  {'class':12s} {'correct':>7s} {'images':>6s} {'errors':>7s} {'cert.':>5s} | confirmed by the exact rank")
tot = defaultdict(int)
for (c, n0, im, n1), rs in sorted(by.items(), key=lambda kv: kv[0][1]):
    ke = sum(r["q_exact"] <= Q for r in rs)
    tot["cert"] += len(rs)
    tot["exact"] += ke
    for m in (5, 10, 20):
        tot[m] += len(rs) if n0 < m else 0
    say(f"  {c:12s} {n0:7d} {im:6d} {n1:7d} {len(rs):5d} | {ke:3d}")
say(f"  {'total':12s} {'':7s} {'':6s} {'':7s} {tot['cert']:5d} | {tot['exact']:3d}")
say(f"  certified in classes with fewer than 5 / 10 / 20 correct objects: {tot[5]} / {tot[10]} / {tot[20]}")
for fam in DIRS + EXTRA:
    rs = [r for r in rows if r["family"] == fam and r["certified"]]
    say(f"  {fam}: certified {len(rs)}, confirmed by the exact rank {sum(r['q_exact'] <= Q for r in rs)}")
sh = [r for r in rows if r["shipped"] and r["certified"]]
say(f"\nshipped by the official run ({len(sh)} situations): in classes with fewer than 5 / 10 / 20 correct objects "
    f"{sum(r['n_ok'] < 5 for r in sh)} / {sum(r['n_ok'] < 10 for r in sh)} / {sum(r['n_ok'] < 20 for r in sh)}; "
    f"not confirmed by the exact rank {sum(r['q_exact'] > Q for r in sh)}")
for c in sorted({r["cls"] for r in sh}, key=lambda c: min(r["n_ok"] for r in sh if r["cls"] == c)):
    rs = [r for r in sh if r["cls"] == c]
    say(f"  {c:12s} correct {rs[0]['n_ok']:3d} | shipped {len(rs):2d}, confirmed {sum(r['q_exact'] <= Q for r in rs):2d}")
say("\nthe certified situations that the exact rank does not confirm (* = shipped by the official run)")
for r in sorted((r for r in rows if r["certified"] and r["q_exact"] > Q), key=lambda r: (r["n_ok"], r["cls"])):
    pb = "—" if r["p_boot"] is None else "%.4f" % r["p_boot"]
    say(f"{'*' if r['shipped'] else ' '} {r['cls']:12s} correct {r['n_ok']:3d} | AUC {r['auc']:.3f} p boot {pb} exact {r['p_exact']:.4f} "
        f"q {r['q_exact']:.3f} | {r['query']}")
json.dump(rows, open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("RANK_FEW_DONE", flush=True)
