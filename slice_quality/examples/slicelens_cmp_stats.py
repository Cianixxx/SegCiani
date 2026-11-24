"""Paired statistical tests for the comparison with SliceLens (§18.6): is each difference larger than chance? CPU only.

Every comparison is paired -- the same errors, victims, images or simulated draws under both methods:
  coverage   errors inside a certified group, VOC val and SBD: SliceLens (slope rule, language branch) against the
             rank test (language branch) against the pipeline (both branches). 95% interval of the difference by an
             image-block bootstrap (errors of one image move together); exact McNemar on the discordant errors
  Pascal-EA  victims inside a certified group, same three columns: exact McNemar on the 144 paired victims
  judges     per-image facts score and rubric score, our narrative against the SliceLens-style text (and the same
             text from the rank-test run): Wilcoxon signed-rank, bootstrap interval of the mean difference
  simulation false alarms and power, slope rule against rank test on the same draws: Wilson intervals
  disagreement  VOC situations certified by one test only: precision (Mann-Whitney) and second-segmenter
             replication (Fisher exact)
Run (server, CPU):  python -u slice_quality/examples/slicelens_cmp_stats.py
Env: OUT (outputs/slicelens_cmp_stats), BOOT (5000)
"""
import json
import os
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, fisher_exact, mannwhitneyu, wilcoxon

OUT = os.environ.get("OUT", "outputs/slicelens_cmp_stats")
BOOT = int(os.environ.get("BOOT", "5000"))
rng = np.random.default_rng(0)
J = lambda p: json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
lines, out = [], {}
say = lambda s="": (lines.append(s), print(s, flush=True))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def mcnemar(a, b):
    """exact McNemar on paired booleans: (only a, only b, p two-sided)"""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def membership(d):
    """error instance -> image, and the errors inside a certified language / cluster slice of run d"""
    pk = pickle.load(open(f"{d}/result_tagged.pkl", "rb"))
    descs = J(f"{d}/image_descriptions.json")
    used = {p["slice_id"] for v in descs.values() for p in v.get("patterns", [])}
    lang, clu, seen = set(), set(), set()
    for s, q in pk["result"].scored:
        if q.slice_id in seen or q.slice_id not in used:
            continue
        seen.add(q.slice_id)
        (clu if s.source == "cluster" else lang).update(m.instance_id for m in s.members)
    return {e.instance_id: e.image_id for e in pk["errors"]}, lang, clu


def coverage(name, slope_run, run):
    img, la, _ = membership(slope_run)
    img2, lb, cb = membership(run)
    assert set(img) == set(img2), f"{name}: the two runs type different errors"
    ids = sorted(img)
    cols = {"SliceLens": np.array([i in la for i in ids]), "+ rank test": np.array([i in lb for i in ids]),
            "pipeline": np.array([i in lb or i in cb for i in ids])}
    ims = sorted(set(img.values()))
    ix = {m: k for k, m in enumerate(ims)}
    gi = np.array([ix[img[i]] for i in ids])
    n_i = np.bincount(gi, minlength=len(ims)).astype(float)
    res = {"n_err": len(ids), "n_img": len(ims), "rate": {k: float(v.mean()) for k, v in cols.items()}}
    W = rng.multinomial(len(ims), np.full(len(ims), 1 / len(ims)), size=BOOT).astype(float)   # image weights
    for a, b in (("SliceLens", "+ rank test"), ("SliceLens", "pipeline"), ("+ rank test", "pipeline")):
        d_i = np.bincount(gi, weights=cols[b].astype(float) - cols[a].astype(float), minlength=len(ims))
        boots = (W @ d_i) / (W @ n_i)
        diff = float(d_i.sum() / n_i.sum())
        n10, n01, p = mcnemar(cols[b], cols[a])
        res[f"{b} vs {a}"] = {"diff": diff, "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                              "p_boot": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))),
                              "only_b": n10, "only_a": n01, "p_mcnemar": p}
        say(f"  {name:8s} {b:12s} vs {a:12s}: {100 * diff:+5.1f} points, 95% CI [{100 * res[f'{b} vs {a}']['ci'][0]:+.1f}, "
            f"{100 * res[f'{b} vs {a}']['ci'][1]:+.1f}] (image bootstrap) | errors only in {b} {n10}, only in {a} {n01}, "
            f"McNemar p {p:.2g}")
    return res


say("=" * 110)
say("PAIRED TESTS FOR THE COMPARISON WITH SLICELENS")
say("=" * 110)
say("\nERRORS INSIDE A CERTIFIED GROUP (same errors under the three columns)")
out["coverage"] = {}
for name, a, b in (("VOC val", "outputs/slice_quality_inst_slope", "outputs/slice_quality_inst"),
                   ("SBD", "outputs/slice_quality_sbd_slope", "outputs/slice_quality_sbd")):
    if os.path.exists(f"{a}/result_tagged.pkl") and os.path.exists(f"{b}/result_tagged.pkl"):
        r = coverage(name, a, b)
        out["coverage"][name] = r
        say(f"  {name:8s} rates: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in r["rate"].items())
            + f" ({r['n_err']} errors, {r['n_img']} images)")

say("\nPASCAL-EA: VICTIMS INSIDE A CERTIFIED GROUP (the same 144 victims)")


def ea_branches(run):
    pl = J(f"{run}/planted_eval.json")
    idx = J(f"{run}/slices_index.json")["slices"]
    member = defaultdict(set)
    for s in idx:
        if s.get("in_descriptions"):
            for ids in s["images"].values():
                for i in ids:
                    member[i].add("cluster" if s["slice_id"].startswith("cluster") else "language")
    errs = pickle.load(open(f"{run}/result_tagged.pkl", "rb"))["errors"]
    link = defaultdict(list)                       # as ea_planted_eval.linked() and ea_branch_coverage.py
    for t in errs:
        keys = {t.instance_id, t.metadata.get("gt_id"), t.metadata.get("primary_gt")} | set(t.metadata.get("gt_ids") or [])
        for k in keys:
            if k:
                link[k].append(t.instance_id)
    return {v: set().union(*[member.get(i, set()) for i in link.get(v, [])]) if link.get(v) else set()
            for v in pl["victims"]}


ea_s, ea_a = ea_branches("outputs/slice_quality_ea"), ea_branches("outputs/slice_quality_ea_auc")
assert set(ea_s) == set(ea_a)
vic = sorted(ea_s)
E = {"SliceLens": [("language" in ea_s[v]) for v in vic], "+ rank test": [("language" in ea_a[v]) for v in vic],
     "pipeline": [bool(ea_a[v]) for v in vic]}
out["ea"] = {"victims": len(vic), "found": {k: int(sum(v)) for k, v in E.items()},
             "ci": {k: wilson(sum(v), len(vic)) for k, v in E.items()}}
for a, b in (("SliceLens", "+ rank test"), ("SliceLens", "pipeline"), ("+ rank test", "pipeline")):
    n10, n01, p = mcnemar(E[b], E[a])
    out["ea"][f"{b} vs {a}"] = {"only_b": n10, "only_a": n01, "p_mcnemar": p}
    say(f"  {b:12s} {sum(E[b]):3d} vs {a:12s} {sum(E[a]):3d} of {len(vic)}: found only by {b} {n10}, only by {a} {n01}, "
        f"exact McNemar p {p:.2g}")

say("\nJUDGES, IMAGE BY IMAGE (our narrative against the SliceLens-style text of the same image)")


def per_image(rows, kind, rubric=None):
    return {r["image"]: r["score"] for r in rows if r.get("kind") == kind and r.get("pairing") == "genuine"
            and "score" in r and (rubric is None or r.get("rubric") == rubric) and r.get("set", "voc") == "voc"}


def paired(name, x, y):
    common = sorted(set(x) & set(y))
    a, b = np.array([x[i] for i in common]), np.array([y[i] for i in common])
    d = a - b
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    r = {"n": len(common), "ours": float(a.mean()), "theirs": float(b.mean()), "diff": float(d.mean()),
         "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], "p_wilcoxon": p,
         "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    say(f"  {name:44s} {r['ours']:.2f} vs {r['theirs']:.2f} on {r['n']} images: difference {r['diff']:+.2f}, 95% CI "
        f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | higher {r['wins']}, lower {r['losses']} | Wilcoxon p {p:.2g}")
    return r


out["judges"] = {}
fc = [r for k in (0, 1) for r in (J(f"outputs/slice_quality_slcmp/judge_claims_shard{k}.json") or {"judgements": []})["judgements"]]
fa = (J("outputs/slice_quality_slcmp_auc/judge_claims_shard0.json") or {"judgements": []})["judgements"]
ours_f = per_image(fc, "narrative", "facts")
if ours_f:
    out["judges"]["facts vs SliceLens"] = paired("facts score, narrative vs SliceLens text", ours_f, per_image(fc, "pattern", "facts"))
    if fa:
        out["judges"]["facts vs + rank test"] = paired("facts score, narrative vs rank-test text", ours_f,
                                                       per_image(fa, "pattern", "facts"))
rb = (J("outputs/slice_quality_slcmp/judge_descriptions.json") or {"judgements": []})["judgements"]
ra = (J("outputs/slice_quality_slcmp_auc/judge_descriptions.json") or {"judgements": []})["judgements"]
if rb:
    ours_r = per_image(rb, "narrative")
    out["judges"]["rubric vs SliceLens"] = paired("rubric, narrative vs SliceLens text", ours_r, per_image(rb, "pattern"))
    if ra:
        out["judges"]["rubric vs + rank test"] = paired("rubric, narrative vs rank-test text", ours_r, per_image(ra, "pattern"))

say("\nSIMULATION (the same draws judged by both rules; Wilson 95% intervals)")
sim = J("outputs/slope_rules_sim.json")
out["sim"] = {}
if sim:
    for (b, n), lab in (((0.0, 400), "false alarms, no association, 400 objects"),
                        ((0.25, 1500), "weak association found, 1,500 objects")):
        r = next(x for x in sim["table"] if x["b"] == b and x["n"] == n)
        k1, k2 = round(r["v1"] * r["draws"]), round(r["AUC"] * r["draws"])
        out["sim"][f"{b}_{n}"] = {"draws": r["draws"], "slope": r["v1"], "auc": r["AUC"],
                                  "slope_ci": wilson(k1, r["draws"]), "auc_ci": wilson(k2, r["draws"])}
        o = out["sim"][f"{b}_{n}"]
        say(f"  {lab:44s} slope {100 * o['slope']:.1f}% [{100 * o['slope_ci'][0]:.1f}, {100 * o['slope_ci'][1]:.1f}] | "
            f"rank test {100 * o['auc']:.1f}% [{100 * o['auc_ci'][0]:.1f}, {100 * o['auc_ci'][1]:.1f}] ({r['draws']} draws)")

say("\nWHERE THE TWO TESTS DISAGREE ON VOC (situations certified by one only)")
dg = J("outputs/slope_vs_auc_disagreement.json")
out["disagree"] = {}
if dg:
    ps = dg["per_situation"]
    ps = ps.items() if isinstance(ps, dict) else ps
    prec = defaultdict(list)
    for _k, v in ps:
        if v.get("precision"):
            prec[v["group"]].append(v["precision"][0])
    rows = {r["group"]: r for r in dg["rows"]}
    ro, so = [g for g in rows if g.startswith(("rank", "AUC"))], [g for g in rows if g.startswith("slope")]
    if ro and so and prec[ro[0]] and prec[so[0]]:
        u = mannwhitneyu(prec[ro[0]], prec[so[0]])
        r1, r2 = rows[ro[0]], rows[so[0]]
        pf = fisher_exact([[r1["replicated"], r1["n_m2"] - r1["replicated"]],
                           [r2["replicated"], r2["n_m2"] - r2["replicated"]]])[1]
        out["disagree"] = {"precision": {ro[0]: float(np.mean(prec[ro[0]])), so[0]: float(np.mean(prec[so[0]])),
                                         "p_mannwhitney": float(u.pvalue)},
                           "replication": {ro[0]: [r1["replicated"], r1["n_m2"]], so[0]: [r2["replicated"], r2["n_m2"]],
                                           "p_fisher": float(pf)}}
        say(f"  precision, mean per situation: {ro[0]} {np.mean(prec[ro[0]]):.2f} vs {so[0]} {np.mean(prec[so[0]]):.2f}, "
            f"Mann-Whitney p {u.pvalue:.2g}")
        say(f"  re-found on the second segmenter: {ro[0]} {r1['replicated']} of {r1['n_m2']} vs {so[0]} "
            f"{r2['replicated']} of {r2['n_m2']}, Fisher p {pf:.2g}")
json.dump(out, open(OUT + ".json", "w"), indent=1)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("CMP_STATS_DONE", flush=True)
