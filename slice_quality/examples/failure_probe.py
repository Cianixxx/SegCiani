"""Failure directions in the embedding space (Daniele, 28 October 2026: "prova il punto 2"). CPU only; it reads the
official VOC run and writes nothing into it but its own report.

For every class with at least ten correct objects (the minimum of the rank test), a linear classifier tries to tell the
wrong objects of the class from its correct ones using the DINOv2 embedding alone. How well it does, as an out-of-fold
AUC with the folds drawn by image, says how much of the failure appearance can explain. Two controls:
  chance   the same classifier, the same C, on labels shuffled within the class (permutation null)
  size     the size of the object alone (log of its box area), the obvious confound of appearance
Three populations of wrong objects, because the embedding of a wrong object is pooled on its predicted mask unless the
object has none:
  all       every wrong object of the class
  real      without hallucinations: only objects that exist in the ground truth
  truemask  only the objects pooled on their true mask (missed, swallowed by a merge): appearance, not error shape
Then: is a class whose errors appearance separates also a class where the pipeline explains more of them? And what
does the direction look like: the error types and the verified attributes of the wrong objects it scores highest.
  python slice_quality/examples/failure_probe.py   (env RUN, PERMS, JOBS)
Out: <RUN>/failure_probe.json and .txt
"""
import json
import os
import pickle
import sys
import warnings
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
RUN = os.environ.get("RUN", os.path.join(HERE, "..", "..", "..", "RUN_UFFICIALI", "slice_quality_inst"))
PERMS = int(os.environ.get("PERMS", "50"))
JOBS = int(os.environ.get("JOBS", "-1"))
CS = (0.001, 0.01, 0.1, 1.0)
MIN_OK = 10
FOLDS = 5


def model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def oof(X, y, g, C, seed=0):
    """out-of-fold scores, folds by image, stratified on the label"""
    s = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(X, y, g):
        s[te] = model(C).fit(X[tr], y[tr]).decision_function(X[te])
    return s


def choose_C(X, y, g):
    """the C with the best inner out-of-fold AUC (folds by image)"""
    best, bc = -1, CS[0]
    for C in CS:
        a = roc_auc_score(y, oof(X, y, g, C, seed=1))
        if a > best:
            best, bc = a, C
    return bc


def boot_ci(y, s, g, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.unique(g)
    by = {i: np.where(g == i)[0] for i in imgs}
    out = []
    for _ in range(B):
        idx = np.concatenate([by[i] for i in rng.choice(imgs, len(imgs))])
        if len(set(y[idx])) == 2:
            out.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def bh(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    q = np.empty(len(ps))
    prev = 1.0
    for r, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * len(ps) / r)
        q[i] = prev
    return q


def run_class(c, wrong, ok, attrs_all):
    objs = wrong + ok
    X = np.stack([o.embedding / np.linalg.norm(o.embedding) for o in objs])
    area = np.array([max(1.0, (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1])) / (520.0 * 520.0) for o in objs])
    S = np.log(area)[:, None]
    y = np.array([1] * len(wrong) + [0] * len(ok))
    g = np.array([o.image_id for o in objs])
    C = choose_C(X, y, g)
    s = oof(X, y, g, C)
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(0)
    null = []
    for k in range(PERMS):
        yp = rng.permutation(y)
        null.append(roc_auc_score(yp, oof(X, yp, g, C, seed=k + 10)))
    null = np.array(null)
    auc_size = roc_auc_score(y, oof(S, y, g, 1.0))
    auc_both = roc_auc_score(y, oof(np.hstack([X, S]), y, g, C))
    # what the direction looks like: the wrong objects it scores highest against those it scores lowest
    sw = s[:len(wrong)]
    hi, lo = sw >= np.quantile(sw, 0.75), sw <= np.quantile(sw, 0.25)
    types = lambda m: Counter(w.seg_subtype.value for w, k in zip(wrong, m) if k)
    att = lambda m: Counter(w.metadata.get("attribute") for w, k in zip(wrong, m) if k and w.metadata.get("attribute"))
    th, tl, ah, al = types(hi), types(lo), att(hi), att(lo)
    lift = {a: ah[a] / max(1, sum(ah.values())) - al[a] / max(1, sum(al.values())) for a in set(ah) | set(al)
            if a not in ("normal", "other", "error")}
    return dict(cls=c, n_wrong=len(wrong), n_ok=len(ok), C=C, auc=auc, ci=boot_ci(y, s, g),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                p=float((1 + (null >= auc).sum()) / (1 + PERMS)), auc_size=auc_size, auc_both=auc_both,
                types_hi=dict(th), types_lo=dict(tl), attr_lift=dict(sorted(lift.items(), key=lambda kv: -kv[1])))


def main():
    R = pickle.load(open(os.path.join(RUN, "result_tagged.pkl"), "rb"))
    errs, refs = R["errors"], R["reference"]
    cls_of = lambda o: o.gt_class or o.pred_class
    on_true_mask = lambda o: o.seg_subtype.value == "fn" or bool(o.metadata.get("swallowed_by") or o.metadata.get("from_gt_mask"))
    ok_by = {}
    for o in refs:
        ok_by.setdefault(cls_of(o), []).append(o)
    classes = sorted(c for c, v in ok_by.items() if len(v) >= MIN_OK)
    pops = {"all": lambda o: True, "real": lambda o: o.seg_subtype.value != "fp", "truemask": on_true_mask}
    # coverage of each class in the official run: wrong objects inside a certified group, either branch or language only
    idx = json.load(open(os.path.join(RUN, "slices_index.json"), encoding="utf-8"))["slices"]
    cov_any, cov_lang = set(), set()
    for s in idx:
        if s.get("in_descriptions"):
            ids = {i for v in s["images"].values() for i in v}
            cov_any |= ids
            if s.get("source") == "language":
                cov_lang |= ids
    out = {"classes": classes, "perms": PERMS, "folds": FOLDS, "min_ok": MIN_OK, "populations": {},
           "n_true_mask": sum(1 for o in errs if on_true_mask(o))}
    lines = ["FAILURE DIRECTIONS IN THE EMBEDDING SPACE (DINOv2, official VOC run)",
             f"linear probe wrong vs correct per class, out-of-fold AUC (folds by image), {PERMS} permutations; "
             f"classes with >= {MIN_OK} correct objects: {len(classes)}", ""]
    for pop, keep in pops.items():
        jobs = []
        for c in classes:
            wrong = [o for o in errs if cls_of(o) == c and keep(o)]
            if len(wrong) >= MIN_OK:
                jobs.append((c, wrong, ok_by[c]))
        res = Parallel(n_jobs=JOBS)(delayed(run_class)(c, w, k, None) for c, w, k in jobs)
        q = bh([r["p"] for r in res])
        for r, qq in zip(res, q):
            r["q"] = float(qq)
            wrong_ids = [o.instance_id for o in errs if cls_of(o) == r["cls"] and pops[pop](o)]
            r["coverage"] = sum(1 for i in wrong_ids if i in cov_any) / len(wrong_ids)
            r["coverage_lang"] = sum(1 for i in wrong_ids if i in cov_lang) / len(wrong_ids)
        n = sum(r["n_wrong"] + r["n_ok"] for r in res)
        summ = dict(n_classes=len(res), mean_auc=float(np.mean([r["auc"] for r in res])),
                    weighted_auc=float(sum(r["auc"] * (r["n_wrong"] + r["n_ok"]) for r in res) / n),
                    mean_auc_size=float(np.mean([r["auc_size"] for r in res])),
                    mean_auc_both=float(np.mean([r["auc_both"] for r in res])),
                    above_chance=sum(1 for r in res if r["q"] <= 0.05),
                    above_size=sum(1 for r in res if r["auc"] > r["auc_size"]))
        if len(res) >= 5:
            rho, pr = spearmanr([r["auc"] for r in res], [r["coverage"] for r in res])
            rl, pl = spearmanr([r["auc"] for r in res], [r["coverage_lang"] for r in res])
            summ.update(spearman_cov=(float(rho), float(pr)), spearman_cov_lang=(float(rl), float(pl)))
        out["populations"][pop] = {"summary": summ, "classes": res}
        lines.append(f"== population '{pop}': {summ['n_classes']} classes | mean AUC {summ['mean_auc']:.3f} "
                     f"(weighted {summ['weighted_auc']:.3f}) | size alone {summ['mean_auc_size']:.3f} | embedding + size "
                     f"{summ['mean_auc_both']:.3f} | above chance (BH q<=0.05) {summ['above_chance']} | above size "
                     f"{summ['above_size']}")
        if "spearman_cov" in summ:
            lines.append(f"   AUC vs share of the class's errors in a certified group: Spearman "
                         f"{summ['spearman_cov'][0]:+.2f} (p {summ['spearman_cov'][1]:.2f}); situations only "
                         f"{summ['spearman_cov_lang'][0]:+.2f} (p {summ['spearman_cov_lang'][1]:.2f})")
        lines.append(f"   {'class':12s} {'wrong':>5s} {'ok':>4s} {'AUC':>6s} {'95% CI':>13s} {'null95':>6s} {'q':>6s} "
                     f"{'size':>6s} {'both':>6s} {'cov':>5s}  top attribute lift | types among the highest-scored")
        for r in sorted(res, key=lambda r: -r["auc"]):
            top = next(iter(r["attr_lift"].items()), ("-", 0))
            th = ", ".join(f"{k} {v}" for k, v in Counter(r["types_hi"]).most_common(3))
            lines.append(f"   {r['cls']:12s} {r['n_wrong']:5d} {r['n_ok']:4d} {r['auc']:6.3f} "
                         f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] {r['null_p95']:6.3f} {r['q']:6.3f} {r['auc_size']:6.3f} "
                         f"{r['auc_both']:6.3f} {100 * r['coverage']:4.0f}%  {top[0]} {top[1]:+.2f} | {th}")
        lines.append("")
    json.dump(out, open(os.path.join(RUN, "failure_probe.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(RUN, "failure_probe.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("FAILURE_PROBE_DONE")


if __name__ == "__main__":
    main()
