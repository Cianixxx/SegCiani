"""A size-aware verdict for SliceLens's slope-trend test. CPU only, on the groundings on disk.

SliceLens calls a hypothesis systematic when the largest positive slope of the error rate
vs the VLM's confidence, over windows of the confidence range, is >= 0.20 -- a fixed
threshold, while the statistic is inflated by noise on small populations (slope_robustness.py:
SBD subsets of VOC's size certify twice as often as the whole of SBD). Here the SAME statistic
gets a null distribution at the population's own size:

  statistic  identical to slicelens.identification.analyze_slope_trend (checked against the
             value each run recorded): confidence bins and windows depend on the confidences
             only, so each window's slope is a linear function of the bins' error rates
  null       the error labels permuted among the instances, the confidences kept: the number
             of errors per bin is multivariate-hypergeometric, B = 20 000 draws
  p          (1 + #{null >= observed}) / (B + 1); systematic when the Benjamini-Hochberg
             q <= 0.05 within the family (run x stage: the 120-instance sample, the re-grounding)

Checks: (1) labels shuffled once, i.e. no real effect anywhere: how often each verdict says
"systematic" at N = 120 and at the class's full N; (2) SBD subsets with VOC's number of images,
as in slope_robustness.py A, with this verdict.
Then copies of the grounding files with is_systematic replaced by the new verdict (only for the
hypotheses re-grounded on their whole class; everything else False: it was never tested on the
whole class) go to <dir>_perm, for two rescores of COPIES of the inst and SBD pickles
(outputs/chain_slope_perm.sh). PHASE=compare reads those rescores.

PHASE=rank: a second verdict with a standard and more powerful statistic for the same question
("is the VLM more confident on the errors than on the correct instances?"): the AUC of the
confidence for telling errors from correct instances, tested one-sided against 0.5 with a
standard error from an image-block bootstrap (objects of one image are not independent), BH
q <= 0.05 per family as above; copies to <dir>_rank, report_rank.txt, results_rank.json.

Run: PHASE=test|rank|compare [VERDICT=slope|rank] python slice_quality/examples/slope_permutation.py
Out: outputs/slope_permutation/{report.txt, results.json, report_rank.txt, results_rank.json}
Env: B (20000), Q (0.05), R (100 subsets), SEED (0), BB (500 block-bootstrap draws), VERDICT
"""
import json
import os
import random
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

from slicelens import SliceLensConfig
from slicelens.identification.slope_trend import _quantile

PHASE = os.environ.get("PHASE", "test")
B = int(os.environ.get("B", "20000"))
Q = float(os.environ.get("Q", "0.05"))
R = int(os.environ.get("R", "100"))
SEED = int(os.environ.get("SEED", "0"))
CFG = SliceLensConfig(task_type="segmentation", grounding="bbox").slope
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
RUNS = {"VOC val": ["/data/outputs", "/data/outputs_extra"], "SBD": ["/data/outputs_sbd"]}
OUT = "outputs/slope_permutation"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))


class Slope:
    """SliceLens's max-positive-slope statistic as a linear map of the bins' error rates."""

    def __init__(self, confs):
        c = np.asarray(confs, float)
        self.ok = len(c) > 0 and c.max() - c.min() > 1e-12
        if not self.ok:
            return
        nb = CFG.num_bins
        width = (c.max() - c.min()) / nb
        self.idx = np.minimum(((c - c.min()) / width).astype(int), nb - 1)
        self.counts = np.bincount(self.idx, minlength=nb)
        sums = np.bincount(self.idx, weights=c, minlength=nb)
        self.kept = np.flatnonzero(self.counts >= CFG.min_bin_count)
        bconf = sums[self.kept] / self.counts[self.kept]
        srt = sorted(c.tolist())
        rows = []
        for i in range(CFG.num_thresholds):
            q = (CFG.max_threshold_quantile * i / (CFG.num_thresholds - 1) if CFG.num_thresholds > 1 else 0.0)
            t = _quantile(srt, q)
            sel = [k for k, x in enumerate(bconf) if x >= t]
            if sum(self.counts[self.kept[k]] for k in sel) < CFG.min_window_size or len(sel) < CFG.min_bins_per_window:
                continue
            xs = bconf[sel]
            if xs.max() - xs.min() < CFG.min_confidence_spread:
                continue
            sxx = ((xs - xs.mean()) ** 2).sum()
            if sxx <= 1e-12:
                continue
            w = np.zeros(len(self.kept))
            w[sel] = (xs - xs.mean()) / sxx
            rows.append(w)
        self.W = np.array(rows) if rows else np.zeros((0, len(self.kept)))

    def stat(self, err_per_bin):
        """err_per_bin: (..., num_bins) error counts -> max(0, max window slope)."""
        if not self.ok or not len(self.W):
            return np.zeros(np.shape(err_per_bin)[:-1])
        rates = err_per_bin[..., self.kept] / self.counts[self.kept]
        return np.maximum(0.0, (rates @ self.W.T).max(-1))

    def observed(self, labels):
        if not self.ok:
            return 0.0
        return float(self.stat(np.bincount(self.idx, weights=np.asarray(labels, float), minlength=CFG.num_bins)))

    def pvalue(self, labels, b=B):
        obs = self.observed(labels)
        k = int(np.sum(labels))
        if not self.ok or not len(self.W) or obs <= 0 or k == 0 or k == len(labels):
            return obs, 1.0
        null = self.stat(rng.multivariate_hypergeometric(self.counts.astype(np.int64), k, size=b))
        return obs, (1 + int((null >= obs - 1e-12).sum())) / (b + 1)


def bh(ps, q=Q):
    """Benjamini-Hochberg: list of q-values and of booleans (q-value <= q)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qv = [1.0] * m
    run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * m / rank)
        qv[i] = run
    return qv, [x <= q for x in qv]


def groundings(dirs):
    """[(cls, stage, query, confs, labels, recorded_slope, recorded_sys, file)] of a run."""
    out = []
    for d in dirs:
        for c in CLASSES:
            for stage, suf in (("sample", "_cc"), ("full", "_full")):
                f = f"{d}/voc_paper_{c}{suf}.json"
                if not os.path.exists(f):
                    continue
                r = json.load(open(f))
                for x in r["candidates"]:
                    if stage == "full" and not (len(x["scored"]) == r["num_instances"] and r["num_instances"] > 120):
                        continue                         # carried over from the sample, not re-grounded
                    out.append((c, stage, x["query"], [s["conf"] for s in x["scored"]],
                                [s["is_error"] for s in x["scored"]], x["max_positive_slope"],
                                bool(x["is_systematic"]), f, [s["instance_id"] for s in x["scored"]]))
    return out


if PHASE == "test":
    say("=" * 78)
    say("SLOPE-TREND TEST WITH A NULL AT THE POPULATION'S OWN SIZE (B = %d permutations, BH q <= %.2f)" % (B, Q))
    say("=" * 78)
    res = {}
    for run, dirs in RUNS.items():
        G = groundings(dirs)
        rows = []
        worst = 0.0
        for c, stage, q, confs, lab, rec, rsys, f, ids in G:
            s = Slope(confs)
            obs, p = s.pvalue(lab)
            worst = max(worst, abs(obs - rec))
            # check 1: the same hypothesis with its labels shuffled once (no real effect)
            sh = list(lab)
            random.Random(zlib.crc32(f"{run}|{c}|{stage}|{q}".encode())).shuffle(sh)
            obs0, p0 = s.pvalue(sh, b=2000)
            rows.append({"cls": c, "stage": stage, "query": q, "n": len(lab), "n_err": int(sum(lab)),
                         "slope": obs, "old_sys": rsys, "p": p, "file": f,
                         "null_slope": obs0, "null_p": p0})
        say(f"\n{run}: statistic identical to the recorded one within {worst:.1e} on {len(rows)} groundings")
        for stage in ("sample", "full"):
            fam = [r for r in rows if r["stage"] == stage]
            qv, sig = bh([r["p"] for r in fam])
            for r, a, b2 in zip(fam, qv, sig):
                r["q"], r["new_sys"] = a, b2
            n = len(fam)
            o = sum(r["old_sys"] for r in fam)
            nw = sum(r["new_sys"] for r in fam)
            raw = sum(r["p"] < Q for r in fam)
            both = sum(r["old_sys"] and r["new_sys"] for r in fam)
            say(f"  {stage:6s} {n:4d} groundings (median N {int(np.median([r['n'] for r in fam])) if fam else 0}) | "
                f"old (slope >= 0.20) {o} ({100 * o / max(n, 1):.0f}%) | new (BH q <= {Q}) {nw} ({100 * nw / max(n, 1):.0f}%), "
                f"p < {Q} {raw} | both {both}, old only {o - both}, new only {nw - both}")
            n0o = sum(r["null_slope"] >= CFG.slope_threshold for r in fam)
            n0n = sum(r["null_p"] < Q for r in fam)
            say(f"         labels shuffled (no real effect): old says systematic {n0o} ({100 * n0o / max(n, 1):.1f}%), "
                f"new p < {Q} {n0n} ({100 * n0n / max(n, 1):.1f}%)")
        res[run] = rows

    # check 2: SBD subsets with VOC's number of images, the hypotheses re-grounded on SBD
    say(f"\nSIZE CHECK: {R} random SBD subsets of 1449 images, the hypotheses re-grounded on SBD (B = 2000)")
    full = [r for r in res["SBD"] if r["stage"] == "full"]
    G = {(c, q): (confs, lab, ids) for c, stage, q, confs, lab, _r, _s, _f, ids in groundings(RUNS["SBD"]) if stage == "full"}
    imgs = sorted({i.split("#")[0] for v in G.values() for i in v[2]})
    rr = random.Random(SEED)
    sub_old, sub_new, sub_raw = [], [], []
    for _ in range(R):
        keep = set(rr.sample(imgs, min(1449, len(imgs))))
        ps, olds = [], []
        for r in full:
            confs, lab, ids = G[(r["cls"], r["query"])]
            m = [k for k, i in enumerate(ids) if i.split("#")[0] in keep]
            s = Slope([confs[k] for k in m])
            obs, p = s.pvalue([lab[k] for k in m], b=2000)
            ps.append(p)
            olds.append(obs >= CFG.slope_threshold)
        sub_old.append(sum(olds))
        sub_new.append(sum(bh(ps)[1]))
        sub_raw.append(sum(p < Q for p in ps))
    k = len(full)
    say(f"  {k} hypotheses | old: {np.mean(sub_old):.1f} systematic per subset ({100 * np.mean(sub_old) / k:.0f}%) | "
        f"new (BH within the subset): {np.mean(sub_new):.1f} ({100 * np.mean(sub_new) / k:.0f}%), p < {Q}: "
        f"{np.mean(sub_raw):.1f} | whole SBD: old {sum(r['old_sys'] for r in full)}, new {sum(r['new_sys'] for r in full)}")
    size_check = {"k": k, "old_mean": float(np.mean(sub_old)), "new_mean": float(np.mean(sub_new)),
                  "raw_mean": float(np.mean(sub_raw))}

    # pending: significant on the sample under the new verdict, never re-grounded
    for run, rows in res.items():
        regr = {(r["cls"], r["query"]) for r in rows if r["stage"] == "full"}
        pend = [r for r in rows if r["stage"] == "sample" and r["new_sys"] and (r["cls"], r["query"]) not in regr]
        say(f"\n{run}: significant on the sample under the new verdict but never re-grounded on the class: {len(pend)}")
        for r in sorted(pend, key=lambda r: r["p"])[:15]:
            say(f"     {r['cls']:12s} p={r['p']:.4f} slope={r['slope']:.2f} | {r['query']}")

    # copies of the grounding files with the new verdict, for the rescores
    verdict = {(r["file"], r["query"]): r["new_sys"] for rows in res.values() for r in rows if r["stage"] == "full"}
    pq = {(r["file"], r["query"]): (r["p"], r["q"]) for rows in res.values() for r in rows}
    for run, dirs in RUNS.items():
        for d in dirs:
            os.makedirs(d + "_perm", exist_ok=True)
            for f in sorted(os.listdir(d)):
                if not (f.startswith("voc_paper_") and f.endswith(".json")):
                    continue
                r = json.load(open(f"{d}/{f}"))
                for x in r["candidates"]:
                    x["is_systematic_slope020"] = x["is_systematic"]
                    x["is_systematic"] = bool(verdict.get((f"{d}/{f}", x["query"]), False))
                    x["perm_p"], x["perm_q"] = pq.get((f"{d}/{f}", x["query"]), (None, None))
                r["systematic_error_slices"] = sorted(x["query"] for x in r["candidates"] if x["is_systematic"])
                json.dump(r, open(f"{d}_perm/{f}", "w"))
    json.dump({"results": {k: [{kk: vv for kk, vv in r.items() if kk != "file"} for r in v] for k, v in res.items()},
               "size_check": size_check, "B": B, "Q": Q}, open(f"{OUT}/results.json", "w"), indent=1)
    open(f"{OUT}/report.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("SLOPE_PERM_TEST_DONE", flush=True)

elif PHASE == "rank":
    from scipy.stats import norm, rankdata
    BB = int(os.environ.get("BB", "500"))

    def auc(x, y):
        n1 = int(y.sum())
        n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            return None
        return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    def rank_test(confs, lab, ids, bb=BB):
        """AUC of the confidence for errors vs correct, one-sided p from an image-block bootstrap SE."""
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

    say("=" * 78)
    say("A SECOND VERDICT: AUC OF THE VLM CONFIDENCE FOR ERRORS VS CORRECT INSTANCES, ONE-SIDED, "
        "IMAGE-BLOCK BOOTSTRAP SE (%d draws), BH q <= %.2f" % (BB, Q))
    say("=" * 78)
    res = {}
    for run, dirs in RUNS.items():
        rows = []
        for c, stage, q, confs, lab, rec, rsys, f, ids in groundings(dirs):
            a, p = rank_test(confs, lab, ids)
            sh = list(lab)
            random.Random(zlib.crc32(f"{run}|{c}|{stage}|{q}".encode())).shuffle(sh)
            a0, p0 = rank_test(confs, sh, ids, bb=200)
            rows.append({"cls": c, "stage": stage, "query": q, "n": len(lab), "n_err": int(sum(lab)),
                         "auc": a, "p": p, "old_sys": rsys, "file": f, "null_p": p0, "slope": rec})
        say(f"\n{run}:")
        for stage in ("sample", "full"):
            fam = [r for r in rows if r["stage"] == stage]
            qv, sig = bh([r["p"] for r in fam])
            for r, a, b2 in zip(fam, qv, sig):
                r["q"], r["new_sys"] = a, b2
            n, o, nw = len(fam), sum(r["old_sys"] for r in fam), sum(r["new_sys"] for r in fam)
            both = sum(r["old_sys"] and r["new_sys"] for r in fam)
            say(f"  {stage:6s} {n:4d} groundings | old (slope >= 0.20) {o} ({100 * o / max(n, 1):.0f}%) | "
                f"AUC verdict {nw} ({100 * nw / max(n, 1):.0f}%) | both {both}, old only {o - both}, new only {nw - both} | "
                f"median AUC of the significant {np.median([r['auc'] for r in fam if r['new_sys']]) if nw else float('nan'):.3f}")
            n0 = sum(r["null_p"] < Q for r in fam)
            say(f"         labels shuffled (no real effect): p < {Q} {n0} ({100 * n0 / max(n, 1):.1f}%)")
        res[run] = rows

    say(f"\nSIZE CHECK: {R} random SBD subsets of 1449 images, the hypotheses re-grounded on SBD (BB = 200)")
    full = [r for r in res["SBD"] if r["stage"] == "full"]
    G = {(c, q): (confs, lab, ids) for c, stage, q, confs, lab, _r, _s, _f, ids in groundings(RUNS["SBD"]) if stage == "full"}
    imgs = sorted({i.split("#")[0] for v in G.values() for i in v[2]})
    rr = random.Random(SEED)
    sub_new = []
    for _ in range(R):
        keep = set(rr.sample(imgs, min(1449, len(imgs))))
        ps = []
        for r in full:
            confs, lab, ids = G[(r["cls"], r["query"])]
            m = [k for k, i in enumerate(ids) if i.split("#")[0] in keep]
            ps.append(rank_test([confs[k] for k in m], [lab[k] for k in m], [ids[k] for k in m], bb=200)[1])
        sub_new.append(sum(bh(ps)[1]))
    say(f"  {len(full)} hypotheses | AUC verdict per subset {np.mean(sub_new):.1f} ({100 * np.mean(sub_new) / len(full):.0f}%) | "
        f"whole SBD {sum(r['new_sys'] for r in full)} | old: subsets 50%, whole 26% (slope_robustness.py)")
    for run, rows in res.items():
        regr = {(r["cls"], r["query"]) for r in rows if r["stage"] == "full"}
        pend = [r for r in rows if r["stage"] == "sample" and r["new_sys"] and (r["cls"], r["query"]) not in regr]
        say(f"\n{run}: significant on the sample under the AUC verdict but never re-grounded on the class: {len(pend)}")
        for r in sorted(pend, key=lambda r: r["p"])[:12]:
            say(f"     {r['cls']:12s} AUC={r['auc']:.2f} p={r['p']:.1e} | {r['query']}")
    verdict = {(r["file"], r["query"]): r["new_sys"] for rows in res.values() for r in rows if r["stage"] == "full"}
    pq = {(r["file"], r["query"]): (r["auc"], r["p"], r["q"]) for rows in res.values() for r in rows}
    for run, dirs in RUNS.items():
        for d in dirs:
            os.makedirs(d + "_rank", exist_ok=True)
            for f in sorted(os.listdir(d)):
                if not (f.startswith("voc_paper_") and f.endswith(".json")):
                    continue
                r = json.load(open(f"{d}/{f}"))
                for x in r["candidates"]:
                    x["is_systematic_slope020"] = x["is_systematic"]
                    x["is_systematic"] = bool(verdict.get((f"{d}/{f}", x["query"]), False))
                    x["auc"], x["auc_p"], x["auc_q"] = pq.get((f"{d}/{f}", x["query"]), (None, None, None))
                r["systematic_error_slices"] = sorted(x["query"] for x in r["candidates"] if x["is_systematic"])
                json.dump(r, open(f"{d}_rank/{f}", "w"))
    json.dump({"results": {k: [{kk: vv for kk, vv in r.items() if kk != "file"} for r in v] for k, v in res.items()},
               "size_check": {"new_mean": float(np.mean(sub_new)), "k": len(full)}, "BB": BB, "Q": Q},
              open(f"{OUT}/results_rank.json", "w"), indent=1)
    open(f"{OUT}/report_rank.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("SLOPE_RANK_DONE", flush=True)

else:   # PHASE=compare: the rescores with the new verdict against the runs of the thesis
    sys.path.insert(0, "docs")
    from _revision_numbers import load_inst_summary
    VERDICT = os.environ.get("VERDICT", "slope")
    SUF = "_perm" if VERDICT == "slope" else "_rank"
    REP, RES = (("report.txt", "results.json") if VERDICT == "slope" else ("report_rank.txt", "results_rank.json"))
    lines = open(f"{OUT}/{REP}", encoding="utf-8").read().rstrip("\n").split("\n")
    say("\n" + "=" * 78)
    say("CERTIFIED SITUATIONS WITH THE %s VERDICT (rescores of copies; the thesis runs untouched)" % VERDICT.upper())
    say("=" * 78)
    comp = {}
    for run, base, new in (("VOC val", "outputs/slice_quality_inst", "outputs/slice_quality_inst" + SUF),
                           ("SBD", "outputs/slice_quality_sbd", "outputs/slice_quality_sbd" + SUF)):
        def cert(d):
            sl = json.load(open(f"{d}/slices_index.json"))["slices"]
            return {(s["slice_id"].split("_")[0], s.get("hypothesis")): s for s in sl
                    if s.get("in_descriptions") and s["source"] == "language"}
        cb, cn = cert(base), cert(new)
        sb, sn = load_inst_summary(base), load_inst_summary(new)
        kept, lost, gained = sorted(set(cb) & set(cn)), sorted(set(cb) - set(cn)), sorted(set(cn) - set(cb))
        say(f"\n{run}: certified language situations {len(cb)} -> {len(cn)} (kept {len(kept)}, lost {len(lost)}, "
            f"new {len(gained)}); cluster slices {sb['slices']['cluster']} -> {sn['slices']['cluster']}")
        say(f"   errors inside a certified group: {100 * sb['cov_union'] / sb['n_err']:.1f}% -> "
            f"{100 * sn['cov_union'] / sn['n_err']:.1f}% (language {100 * sb['cov_language'] / sb['n_err']:.1f}% -> "
            f"{100 * sn['cov_language'] / sn['n_err']:.1f}%) | images with a situational description "
            f"{sb['situational_images']} -> {sn['situational_images']}")
        for tag, ks, src in (("-", lost, cb), ("+", gained, cn)):
            for c, h in ks:
                s = src[(c, h)]
                say(f"   {tag} {c:12s} {h} | {s['size']} errors, {s['n_images']} images, {s.get('dominant_error_type')}")
        comp[run] = {"base": len(cb), "new": len(cn), "kept": kept, "lost": lost, "gained": gained,
                     "summary_base": {k: v for k, v in sb.items() if k != "tax"},
                     "summary_new": {k: v for k, v in sn.items() if k != "tax"}}
    js = json.load(open(f"{OUT}/{RES}"))
    js["compare"] = comp
    json.dump(js, open(f"{OUT}/{RES}", "w"), indent=1)
    open(f"{OUT}/{REP}", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("SLOPE_PERM_COMPARE_DONE", flush=True)
