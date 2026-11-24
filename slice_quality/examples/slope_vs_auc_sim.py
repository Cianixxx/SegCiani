"""SliceLens's slope rule against the AUC test, on effects of KNOWN strength. CPU only.

The two tests have only ever been compared on real labels, where the truth is unknown. Here the truth is
planted. For real confidence vectors of Qwen (whole-class groundings of VOC val and SBD: the real scale,
ties and clumps of the VLM), the error labels are replaced by synthetic ones whose dependence on the
confidence is known:
    P(error) = sigmoid(a + b * z),   z = normal score of the confidence's rank within the objects,
with a set so that the error rate equals the real one of that class. b = 0 is the null -- the phrase has
nothing to do with the error, every "systematic" verdict is a false alarm; b > 0 is a real association,
stronger as b grows (its size is also reported as the AUC it produces, to read b in familiar units).
Each cell (pair, b, N): the class's objects are subsampled image by image to N (the size of a VOC class
up to a large SBD one), labels drawn R times, and both verdicts taken:
    slope   slicelens.identification.analyze_slope_trend, default config (tau 0.2) -- SliceLens as run
    AUC     one-sided, image-block bootstrap SE (BB draws), alpha 0.05 -- per cell a single test, so no BH
Also, on the REAL labels of every whole-class grounding: how many slope-systematic phrases have
AUC < 0.5, i.e. the wrong objects get LESS of a "yes" than the correct ones.

Run (CPU, on the server):  python -u slice_quality/examples/slope_vs_auc_sim.py
Env: DIRS (colon-separated grounding dirs, label=path), R (20), BB (200), PAIRS (40), WORKERS (40), OUT
"""
import glob
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm, rankdata

from slicelens.identification import analyze_slope_trend

DIRS = os.environ.get("DIRS", "VOC=/data/outputs_rank:VOC=/data/outputs_extra_rank:SBD=/data/outputs_sbd_rank")
R = int(os.environ.get("R", "20"))
BB = int(os.environ.get("BB", "200"))
PAIRS = int(os.environ.get("PAIRS", "40"))
WORKERS = int(os.environ.get("WORKERS", "40"))
OUT = os.environ.get("OUT", "outputs/slope_vs_auc_sim")
B_LIST = (0.0, 0.25, 0.5, 1.0)
N_LIST = (120, 400, 1500, 5000)
ALPHA = 0.05


def auc(x, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(x)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rank_test(x, y, img, rng):
    """The pipeline's AUC verdict: one-sided p from an image-block bootstrap SE (slope_permutation.py)."""
    a = auc(x, y)
    if a is None:
        return 0.5, 1.0
    groups = defaultdict(list)
    for k, i in enumerate(img):
        groups[i].append(k)
    gl = [np.array(v) for v in groups.values()]
    boots = []
    for _ in range(BB):
        sel = np.concatenate([gl[j] for j in rng.integers(0, len(gl), len(gl))])
        v = auc(x[sel], y[sel])
        if v is not None:
            boots.append(v)
    se = float(np.std(boots)) if len(boots) > 10 else 0.0
    return a, (float(norm.sf((a - 0.5) / se)) if se > 0 else (0.0 if a > 0.5 else 1.0))


def slope_sys(x, y):
    cand = SimpleNamespace(scored_instances=[SimpleNamespace(slice_confidence=float(c),
                                                             instance=SimpleNamespace(is_error=bool(e)))
                                             for c, e in zip(x, y)])
    r = analyze_slope_trend(cand)
    return bool(r.is_systematic), float(r.max_positive_slope)


def load():
    pairs = {}
    for spec in DIRS.split(":"):
        ds, d = spec.split("=", 1)
        for f in sorted(glob.glob(f"{d}/voc_paper_*_full.json")):
            r = json.load(open(f))
            for x in r["candidates"]:
                sc = x["scored"]
                if len(sc) != r["num_instances"] or (ds, r["label"], x["query"]) in pairs:
                    continue
                pairs[(ds, r["label"], x["query"])] = dict(
                    ds=ds, cls=r["label"], q=x["query"], conf=np.array([s["conf"] for s in sc], float),
                    img=np.array([s["instance_id"].split("#")[0] for s in sc]),
                    err=np.array([bool(s["is_error"]) for s in sc]),
                    slope_sys=bool(x.get("is_systematic_slope020", x["is_systematic"])),
                    slope=float(x["max_positive_slope"]))
    return list(pairs.values())


def subsample(p, n, rng):
    """Whole images, in random order, until n objects (all of them when the class is smaller)."""
    if len(p["conf"]) <= n:
        return np.arange(len(p["conf"]))
    imgs = np.unique(p["img"])
    rng.shuffle(imgs)
    by = defaultdict(list)
    for k, i in enumerate(p["img"]):
        by[i].append(k)
    sel = []
    for i in imgs:
        sel += by[i]
        if len(sel) >= n:
            break
    return np.array(sel)


def cell(task):
    pi, b, n, seed = task
    p = PAIRS_DATA[pi]
    rng = np.random.default_rng(seed)
    idx = subsample(p, n, rng)
    x, img = p["conf"][idx], p["img"][idx]
    rate = float(p["err"].mean())
    z = norm.ppf(rankdata(x) / (len(x) + 1))
    lo, hi = -20.0, 20.0                                   # a: the planted labels keep the real error rate
    for _ in range(60):
        a = (lo + hi) / 2
        if (1 / (1 + np.exp(-(a + b * z)))).mean() < rate:
            lo = a
        else:
            hi = a
    pr = 1 / (1 + np.exp(-(a + b * z)))
    out = []
    for _ in range(R):
        y = rng.random(len(x)) < pr
        if y.all() or not y.any():
            continue
        s_sys, _ = slope_sys(x, y)
        a_val, pval = rank_test(x, y, img, rng)
        out.append((s_sys, pval <= ALPHA, a_val))
    return dict(ds=p["ds"], b=b, n=n, n_real=len(idx), res=out)


PAIRS_DATA = []
if __name__ == "__main__":
    t0 = time.time()
    PAIRS_DATA[:] = load()
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    lines = []
    say = lambda s="": (lines.append(s), print(s, flush=True))
    say("=" * 96)
    say("SLOPE RULE vs AUC TEST ON PLANTED EFFECTS (real Qwen confidences, synthetic error labels)")
    say("=" * 96)
    per_ds = defaultdict(int)
    for p in PAIRS_DATA:
        per_ds[p["ds"]] += 1
    say(f"whole-class groundings: {len(PAIRS_DATA)} ({dict(per_ds)}) | R {R} label draws per cell, BB {BB}, "
        f"alpha {ALPHA}")

    # ---- the real labels: slope-systematic phrases that point the wrong way
    real = []
    for p in PAIRS_DATA:                 # the slope verdict recomputed with the same function as below
        p["slope_sys"], p["slope"] = slope_sys(p["conf"], p["err"])
        real.append((p, auc(p["conf"], p["err"])))
    say("\nREAL LABELS: phrases the slope rule calls systematic although the WRONG objects get LESS of a 'yes'")
    for d in sorted({p["ds"] for p in PAIRS_DATA}):
        rr = [(p, a) for p, a in real if p["ds"] == d and a is not None]
        ss = [(p, a) for p, a in rr if p["slope_sys"]]
        inv = [(p, a) for p, a in ss if a < 0.5]
        say(f"  {d}: {len(rr)} phrases | slope-systematic {len(ss)} | of which AUC < 0.5: {len(inv)} "
            f"({100 * len(inv) / max(len(ss), 1):.0f}%)")
        for p, a in sorted(inv, key=lambda t: t[1])[:8]:
            say(f"      {p['cls']:12s} AUC {a:.2f} slope {p['slope']:.2f} | {p['q'][:60]}")

    # ---- planted effects
    rng = np.random.default_rng(0)
    tasks = []
    for n in N_LIST:
        elig = [i for i, p in enumerate(PAIRS_DATA) if len(p["conf"]) >= n]
        pick = rng.choice(elig, size=min(PAIRS, len(elig)), replace=False) if elig else []
        for pi in pick:
            for b in B_LIST:
                tasks.append((int(pi), b, n, int(rng.integers(1 << 31))))
    say(f"\nPLANTED: {len(tasks)} cells ({len(B_LIST)} strengths x {len(N_LIST)} sizes x up to {PAIRS} phrases)")
    with Pool(WORKERS) as pool:
        res = pool.map(cell, tasks, chunksize=1)
    agg = defaultdict(lambda: [0, 0, 0, [], []])          # (b, n) -> [draws, slope hits, auc hits, aucs, n_real]
    for r in res:
        g = agg[(r["b"], r["n"])]
        for s_sys, a_sys, a_val in r["res"]:
            g[0] += 1
            g[1] += s_sys
            g[2] += a_sys
            g[3].append(a_val)
        g[4].append(r["n_real"])
    say(f"\n{'strength b':>10s} {'AUC made':>9s} {'objects':>8s} | {'slope says systematic':>22s} | {'AUC test says systematic':>25s}")
    table = []
    for b in B_LIST:
        for n in N_LIST:
            g = agg.get((b, n))
            if not g or not g[0]:
                continue
            row = dict(b=b, n=n, n_mean=float(np.mean(g[4])), draws=g[0], slope=g[1] / g[0], auc=g[2] / g[0],
                       auc_mean=float(np.mean(g[3])))
            table.append(row)
            say(f"{b:10.2f} {row['auc_mean']:9.2f} {row['n_mean']:8.0f} | {100 * row['slope']:21.1f}% | "
                f"{100 * row['auc']:24.1f}%")
        say()
    say("b = 0: no association, every 'systematic' is a FALSE ALARM (a correct test stays at 5%).")
    say("b > 0: a real association; the share found is the test's POWER, and it should grow with N.")
    json.dump({"table": table, "R": R, "BB": BB, "alpha": ALPHA, "pairs": len(PAIRS_DATA),
               "real": [{"ds": p["ds"], "cls": p["cls"], "q": p["q"], "auc": a, "slope": p["slope"],
                         "slope_sys": p["slope_sys"]} for p, a in real]},
              open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    say(f"done in {time.time() - t0:.0f}s -> {OUT}.txt")
    print("SLOPE_VS_AUC_SIM_DONE", flush=True)
