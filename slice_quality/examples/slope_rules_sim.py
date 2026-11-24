"""Every SliceLens decision rule against the AUC test, on effects of KNOWN strength. CPU only.

slope_vs_auc_sim.py compared the AUC test with the slope rule of SliceLens's first version (the one this
thesis implements: 10 confidence bins, windows above 10 quantile thresholds, max slope > 0.2). The final
version of the paper (arXiv 2512.24592v4, Sec. 3.4) words the rule differently and adds a Spearman variant,
leaving some parameters open. Here every reading is a baseline, with the open choices DECLARED:
    v1          the implemented rule (slicelens.identification.analyze_slope_trend, default config)
    v4-slope    units with confidence > 0.5, sorted; windows of W consecutive units, stride 1 (the paper:
                "a sliding window over the ordered units", windows under 5 units discarded; W not given):
                OLS slope of the error on the confidence in each window, systematic if the max > 0.2.
                W = 10 and W = 30.
    v4-rho      Spearman rho between confidence and error over all units with confidence > 0.5:
                systematic if rho >= 0.5 and p < 0.05 (the paper's thresholds; one scale = all units)
    v4-rho-ms   the same in windows of 20 and 50 units (stride half a window), systematic if ANY window
                passes ("multi-scale"; the paper does not say how scales are combined)
    AUC         the thesis's test: one-sided, image-block bootstrap SE, alpha 0.05
Labels as in slope_vs_auc_sim.py: P(error) = sigmoid(a + b z) on the real Qwen confidences, the class's real
error rate kept; b = 0 is the null. And on the REAL labels: how many whole-class phrases each rule certifies,
and how many of those point the wrong way (AUC < 0.5).
Run (CPU):  python -u slice_quality/examples/slope_rules_sim.py
Env: DIRS, R (50), BB (200), PAIRS (60), WORKERS (24), OUT (outputs/slope_rules_sim)
"""
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm, rankdata, spearmanr

import slope_vs_auc_sim as base

R = int(os.environ.get("R", "50"))
PAIRS = int(os.environ.get("PAIRS", "60"))
WORKERS = int(os.environ.get("WORKERS", "24"))
OUT = os.environ.get("OUT", "outputs/slope_rules_sim")
base.R, base.PAIRS = R, PAIRS
B_LIST, N_LIST = (0.0, 0.25, 0.5, 1.0), (120, 400, 1500, 5000)
RULES = ("v1", "v4-slope W10", "v4-slope W30", "v4-rho", "v4-rho-ms", "AUC")
GAMMA, TAU, MINW = 0.5, 0.2, 5


def v4_slope(x, y, w):
    keep = x > GAMMA
    xs, ys = x[keep], y[keep].astype(float)
    if len(xs) < MINW:
        return False
    o = np.argsort(-xs, kind="stable")
    xs, ys = xs[o], ys[o]
    w = min(w, len(xs))
    best = -np.inf
    for s in range(0, len(xs) - w + 1):                  # stride 1
        a, b = xs[s:s + w], ys[s:s + w]
        va = a.var()
        if va <= 1e-12:
            continue
        best = max(best, float(((a - a.mean()) * (b - b.mean())).mean() / va))
    return best > TAU


def rho_ok(a, b):
    if len(a) < MINW or a.std() == 0 or b.std() == 0:
        return False
    r, p = spearmanr(a, b)
    return bool(r >= 0.5 and p < 0.05)


def v4_rho(x, y):
    keep = x > GAMMA
    return rho_ok(x[keep], y[keep].astype(float))


def v4_rho_ms(x, y):
    keep = x > GAMMA
    xs, ys = x[keep], y[keep].astype(float)
    o = np.argsort(-xs, kind="stable")
    xs, ys = xs[o], ys[o]
    for w in (20, 50):
        if len(xs) < w:
            if rho_ok(xs, ys):
                return True
            continue
        for s in range(0, len(xs) - w + 1, w // 2):
            if rho_ok(xs[s:s + w], ys[s:s + w]):
                return True
    return False


def verdicts(x, y, img, rng):
    s1, _ = base.slope_sys(x, y)
    _, p = base.rank_test(x, y, img, rng)
    return (s1, v4_slope(x, y, 10), v4_slope(x, y, 30), v4_rho(x, y), v4_rho_ms(x, y), p <= base.ALPHA)


def cell(task):
    pi, b, n, seed = task
    p = base.PAIRS_DATA[pi]
    rng = np.random.default_rng(seed)
    idx = base.subsample(p, n, rng)
    x, img = p["conf"][idx], p["img"][idx]
    rate = float(p["err"].mean())
    z = norm.ppf(rankdata(x) / (len(x) + 1))
    lo, hi = -20.0, 20.0
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
        out.append(verdicts(x, y, img, rng))
    return dict(b=b, n=n, n_real=len(idx), res=out)


if __name__ == "__main__":
    import json
    t0 = time.time()
    base.PAIRS_DATA[:] = base.load()
    lines = []
    say = lambda s="": (lines.append(s), print(s, flush=True))
    say("=" * 110)
    say("EVERY SLICELENS RULE vs THE AUC TEST, ON PLANTED EFFECTS (real Qwen confidences, synthetic labels)")
    say("=" * 110)
    say(f"whole-class groundings: {len(base.PAIRS_DATA)} | R {R} draws per cell, up to {PAIRS} phrases per size")
    say("\nREAL LABELS: phrases each rule certifies on whole classes, and how many point the wrong way (AUC < 0.5)")
    rng0 = np.random.default_rng(1)
    real = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for p in base.PAIRS_DATA:
        a = base.auc(p["conf"], p["err"])
        v = verdicts(p["conf"], p["err"], p["img"], rng0)
        for rule, sys_ in zip(RULES, v):
            if sys_:
                real[p["ds"]][rule][0] += 1
                real[p["ds"]][rule][1] += a is not None and a < 0.5
    for ds in sorted(real):
        n = sum(1 for p in base.PAIRS_DATA if p["ds"] == ds)
        say(f"  {ds} ({n} phrases): " + " | ".join(f"{r} {real[ds][r][0]} ({real[ds][r][1]} wrong way)" for r in RULES))
    rng = np.random.default_rng(0)
    tasks = []
    for n in N_LIST:
        elig = [i for i, p in enumerate(base.PAIRS_DATA) if len(p["conf"]) >= n]
        for pi in (rng.choice(elig, size=min(PAIRS, len(elig)), replace=False) if elig else []):
            for b in B_LIST:
                tasks.append((int(pi), b, n, int(rng.integers(1 << 31))))
    say(f"\nPLANTED: {len(tasks)} cells")
    with Pool(WORKERS) as pool:
        res = pool.map(cell, tasks, chunksize=1)
    agg = defaultdict(lambda: [0, np.zeros(len(RULES))])
    for r in res:
        g = agg[(r["b"], r["n"])]
        for v in r["res"]:
            g[0] += 1
            g[1] += np.array(v, float)
    say(f"\n{'b':>5s} {'objects':>8s} {'draws':>6s} | " + " | ".join(f"{r:>13s}" for r in RULES))
    table = []
    for b in B_LIST:
        for n in N_LIST:
            g = agg.get((b, n))
            if not g or not g[0]:
                continue
            rates = g[1] / g[0]
            table.append(dict(b=b, n=n, draws=g[0], **{r: float(v) for r, v in zip(RULES, rates)}))
            say(f"{b:5.2f} {n:8d} {g[0]:6d} | " + " | ".join(f"{100 * v:12.1f}%" for v in rates))
        say()
    say("b = 0: every 'systematic' is a false alarm (a correct test stays at 5%); b > 0: the share found is the power.")
    json.dump({"table": table, "rules": RULES, "R": R, "real": {d: {r: v for r, v in x.items()} for d, x in real.items()}},
              open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    say(f"done in {time.time() - t0:.0f}s -> {OUT}.txt")
    print("SLOPE_RULES_SIM_DONE", flush=True)
