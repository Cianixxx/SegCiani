"""Does the slope rule of SliceLens depend on the share of wrong objects in the class? The simulation of
slope_rules_sim.py (real Qwen confidences of whole-class groundings, P(error) = sigmoid(a + b z), effects of known
strength b, b = 0 the null), with one change: the class's error rate is set to a given share instead of being kept.
Two shares: GESD's (wrong objects / all objects of the GESD substrate, about 37%) and VOC val's and SBD's (about 91%).
Rules: v1 (the implemented slope rule) and the rank test. CPU only.
Run (server):  python -u slice_quality/examples/slope_baserate_sim.py
Env: DIRS (as slope_vs_auc_sim.py), RATES ("gesd,vocsbd" or numbers), R (50), PAIRS (60), WORKERS (24),
     GESD_SUBSTRATE (/gesd/substrate.pkl), OUT (outputs/slope_baserate_sim)
"""
import json
import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm, rankdata

import slope_vs_auc_sim as base

R = int(os.environ.get("R", "50"))
PAIRS = int(os.environ.get("PAIRS", "60"))
WORKERS = int(os.environ.get("WORKERS", "24"))
OUT = os.environ.get("OUT", "outputs/slope_baserate_sim")
SUB = os.environ.get("GESD_SUBSTRATE", "/gesd/substrate.pkl")
B_LIST, N_LIST = (0.0, 0.25, 0.5), (120, 400, 1500, 5000)
base.R, base.PAIRS = R, PAIRS


def cell(task):
    pi, b, n, rate, seed = task
    p = base.PAIRS_DATA[pi]
    rng = np.random.default_rng(seed)
    idx = base.subsample(p, n, rng)
    x, img = p["conf"][idx], p["img"][idx]
    z = norm.ppf(rankdata(x) / (len(x) + 1))
    lo, hi = -20.0, 20.0
    for _ in range(60):                                     # intercept that gives the chosen error rate
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
        s1, _ = base.slope_sys(x, y)
        _, pv = base.rank_test(x, y, img, rng)
        out.append((bool(s1), bool(pv <= base.ALPHA)))
    return dict(b=b, n=n, rate=rate, res=out)


if __name__ == "__main__":
    t0 = time.time()
    base.PAIRS_DATA[:] = base.load()
    lines = []
    say = lambda s="": (lines.append(s), print(s, flush=True))
    d = pickle.load(open(SUB, "rb"))
    g_err, g_ref = len(d["errors"]), len(d["reference"])
    vs_err = np.mean([p["err"].mean() for p in base.PAIRS_DATA])
    named = {"gesd": g_err / (g_err + g_ref), "vocsbd": float(vs_err)}
    rates = [named.get(r, None) or float(r) for r in os.environ.get("RATES", "gesd,vocsbd").split(",")]
    say("=" * 110)
    say("THE SLOPE RULE AND THE RANK TEST AT TWO SHARES OF WRONG OBJECTS (real Qwen confidences, synthetic labels)")
    say("=" * 110)
    say(f"GESD substrate: {g_err} wrong, {g_ref} correct -> {100 * named['gesd']:.1f}% wrong | whole-class groundings of "
        f"VOC val and SBD: mean share wrong {100 * vs_err:.1f}% | R {R} draws per cell, up to {PAIRS} phrases per size")
    rng = np.random.default_rng(0)
    tasks = []
    for n in N_LIST:
        elig = [i for i, p in enumerate(base.PAIRS_DATA) if len(p["conf"]) >= n]
        for pi in (rng.choice(elig, size=min(PAIRS, len(elig)), replace=False) if elig else []):
            for b in B_LIST:
                for rate in rates:
                    tasks.append((int(pi), b, n, rate, int(rng.integers(1 << 31))))
    with Pool(WORKERS) as pool:
        res = pool.map(cell, tasks, chunksize=1)
    agg = defaultdict(lambda: [0, np.zeros(2)])
    for r in res:
        g = agg[(r["b"], r["n"], r["rate"])]
        for v in r["res"]:
            g[0] += 1
            g[1] += np.array(v, float)
    say(f"\n{'b':>5s} {'objects':>8s} {'wrong':>6s} {'draws':>6s} | {'slope rule':>10s} | {'rank test':>9s}")
    table = []
    for b in B_LIST:
        for n in N_LIST:
            for rate in rates:
                g = agg.get((b, n, rate))
                if not g or not g[0]:
                    continue
                s, k = g[1] / g[0]
                table.append(dict(b=b, n=n, rate=rate, draws=g[0], slope=float(s), auc=float(k)))
                say(f"{b:5.2f} {n:8d} {100 * rate:5.1f}% {g[0]:6d} | {100 * s:9.1f}% | {100 * k:8.1f}%")
        say()
    say("b = 0: every 'systematic' is a false alarm (a correct test stays at 5%); b > 0: the share found is the power.")
    json.dump({"table": table, "rates": named, "gesd_wrong": g_err, "gesd_correct": g_ref, "R": R},
              open(OUT + ".json", "w"), indent=1)
    open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    say(f"done in {time.time() - t0:.0f}s -> {OUT}.txt")
    print("SLOPE_BASERATE_SIM_DONE", flush=True)
