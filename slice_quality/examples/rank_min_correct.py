"""The rank verdict with a minimum of correct objects (Daniele, 2026-10-24: "almeno 10 corretti"). CPU only.

The rank test compares P(yes) of a class's wrong objects with that of its correct ones. With very few correct objects
(VOC pottedplant 1 of 429, chair 3 of 676) the image-block bootstrap has almost nothing to resample on the correct
side, its spread is too small and the p-value comes out near 0 (rank_few_correct.py: an exact rank test confirms
none of pottedplant's 14). So on the whole class a hypothesis is TESTED only when its class has at least MIN_CORRECT
correct objects; the others are not certified ("not testable").

Nothing is recomputed: the p-values already in the rank copies are kept, the hypotheses of the classes below the
minimum leave the family, and BH is redone over the rest, within the same family as before. MIN_CORRECT=0
reproduces the copies as they are (checked on every run).
Env: FAMILY (rank-copy directories of ONE family, ':'-separated), SUFFIX (_m10: the new copies go to <dir><SUFFIX>),
     MIN_CORRECT (10), Q (0.05), OUT (report, without extension)
"""
import json
import os

FAMILY = os.environ["FAMILY"].split(":")
SUFFIX = os.environ.get("SUFFIX", "_m10")
MIN = int(os.environ.get("MIN_CORRECT", "10"))
Q = float(os.environ.get("Q", "0.05"))
OUT = os.environ.get("OUT", "outputs/rank_min_correct")


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    qv, run = [1.0] * m, 1.0
    for r in range(m, 0, -1):
        i = order[r - 1]
        run = min(run, ps[i] * m / r)
        qv[i] = run
    return qv


files = {d: sorted(f for f in os.listdir(d) if f.startswith("voc_paper_") and f.endswith(".json")) for d in FAMILY}
data = {(d, f): json.load(open(os.path.join(d, f))) for d in FAMILY for f in files[d]}
tested = []                                       # (key, candidate index, p, n_ok)
for (d, f), r in data.items():
    if not f.endswith("_full.json"):
        continue
    for k, c in enumerate(r["candidates"]):
        if len(c["scored"]) == r["num_instances"] and c.get("auc_p") is not None:
            tested.append(((d, f), k, c["auc_p"], sum(1 for s in c["scored"] if not s["is_error"])))


def verdict(min_ok):
    keep = [t for t in tested if t[3] >= min_ok]
    return {(t[0], t[1]): q for t, q in zip(keep, bh([t[2] for t in keep]))}


# the family as it was: same p-values, same BH, same verdict
v0 = verdict(0)
miss = sum((v0[(t[0], t[1])] <= Q) != bool(data[t[0]]["candidates"][t[1]]["is_systematic"]) for t in tested)
assert miss == 0, f"{miss} verdicts not reproduced from the stored p-values: is this one family?"
v = verdict(MIN)
lines = [f"family {':'.join(FAMILY)}: {len(tested)} hypotheses tested on whole classes; minimum {MIN} correct objects",
         f"  certified before {sum(q <= Q for q in v0.values())}, now {sum(q <= Q for q in v.values())} "
         f"({len(tested) - len(v)} hypotheses in classes below the minimum, no longer tested)"]
for t in sorted(tested, key=lambda t: (data[t[0]]["label"], t[2])):
    before, now = v0[(t[0], t[1])] <= Q, (t[0], t[1]) in v and v[(t[0], t[1])] <= Q
    if before != now:
        lines.append(f"  {'lost' if before else 'NEW '} {data[t[0]]['label']:12s} correct {t[3]:3d} p {t[2]:.4f} | "
                     f"{data[t[0]]['candidates'][t[1]]['query']}")
for (d, f), r in data.items():
    for k, c in enumerate(r["candidates"]):
        if f.endswith("_full.json") and ((d, f), k) in v0:
            c["auc_q"] = v.get(((d, f), k))
            c["is_systematic"] = bool(c["auc_q"] is not None and c["auc_q"] <= Q)
            if c["auc_q"] is None:
                c["rank_untested"] = f"fewer than {MIN} correct objects in the class"
    r["systematic_error_slices"] = sorted(c["query"] for c in r["candidates"] if c["is_systematic"])
    os.makedirs(d + SUFFIX, exist_ok=True)
    json.dump(r, open(os.path.join(d + SUFFIX, f), "w"))
lines.append(f"  copies -> {', '.join(d + SUFFIX for d in FAMILY)}")
print("\n".join(lines), flush=True)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("RANK_MIN_DONE", flush=True)
