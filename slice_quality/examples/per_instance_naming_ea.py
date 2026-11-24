"""Does the language branch NAME a planted cause object by object? CPU only, saved groundings.

Pascal-EA (val-EA): 691 VOC val subjects re-rendered "made of paper / metal / plastic / wood"; the
victims are those the segmenter loses (recall drop >= 0.3, ea_planted_eval.py). At group level no
certified situation names the material (thesis §13). Here every instance gets a description of its
own: the hypotheses Qwen confirms on it (P(yes) >= TAU), highest first, under three rules:

  all     every confirmed hypothesis
  auc     only hypotheses associated with the errors of that class on the ORIGINAL VOC val sample
          (AUC verdict, slope_permutation.py PHASE=rank, BH q <= 0.05) -- learnt without the edits
  slope   only hypotheses SliceLens's slope test called systematic on the original sample

A description names the cause when one of its top-k phrases contains a word of CAUSE_WORDS, fixed
before looking at the results. Groups: victims, edited objects the edit did not hurt, objects of
unedited images; for the victims also the same object BEFORE the edit (July groundings of VOC val).
Only instances inside the 120-instance samples carry every hypothesis, so only those are counted.

Run: python slice_quality/examples/per_instance_naming_ea.py
Out: outputs/slice_quality_ea/per_instance_naming.{txt,json}
Env: TAU (0.5), RANK (outputs/slope_permutation/results_rank.json)
"""
import json
import os
from collections import Counter, defaultdict

TAU = float(os.environ.get("TAU", "0.5"))
RANK = os.environ.get("RANK", "outputs/slope_permutation/results_rank.json")
CAUSE_WORDS = ("toy", "statue", "model", "stuffed", "figurine", "sculpture", "fake", "artificial",
               "paper", "metal", "plastic", "wood", "replica", "miniature")
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
EXTRA = os.environ.get("EXTRA", "")      # groundings of examples/ground_ea_naming.py: all victims
OUT = "outputs/slice_quality_ea/per_instance_naming" + ("_all" if EXTRA else "")
lines = []
say = lambda s="": (lines.append(s), print(s, flush=True))


def groundings(dirs):
    """{instance: {hypothesis: (P(yes), is_error, class)}} on the 120-instance samples."""
    out = defaultdict(dict)
    for d in dirs:
        for c in CLASSES:
            f = f"{d}/voc_paper_{c}_cc.json"
            if os.path.exists(f):
                for x in json.load(open(f))["candidates"]:
                    for s in x["scored"]:
                        out[s["instance_id"]][x["query"]] = (s["conf"], s["is_error"], c, bool(x["is_systematic"]))
    return out


cause = lambda q: any(w in q.lower() for w in CAUSE_WORDS)
pl = json.load(open("outputs/slice_quality_ea/planted_eval.json"))
victims, survivors = set(pl["victims"]), set(pl["survivors"])
edited = {l.strip() for l in open("/data/voc2012_ea/edited_stems.txt") if l.strip()}
after = groundings(["/data/outputs_ea"])
before = groundings(["/data/outputs", "/data/outputs_extra"])
rk = json.load(open(RANK))["results"]["VOC val"]
auc_ok = {(r["cls"], r["query"]) for r in rk if r["stage"] == "sample" and r["new_sys"]}
slope_ok = {(c, q) for i, h in before.items() for q, (p, e, c, sy) in h.items() if sy}


def describe(h, rule):
    """the confirmed hypotheses of one instance under a rule, highest P(yes) first."""
    keep = []
    for q, (p, e, c, sy) in h.items():
        if p < TAU:
            continue
        if rule == "auc" and (c, q) not in auc_ok:
            continue
        if rule == "slope" and (c, q) not in slope_ok:
            continue
        keep.append((p, q))
    return [q for p, q in sorted(keep, reverse=True)]


def group(i):
    stem = i.split("#")[0]
    if i in victims:
        return "victim"
    if stem in edited:
        return "edited, not lost" if i in survivors else "edited, other"
    return "unedited image"


say("=" * 78)
say(f"PER-INSTANCE NAMING ON val-EA: does an object's own description name the planted cause? (P(yes) >= {TAU})")
say("=" * 78)
say(f"cause words: {', '.join(CAUSE_WORDS)}")
res = {}
for rule in ("all", "auc", "slope"):
    say(f"\nrule '{rule}':")
    res[rule] = {}
    by = defaultdict(list)
    for i, h in after.items():
        by[group(i)].append(describe(h, rule))
    for g in ("victim", "edited, not lost", "edited, other", "unedited image"):
        ds = by.get(g, [])
        if not ds:
            continue
        n = len(ds)
        k1 = sum(1 for d in ds if d[:1] and cause(d[0]))
        k3 = sum(1 for d in ds if any(cause(q) for q in d[:3]))
        ka = sum(1 for d in ds if any(cause(q) for q in d))
        np_ = sum(len(d) for d in ds) / n
        say(f"   {g:17s} n={n:5d} | cause in top-1 {k1:4d} ({100 * k1 / n:4.1f}%) | top-3 {k3:4d} ({100 * k3 / n:4.1f}%) | "
            f"anywhere {ka:4d} ({100 * ka / n:4.1f}%) | phrases per object {np_:.1f}")
        res[rule][g] = {"n": n, "top1": k1, "top3": k3, "any": ka, "phrases": np_}
    # the same victims before the edit (original image, July groundings)
    vs = [i for i in after if i in victims and i in before]
    if vs:
        b3 = sum(1 for i in vs if any(cause(q) for q in describe(before[i], rule)[:3]))
        a3 = sum(1 for i in vs if any(cause(q) for q in describe(after[i], rule)[:3]))
        say(f"   same victims before -> after the edit (n={len(vs)}): cause in top-3 {b3} -> {a3}")
        res[rule]["paired"] = {"n": len(vs), "before_top3": b3, "after_top3": a3}

if EXTRA:
    # every victim and survivor, with every hypothesis the 'auc' rule can report grounded on it
    ext = defaultdict(dict)
    for f in sorted(os.listdir(EXTRA)):
        if f.startswith("groundings_") and f.endswith(".json"):
            c = f[len("groundings_"):-len(".json")]
            for q, m in json.load(open(f"{EXTRA}/{f}")).items():
                for i, p in m.items():
                    ext[i][q] = (p, None, c, False)
    _tm = json.load(open(f"{EXTRA}/targets_map.json"))          # victim/survivor id -> substrate instance
    ext = {t: ext[s] for t, s in _tm.items() if s in ext}
    say(f"\nALL VICTIMS AND SURVIVORS, rule 'auc' (groundings of {EXTRA}):")
    res["auc_all"] = {}
    for g, ids in (("victim", victims), ("edited, not lost", survivors)):
        ds = [describe(ext[i], "auc") for i in ids if i in ext]
        n = len(ds)
        if not n:
            continue
        k1 = sum(1 for d in ds if d[:1] and cause(d[0]))
        k3 = sum(1 for d in ds if any(cause(q) for q in d[:3]))
        say(f"   {g:17s} n={n:5d} of {len(ids)} | cause in top-1 {k1:4d} ({100 * k1 / n:4.1f}%) | top-3 {k3:4d} "
            f"({100 * k3 / n:4.1f}%) | phrases per object {sum(len(d) for d in ds) / n:.1f}")
        res["auc_all"][g] = {"n": n, "of": len(ids), "top1": k1, "top3": k3}
    meta = {r["stem"]: r for r in json.load(open("outputs/pascal_ea/per_image.json"))["rows"]}
    for key in ("material_v3", "voc_class"):
        agg = defaultdict(lambda: [0, 0])
        for i in victims:
            if i in ext:
                v = meta.get(i.split("#")[0], {}).get(key, "?")
                agg[v][1] += 1
                agg[v][0] += int(any(cause(q) for q in describe(ext[i], "auc")[:1]))
        say(f"   victims by {key}: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(agg.items(), key=lambda kv: -kv[1][1])))
        res["auc_all"]["by_" + key] = dict(agg)

say("\nexamples, rule 'auc' (victim: top-3 phrases):")
ex = []
for i in sorted(i for i in after if i in victims):
    d = describe(after[i], "auc")[:3]
    c = next(iter(after[i].values()))[2]
    ex.append({"instance": i, "class": c, "top3": d})
    say(f"   {i:20s} {c:9s} | " + ("; ".join(d) if d else "(nothing confirmed)"))
json.dump({"tau": TAU, "cause_words": CAUSE_WORDS, "results": res, "examples": ex,
           "n_auc_pairs": len(auc_ok), "n_slope_pairs": len(slope_ok)},
          open(OUT + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
open(OUT + ".txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("PER_INSTANCE_NAMING_DONE")
