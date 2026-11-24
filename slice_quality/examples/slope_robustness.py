"""Why does SBD certify fewer language slices than VOC val? Two checks, CPU only, on the
groundings already on disk (the hypotheses re-grounded on their whole class, *_full.json),
with SliceLens's own slope-trend test (max positive slope of error rate vs confidence
> 0.20).

  A  SAME SIZE. Random subsets of SBD with as many images as VOC val (1 449), the same
     re-grounded hypotheses: if the slopes stay weak, the difference is in the data; if
     they come back to VOC's level, the larger population is what dilutes them (and
     VOC's own slices would thin out with more data).
  B  WITHOUT BOUNDARY ERRORS. The whole population again, but an instance counts as an
     error only if its type is not "boundary" (variant: neither boundary nor extent) --
     the errors that depend on how the outline was traced rather than on the scene. If
     the slopes rise on SBD, its coarse annotations are what flattens them. Run on VOC
     too, for the comparison.

Run: python slice_quality/examples/slope_robustness.py  ->  outputs/slice_quality_sbd/slope_robustness.{txt,json}
Env: R (100 subsets), N_IMG (1449), SEED (0).
"""
import json
import os
import pickle
import random
import statistics as st
from types import SimpleNamespace

from slicelens import Hypothesis, PromptType, SliceLensConfig
from slicelens.data_types import CandidateSlice, ScoredInstance
from slicelens.identification import analyze_slope_trend

R = int(os.environ.get("R", "100"))
N_IMG = int(os.environ.get("N_IMG", "1449"))
SEED = int(os.environ.get("SEED", "0"))
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]
RUNS = {"VOC val": ("/data/outputs", "/data/voc2012/substrate_cc.pkl"),
        "SBD": ("/data/outputs_sbd", "/data/sbd/substrate_cc.pkl")}
SLOPE = SliceLensConfig(task_type="segmentation", grounding="bbox").slope


def regrounded(d):
    """[(class, query, [(instance_id, conf, is_error)])] of the hypotheses re-grounded on the class."""
    out = []
    for c in CLASSES:
        f = f"{d}/voc_paper_{c}_full.json"
        if not os.path.exists(f):
            continue
        b = json.load(open(f))
        n = b["num_instances"]
        for cand in b["candidates"]:
            if len(cand["scored"]) == n and n > 120:
                out.append((c, cand["query"], [(x["instance_id"], x["conf"], int(x["is_error"])) for x in cand["scored"]]))
    return out


def slope(points):
    """max positive slope of SliceLens's test on [(conf, is_error)]."""
    cand = CandidateSlice(hypothesis=Hypothesis(query="q", prompt_type=PromptType.SEARCH, source="manual"),
                          scored_instances=[ScoredInstance(instance=SimpleNamespace(is_error=bool(e)), slice_confidence=c)
                                            for c, e in points])
    return analyze_slope_trend(cand, SLOPE).max_positive_slope


lines, out = [], {}
say = lambda s="": (lines.append(s), print(s, flush=True))
say("=" * 78)
say("WHY FEWER CERTIFIED LANGUAGE SLICES ON SBD? (slope-trend test, threshold 0.20)")
say("=" * 78)
hyps, types = {}, {}
for lab, (d, sub) in RUNS.items():
    hyps[lab] = regrounded(d)
    s = pickle.load(open(sub, "rb"))
    types[lab] = {t.instance_id: t.seg_subtype.value for t in s["errors"] + s["reference"]}
    full = [slope([(c, e) for _i, c, e in sc]) for _c, _q, sc in hyps[lab]]
    out[lab] = {"n": len(full), "systematic": sum(x > 0.2 for x in full), "median_slope": st.median(full)}
    imgs = {i.split("#")[0] for _c, _q, sc in hyps[lab] for i, _x, _e in sc}
    say(f"{lab:8s} re-grounded hypotheses {len(full)} | systematic on the whole population "
        f"{out[lab]['systematic']} ({100 * out[lab]['systematic'] / len(full):.0f}%) | median slope "
        f"{out[lab]['median_slope']:.2f} | images {len(imgs)}")

# ---- A: SBD at VOC's size --------------------------------------------------------
say()
say(f"A. SAME SIZE: {R} random subsets of SBD with {N_IMG} images each, same {len(hyps['SBD'])} hypotheses")
all_imgs = sorted({i.split("#")[0] for _c, _q, sc in hyps["SBD"] for i, _x, _e in sc})
rng = random.Random(SEED)
n_sys, med = [], []
for r in range(R):
    keep = set(rng.sample(all_imgs, min(N_IMG, len(all_imgs))))
    sl = [slope([(c, e) for i, c, e in sc if i.split("#")[0] in keep]) for _c, _q, sc in hyps["SBD"]]
    n_sys.append(sum(x > 0.2 for x in sl))
    med.append(st.median(sl))
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
out["A"] = {"systematic_mean": st.mean(n_sys), "p05": q(n_sys, 0.05), "p95": q(n_sys, 0.95), "median_slope_mean": st.mean(med)}
k = len(hyps["SBD"])
say(f"   systematic per subset: mean {st.mean(n_sys):.1f} of {k} ({100 * st.mean(n_sys) / k:.0f}%), "
    f"5-95% range {q(n_sys, 0.05)}-{q(n_sys, 0.95)} | median slope {st.mean(med):.2f}")
say(f"   compare: SBD whole {out['SBD']['systematic']}/{k} ({100 * out['SBD']['systematic'] / k:.0f}%), "
    f"VOC val whole {out['VOC val']['systematic']}/{out['VOC val']['n']} "
    f"({100 * out['VOC val']['systematic'] / out['VOC val']['n']:.0f}%)")

# ---- B: errors that do not depend on the traced outline ----------------------------
say()
say("B. WITHOUT OUTLINE ERRORS: an instance counts as an error only if its type is not excluded")
out["B"] = {}
for excl in (("boundary",), ("boundary", "extent")):
    for lab in RUNS:
        sl, base = [], []
        for _c, _q, sc in hyps[lab]:
            pts = [(c, int(e and types[lab].get(i) not in excl)) for i, c, e in sc]
            sl.append(slope(pts))
            base.append(sum(e for _c2, e in pts) / len(pts))
        ns = sum(x > 0.2 for x in sl)
        out["B"][f"{lab} | no {'+'.join(excl)}"] = {"systematic": ns, "n": len(sl), "median_slope": st.median(sl),
                                                     "error_rate": st.mean(base)}
        say(f"   {lab:8s} without {'+'.join(excl):17s} systematic {ns:2d}/{len(sl)} ({100 * ns / len(sl):.0f}%) | "
            f"median slope {st.median(sl):.2f} | error rate {100 * st.mean(base):.0f}%")
json.dump({"results": out, "report": lines}, open("outputs/slice_quality_sbd/slope_robustness.json", "w"), indent=1)
open("outputs/slice_quality_sbd/slope_robustness.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("SLOPE_ROBUSTNESS_DONE", flush=True)
