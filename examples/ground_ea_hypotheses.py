"""Language branch on val-EA (Pascal-EA material edits in place of 690 val images).

Exactly the protocol of the thesis runs (run_voc.py --substrate cc --max-instances
120, per class): the cc substrate of val-EA (build_voc_ea.py), the same hypotheses
-- combined_<cls>.txt (7 canonical + the class-specific ones of July) plus the 8
extra generic ones (slice_quality.describe._GENERIC_EXTRA) --, the same
deterministic 120-instance sample (random.Random(0)), the same Qwen grounding and
the same slope-trend test. The output has the run_voc.py format, so the rescore
reads it unchanged.

The sample (PHASE=sample) is a random sample of 120 instances of the class, drawn
in two strata with proportional allocation: the instances of untouched images and
those of edited images, in the proportion they have in the class. The untouched
stratum is drawn from the July sample (itself a simple random sample of the same
instances), topped up at random if it is short. Proportional stratified sampling
is a valid random sample of the class, and coupling it to July's makes the two
runs differ only where the images differ -- not by the luck of the draw -- while
reusing the July groundings (SAMPLING=srs restores the plain run_voc.py draw).

One difference, about cost only: a grounding P(yes | query, region) depends on
the crop and the box alone. For an instance of an UNTOUCHED image whose id, box
and type are identical to the thesis substrate's, the grounding already computed
in July/October (any voc_paper_<cls>{_full,_cc}.json in CACHE_DIRS) is reused
instead of asking Qwen again. Instances of edited images are always grounded
fresh. Every grounding used here (reused or new) is saved to
OUT_DIR/groundings_<cls>.json for the full-population re-grounding that follows.

Classes are claimed from a shared queue (mkdir lock), so a second GPU can join.

The same script runs the thesis protocol on a NEW dataset with nothing to reuse (the
SBD run): EDITED="" OLD_SUBSTRATE="" CACHE_DIRS="" SAMPLING=srs -- every instance is
grounded fresh and the sample is run_voc.py's own random.Random(0) draw.
Run (GPU container, Qwen mounted):
  OUT_DIR=/data/outputs_ea python examples/ground_ea_hypotheses.py
Env: OUT_DIR, SUBSTRATE, OLD_SUBSTRATE, EDITED, CACHE_DIRS, CLASSES, MAX_INSTANCES,
     QWEN_4BIT, PHASE (sample | full: full = re-ground the hypotheses validated in
     SLICES_INDEX on every instance of their class, like reground_validated.py),
     SAMPLING (coupled | srs), DATASET (name written in the output), FULL_HYPS (json {class: [queries]}:
     the list re-grounded in PHASE=full instead of the certified language slices of SLICES_INDEX).
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from slicelens import Hypothesis, PromptType, SliceLensConfig  # noqa: E402
from slicelens.data_types import CandidateSlice, ScoredInstance  # noqa: E402
from slicelens.identification import analyze_slope_trend, identify_error_slices  # noqa: E402
from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances  # noqa: E402
from slice_quality.describe import _GENERIC_EXTRA  # noqa: E402

OUT_DIR = Path(os.environ.get("OUT_DIR", "/data/outputs_ea"))
SUBSTRATE = os.environ.get("SUBSTRATE", "/data/voc2012_ea/substrate_cc.pkl")
OLD_SUBSTRATE = os.environ.get("OLD_SUBSTRATE", "/data/voc2012/substrate_cc.pkl")
EDITED = os.environ.get("EDITED", "/data/voc2012_ea/edited_stems.txt")
CACHE_DIRS = [d for d in os.environ.get(
    "CACHE_DIRS", "/data/outputs:/data/outputs_extra:/data/outputs_extra_full").split(":") if d]
CLASSES = os.environ.get("CLASSES", "").split() or [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]
MAX_INSTANCES = int(os.environ.get("MAX_INSTANCES", "120"))
QWEN_4BIT = os.environ.get("QWEN_4BIT", "0") == "1"
PHASE = os.environ.get("PHASE", "sample")
SLICES_INDEX = os.environ.get("SLICES_INDEX", "outputs/slice_quality_ea/slices_index.json")
DATASET = os.environ.get("DATASET", "val-EA")

_vlm = None


def vlm():
    global _vlm
    if _vlm is None:
        from slicelens.backends import QwenVLM
        t0 = time.time()
        _vlm = QwenVLM(model="/models/qwen", grounding="bbox", load_in_4bit=QWEN_4BIT)
        print(f"Qwen loaded ({time.time() - t0:.0f}s), 4bit={QWEN_4BIT}", flush=True)
    return _vlm


def sig(t):
    return (tuple(round(float(x), 1) for x in t.bbox), t.seg_subtype.value, t.gt_class, t.pred_class)


def load_cache(cls, reusable):
    """(instance_id, query) -> conf from the thesis runs, for reusable ids only."""
    cache = {}
    for d in CACHE_DIRS:
        for suffix in ("_cc", "_full"):           # _full last: the full-population value wins
            f = Path(d) / f"voc_paper_{cls}{suffix}.json"
            if not f.exists():
                continue
            for c in json.loads(f.read_text()).get("candidates", []):
                for it in c.get("scored", []):
                    iid = it.get("instance_id")
                    if iid in reusable and "conf" in it:
                        cache[(iid, c["query"])] = float(it["conf"])
    own = OUT_DIR / f"groundings_{cls}.json"      # this run's own groundings (resume)
    if own.exists():
        for q, m in json.loads(own.read_text()).items():
            for iid, conf in m.items():
                cache[(iid, q)] = conf
    return cache


def queries_for(cls):
    tpl = os.environ.get("QUERIES_TEMPLATE")      # only this file's hypotheses (e.g. the data-driven
    if tpl:                                       # ones of a class), without the extra generic ones
        return [l.strip() for l in open(tpl.format(cls=cls), encoding="utf-8")
                if l.strip() and not l.startswith("#")]
    qs = []
    for ln in open(f"/data/combined_{cls}.txt", encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and ln not in qs:
            qs.append(ln)
    for q in _GENERIC_EXTRA:
        if q not in qs:
            qs.append(q)
    return qs


def coupled_sample(insts, cls, reusable):
    """Proportional stratified sample (untouched / edited), the untouched stratum
    taken from the July sample of the class; same size as run_voc.py's."""
    july = set()
    f = Path("/data/outputs") / f"voc_paper_{cls}_cc.json"
    if f.exists():
        for c in json.loads(f.read_text()).get("candidates", []):
            july |= {it["instance_id"] for it in c.get("scored", []) if it.get("instance_id")}
    old = [i for i in insts if i.instance_id in reusable]
    new = [i for i in insts if i.instance_id not in reusable]
    k_new = round(MAX_INSTANCES * len(new) / len(insts))
    k_old = MAX_INSTANCES - k_new
    rng = random.Random(0)
    in_july = [i for i in old if i.instance_id in july]
    if len(in_july) >= k_old:
        pick_old = rng.sample(in_july, k_old)
    else:
        rest = [i for i in old if i.instance_id not in july]
        pick_old = in_july + rng.sample(rest, min(len(rest), k_old - len(in_july)))
    pick_new = rng.sample(new, min(len(new), k_new))
    keep = {i.instance_id for i in pick_old + pick_new}
    print(f"[{cls}] sample: {len(pick_old)} untouched ({sum(1 for i in pick_old if i.instance_id in july)} "
          f"from the July sample) + {len(pick_new)} from edited images; class: {len(old)} untouched, "
          f"{len(new)} edited", flush=True)
    return [i for i in insts if i.instance_id in keep]


def validated(cls):
    if os.environ.get("FULL_HYPS"):               # an explicit list per class (GESD point B)
        return json.load(open(os.environ["FULL_HYPS"], encoding="utf-8")).get(cls, [])
    data = json.load(open(SLICES_INDEX, encoding="utf-8"))
    out = []
    for s in data["slices"]:
        if (s.get("source") == "language" and s.get("in_descriptions") and s.get("hypothesis")
                and s["slice_id"].rsplit("_", 1)[0] == cls and s["hypothesis"] not in out):
            out.append(s["hypothesis"])
    return out


def finished(cls, tag):
    return (OUT_DIR / f"voc_paper_{cls}{tag}.json").exists() or (OUT_DIR / f".none{tag}_{cls}").exists()


def claim(cls, tag):
    if finished(cls, tag):
        return False
    try:
        os.mkdir(OUT_DIR / f".claim{tag}_{cls}")
        return True
    except FileExistsError:
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edited = {l.strip() for l in open(EDITED) if l.strip()} if EDITED else set()
    d = pickle.load(open(SUBSTRATE, "rb"))
    errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
    old_sig = {}
    if OLD_SUBSTRATE:                             # nothing to reuse on a new dataset
        old = pickle.load(open(OLD_SUBSTRATE, "rb"))
        old_sig = {t.instance_id: sig(t) for t in old["errors"] + old["reference"]}
    reusable = {t.instance_id for t in errors + reference
                if t.image_id not in edited and old_sig.get(t.instance_id) == sig(t)}
    print(f"{DATASET} substrate {len(errors)} errors / {len(reference)} reference; "
          f"{len(reusable)} instances reusable from the thesis runs; phase {PHASE}", flush=True)
    cfg = SliceLensConfig(task_type="segmentation", grounding="bbox")
    tag = "_cc" if PHASE == "sample" else "_full"
    dry = os.environ.get("DRY") == "1"            # count the Qwen calls needed, call nothing
    tot_new = 0
    for cls in CLASSES:
        if not dry and not claim(cls, tag):
            continue
        t1 = time.time()
        insts = typed_to_slicelens_instances(errors, reference, image_paths, only_label=cls)
        n_all = len(insts)
        if PHASE == "sample":
            queries = queries_for(cls)
            if MAX_INSTANCES and len(insts) > MAX_INSTANCES:
                if os.environ.get("SAMPLING", "coupled") == "srs":
                    insts = random.Random(0).sample(insts, MAX_INSTANCES)
                else:
                    insts = coupled_sample(insts, cls, reusable)
        else:
            queries = validated(cls)
            if not queries:
                print(f"[{cls}] no validated hypothesis: nothing to re-ground", flush=True)
                (OUT_DIR / f".none{tag}_{cls}").write_text("no validated hypothesis\n")
                continue
        cache = load_cache(cls, reusable)
        if dry:
            miss = sum(1 for q in queries for i in insts if (i.instance_id, q) not in cache)
            tot_new += miss
            print(f"DRY [{cls}] {len(queries)} hypotheses x {len(insts)} instances: {miss} new groundings "
                  f"(~{miss * 1.6 / 3600:.1f} h)", flush=True)
            continue
        n_err = sum(i.is_error for i in insts)
        print(f"\n=== {time.strftime('%H:%M')} [{cls}] {len(queries)} hypotheses x {len(insts)} "
              f"instances ({n_err} errors, class population {n_all}) ===", flush=True)
        own = defaultdict(dict)
        if (OUT_DIR / f"groundings_{cls}.json").exists():
            for q, m in json.loads((OUT_DIR / f"groundings_{cls}.json").read_text()).items():
                own[q].update(m)
        hits = new = 0
        candidates = []
        for qi, q in enumerate(queries, 1):
            scored = []
            for inst in insts:
                k = (inst.instance_id, q)
                if k in cache:
                    conf = cache[k]
                    hits += 1
                else:
                    conf = vlm().slice_confidence(inst.image_path, inst.region, q)
                    cache[k] = conf
                    new += 1
                own[q][inst.instance_id] = round(float(conf), 4)
                scored.append(ScoredInstance(instance=inst, slice_confidence=conf))
            scored.sort(key=lambda s: s.slice_confidence, reverse=True)
            candidates.append(CandidateSlice(hypothesis=Hypothesis(query=q, prompt_type=PromptType.SEARCH,
                                                                   source="manual"),
                                             scored_instances=scored))
            (OUT_DIR / f"groundings_{cls}.json").write_text(json.dumps(own))
            print(f"  [{cls}] {qi}/{len(queries)} '{q[:40]}' | reused {hits}, new {new} | "
                  f"{(time.time() - t1) / 60:.1f} min", flush=True)
        systematic = {es.hypothesis.query for es in identify_error_slices(candidates, cfg.slope)}
        report = []
        for cand in candidates:
            trend = analyze_slope_trend(cand, cfg.slope)
            topk = cand.top_k(10)
            report.append({
                "query": cand.hypothesis.query, "is_systematic": cand.hypothesis.query in systematic,
                "max_positive_slope": trend.max_positive_slope,
                "top_k_error_rate": sum(s.instance.is_error for s in topk) / max(len(topk), 1),
                "scored": [{"name": s.instance.metadata.get("name"), "instance_id": s.instance.instance_id,
                            "label": s.instance.label, "iou": s.instance.metadata.get("iou"),
                            "conf": round(s.slice_confidence, 4), "is_error": int(s.instance.is_error)}
                           for s in cand.scored_instances],
                "top_instances": [{"name": s.instance.metadata.get("name"), "label": s.instance.label,
                                   "instance_id": s.instance.instance_id,
                                   "iou": s.instance.metadata.get("iou"), "bbox": s.instance.region.bbox,
                                   "slice_confidence": s.slice_confidence, "is_error": s.instance.is_error}
                                  for s in topk]})
        if PHASE == "full":                       # keep the non-validated candidates, as reground_validated.py
            prev = OUT_DIR / f"voc_paper_{cls}_cc.json"
            if prev.exists():
                done = {c["query"] for c in report}
                report += [c for c in json.loads(prev.read_text())["candidates"] if c["query"] not in done]
        (OUT_DIR / f"voc_paper_{cls}{tag}.json").write_text(json.dumps({
            "label": cls, "num_instances": len(insts), "num_errors": int(n_err), "iou_threshold": 0.5,
            "substrate": "cc" if PHASE == "sample" else "cc-full", "dataset": DATASET,
            "systematic_error_slices": sorted(systematic), "groundings_reused": hits,
            "groundings_new": new, "candidates": report}, indent=1))
        print(f"=== done {cls} in {(time.time() - t1) / 60:.1f} min: {len(systematic)} systematic, "
              f"reused {hits}, new {new} ===", flush=True)
    if dry:
        print(f"DRY total: {tot_new} new groundings (~{tot_new * 1.6 / 3600:.1f} h of GPU at 1.6 s each)", flush=True)
        return
    left = [c for c in CLASSES if not finished(c, tag)]
    print(f"worker finished; classes still without {tag}: {left or 'none'}", flush=True)
    print("EA_GROUNDING_DONE" if not left else "EA_WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
