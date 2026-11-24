"""Groundings for the per-instance naming measure on val-EA (slice_quality/examples/
per_instance_naming_ea.py), extended from the 45 victims inside the 120-instance samples to all of
them. GPU: Qwen2.5-VL, bf16 by default as the val-EA groundings it is compared with.

Instances: the 144 victims (edited subjects the segmenter loses) and the 371 survivors (edited
subjects it does not), from outputs/slice_quality_ea/planted_eval.json. Hypotheses: for each class,
those associated with the errors of that class on the ORIGINAL VOC val sample (AUC verdict,
outputs/slope_permutation/results_rank.json) -- the only ones the 'auc' description rule can
report, so the description of every instance is complete under that rule. Values already in the
val-EA groundings (/data/outputs_ea) are reused, never recomputed.

Out: OUT_DIR/groundings_<cls>.json {hypothesis: {instance_id: P(yes)}} (written after every
hypothesis: a stopped run resumes), OUT_DIR/.done_<cls>; prints EA_NAMING_DONE at the end.
Run: OUT_DIR=/data/outputs_ea_naming python examples/ground_ea_naming.py   (DRY=1: count only)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances  # noqa: E402

OUT_DIR = Path(os.environ.get("OUT_DIR", "/data/outputs_ea_naming"))
SUBSTRATE = os.environ.get("SUBSTRATE", "/data/voc2012_ea/substrate_cc.pkl")
RANK = os.environ.get("RANK", "outputs/slope_permutation/results_rank.json")
PLANTED = os.environ.get("PLANTED", "outputs/slice_quality_ea/planted_eval.json")
REUSE = os.environ.get("REUSE", "/data/outputs_ea")
MASKS = os.environ.get("MASKS", "/ds/pascal_ea_material/binary_masks")
PER_IMAGE = os.environ.get("PER_IMAGE", "outputs/pascal_ea/per_image.json")
QWEN_4BIT = os.environ.get("QWEN_4BIT", "0") == "1"
DRY = os.environ.get("DRY") == "1"
_vlm = None


def vlm():
    global _vlm
    if _vlm is None:
        from slicelens.backends import QwenVLM
        t0 = time.time()
        _vlm = QwenVLM(model="/models/qwen", grounding="bbox", load_in_4bit=QWEN_4BIT)
        print(f"Qwen loaded ({time.time() - t0:.0f}s), 4bit={QWEN_4BIT}", flush=True)
    return _vlm


def subject_box(stem):
    """bbox of the edited subject (Pascal-EA binary mask) in the 520 centre crop the substrate uses."""
    import numpy as np
    from PIL import Image
    from slicelens.adapters.voc_seg import _resize_center_crop
    f = f"{MASKS}/{stem}.png"
    if not os.path.exists(f):
        return None
    a = np.array(_resize_center_crop(Image.open(f).convert("L"), 520, Image.NEAREST)) > 127
    ys, xs = np.nonzero(a)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)) if len(xs) else None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def target_map(targets, errors, reference):
    """victim / survivor id ('stem#gK', defined on the ground-truth subject) -> substrate instance.
    A subject the segmenter got right is a PREDICTION ('#pK') in the substrate, not '#gK': it is
    found as the instance of the subject's class, in the same image, whose box overlaps the
    subject's mask box most (IoU >= 0.3)."""
    meta = {r["stem"]: r for r in json.load(open(PER_IMAGE))["rows"]}
    by_img = {}
    for t in errors + reference:
        by_img.setdefault(t.image_id, []).append(t)
    ids = {t.instance_id for t in errors + reference}
    out = {}
    for tg in targets:
        if tg in ids:
            out[tg] = tg
            continue
        stem = tg.split("#")[0]
        cls, box = meta.get(stem, {}).get("voc_class"), subject_box(stem)
        if not cls or box is None:
            continue
        cands = [(iou(t.bbox, box), t.instance_id) for t in by_img.get(stem, [])
                 if t.bbox is not None and (t.gt_class or t.pred_class) == cls]
        best = max(cands, default=(0.0, None))
        if best[0] >= 0.3:
            out[tg] = best[1]
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl = json.load(open(PLANTED))
    targets = set(pl["victims"]) | set(pl["survivors"])
    rk = json.load(open(RANK))["results"]["VOC val"]
    hyps = {}
    for r in rk:
        if r["stage"] == "sample" and r["new_sys"]:
            hyps.setdefault(r["cls"], []).append(r["query"])
    d = pickle.load(open(SUBSTRATE, "rb"))
    errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
    tmap = target_map(targets, errors, reference)
    (OUT_DIR / "targets_map.json").write_text(json.dumps(tmap, indent=1))
    pv, ps = set(pl["victims"]), set(pl["survivors"])
    print(f"targets mapped to a substrate instance: victims {sum(t in tmap for t in pv)}/{len(pv)}, "
          f"survivors {sum(t in tmap for t in ps)}/{len(ps)} "
          f"(directly {sum(1 for t, s in tmap.items() if t == s)}, by the subject mask {sum(1 for t, s in tmap.items() if t != s)})", flush=True)
    wanted = set(tmap.values())
    plan, total = {}, 0
    for cls, qs in sorted(hyps.items()):
        insts = [i for i in typed_to_slicelens_instances(errors, reference, image_paths, only_label=cls)
                 if i.instance_id in wanted]
        if not insts:
            continue
        cache = {}
        for suffix in ("_cc", "_full"):
            f = Path(REUSE) / f"voc_paper_{cls}{suffix}.json"
            if f.exists():
                for c in json.loads(f.read_text()).get("candidates", []):
                    for it in c.get("scored", []):
                        cache[(it["instance_id"], c["query"])] = float(it["conf"])
        own = OUT_DIR / f"groundings_{cls}.json"
        done = json.loads(own.read_text()) if own.exists() else {}
        for q, m in done.items():
            for iid, v in m.items():
                cache[(iid, q)] = v
        miss = sum(1 for q in qs for i in insts if (i.instance_id, q) not in cache)
        plan[cls] = (insts, qs, cache, done)
        total += miss
        print(f"  {cls:12s} {len(insts):4d} instances x {len(qs):2d} hypotheses | to ground {miss}", flush=True)
    print(f"Qwen calls needed: {total}", flush=True)
    if DRY:
        return
    (OUT_DIR / "plan_classes.json").write_text(json.dumps(sorted(plan)))
    t0, n = time.time(), 0
    order = sorted(plan, key=lambda c: -sum(1 for q in plan[c][1] for i in plan[c][0]
                                            if (i.instance_id, q) not in plan[c][2]))
    for cls in order:                       # a queue two GPUs share: the costliest classes first
        insts, qs, cache, done = plan[cls]
        if (OUT_DIR / f".done_{cls}").exists():
            continue
        try:
            os.mkdir(OUT_DIR / f".claim_{cls}")
        except FileExistsError:
            continue
        out = {q: dict(done.get(q, {})) for q in qs}
        for q in qs:
            for inst in insts:
                k = (inst.instance_id, q)
                if k not in cache:
                    cache[k] = vlm().slice_confidence(inst.image_path, inst.region, q)
                    n += 1
                out[q][inst.instance_id] = round(float(cache[k]), 4)
            (OUT_DIR / f"groundings_{cls}.json").write_text(json.dumps(out))
            if n:
                print(f"  {time.strftime('%H:%M')} {cls}: {q[:50]} | {n}/{total} calls, "
                      f"{(time.time() - t0) / n:.2f} s/call", flush=True)
        (OUT_DIR / f".done_{cls}").touch()
    left = [c for c in plan if not (OUT_DIR / f".done_{c}").exists()]
    print(f"worker finished; classes not done yet: {left or 'none'}", flush=True)
    print("EA_NAMING_DONE" if not left else "EA_NAMING_WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
