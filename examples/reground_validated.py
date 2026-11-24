"""Re-ground the VALIDATED hypotheses on ALL instances of their classes.

Motivation (usage improvement #1): a slope-trend-verified hypothesis is a
reusable predicate, but the per-class grounding runs only evaluated it on a
<=120-instance sample -- most instances of the class were never tested, so
images containing the situation get no description. This script grounds ONLY
the validated hypotheses (few) on EVERY instance of their classes (errors +
correct, no sampling), and writes run-compatible JSONs
(voc_paper_<class>_full.json) that the rescore ingests with exact id-joins.

Inputs: the slice index produced by the rescore (in_descriptions == true marks
the validated language slices) and the cc substrate cache (built by
run_voc.py --substrate-cache; CPU).

Run (GPU container, Qwen mounted, ~1.5 s/eval):
    python examples/reground_validated.py \
        --slices-index outputs/slice_quality/slices_index.json \
        --substrate-cache /data/voc2012/substrate_cc.pkl \
        --vlm-backend qwen --vlm-model /models/qwen \
        --out-dir /data/outputs
"""

from __future__ import annotations

import argparse
import json
import re
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens import (  # noqa: E402
    Hypothesis, PromptType, SliceLens, SliceLensConfig, SlopeTrendConfig,
)
from slicelens.identification import analyze_slope_trend  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ground validated hypotheses on full classes.")
    p.add_argument("--slices-index", required=True)
    p.add_argument("--substrate-cache", required=True)
    p.add_argument("--vlm-backend", default="qwen", choices=["qwen", "hf", "mock"])
    p.add_argument("--vlm-model", default=None)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--grounding", default="bbox", choices=["bbox", "crop", "point"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--small-data", action="store_true",
                   help="relaxed slope-trend bins (same flag as run_voc.py)")
    p.add_argument("--classes", default=None,
                   help="comma-separated subset of validated classes to process "
                        "(default: all). Lets two GPUs split the workload without "
                        "duplicating work -- run one process per GPU with a "
                        "disjoint --classes list.")
    return p.parse_args()


def validated_hypotheses(index_path: str) -> dict[str, list[str]]:
    """{class -> [hypothesis, ...]} for language slices used in descriptions."""
    data = json.load(open(index_path, encoding="utf-8"))
    out: dict[str, list[str]] = defaultdict(list)
    for s in data["slices"]:
        if (s.get("source") == "language" and s.get("in_descriptions")
                and s.get("hypothesis")):
            cls = re.sub(r"_(extra|dd|pend)$", "", s["slice_id"].rsplit("_", 1)[0])   # "<cls>_extra_<n>", "<cls>_dd_<n>", "<cls>_pend_<n>" too
            if s["hypothesis"] not in out[cls]:
                out[cls].append(s["hypothesis"])
    return dict(out)


def main() -> None:
    args = parse_args()
    from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances

    hyps_by_cls = validated_hypotheses(args.slices_index)
    n_h = sum(len(v) for v in hyps_by_cls.values())
    print(f"validated hypotheses: {n_h} over {len(hyps_by_cls)} classes "
          f"-> {sorted(hyps_by_cls)}")
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
        hyps_by_cls = {c: v for c, v in hyps_by_cls.items() if c in wanted}
        missing = wanted - set(hyps_by_cls)
        if missing:
            print(f"--classes requested but not validated (skipped): {sorted(missing)}")
        print(f"restricted to {len(hyps_by_cls)} classes: {sorted(hyps_by_cls)}")
    if not hyps_by_cls:
        return

    d = pickle.load(open(args.substrate_cache, "rb"))
    errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
    print(f"substrate: {len(errors)} errors, {len(reference)} reference")

    config = SliceLensConfig(task_type="segmentation", grounding=args.grounding)
    if args.small_data:
        config.slope = SlopeTrendConfig(
            num_bins=4, min_bin_count=3, min_window_size=12,
            min_confidence_spread=0.05, num_thresholds=6, slope_threshold=0.2)

    if args.vlm_backend == "mock":
        from slicelens.backends import MockVLM
        vlm = MockVLM()
    elif args.vlm_backend == "qwen":
        from slicelens.backends import QwenVLM
        vlm = QwenVLM(model=args.vlm_model, grounding=args.grounding,
                      load_in_4bit=args.load_in_4bit)
    else:
        from slicelens.backends import HFVLM
        vlm = HFVLM(model=args.vlm_model, grounding=args.grounding,
                    load_in_4bit=args.load_in_4bit)
    sl = SliceLens(None, vlm, config)

    # Pre-load instances per class and order by DESCENDING eval cost (hypotheses
    # x instances): under a hard time budget, the heaviest-payoff classes (e.g.
    # person) get done first instead of being starved by alphabetical order.
    per_class = {}
    for cls in hyps_by_cls:
        insts = typed_to_slicelens_instances(errors, reference, image_paths, only_label=cls)
        per_class[cls] = insts
    order = sorted(hyps_by_cls, key=lambda c: -len(hyps_by_cls[c]) * len(per_class[c]))
    print("class order (heaviest first): " + ", ".join(
        f"{c}({len(hyps_by_cls[c]) * len(per_class[c])})" for c in order))

    for cls in order:
        queries = hyps_by_cls[cls]
        instances = per_class[cls]
        n_err = sum(i.is_error for i in instances)
        print(f"\n[{cls}] {len(queries)} hypotheses x {len(instances)} instances "
              f"({n_err} errors) ~= {len(queries) * len(instances)} evals", flush=True)
        search = [Hypothesis(query=q, prompt_type=PromptType.SEARCH,
                             source="validated-reground") for q in queries]
        candidates = sl.discover_candidate_slices(search, instances)
        systematic = {es.hypothesis.query for es in sl.identify_error_slices(candidates)}

        report_cands = []
        for cand in candidates:
            trend = analyze_slope_trend(cand, config.slope)
            topk = cand.top_k(10)
            report_cands.append({
                "query": cand.hypothesis.query,
                "is_systematic": cand.hypothesis.query in systematic,
                "max_positive_slope": trend.max_positive_slope,
                "scored": [{"name": s.instance.metadata.get("name"),
                            "instance_id": s.instance.instance_id,
                            "label": s.instance.label,
                            "iou": s.instance.metadata.get("iou"),
                            "conf": round(s.slice_confidence, 4),
                            "is_error": int(s.instance.is_error)}
                           for s in cand.scored_instances],
                "top_instances": [
                    {"name": s.instance.metadata.get("name"), "label": s.instance.label,
                     "instance_id": s.instance.instance_id,
                     "iou": s.instance.metadata.get("iou"),
                     "bbox": s.instance.region.bbox,
                     "slice_confidence": s.slice_confidence,
                     "is_error": s.instance.is_error}
                    for s in topk],
            })

        # The _full file REPLACES the class's previous run in the rescore, so it
        # must not lose the non-validated candidates (dedup aliases / future
        # recalibrations need them): copy them over verbatim.
        regrounded = {c["query"] for c in report_cands}
        for suffix in ("_cc", ""):
            prev = Path(args.out_dir) / f"voc_paper_{cls}{suffix}.json"
            if prev.exists():
                old = json.loads(prev.read_text())
                carried = [c for c in old.get("candidates", [])
                           if c.get("query") not in regrounded]
                report_cands.extend(carried)
                print(f"[{cls}] carried over {len(carried)} non-validated candidates "
                      f"from {prev.name}")
                break

        out = Path(args.out_dir) / f"voc_paper_{cls}_full.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "label": cls, "num_instances": len(instances), "num_errors": int(n_err),
            "substrate": "cc-full", "systematic_error_slices": sorted(systematic),
            "candidates": report_cands}, indent=2))
        print(f"[{cls}] wrote {out} ({sum(1 for c in report_cands if c.get('is_systematic'))}"
              f"/{len(report_cands)} systematic)", flush=True)

    print("REGROUND_DONE", flush=True)


if __name__ == "__main__":
    main()
