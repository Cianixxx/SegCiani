"""Run SliceLens on a Pascal-VOC semantic-segmentation dataset.

Discovers systematic error slices of a segmentation model (e.g. MobileNet) by
comparing its predicted masks to the GT, then ranking instances with a VLM
(Eq. 1) and running the slope-trend analysis.

Example (in the SliceLens container on the GPU server, Gemma 4 12B local):

    python examples/run_voc.py \\
        --dataset-root /work/dataset --crops-dir /work/outputs/voc_crops \\
        --vlm-backend hf --vlm-model /models/gemma --load-in-4bit \\
        --label aeroplane --small-data \\
        --queries "a small or distant aeroplane" \\
                  "an aeroplane with thin elongated wings" \\
                  "an aeroplane against a cluttered background" \\
        --out /work/outputs/voc_aeroplane_slices.json

Omit --label to see the class distribution and run on all classes; omit
--queries to generate hypotheses with the LLM (needs GOOGLE_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens import (  # noqa: E402
    Hypothesis, PromptType, SliceLens, SliceLensConfig, SliceLensResult,
    SlopeTrendConfig,
)
from slicelens.adapters import class_distribution, load_voc_seg_instances  # noqa: E402
from slicelens.backends import GeminiLLM  # noqa: E402
from slicelens.identification import analyze_slope_trend  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SliceLens on Pascal-VOC segmentation.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--crops-dir", required=True, help="where to write center-cropped images")
    p.add_argument("--label", default=None, help="focus one VOC class (e.g. aeroplane)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU below this = segmentation error")
    p.add_argument("--names-file", default=None,
                   help="process every image in this split list (e.g. VOC2012 "
                        "ImageSets/Segmentation/trainval.txt) instead of the answer_gts subset")
    p.add_argument("--substrate", default="class", choices=["class", "cc"],
                   help="'class' = one instance per (image, class) [legacy]; "
                        "'cc' = connected components typed by slice_quality BLOCK 1 "
                        "(shared instance_ids -> exact matching downstream)")
    p.add_argument("--substrate-cache", default=None,
                   help="pickle path for the cc substrate (errors/reference/paths). "
                        "Loaded if it exists, written after the first build -- saves "
                        "~20 min per class in multi-class batches. DELETE the file "
                        "after changing typed_error/config, it is not invalidated "
                        "automatically.")
    p.add_argument("--vlm-backend", default="hf",
                   choices=["hf", "qwen", "gemini", "mock"])
    p.add_argument("--vlm-model", default=None, help="HF model id/path (for --vlm-backend hf)")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--grounding", default="bbox", choices=["bbox", "crop", "point"],
                   help="how the region is given to the VLM; 'crop' (image crop) "
                        "is required for non-grounding-tuned VLMs like Gemma")
    p.add_argument("--llm-model", default="gemini-2.5-pro")
    p.add_argument("--queries", nargs="*", default=None,
                   help="explicit hypotheses (skip LLM generation)")
    p.add_argument("--queries-file", default=None,
                   help="text file with one hypothesis per line (skip LLM generation)")
    p.add_argument("--max-hypotheses", type=int, default=8)
    p.add_argument("--max-instances", type=int, default=0)
    p.add_argument("--small-data", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    queries = list(args.queries or [])
    if args.queries_file:
        with open(args.queries_file, encoding="utf-8") as f:
            queries += [ln.strip() for ln in f
                        if ln.strip() and not ln.lstrip().startswith("#")]

    if args.substrate == "cc":
        # Shared substrate: instances = BLOCK-1 connected components, so the
        # saved run carries the SAME instance_ids as the slice_quality pipeline
        # (exact join downstream instead of the bbox-IoU heuristic).
        import pickle

        from slice_quality.adapters.to_slicelens import typed_to_slicelens_instances
        from slice_quality.adapters.voc_seg_samples import build_voc_samples
        from slice_quality.typed_error import decompose_dataset

        if not args.names_file:
            raise SystemExit("--substrate cc requires --names-file (full-split mode)")
        if args.substrate_cache and Path(args.substrate_cache).exists():
            d = pickle.load(open(args.substrate_cache, "rb"))
            errors, reference, image_paths = d["errors"], d["reference"], d["image_paths"]
            print(f"substrate cache hit: {args.substrate_cache} "
                  f"({len(errors)} errors, {len(reference)} reference)")
        else:
            path = (args.names_file if Path(args.names_file).is_absolute()
                    else str(Path(args.dataset_root) / args.names_file))
            names = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
            samples, image_paths = build_voc_samples(args.dataset_root, args.crops_dir, names)
            errors, reference = decompose_dataset(samples)
            if args.substrate_cache:
                pickle.dump({"errors": errors, "reference": reference,
                             "image_paths": image_paths},
                            open(args.substrate_cache, "wb"))
                print(f"substrate cache written -> {args.substrate_cache}")
        instances = typed_to_slicelens_instances(
            errors, reference, image_paths, only_label=args.label)
    else:
        instances = load_voc_seg_instances(
            args.dataset_root, args.crops_dir, iou_threshold=args.iou, only_label=args.label,
            names_file=args.names_file,
        )
    if not args.label:
        print("Class distribution (pick one with --label):")
        for cls, d in class_distribution(instances).items():
            print(f"  {cls:14s} n={d['n']:4d}  errori={d['errors']:4d}")
    n_err = sum(i.is_error for i in instances)
    print(f"\nUsing {len(instances)} instances"
          f"{' for ' + args.label if args.label else ''} "
          f"({n_err} errors, {len(instances) - n_err} correct).")
    if not instances:
        return
    if args.max_instances and len(instances) > args.max_instances:
        import random
        instances = random.Random(0).sample(instances, args.max_instances)

    config = SliceLensConfig(task_type="segmentation", grounding=args.grounding)
    if args.small_data:
        config.slope = SlopeTrendConfig(
            num_bins=4, min_bin_count=3, min_window_size=12,
            min_confidence_spread=0.05, num_thresholds=6, slope_threshold=0.2)

    if args.vlm_backend == "mock":
        from slicelens.backends import MockLLM, MockVLM
        llm, vlm = MockLLM(), MockVLM()
    elif args.vlm_backend == "hf":
        from slicelens.backends import HFVLM
        if not args.vlm_model:
            raise SystemExit("--vlm-backend hf requires --vlm-model")
        llm = None if queries else GeminiLLM(model=args.llm_model)
        vlm = HFVLM(model=args.vlm_model, grounding=args.grounding,
                    load_in_4bit=args.load_in_4bit)
    elif args.vlm_backend == "qwen":
        from slicelens.backends import QwenVLM
        if not args.vlm_model:
            raise SystemExit("--vlm-backend qwen requires --vlm-model")
        llm = None if queries else GeminiLLM(model=args.llm_model)
        vlm = QwenVLM(model=args.vlm_model, grounding=args.grounding,
                      load_in_4bit=args.load_in_4bit)
    else:  # gemini
        from slicelens.backends import GeminiVLM
        llm, vlm = GeminiLLM(model=args.llm_model), GeminiVLM()

    sl = SliceLens(llm, vlm, config)

    # Hypotheses: explicit --queries, or LLM-generated for the segmentation task.
    if queries:
        search = [Hypothesis(query=q, prompt_type=PromptType.SEARCH, source="manual")
                  for q in queries]
        hypotheses = search
    else:
        hypotheses = sl.generate_hypotheses(instances=None, label=args.label or "object")
        search = [h for h in hypotheses if h.prompt_type == PromptType.SEARCH]
        if args.max_hypotheses:
            search = search[: args.max_hypotheses]

    print(f"\nVerifying {len(search)} hypotheses on {len(instances)} instances "
          f"(~{len(search) * len(instances)} VLM evals)...")
    candidates = sl.discover_candidate_slices(search, instances)
    error_slices = sl.identify_error_slices(candidates)
    systematic = {es.hypothesis.query for es in error_slices}
    print(f"\nSystematic error slices: {len(error_slices)}\n")

    print(f"  {'hypothesis':46s} {'conf min..max':>13s} {'slope':>7s} {'sys':>4s} {'top%err':>8s}")
    report_cands = []
    for cand in sorted(candidates, key=lambda c: max(c.confidences() or [0]), reverse=True):
        trend = analyze_slope_trend(cand, config.slope)
        confs = cand.confidences()
        cmin, cmax = (min(confs), max(confs)) if confs else (0.0, 0.0)
        topk = cand.top_k(10)
        topk_err = sum(s.instance.is_error for s in topk) / max(len(topk), 1)
        is_sys = cand.hypothesis.query in systematic
        print(f"  {cand.hypothesis.query[:46]:46s} {cmin:5.2f}..{cmax:0.2f} "
              f"{trend.max_positive_slope:7.3f} {('YES' if is_sys else '-'):>4s} {topk_err:8.2f}")
        report_cands.append({
            "query": cand.hypothesis.query, "is_systematic": is_sys,
            "max_positive_slope": trend.max_positive_slope,
            "top_k_error_rate": topk_err,
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
                 "iou": s.instance.metadata.get("iou"), "bbox": s.instance.region.bbox,
                 "slice_confidence": s.slice_confidence, "is_error": s.instance.is_error}
                for s in topk],
        })

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "label": args.label, "num_instances": len(instances),
            "num_errors": int(n_err), "iou_threshold": args.iou,
            "substrate": args.substrate,
            "systematic_error_slices": sorted(systematic),
            "candidates": report_cands}, indent=2))
        print(f"\nWrote report -> {out}")


if __name__ == "__main__":
    main()
