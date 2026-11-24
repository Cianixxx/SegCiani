"""Run SliceLens on real COCO detection results (Gemini + Qwen2.5-VL).

Example (on the GPU server, after `pip install -e ".[gemini,qwen]"` and setting
GOOGLE_API_KEY)::

    python examples/run_coco.py \\
        --gt        /data/coco/annotations/instances_val2017.json \\
        --results   /data/coco/preds/faster_rcnn_val2017.bbox.json \\
        --image-root /data/coco/val2017 \\
        --label bicycle \\
        --out outputs/bicycle_slices.json

``--results`` is a standard COCO detection-results file (the
``[{image_id, category_id, bbox, score}, ...]`` list that pycocotools and most
detectors emit).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens import (  # noqa: E402
    PromptType,
    SliceLens,
    SliceLensConfig,
    SliceLensResult,
    SlopeTrendConfig,
)
from slicelens.adapters import (  # noqa: E402
    load_coco_detection_errors,
    load_coco_segmentation_errors,
)
from slicelens.backends import GeminiLLM, QwenVLM  # noqa: E402
from slicelens.identification import analyze_slope_trend  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SliceLens on COCO detection results.")
    p.add_argument("--gt", required=True, help="COCO ground-truth json")
    p.add_argument("--results", required=True, help="COCO detection-results json")
    p.add_argument("--image-root", required=True, help="directory with the images")
    p.add_argument("--label", required=True, help="target class, e.g. 'bicycle'")
    p.add_argument("--task", default="detection", choices=["detection", "segmentation"])
    p.add_argument("--grounding", default="bbox", choices=["bbox", "point"])
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--no-data-driven", action="store_true",
                   help="skip the (slower) data-driven hypothesis branch")
    p.add_argument("--llm-model", default="gemini-2.5-pro")
    p.add_argument("--vlm-backend", default="qwen",
                   choices=["qwen", "gemini", "hf", "mock"],
                   help="qwen = Qwen2.5-VL (logits); gemini = API (logprobs); "
                        "hf = any HuggingFace open VLM via --vlm-model (e.g. Gemma); "
                        "mock = offline plumbing test (no GPU/API)")
    p.add_argument("--vlm-model", default=None,
                   help="VLM model id (default: Qwen2.5-VL-7B for qwen, "
                        "gemini-2.5-flash for gemini)")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantize the Qwen VLM (needed on ~11GB GPUs, e.g. westworld)")
    # Caps to keep an API run tractable (the VLM is called per instance x hypothesis).
    p.add_argument("--max-instances", type=int, default=0,
                   help="randomly subsample to N instances (0 = all)")
    p.add_argument("--max-hypotheses", type=int, default=0,
                   help="verify only the first N search hypotheses (0 = all)")
    p.add_argument("--queries", nargs="*", default=None,
                   help="verify these explicit hypotheses instead of LLM-generated "
                        "ones (targeted + cheap), e.g. --queries \"a small bicycle\" "
                        "\"a bicycle occluded by a person\"")
    p.add_argument("--vlm-samples", type=int, default=0,
                   help="Gemini sampling-fallback draws per instance (0 = backend default)")
    p.add_argument("--small-data", action="store_true",
                   help="loosen slope-trend thresholds for small runs (~tens of instances)")
    p.add_argument("--out", default=None, help="path to write a JSON report")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    load_errors = (
        load_coco_segmentation_errors
        if args.task == "segmentation"
        else load_coco_detection_errors
    )
    instances = load_errors(
        gt_json_path=args.gt,
        results_json_path=args.results,
        image_root=args.image_root,
        target_label=args.label,
        iou_threshold=args.iou,
        score_threshold=args.score_threshold,
    )
    if not instances:
        print("No instances for this label -- check the label / files.")
        return
    if args.max_instances and len(instances) > args.max_instances:
        instances = random.Random(0).sample(instances, args.max_instances)
    n_err = sum(i.is_error for i in instances)
    print(f"Using {len(instances)} '{args.label}' instances "
          f"({n_err} errors, {len(instances) - n_err} correct).")

    # Resolve the VLM model id + backend.
    if args.vlm_backend == "gemini":
        vlm_model = args.vlm_model or "gemini-2.5-flash"
    elif args.vlm_backend == "mock":
        vlm_model = "mock"
    elif args.vlm_backend == "hf":
        if not args.vlm_model:
            raise SystemExit("--vlm-backend hf requires --vlm-model <hf-model-id> "
                             "(e.g. google/gemma-3-... or a multimodal VLM).")
        vlm_model = args.vlm_model
    else:
        vlm_model = args.vlm_model or "Qwen/Qwen2.5-VL-7B-Instruct"

    config = SliceLensConfig(
        task_type=args.task,
        grounding=args.grounding,
        precision_k=args.k,
        use_data_driven=not args.no_data_driven,
        llm_model=args.llm_model,
        vlm_model=vlm_model,
    )
    if args.small_data:
        config.slope = SlopeTrendConfig(
            num_bins=4, min_bin_count=3, min_window_size=12,
            min_confidence_spread=0.05, num_thresholds=6,
            max_threshold_quantile=0.8, slope_threshold=0.2,
        )

    if args.vlm_backend == "mock":
        from slicelens.backends import MockLLM, MockVLM

        llm, vlm = MockLLM(), MockVLM()
    elif args.vlm_backend == "gemini":
        from slicelens.backends import GeminiVLM

        vlm_kwargs: dict = {"model": vlm_model, "grounding": config.grounding}
        if args.vlm_samples:
            vlm_kwargs["n_samples"] = args.vlm_samples
        llm, vlm = GeminiLLM(model=config.llm_model), GeminiVLM(**vlm_kwargs)
    elif args.vlm_backend == "hf":
        from slicelens.backends import HFVLM

        llm = GeminiLLM(model=config.llm_model)
        vlm = HFVLM(model=vlm_model, grounding=config.grounding,
                    load_in_4bit=args.load_in_4bit)
    else:
        llm = GeminiLLM(model=config.llm_model)
        vlm = QwenVLM(
            model=vlm_model,
            grounding=config.grounding,
            load_in_4bit=args.load_in_4bit,
        )

    sl = SliceLens(llm, vlm, config)

    # Hypotheses: either explicit (--queries) or LLM-generated (capped).
    if args.queries:
        from slicelens import Hypothesis

        hypotheses = [
            Hypothesis(query=q, prompt_type=PromptType.SEARCH, source="manual")
            for q in args.queries
        ]
        search = hypotheses
    else:
        hypotheses = sl.generate_hypotheses(instances=instances, label=args.label)
        search = [h for h in hypotheses if h.prompt_type == PromptType.SEARCH]
        if args.max_hypotheses:
            search = search[: args.max_hypotheses]

    print(f"Verifying {len(search)} hypotheses on {len(instances)} instances "
          f"(~{len(search) * len(instances)} VLM evals)...")
    candidates = sl.discover_candidate_slices(search, instances)
    error_slices = sl.identify_error_slices(candidates)
    systematic = {es.hypothesis.query for es in error_slices}
    print(f"\nGenerated {len(hypotheses)} hypotheses | verified {len(candidates)} | "
          f"systematic error slices: {len(error_slices)}\n")

    # Diagnostics for EVERY candidate (so a 0-slice result is still interpretable).
    print(f"  {'hypothesis':44s} {'conf min..max':>13s} {'slope':>7s} {'sys':>4s} {'top%err':>8s}")
    candidates_report = []
    for cand in sorted(candidates, key=lambda c: max(c.confidences() or [0]), reverse=True):
        trend = analyze_slope_trend(cand, config.slope)
        confs = cand.confidences()
        cmin, cmax = (min(confs), max(confs)) if confs else (0.0, 0.0)
        topk = cand.top_k(args.k)
        topk_err = sum(s.instance.is_error for s in topk) / max(len(topk), 1)
        is_sys = cand.hypothesis.query in systematic
        print(f"  {cand.hypothesis.query[:44]:44s} {cmin:5.2f}..{cmax:0.2f} "
              f"{trend.max_positive_slope:7.3f} {('YES' if is_sys else '-'):>4s} {topk_err:8.2f}")
        candidates_report.append({
            "query": cand.hypothesis.query,
            "source": cand.hypothesis.source,
            "is_systematic": is_sys,
            "max_positive_slope": trend.max_positive_slope,
            "confidence_min": cmin, "confidence_max": cmax,
            "top_k_error_rate": topk_err,
            "curve": {"confidence": trend.curve_confidence,
                      "error_rate": trend.curve_error_rate},
            # Compact (confidence, is_error) for every instance -> re-analyse offline
            # (re-tune the slope params) for FREE, no extra API calls.
            "scored": [[round(s.slice_confidence, 4), int(s.instance.is_error)]
                       for s in cand.scored_instances],
            "top_instances": [
                {"image_path": s.instance.image_path, "bbox": s.instance.region.bbox,
                 "slice_confidence": s.slice_confidence,
                 "is_error": s.instance.is_error,
                 "error_type": s.instance.error_type.value}
                for s in topk
            ],
        })

    report = {
        "label": args.label, "task": args.task,
        "num_instances": len(instances),
        "num_errors": int(n_err),
        "systematic_error_slices": sorted(systematic),
        "candidates": candidates_report,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nWrote report -> {out_path}")


if __name__ == "__main__":
    main()
