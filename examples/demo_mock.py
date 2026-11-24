"""End-to-end SliceLens demo on synthetic data with the offline mock backends.

Run (no GPU / API key needed)::

    python examples/demo_mock.py

It exercises the full pipeline of Figure 3 -- hypothesis generation, grounded
candidate-slice discovery (Eq. 1), and slope-trend identification (Section
3.2.2) -- and reports Precision@10 (Eq. 2) against the planted ground-truth
slice "bicycle obscured by a person".
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens import SliceLens, SliceLensConfig  # noqa: E402
from slicelens.backends import MockLLM, MockVLM  # noqa: E402
from slicelens.evaluation import precision_at_k  # noqa: E402
from slicelens.identification import analyze_slope_trend  # noqa: E402
from slicelens.synthetic import (  # noqa: E402
    GT_SLICE_NAME,
    make_synthetic_detection_instances,
)


def _bar(value: float, width: int = 30) -> str:
    filled = int(round(value * width))
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    instances, gt_ids = make_synthetic_detection_instances(seed=0)
    overall_error_rate = sum(i.is_error for i in instances) / len(instances)
    print(f"Synthetic dataset: {len(instances)} bicycle-detection instances")
    print(f"  ground-truth slice : {GT_SLICE_NAME!r} ({len(gt_ids)} instances)")
    print(f"  overall error rate : {overall_error_rate:.2f}\n")

    config = SliceLensConfig(task_type="detection", grounding="bbox", precision_k=10)
    # A reasonably confident mock VLM (like a real one): the between-scenario
    # confidence gaps drive the trend, while within-scenario spread stays small.
    sl = SliceLens(MockLLM(), MockVLM(logit_scale=2.5, noise=0.15), config)

    # --- Stage 1: hypotheses -------------------------------------------
    hypotheses = sl.generate_hypotheses(instances=instances, label="bicycle")
    search_h = [h for h in hypotheses if h.prompt_type.value == "search"]
    print(f"Stage 1 - generated {len(hypotheses)} hypotheses "
          f"({len(search_h)} verifiable 'search', "
          f"{len(hypotheses) - len(search_h)} 'cluster' seeds):")
    for h in hypotheses:
        print(f"    [{h.source:9s}/{h.prompt_type.value:7s}] {h.query}")
    print()

    # --- Stage 2: candidate slices -------------------------------------
    candidates = sl.discover_candidate_slices(hypotheses, instances)
    print(f"Stage 2 - discovered {len(candidates)} candidate slices "
          "(ranked by slice confidence, Eq. 1)\n")

    # --- Stage 3: error-slice identification ---------------------------
    error_slices = sl.identify_error_slices(candidates)

    print("Per-candidate analysis:")
    print(f"    {'query':40s} {'maxSlope':>9s} {'systematic':>11s} {'P@10':>6s}")
    for cand in candidates:
        trend = analyze_slope_trend(cand, config.slope)
        p10 = precision_at_k(cand, gt_ids, k=config.precision_k)
        flag = "YES" if trend.is_systematic else "no"
        print(f"    {cand.hypothesis.query[:40]:40s} "
              f"{trend.max_positive_slope:9.3f} {flag:>11s} {p10:6.2f}")
    print()

    print(f"Stage 3 - {len(error_slices)} systematic error slice(s) identified:")
    for es in error_slices:
        p10 = precision_at_k(es.candidate, gt_ids, k=config.precision_k)
        print(f"    * {es.hypothesis.query!r}")
        print(f"        max positive slope : {es.max_positive_slope:.3f}")
        print(f"        Precision@10       : {p10:.2f}")
    print()

    # --- Detailed slope-trend curve for the strongest slice ------------
    if error_slices:
        top = error_slices[0]
        trend = analyze_slope_trend(top.candidate, config.slope)
        print(f"Slope-trend curve for {top.hypothesis.query!r} "
              "(error rate vs. slice confidence):")
        for conf, rate in zip(trend.curve_confidence, trend.curve_error_rate):
            print(f"    conf={conf:0.2f}  err={rate:0.2f}  |{_bar(rate)}|")
        print("\n  -> error rate increases with slice confidence "
              "=> a systematic error slice (Section 3.2.2).")


if __name__ == "__main__":
    main()
