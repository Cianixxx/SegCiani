"""Re-run slope-trend analysis on a saved run_coco report -- offline, no API.

run_coco.py saves each candidate's (confidence, is_error) pairs in the report's
"scored" field. This lets you re-tune the slope-trend thresholds (or just inspect
the error-rate-vs-confidence curve) WITHOUT paying for the VLM again.

    python scripts/analyze_scored.py outputs/bicycle_slices.json
    python scripts/analyze_scored.py outputs/bicycle_slices.json --slope-threshold 0.1 --num-bins 5 --min-bin-count 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicelens.config import SlopeTrendConfig  # noqa: E402
from slicelens.data_types import (  # noqa: E402
    CandidateSlice,
    Hypothesis,
    Instance,
    Region,
    ScoredInstance,
)
from slicelens.identification import analyze_slope_trend  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Offline slope re-analysis of a run_coco report.")
    p.add_argument("report", help="path to the run_coco JSON report")
    p.add_argument("--slope-threshold", type=float, default=0.2)
    p.add_argument("--num-bins", type=int, default=4)
    p.add_argument("--min-bin-count", type=int, default=3)
    p.add_argument("--min-window-size", type=int, default=12)
    p.add_argument("--min-confidence-spread", type=float, default=0.05)
    p.add_argument("--num-thresholds", type=int, default=6)
    p.add_argument("--max-threshold-quantile", type=float, default=0.8)
    args = p.parse_args()

    cfg = SlopeTrendConfig(
        num_bins=args.num_bins, min_bin_count=args.min_bin_count,
        min_window_size=args.min_window_size,
        min_confidence_spread=args.min_confidence_spread,
        num_thresholds=args.num_thresholds,
        max_threshold_quantile=args.max_threshold_quantile,
        slope_threshold=args.slope_threshold,
    )
    data = json.loads(Path(args.report).read_text())
    cands = data.get("candidates", [])
    print(f"Re-analysing {len(cands)} candidates from {args.report}")
    print(f"  cfg: bins={cfg.num_bins} min_bin={cfg.min_bin_count} "
          f"min_win={cfg.min_window_size} spread={cfg.min_confidence_spread} "
          f"slope_thr={cfg.slope_threshold}\n")
    print(f"  {'hypothesis':44s} {'n':>4s} {'conf min..max':>13s} {'slope':>7s} {'sys':>4s}")

    for c in cands:
        scored = []
        for i, pair in enumerate(c.get("scored", [])):
            conf, err = float(pair[0]), int(pair[1])
            inst = Instance(instance_id=str(i), image_path="x",
                            region=Region(point=(0, 0)), is_error=bool(err))
            scored.append(ScoredInstance(instance=inst, slice_confidence=conf))
        scored.sort(key=lambda s: s.slice_confidence, reverse=True)
        cand = CandidateSlice(hypothesis=Hypothesis(query=c["query"]),
                              scored_instances=scored)
        res = analyze_slope_trend(cand, cfg)
        confs = [s.slice_confidence for s in scored]
        cmin, cmax = (min(confs), max(confs)) if confs else (0.0, 0.0)
        print(f"  {c['query'][:44]:44s} {len(scored):4d} {cmin:5.2f}..{cmax:0.2f} "
              f"{res.max_positive_slope:7.3f} {('YES' if res.is_systematic else '-'):>4s}")
        if res.curve_confidence:
            curve = "  ".join(f"{cc:0.2f}:{rr:0.2f}"
                              for cc, rr in zip(res.curve_confidence, res.curve_error_rate))
            print(f"       error-rate vs conf: {curve}")


if __name__ == "__main__":
    main()
