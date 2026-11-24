"""Automated error slice identification via slope trend analysis (Section 3.2.2).

Existing methods threshold the *average* error rate, which is sensitive to the
threshold and ignores the reliability of the slice. SliceLens instead asks a
structural question: **does the error likelihood increase monotonically with the
slice confidence?** If so, the slice captures a genuine systematic failure.

Algorithm (faithful to the paper's description):

1. Each instance contributes a point ``(confidence, is_error)``.
2. A sliding threshold ``t`` defines a set of *overlapping* high-confidence
   windows ``W_t = {i : confidence_i >= t}``.
3. Inside each window we estimate the *error-rate vs. confidence* curve by
   binning the confidences and averaging the error labels per bin, then fit a
   local line by least squares and read its slope.
4. The **maximum positive slope** over all windows is recorded.
5. The slice is flagged as a systematic error when that slope exceeds
   ``slope_threshold`` (the paper reports the decision is robust to this value).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SlopeTrendConfig
from ..data_types import CandidateSlice, ErrorSlice


@dataclass
class WindowFit:
    """Diagnostics for one sliding-threshold window."""

    threshold: float
    size: int
    slope: float
    bin_confidences: list[float] = field(default_factory=list)
    bin_error_rates: list[float] = field(default_factory=list)


@dataclass
class SlopeTrendResult:
    max_positive_slope: float
    is_systematic: bool
    windows: list[WindowFit] = field(default_factory=list)
    # Whole-slice error-rate curve (for visualisation, cf. Figure 17).
    curve_confidence: list[float] = field(default_factory=list)
    curve_error_rate: list[float] = field(default_factory=list)


def _linfit_slope(xs: list[float], ys: list[float]) -> float | None:
    """Ordinary least-squares slope of ``ys`` on ``xs`` (None if degenerate)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:  # all x identical -> slope undefined
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def _binned_curve(
    points: list[tuple[float, float]], num_bins: int
) -> tuple[list[float], list[float], list[int]]:
    """Bin by confidence and return (mean_confidence, error_rate, count) per bin.

    The binning is done *once* over the full confidence range; windows are later
    taken as tails of this stable curve. Only non-empty bins are returned.
    """
    if not points:
        return [], [], []
    cmin = min(p[0] for p in points)
    cmax = max(p[0] for p in points)
    if cmax - cmin <= 1e-12:
        return [], [], []
    width = (cmax - cmin) / num_bins
    sums = [0.0] * num_bins
    counts = [0] * num_bins
    err = [0.0] * num_bins
    for conf, is_err in points:
        idx = min(int((conf - cmin) / width), num_bins - 1)
        sums[idx] += conf
        err[idx] += is_err
        counts[idx] += 1
    confs: list[float] = []
    rates: list[float] = []
    ns: list[int] = []
    for i in range(num_bins):
        if counts[i] > 0:
            confs.append(sums[i] / counts[i])
            rates.append(err[i] / counts[i])
            ns.append(counts[i])
    return confs, rates, ns


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def analyze_slope_trend(
    candidate: CandidateSlice,
    config: SlopeTrendConfig | None = None,
) -> SlopeTrendResult:
    """Run slope-trend analysis on a candidate slice."""
    cfg = config or SlopeTrendConfig()
    points = [
        (s.slice_confidence, 1.0 if s.instance.is_error else 0.0)
        for s in candidate.scored_instances
    ]

    # Estimate the error-rate-vs-confidence curve once over the full range, then
    # drop low-count bins whose error rate is too noisy to trust.
    raw_conf, raw_rate, raw_count = _binned_curve(points, cfg.num_bins)
    bin_conf, bin_rate, bin_count = [], [], []
    for c, r, n in zip(raw_conf, raw_rate, raw_count):
        if n >= cfg.min_bin_count:
            bin_conf.append(c)
            bin_rate.append(r)
            bin_count.append(n)

    windows: list[WindowFit] = []
    max_pos_slope = 0.0

    # Sliding thresholds at evenly spaced confidence quantiles, so each window
    # ("confidence exceeds a given value") keeps a meaningful amount of data.
    confs_sorted = sorted(p[0] for p in points)
    thresholds: list[float] = []
    if confs_sorted and cfg.num_thresholds >= 1:
        for i in range(cfg.num_thresholds):
            q = (cfg.max_threshold_quantile * i / (cfg.num_thresholds - 1)
                 if cfg.num_thresholds > 1 else 0.0)
            thresholds.append(_quantile(confs_sorted, q))

    for t in thresholds:
        # Window = tail of the binned curve whose bins sit above the threshold.
        sel = [
            (c, r, n) for c, r, n in zip(bin_conf, bin_rate, bin_count) if c >= t
        ]
        total = sum(n for _, _, n in sel)
        if total < cfg.min_window_size:
            continue
        if len(sel) < cfg.min_bins_per_window:
            continue
        xs = [c for c, _, _ in sel]
        ys = [r for _, r, _ in sel]
        if max(xs) - min(xs) < cfg.min_confidence_spread:
            continue
        slope = _linfit_slope(xs, ys)
        if slope is None:
            continue
        windows.append(
            WindowFit(
                threshold=t, size=total, slope=slope,
                bin_confidences=xs, bin_error_rates=ys,
            )
        )
        if slope > max_pos_slope:
            max_pos_slope = slope

    return SlopeTrendResult(
        max_positive_slope=max_pos_slope,
        is_systematic=max_pos_slope >= cfg.slope_threshold,
        windows=windows,
        curve_confidence=bin_conf,
        curve_error_rate=bin_rate,
    )


def identify_error_slices(
    candidates: list[CandidateSlice],
    config: SlopeTrendConfig | None = None,
) -> list[ErrorSlice]:
    """Classify each candidate slice and return all systematic error slices.

    Results are sorted by descending max-positive-slope (strongest trend first).
    """
    cfg = config or SlopeTrendConfig()
    error_slices: list[ErrorSlice] = []
    for cand in candidates:
        result = analyze_slope_trend(cand, cfg)
        error_slices.append(
            ErrorSlice(
                candidate=cand,
                max_positive_slope=result.max_positive_slope,
                is_systematic=result.is_systematic,
            )
        )
    error_slices.sort(key=lambda e: e.max_positive_slope, reverse=True)
    return [e for e in error_slices if e.is_systematic]
