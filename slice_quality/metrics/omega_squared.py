"""Axis 1/6 -- omega^2 (performance disparity).  Source: SEG4SEG.

Effect size: the proportion of variance in per-instance performance explained by
the partition. Robust to small/lucky clusters (unlike best-worst difference).

Used here in the *one-vs-rest* form: how much does "in this slice vs the rest"
explain performance? Returns a scalar in [0, 1] (negative estimates clipped to 0).

Input  : slice performance values (y_i) + population performance values.
Output : omega^2 in [0, 1].
"""

from __future__ import annotations

import numpy as np


def omega_squared(slice_perf: np.ndarray, rest_perf: np.ndarray) -> float:
    """One-way ANOVA omega^2 for the 2-group partition {slice, rest}."""
    slice_perf = np.asarray(slice_perf, dtype=float)
    rest_perf = np.asarray(rest_perf, dtype=float)
    groups = [slice_perf, rest_perf]
    N = sum(len(g) for g in groups)
    K = len(groups)
    if N <= K or any(len(g) == 0 for g in groups):
        return 0.0

    grand = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_total = ss_between + ss_within
    ms_within = ss_within / (N - K)

    denom = ss_total + ms_within
    if denom <= 1e-12:
        return 0.0
    omega2 = (ss_between - (K - 1) * ms_within) / denom
    return float(max(0.0, omega2))
