"""Axis 3/6 -- misprediction coherence.  Source: CB-SLICE (Eq. 8).

MC = 1 - H(p_j) / log L, where p_j is the distribution of the *wrong outputs*
inside the slice and L the number of distinct outcomes. High = the slice fails
the same way (one systematic mistake).

This axis is the antidote to circularity (B4): it uses ONLY output labels, never
embeddings, so it cannot agree with the clustering "by construction".

For instance seg we use the confusion pair (gt -> pred) when available, falling
back to the seg_subtype for FP/FN (which have no confusion pair).

Input  : list of categorical outcome labels (one per member) + optional weights.
Output : coherence in [0, 1].
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional, Sequence

from ..data_types import TypedInstance


def _outcome_label(inst: TypedInstance) -> str:
    pair = inst.confusion_pair
    if pair is not None:
        return f"{pair[0]}->{pair[1]}"
    if inst.pred_class is not None:
        return f"pred:{inst.pred_class}"
    return f"sub:{inst.seg_subtype.value}"   # FN / etc.


def misprediction_coherence(
    members: Sequence[TypedInstance], weights: Optional[Sequence[float]] = None
) -> float:
    """1 - normalised entropy of the outcome distribution inside the slice."""
    if not members:
        return 0.0
    weights = list(weights) if weights is not None else [1.0] * len(members)
    dist: dict[str, float] = defaultdict(float)
    for inst, w in zip(members, weights):
        dist[_outcome_label(inst)] += w
    total = sum(dist.values())
    if total <= 0:
        return 0.0
    probs = [v / total for v in dist.values()]
    L = len(probs)
    if L <= 1:
        return 1.0                       # everyone fails identically
    H = -sum(p * math.log(p) for p in probs if p > 0)
    return float(1.0 - H / math.log(L))
