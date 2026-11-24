"""Axis 4/6 -- error-type purity.  Source: new (from Beyond IoU typed errors).

Same entropy form as misprediction_coherence, but over the *error TYPE* of each
member (boundary / extent / confusion / fp / fn / split / merge) instead of the
predicted class. High = one failure mechanism dominates -> the slice is
repairable with a SINGLE intervention, and the dominant type IS the repair key
(see recommendation.py).

Distinct from:
  * misprediction_coherence (#3) -- the wrong OUTPUT (class), not the mechanism.
  * slice_purity (#5)            -- a semantic ATTRIBUTE of the object.

Input  : slice members.
Output : (purity in [0,1], dominant SegErrorType).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from ..data_types import SegErrorType, TypedInstance


def error_type_purity(members: Sequence[TypedInstance]) -> tuple[float, SegErrorType]:
    if not members:
        return 0.0, SegErrorType.CORRECT
    counts = Counter(m.seg_subtype for m in members)
    dominant, _ = counts.most_common(1)[0]

    total = len(members)
    probs = [c / total for c in counts.values()]
    L = len(probs)
    if L <= 1:
        return 1.0, dominant
    H = -sum(p * math.log(p) for p in probs if p > 0)
    purity = 1.0 - H / math.log(L)
    return float(purity), dominant
