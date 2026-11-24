"""Axis 5/6 -- slice purity (attribute homogeneity).  Source: SEG4SEG (Def. 1).

Fraction of the slice that shares the most common value of a *semantic attribute*
(occluded / small / dark / ...). High = the slice is one identifiable situation.
Paired with N_min (enforced in BLOCK 6) which excludes tiny clusters.

Two departures from a naive implementation, both forced by the fact that our
attribute channel is a NOISY VLM with a small vocabulary, not ground-truth
metadata (the retag experiment of 2026-07-18 exposed both):

1. NON-ATTRIBUTES are excluded. "normal" (Qwen's catch-all for "nothing
   notable"), "other" and "error" assert the ABSENCE of an attribute. Counting
   them cuts both ways: a slice that is 80% "normal" would look pure while
   saying nothing (a vacuous certification), and a slice where 70% of the
   INFORMATIVE tags agree gets diluted below threshold by the catch-all.
   The same skip-set is already used by describe.py for the descriptions.

2. The ABSOLUTE threshold (SEG4SEG's p_theta = 0.80) does not transfer to a
   noisy channel: even a perfectly coherent slice can sit below 0.8 purely from
   VLM tag noise. ``purity_permutation_p`` therefore tests purity AGAINST
   CHANCE: is the slice more attribute-coherent than random same-size groups of
   same-subtype errors? The null draws carry the same VLM noise and the same
   vocabulary limits, so the comparison is fair by construction. Note this test
   is genuinely non-circular for cluster slices: attributes are not part of the
   clustering features (DINOv2 + typed error), so coherence here is independent
   evidence.

Input  : list of attribute values (one per member; None allowed).
Output : max informative-class proportion in [0, 1]; permutation p-value.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

# Values that carry no usable evidence -- excluded from purity.
# Keep in sync with describe._SKIP_ATTR.
#   "normal"/"other"/"error" assert the ABSENCE of an attribute (see note 1 above).
#   "dark" is excluded for a different reason, and only after it was measured: the
#   description-precision evaluation (report S6.2) found phrases built on it are not
#   confirmed by the VLM on their own instances in EITHER elicitation mode -- a dark
#   sofa and a dark sheep both score 0.00, at or below the matched-chance control.
#   It was dropped from the user-facing text first; keeping it as evidence for
#   CERTIFYING while refusing to SAY it would be incoherent, so it is dropped here too.
#   The cost is real (237 instances carry the tag) and is paid deliberately.
NON_ATTRIBUTES = ("normal", "other", "error", "dark")


def _informative(attribute_values: Sequence[Optional[str]],
                 exclude: Sequence[str] = NON_ATTRIBUTES) -> list[str]:
    return [v for v in attribute_values if v is not None and v not in exclude]


def slice_purity(attribute_values: Sequence[Optional[str]],
                 exclude: Sequence[str] = NON_ATTRIBUTES) -> float:
    vals = _informative(attribute_values, exclude)
    if not vals:
        return 0.0
    counts = Counter(vals)
    _, top = counts.most_common(1)[0]
    return float(top / len(vals))


def purity_permutation_p(
    attribute_values: Sequence[Optional[str]],
    pool: Sequence[Optional[str]],
    n_boot: int = 200,
    rng: Optional[np.random.Generator] = None,
    exclude: Sequence[str] = NON_ATTRIBUTES,
) -> tuple[float, float, int]:
    """(p_value, observed_purity, n_informative) of the slice vs chance.

    ``pool``: attribute values of the peer errors (same subtype) to draw the
    null from. p = P(random same-size draw is at least as pure as the slice).
    Untestable cases (no informative tags, or a pool too small to draw
    meaningfully distinct subsets from) return p = 1.0 -- no free pass.
    """
    rng = rng or np.random.default_rng(0)
    vals = _informative(attribute_values, exclude)
    pool_vals = _informative(pool, exclude)
    n = len(vals)
    if n == 0 or len(pool_vals) < max(2 * n, n + 5):
        return 1.0, slice_purity(attribute_values, exclude), n

    obs = slice_purity(vals, exclude=())
    pool_arr = np.array(pool_vals, dtype=object)
    count = 0
    for _ in range(n_boot):
        draw = rng.choice(pool_arr, size=n, replace=False)
        c = Counter(draw)
        if c.most_common(1)[0][1] / n >= obs:
            count += 1
    p = float((1 + count) / (n_boot + 1))
    return p, obs, n
