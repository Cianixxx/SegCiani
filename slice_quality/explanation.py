"""BLOCK 8 -- Slice explanation (the diagnostic card).

Source : CB-SLICE (ECSA / keyword-concepts) -- adapted (no CBM available).
Input  : a kept Slice + the population (for over-representation) .
Output : an ExplanationCard: deterministic facts (faithful) + over-represented
         semantic attributes (interpretable, VLM-dependent).

The card is a TEMPLATE filled by aggregating member properties -- not free text.
  * deterministic : count, dominant confusion pair, dominant error type (masks).
  * semantic      : attributes that are MORE frequent in the slice than in the
                    population (discriminative, like CB-SLICE keyword-concepts).

Where to modify
---------------
* ``ATTRIBUTE_KEYS``      : which metadata tags (from the VLM) to consider.
* ``over_representation`` : the (slice% - pop%) selection rule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .data_types import Slice, TypedInstance
from .metrics.error_type_purity import error_type_purity

# VLM/metadata tag keys to scan for the semantic part of the explanation.
ATTRIBUTE_KEYS = ("attribute", "scene", "occlusion", "size", "lighting")


@dataclass
class ExplanationCard:
    slice_id: str
    size: int
    dominant_error_type: str
    top_confusion_pairs: list[tuple[str, int]] = field(default_factory=list)
    over_represented: list[tuple[str, float, float]] = field(default_factory=list)  # (tag, slice%, pop%)
    sentence: str = ""


def _tag_freq(members: Sequence[TypedInstance]) -> Counter:
    c: Counter = Counter()
    for m in members:
        for key in ATTRIBUTE_KEYS:
            v = m.metadata.get(key)
            if v is not None:
                c[f"{key}={v}"] += 1
    return c


def over_representation(
    slice_members: Sequence[TypedInstance],
    population: Sequence[TypedInstance],
    min_gap: float = 0.25,
) -> list[tuple[str, float, float]]:
    """Tags more frequent in the slice than in the population (discriminative)."""
    n_s, n_p = max(1, len(slice_members)), max(1, len(population))
    fs, fp = _tag_freq(slice_members), _tag_freq(population)
    out = []
    for tag, cnt in fs.items():
        s_pct = cnt / n_s
        p_pct = fp.get(tag, 0) / n_p
        if s_pct - p_pct >= min_gap:
            out.append((tag, s_pct, p_pct))
    out.sort(key=lambda t: t[1] - t[2], reverse=True)
    return out


def explain_slice(slice_: Slice, population: Sequence[TypedInstance]) -> ExplanationCard:
    _, dominant = error_type_purity(slice_.members)
    pairs = Counter(m.confusion_pair for m in slice_.members if m.confusion_pair)
    top_pairs = [(f"{a}->{b}", n) for (a, b), n in pairs.most_common(3)]
    over = over_representation(slice_.members, population)

    # template (deterministic facts + over-represented attributes)
    attr_phrase = ", ".join(t for t, _, _ in over[:3]) or "no distinctive attribute"
    conf_phrase = top_pairs[0][0] if top_pairs else dominant.value
    sentence = (f"{slice_.size} instances: {conf_phrase}, "
                f"{attr_phrase} - error type {dominant.value}")

    return ExplanationCard(
        slice_id=slice_.slice_id, size=slice_.size,
        dominant_error_type=dominant.value,
        top_confusion_pairs=top_pairs, over_represented=over, sentence=sentence,
    )
