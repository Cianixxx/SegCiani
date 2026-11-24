"""Grounded candidate slice discovery (Section 3.2.1).

For a hypothesis ``q``, SliceLens scores *every* instance with the VLM slice
confidence P(yes | q, r) and ranks them. The top-ranked instances form the
candidate slice -- "the retrieved examples form a slice that illustrates various
scenarios" matching the hypothesis.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..backends.base import VLMBackend
from ..data_types import (
    CandidateSlice,
    Hypothesis,
    Instance,
    ScoredInstance,
)


def discover_candidate_slice(
    hypothesis: Hypothesis,
    instances: Sequence[Instance],
    vlm: VLMBackend,
) -> CandidateSlice:
    """Score and rank all ``instances`` for a single ``hypothesis``."""
    scored: list[ScoredInstance] = []
    for inst in instances:
        conf = vlm.slice_confidence(inst.image_path, inst.region, hypothesis.query)
        scored.append(ScoredInstance(instance=inst, slice_confidence=conf))
    scored.sort(key=lambda s: s.slice_confidence, reverse=True)
    return CandidateSlice(hypothesis=hypothesis, scored_instances=scored)


def discover_candidate_slices(
    hypotheses: Iterable[Hypothesis],
    instances: Sequence[Instance],
    vlm: VLMBackend,
) -> list[CandidateSlice]:
    """Run :func:`discover_candidate_slice` for every hypothesis."""
    return [discover_candidate_slice(h, instances, vlm) for h in hypotheses]
