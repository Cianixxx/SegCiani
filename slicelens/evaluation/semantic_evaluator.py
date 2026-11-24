"""LLM-based semantic relevance evaluation (Section 5.3.1, prompt in A.3).

Used to evaluate *hypothesis generation* quality: for each discovered slice
description, an LLM judge decides which ground-truth error pattern (if any) it
semantically matches. From the matches we compute:

* **recall**    -- fraction of GT patterns that were matched by some discovery
* **precision** -- fraction of discoveries that matched a GT pattern

(The paper reports recall 0.90 / precision 0.74 for SliceLens hypotheses.)
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..backends.base import LLMBackend
from ..prompts import SEMANTIC_RELEVANCE_PROMPT


def match_to_ground_truth(
    algorithm_output: str,
    ground_truth_patterns: Sequence[str],
    llm: LLMBackend,
) -> int:
    """Return the index of the matching GT pattern, or -1 if none (A.3).

    ``algorithm_output`` is a discovered slice description; the GT patterns are
    presented to the judge as a 1-indexed numbered list. The returned value is a
    0-based index into ``ground_truth_patterns`` (or -1).
    """
    numbered = "\n".join(
        f"{i + 1}. {pat}" for i, pat in enumerate(ground_truth_patterns)
    )
    prompt = SEMANTIC_RELEVANCE_PROMPT.format(
        algorithm_output=algorithm_output, ground_truth_list=numbered
    )
    response = llm.chat(system="", user=prompt)
    m = re.search(r"-?\d+", response)
    if not m:
        return -1
    one_based = int(m.group())
    if one_based <= 0 or one_based > len(ground_truth_patterns):
        return -1
    return one_based - 1


def evaluate_hypotheses(
    discovered: Sequence[str],
    ground_truth_patterns: Sequence[str],
    llm: LLMBackend,
) -> dict[str, object]:
    """Compute recall / precision of discovered slices vs GT patterns (A.3)."""
    matched_gt: set[int] = set()
    matched_discoveries = 0
    assignments: list[tuple[str, int]] = []
    for desc in discovered:
        idx = match_to_ground_truth(desc, ground_truth_patterns, llm)
        assignments.append((desc, idx))
        if idx >= 0:
            matched_gt.add(idx)
            matched_discoveries += 1

    recall = len(matched_gt) / len(ground_truth_patterns) if ground_truth_patterns else 0.0
    precision = matched_discoveries / len(discovered) if discovered else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "matched_gt_indices": sorted(matched_gt),
        "assignments": assignments,
    }
