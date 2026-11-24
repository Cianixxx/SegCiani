"""Evaluation metrics.

Precision@k -- Equation (2) of the paper::

    Precision@k = |S_gt  ∩  S_pred^(k)| / k

where ``S_pred^(k)`` is the set of top-k instances of the candidate slice ranked
by slice confidence. Following the paper we use ``k = 10``, or the actual number
of retrieved samples if fewer than 10 are available.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..data_types import CandidateSlice, Instance


def precision_at_k(
    candidate: CandidateSlice,
    gt_instance_ids: set[str],
    k: int = 10,
) -> float:
    """Precision@k of a candidate slice against a ground-truth instance set."""
    top = candidate.top_k(k)
    if not top:
        return 0.0
    eff_k = min(k, len(top))
    hits = sum(1 for s in top if s.instance.instance_id in gt_instance_ids)
    return hits / eff_k


def precision_at_k_by_predicate(
    candidate: CandidateSlice,
    is_in_gt: Callable[[Instance], bool],
    k: int = 10,
) -> float:
    """Precision@k where GT membership is given by a predicate on the instance.

    Convenient when the ground-truth slice is defined by metadata (e.g. an
    instance carries the tag of the true failure mode) rather than an id set.
    """
    top = candidate.top_k(k)
    if not top:
        return 0.0
    eff_k = min(k, len(top))
    hits = sum(1 for s in top if is_in_gt(s.instance))
    return hits / eff_k


def best_precision_at_k(
    candidates: Sequence[CandidateSlice],
    gt_instance_ids: set[str],
    k: int = 10,
) -> tuple[float, CandidateSlice | None]:
    """Best Precision@k over candidate slices for one GT slice (cf. Table 7).

    Returns the score and the candidate that achieved it (its hypothesis is the
    "Best Predicted Slice").
    """
    best_score = 0.0
    best_cand: CandidateSlice | None = None
    for cand in candidates:
        score = precision_at_k(cand, gt_instance_ids, k)
        if best_cand is None or score > best_score:
            best_score, best_cand = score, cand
    return best_score, best_cand


def average_precision_at_k(
    candidates: Sequence[CandidateSlice],
    gt_slices: dict[str, set[str]],
    k: int = 10,
) -> dict[str, object]:
    """Aggregate report mirroring Table 7/8 (average P@k, perfect/valid matches).

    ``gt_slices`` maps each ground-truth slice name to its set of instance ids.
    For each GT slice we take the best-matching candidate (best Precision@k).
    """
    per_slice: dict[str, dict[str, object]] = {}
    scores: list[float] = []
    for name, gt_ids in gt_slices.items():
        score, cand = best_precision_at_k(candidates, gt_ids, k)
        per_slice[name] = {
            "precision_at_k": score,
            "best_predicted_slice": cand.hypothesis.query if cand else None,
            "gt_size": len(gt_ids),
        }
        scores.append(score)
    n = len(scores)
    return {
        "per_slice": per_slice,
        "average_precision_at_k": (sum(scores) / n) if n else 0.0,
        "perfect_matches": sum(1 for s in scores if s >= 1.0),
        "valid_matches": sum(1 for s in scores if s > 0.0),
        "num_gt_slices": n,
        "k": k,
    }
