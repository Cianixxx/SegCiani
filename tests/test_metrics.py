"""Tests for Precision@k (Equation 2)."""

from slicelens.data_types import (
    CandidateSlice,
    Hypothesis,
    Instance,
    Region,
    ScoredInstance,
)
from slicelens.evaluation import average_precision_at_k, precision_at_k


def _slice(ranked_ids):
    """Build a candidate slice whose instances are already in ranked order."""
    scored = [
        ScoredInstance(
            instance=Instance(
                instance_id=i, image_path="x", region=Region(point=(0, 0))
            ),
            slice_confidence=1.0 - n / 100.0,
        )
        for n, i in enumerate(ranked_ids)
    ]
    return CandidateSlice(hypothesis=Hypothesis(query="q"), scored_instances=scored)


def test_precision_perfect():
    cand = _slice([f"i{n}" for n in range(10)])
    gt = {f"i{n}" for n in range(10)}
    assert precision_at_k(cand, gt, k=10) == 1.0


def test_precision_half():
    cand = _slice([f"i{n}" for n in range(10)])
    gt = {f"i{n}" for n in range(5)}  # only first 5 are correct
    assert precision_at_k(cand, gt, k=10) == 0.5


def test_precision_uses_top_k_only():
    # First 10 are wrong, last 10 are right -> P@10 = 0.
    cand = _slice([f"w{n}" for n in range(10)] + [f"r{n}" for n in range(10)])
    gt = {f"r{n}" for n in range(10)}
    assert precision_at_k(cand, gt, k=10) == 0.0


def test_precision_fewer_than_k_uses_actual_size():
    cand = _slice(["a", "b", "c"])
    gt = {"a", "b", "c"}
    # Only 3 retrieved -> effective k = 3, all correct -> 1.0
    assert precision_at_k(cand, gt, k=10) == 1.0


def test_average_report():
    cand = _slice([f"i{n}" for n in range(10)])
    report = average_precision_at_k([cand], {"slice_a": {f"i{n}" for n in range(10)}}, k=10)
    assert report["average_precision_at_k"] == 1.0
    assert report["perfect_matches"] == 1
    assert report["valid_matches"] == 1
    assert report["per_slice"]["slice_a"]["best_predicted_slice"] == "q"
