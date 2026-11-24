"""Tests for slope-trend analysis (Section 3.2.2)."""

import random

from slicelens.config import SlopeTrendConfig
from slicelens.data_types import (
    CandidateSlice,
    Hypothesis,
    Instance,
    Region,
    ScoredInstance,
)
from slicelens.identification import analyze_slope_trend


def _candidate(pairs):
    """pairs: list of (confidence, is_error)."""
    scored = [
        ScoredInstance(
            instance=Instance(
                instance_id=str(n),
                image_path="x",
                region=Region(point=(0, 0)),
                is_error=bool(err),
            ),
            slice_confidence=conf,
        )
        for n, (conf, err) in enumerate(pairs)
    ]
    scored.sort(key=lambda s: s.slice_confidence, reverse=True)
    return CandidateSlice(hypothesis=Hypothesis(query="q"), scored_instances=scored)


def test_monotonic_increasing_is_systematic():
    rng = random.Random(1)
    pairs = []
    for _ in range(400):
        conf = rng.random()
        # Error probability rises with confidence -> systematic.
        pairs.append((conf, 1 if rng.random() < 0.05 + 0.9 * conf else 0))
    result = analyze_slope_trend(_candidate(pairs), SlopeTrendConfig())
    assert result.is_systematic
    assert result.max_positive_slope > 0.2


def test_flat_error_rate_is_not_systematic():
    rng = random.Random(2)
    # Error independent of confidence -> no trend.
    pairs = [(rng.random(), 1 if rng.random() < 0.3 else 0) for _ in range(400)]
    result = analyze_slope_trend(_candidate(pairs), SlopeTrendConfig())
    assert not result.is_systematic


def test_decreasing_error_rate_is_not_systematic():
    rng = random.Random(3)
    pairs = []
    for _ in range(400):
        conf = rng.random()
        # Error probability *falls* with confidence.
        pairs.append((conf, 1 if rng.random() < 0.95 - 0.9 * conf else 0))
    result = analyze_slope_trend(_candidate(pairs), SlopeTrendConfig())
    assert not result.is_systematic


def test_constant_confidence_yields_no_windows():
    # All identical confidence -> no spread -> nothing to fit.
    pairs = [(0.5, n % 2) for n in range(400)]
    result = analyze_slope_trend(_candidate(pairs), SlopeTrendConfig())
    assert result.max_positive_slope == 0.0
    assert not result.is_systematic
