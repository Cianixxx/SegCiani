"""Tests for the slice confidence softmax (Equation 1)."""

import math

from slicelens.verification.slice_confidence import softmax_yes_no


def test_equal_logits_give_half():
    assert abs(softmax_yes_no(1.0, 1.0) - 0.5) < 1e-9


def test_monotonic_in_yes_logit():
    assert softmax_yes_no(2.0, 0.0) > softmax_yes_no(1.0, 0.0) > softmax_yes_no(0.0, 0.0)


def test_matches_closed_form():
    z_yes, z_no = 1.3, -0.4
    expected = math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))
    assert abs(softmax_yes_no(z_yes, z_no) - expected) < 1e-9


def test_numerically_stable_for_large_logits():
    # Should not overflow.
    v = softmax_yes_no(1000.0, -1000.0)
    assert 0.999 < v <= 1.0


def test_bounded():
    for zy in (-5, 0, 5):
        for zn in (-5, 0, 5):
            assert 0.0 <= softmax_yes_no(zy, zn) <= 1.0
