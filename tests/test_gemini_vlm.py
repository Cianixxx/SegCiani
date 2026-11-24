"""Tests for the API-VLM logprob parsing + Eq. (1) math (no SDK needed).

The Gemini response is mimicked with plain dicts (``extract_yes_no_logprobs``
and ``confidence_from_logprobs`` are pure functions). This validates the part
unique to the API backend; the actual API wiring needs a key and is exercised
separately on the server.
"""

import math

import pytest

from slicelens.backends.gemini_vlm import (
    _parse_yes_no,
    confidence_from_logprobs,
    extract_yes_no_logprobs,
)


def _resp(step_tokens):
    """Build a fake Gemini response. step_tokens: list of [(token, logprob), ...]."""
    return {
        "candidates": [
            {
                "logprobs_result": {
                    "top_candidates": [
                        {"candidates": [{"token": t, "log_probability": lp}
                                        for t, lp in step]}
                        for step in step_tokens
                    ]
                }
            }
        ]
    }


def test_extract_both_tokens():
    resp = _resp([[("yes", -0.05), ("no", -3.0)]])
    lp_yes, lp_no = extract_yes_no_logprobs(resp)
    assert lp_yes == -0.05 and lp_no == -3.0


def test_extract_handles_subword_and_case():
    resp = _resp([[("▁Yes", -0.1), (" no", -2.0)]])
    lp_yes, lp_no = extract_yes_no_logprobs(resp)
    assert lp_yes == -0.1 and lp_no == -2.0


def test_extract_skips_leading_step_without_yes_no():
    # First emitted token is irrelevant; yes/no appear at the next step.
    resp = _resp([[("```", -0.2)], [("yes", -0.3), ("no", -1.5)]])
    lp_yes, lp_no = extract_yes_no_logprobs(resp)
    assert lp_yes == -0.3 and lp_no == -1.5


def test_extract_one_sided():
    resp = _resp([[("yes", -0.01), ("maybe", -5.0)]])
    lp_yes, lp_no = extract_yes_no_logprobs(resp)
    assert lp_yes == -0.01 and lp_no is None


def test_extract_no_logprobs():
    assert extract_yes_no_logprobs({"candidates": [{}]}) == (None, None)
    assert extract_yes_no_logprobs({}) == (None, None)


def test_confidence_matches_softmax():
    # High yes logprob, low no -> high confidence.
    c = confidence_from_logprobs(-0.05, -3.0)
    expected = math.exp(-0.05) / (math.exp(-0.05) + math.exp(-3.0))
    assert abs(c - expected) < 1e-9
    assert c > 0.9


def test_confidence_one_sided_yes_is_high():
    assert confidence_from_logprobs(-0.01, None) > 0.99


def test_confidence_one_sided_no_is_low():
    assert confidence_from_logprobs(None, -0.01) < 0.01


def test_confidence_requires_at_least_one():
    with pytest.raises(ValueError):
        confidence_from_logprobs(None, None)


def test_parse_yes_no_lenient():
    assert _parse_yes_no("Yes") == "yes"
    assert _parse_yes_no("No.") == "no"
    assert _parse_yes_no("Yes, the bicycle is occluded by a person.") == "yes"
    assert _parse_yes_no("The answer is no.") == "no"
    assert _parse_yes_no("maybe") is None
    assert _parse_yes_no("") is None
    # must not false-match inside other words
    assert _parse_yes_no("nobody is here") is None
    assert _parse_yes_no("yesterday it rained") is None


def test_end_to_end_parse_then_confidence():
    resp = _resp([[("yes", -2.0), ("no", -0.1)]])  # model leans "no"
    lp_yes, lp_no = extract_yes_no_logprobs(resp)
    conf = confidence_from_logprobs(lp_yes, lp_no)
    assert conf < 0.2  # low slice confidence
