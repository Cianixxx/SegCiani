"""Tests for robust JSON extraction from LLM responses."""

import pytest

from slicelens.utils import extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    text = "Here you go:\n```json\n{\"a\": [1, 2]}\n```\nThanks!"
    assert extract_json(text) == {"a": [1, 2]}


def test_fenced_without_lang():
    assert extract_json("```\n{\"x\": true}\n```") == {"x": True}


def test_json_with_surrounding_prose():
    text = 'Sure. {"results": ["a", "b"]} Hope that helps.'
    assert extract_json(text) == {"results": ["a", "b"]}


def test_array_json():
    assert extract_json("prefix [1, 2, 3] suffix") == [1, 2, 3]


def test_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here at all")
