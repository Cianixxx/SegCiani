"""Offline mock backends.

These let the *entire* pipeline run end-to-end with no GPU, no API key and no
heavy dependencies, so the algorithms (slope-trend, Precision@k, ranking) can be
exercised and tested deterministically.

The mock VLM needs a ground-truth signal to fake a believable confidence. We use
the convention that synthetic "images" are URIs that carry their ground-truth
visual tags, e.g.::

    synthetic://inst_0?tags=bicycle|person|occlusion

Real backends ignore this convention and read pixels instead. Crucially, the
mock derives pseudo-logits ``z_yes`` / ``z_no`` and runs them through the *same*
softmax helper as the real Qwen backend, so the Eq. (1) code path is identical.
"""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import parse_qs, urlparse

from ..data_types import Region
from .base import LLMBackend, VLMBackend

_STOPWORDS = {
    "a", "an", "the", "of", "with", "by", "in", "on", "and", "or", "to", "is",
    "are", "that", "this", "photo", "image", "picture", "showing", "being",
    "for", "from", "as", "at", "it", "its", "no",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _tags_from_uri(image_path: str) -> set[str]:
    """Recover ground-truth tags from a ``synthetic://...?tags=a|b|c`` URI."""
    parsed = urlparse(image_path)
    qs = parse_qs(parsed.query)
    raw = qs.get("tags", [""])[0]
    return {t.strip().lower() for t in raw.split("|") if t.strip()}


def _hash_unit(*parts: str) -> float:
    """Deterministic pseudo-random float in [0, 1) from string parts."""
    h = hashlib.sha256("\x00".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class MockLLM(LLMBackend):
    """A deterministic stand-in for an LLM (e.g. Gemini).

    It inspects the prompt to decide which stage is calling it and returns a
    plausible, well-formed response. The knowledge-driven hypotheses include a
    query that matches the planted failure mode so the downstream demo finds a
    real systematic slice.
    """

    def chat(self, system: str, user: str) -> str:
        blob = (system + "\n" + user).lower()

        if "computer vision failure analysis" in blob:
            return self._knowledge_hypotheses(user)
        if "determine the most relevant" in blob:
            return self._extract_attribute(user)
        if "text clustering assistant" in blob:
            return self._cluster_keywords(user)
        if "text to image retrieval prompt" in blob:
            return self._refine_queries(user)
        if "ground truth error patterns" in blob:
            return self._semantic_match(user)
        return "none"

    # -- stage-specific canned responses ----------------------------------
    def _label_from_user(self, user: str) -> str:
        m = re.search(r"detect(?:\s+and\s+segment)?\s+([a-zA-Z_ ]+?)[.\n]", user)
        return m.group(1).strip() if m else "object"

    def _knowledge_hypotheses(self, user: str) -> str:
        label = self._label_from_user(user)
        data = {
            "title": f"Possible failure reasons for {label} detection model",
            "hypothesis": {
                "Contextual Interference": [
                    {
                        "title": f"{label} occluded by a person",
                        "description": (
                            f"A {label} partially hidden by a person is hard to "
                            "detect because key parts are missing."
                        ),
                        "prompts": [
                            {"prompt": f"{label} obscured by a person", "type": "search"},
                            {"prompt": f"{label} occlusion", "type": "cluster"},
                        ],
                    }
                ],
                "Object Attributes": [
                    {
                        "title": f"Very small {label}",
                        "description": f"A {label} occupying few pixels is easily missed.",
                        "prompts": [
                            {"prompt": f"a very small {label}", "type": "search"},
                            {"prompt": f"{label} sizes", "type": "cluster"},
                        ],
                    }
                ],
                "Background Descriptions": [
                    {
                        "title": "Similar background objects",
                        "description": (
                            f"Background objects similar to {label} cause false positives."
                        ),
                        "prompts": [
                            {"prompt": f"objects similar to {label}", "type": "search"},
                        ],
                    }
                ],
            },
        }
        return json.dumps(data, indent=2)

    def _extract_attribute(self, user: str) -> str:
        # Pull the caption out of the prompt and echo its most salient keyword.
        m = re.search(r"caption:\s*(.+)$", user, flags=re.IGNORECASE | re.DOTALL)
        caption = m.group(1) if m else user
        kws = sorted(_keywords(caption))
        return kws[-1] if kws else "none"

    def _cluster_keywords(self, user: str) -> str:
        m = re.search(r"keyword list:\s*(.+)$", user, flags=re.IGNORECASE | re.DOTALL)
        raw = m.group(1) if m else user
        kws = sorted(set(re.findall(r"[a-zA-Z_]+", raw)) - _STOPWORDS)
        return json.dumps({"attributes": kws[:8]})

    def _refine_queries(self, user: str) -> str:
        items = re.findall(r"[a-zA-Z_]{3,}", user.split("Input:")[-1])
        items = [w for w in items if w.lower() not in _STOPWORDS][:6]
        return json.dumps({"results": [f"a photo of {w}" for w in items]})

    def _semantic_match(self, user: str) -> str:
        # Match the algorithm output to the most keyword-overlapping GT line.
        algo = ""
        m = re.search(r"algorithm output:\s*(.+?)\n", user, flags=re.IGNORECASE)
        if m:
            algo = m.group(1)
        algo_kw = _keywords(algo)
        best_idx, best_overlap = -1, 0.0
        for line in user.splitlines():
            lm = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
            if not lm:
                continue
            idx, text = int(lm.group(1)), lm.group(2)
            gt_kw = _keywords(text)
            if not gt_kw:
                continue
            overlap = len(algo_kw & gt_kw) / len(gt_kw)
            if overlap > best_overlap:
                best_idx, best_overlap = idx, overlap
        return str(best_idx if best_overlap >= 0.5 else -1)


class MockVLM(VLMBackend):
    """A deterministic stand-in for a VLM (e.g. Qwen2.5-VL).

    ``slice_confidence`` builds pseudo-logits from the keyword overlap between the
    query and the instance's ground-truth tags, then runs the real Eq. (1)
    softmax. This produces a confidence that genuinely correlates with the
    planted failure mode, so the slope-trend analysis has something to find.
    """

    def __init__(self, logit_scale: float = 3.0, noise: float = 0.4) -> None:
        self.logit_scale = logit_scale
        self.noise = noise

    def caption_region(self, image_path: str, region: Region, attribute: str) -> str:
        tags = sorted(_tags_from_uri(image_path))
        box = region.bbox if region.bbox is not None else region.point
        return (
            f"A detailed view of the region {box} relevant to {attribute}: "
            f"the region shows {', '.join(tags) if tags else 'an object'}."
        )

    def slice_confidence(self, image_path: str, region: Region, query: str) -> float:
        from ..verification.slice_confidence import softmax_yes_no

        tags = _tags_from_uri(image_path)
        q_kw = _keywords(query)
        if not q_kw:
            overlap = 0.0
        else:
            # A query keyword counts as matched if it appears among the tags.
            matched = sum(1 for w in q_kw if w in tags)
            overlap = matched / len(q_kw)

        centered = 2.0 * overlap - 1.0  # in [-1, 1]
        noise = (_hash_unit(image_path, query) - 0.5) * 2.0 * self.noise
        z_yes = self.logit_scale * centered + noise
        z_no = -self.logit_scale * centered - noise
        return softmax_yes_no(z_yes, z_no)
