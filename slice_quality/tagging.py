"""VLM attribute tagging -> activates slice_purity (#5) + semantic explanation (#8).

Fills ``inst.metadata["attribute"]`` with ONE word from a small, failure-relevant
vocabulary, via the SliceLens Qwen backend's ``caption_region`` (reused, proven
path). A *controlled vocabulary* (not free captions) is what makes slice_purity
meaningful: the attribute must match across instances to form pure slices.

This is the interpretable, VLM-dependent half of the explanation (bottleneck B1):
if Qwen mis-tags, the attribute is wrong, but the deterministic error type stays
correct. Keep the vocabulary short and editable.

Where to modify
---------------
* ``DEFAULT_VOCAB`` : the attribute classes (edit per dataset / failure taxonomy).
* ``_ATTR``         : the instruction handed to caption_region.
"""

from __future__ import annotations

from typing import Optional

from slicelens.data_types import Region

DEFAULT_VOCAB = ["small", "occluded", "dark", "truncated", "cluttered", "blurry", "normal"]

_ATTR = ("the single most failure-relevant visual property of the object in the box; "
         "reply with EXACTLY ONE word from this list: " + ", ".join(DEFAULT_VOCAB))


def _closest(raw: str, vocab: list[str]) -> str:
    r = raw.lower()
    for v in vocab:
        if v in r:
            return v
    return "other"


# --- verification-mode tagging ---------------------------------------------
# Measured problem with the captioning path above (2026-07-25): asked to pick one
# word, Qwen answers "normal" for 56% of instances and its "dark"/"blurry" labels
# are not reproduced when the SAME model is asked to verify them (precision 0.00
# for "a dark person", "a blurry object"; §6.2). The corruption experiment found
# the same failure from the other side: deliberately shrunk chairs come back
# "normal" 12 times out of 14. Free-form choice over a closed list is simply not
# the mode this model is reliable in -- grounded yes/no verification is (0.784).
# So here each attribute is asked as its own yes/no question, exactly like the
# language branch's grounding, and the winner is the argmax over P(yes | a, r).
# Every score is kept in metadata["attr_scores"] so thresholds can be re-tuned
# offline without paying for the GPU pass again.
VERIFY_QUERIES = {
    "small": "a small object, occupying only a small part of the image",
    "occluded": "an object partly hidden behind another object",
    "dark": "an object in shadow, or in poor lighting",
    "truncated": "an object cut off by the image border",
    "cluttered": "an object surrounded by many other objects",
    "blurry": "an object that is blurry or out of focus",
}
VERIFY_MIN = 0.5          # below this, no attribute is claimed -> "normal"


def tag_instances_verify(
    instances,
    image_paths: dict[str, str],
    vlm,
    queries: dict = VERIFY_QUERIES,
    min_score: float = VERIFY_MIN,
    verbose: bool = True,
):
    """Tag by grounded verification: argmax_a P(yes | a, region), else "normal".

    Costs len(queries) forward passes per instance (no generation), against one
    generation of <=10 tokens for the captioning path.
    """
    n = 0
    for inst in instances:
        if inst.bbox is None:
            continue
        path = image_paths.get(inst.image_id)
        if not path:
            continue
        region = Region(bbox=inst.bbox)
        scores = {}
        try:
            for attr, q in queries.items():
                scores[attr] = round(float(vlm.slice_confidence(path, region, q)), 4)
        except Exception:
            inst.metadata["attribute"] = "error"
            continue
        best, p = max(scores.items(), key=lambda kv: kv[1])
        inst.metadata["attr_scores"] = scores
        inst.metadata["attribute"] = best if p >= min_score else "normal"
        inst.metadata["attribute_mode"] = "verify"
        n += 1
        if verbose and n % 25 == 0:
            print(f"  verified {n} instances ({n * len(queries)} passes) ...",
                  flush=True)
    return instances


def tag_instances(
    instances,
    image_paths: dict[str, str],
    vlm,
    vocab: list[str] = DEFAULT_VOCAB,
    attribute: str = _ATTR,
    limit: Optional[int] = None,
    verbose: bool = True,
):
    """Tag instances in place; returns them. ``vlm`` is a SliceLens VLMBackend."""
    n = 0
    for inst in instances:
        if limit is not None and n >= limit:
            break
        if inst.bbox is None:
            continue
        path = image_paths.get(inst.image_id)
        if not path:
            continue
        try:
            raw = vlm.caption_region(path, Region(bbox=inst.bbox), attribute)
            inst.metadata["attribute"] = _closest(raw, vocab)
        except Exception:
            inst.metadata["attribute"] = "error"
        n += 1
        if verbose and n % 25 == 0:
            print(f"  tagged {n} instances ...", flush=True)
    return instances
