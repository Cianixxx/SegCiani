"""Bridge: SliceLens grounding output -> slice_quality language-branch slices.

The SliceLens pipeline (../slicelens) produces ``CandidateSlice``s: for each
hypothesis ``q`` it ranks instances by slice confidence P(yes | q, r). To score
those with our Slice-Quality Vector we must re-express them as ``Slice`` objects
whose members are *our* ``TypedInstance``s (which carry the typed error + embedding).

We link a SliceLens scored instance to a TypedInstance by (image_id, bbox IoU).
This is exactly where the manifold-compactness axis becomes non-circular: a
language slice was NOT formed by embedding proximity, so its compactness is a
genuine test.

IMPORTANT: the two sides must share a coordinate system. Use the SAME adapter
conventions (e.g. COCO original-image coords on both sides, or the 520 crop on
both sides for VOC).

Where to modify
---------------
* ``image_id_of`` : how to derive our image_id from a SliceLens Instance.
* ``conf_threshold`` / ``top_k`` : which scored instances form the slice.
* ``bbox_iou_threshold`` : how strict the instance linking is.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Callable, Optional, Sequence

from .data_types import Slice, TypedInstance


def dedup_language_slices(
    slices: Sequence[Slice],
    *,
    jaccard_min: float = 0.6,
) -> list[Slice]:
    """Merge near-duplicate language slices (paraphrase hypotheses). Improvement #7.

    The LLM often emits paraphrases ("a dark cow" / "a cow in shadow") that the
    VLM grounds onto largely the SAME instances -> near-identical slices. They
    clutter the per-image descriptions with redundant rows and inflate the
    multiple-testing burden (every extra slice makes the BH-FDR threshold
    stricter for all).

    Grouping: connected components of the graph where two slices are linked when
    the Jaccard of their member-id sets >= ``jaccard_min``. Slices of different
    classes never share members -> never merge (open-world safe: purely
    member-based, no cluster-branch or embedding input).

    Representative per group: verified-systematic first (slope-trend flag --
    otherwise a vague paraphrase with inflated confidence can swallow the
    verified one), then highest mean slice confidence, then larger size. The
    other hypotheses are recorded in ``rep.metadata["aliases"]`` (shown by the
    gallery), members are NOT unioned -- the representative stays exactly the
    slice SliceLens grounded.
    """
    slices = list(slices)
    n = len(slices)
    if n <= 1:
        return slices

    ids = [frozenset(m.instance_id for m in s.members) for s in slices]

    # union-find over Jaccard-linked pairs (n is ~1e2: O(n^2) is fine)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            inter = len(ids[i] & ids[j])
            if inter == 0:
                continue
            union = len(ids[i] | ids[j])
            if inter / union >= jaccard_min:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    def _conf(s: Slice) -> float:
        return (sum(s.membership.values()) / len(s.membership)) if s.membership else 0.0

    def _sys(s: Slice) -> bool:
        return bool((getattr(s, "metadata", None) or {}).get("is_systematic"))

    out: list[Slice] = []
    for members_idx in groups.values():
        rep_i = max(members_idx,
                    key=lambda i: (_sys(slices[i]), _conf(slices[i]), slices[i].size))
        rep = slices[rep_i]
        aliases = [slices[i].hypothesis for i in members_idx
                   if i != rep_i and slices[i].hypothesis
                   and slices[i].hypothesis != rep.hypothesis]
        if aliases:
            existing = rep.metadata.get("aliases", [])
            rep.metadata["aliases"] = existing + [a for a in aliases
                                                  if a not in existing]
        out.append(rep)
    return out


def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _default_image_id(slicelens_instance) -> str:
    """COCO: '000000087038.jpg' -> '87038'; otherwise the file stem."""
    stem = os.path.splitext(os.path.basename(slicelens_instance.image_path))[0]
    try:
        return str(int(stem))           # strip COCO zero-padding
    except ValueError:
        return stem


def build_language_slices(
    candidate_slices: Sequence,                       # list[slicelens CandidateSlice]
    typed_instances: Sequence[TypedInstance],
    *,
    image_id_of: Callable = _default_image_id,
    conf_threshold: float = 0.5,
    top_k: Optional[int] = None,
    bbox_iou_threshold: float = 0.5,
) -> list[Slice]:
    """Map SliceLens candidate slices onto our TypedInstances.

    Returns a list[Slice] with ``source == "language"`` ready for ``run_pipeline``
    (pass as ``language_slices``).
    """
    by_img: dict[str, list[TypedInstance]] = defaultdict(list)
    for t in typed_instances:
        by_img[t.image_id].append(t)

    slices: list[Slice] = []
    for ci, cand in enumerate(candidate_slices):
        scored = list(cand.scored_instances)
        chosen = (scored[:top_k] if top_k
                  else [s for s in scored if s.slice_confidence >= conf_threshold])

        members: list[TypedInstance] = []
        membership: dict[str, float] = {}
        seen: set[str] = set()
        for si in chosen:
            box = getattr(si.instance.region, "bbox", None)
            if box is None:
                continue
            iid = image_id_of(si.instance)
            best, best_iou = None, bbox_iou_threshold
            for t in by_img.get(iid, []):
                if t.bbox is None:
                    continue
                iou = _bbox_iou(box, t.bbox)
                if iou >= best_iou:
                    best, best_iou = t, iou
            if best is not None and best.instance_id not in seen:
                members.append(best)
                membership[best.instance_id] = float(si.slice_confidence)
                seen.add(best.instance_id)

        if members:
            slices.append(Slice(
                slice_id=f"lang_{ci}",
                members=members,
                source="language",
                hypothesis=cand.hypothesis.query,
                membership=membership,
            ))
    return slices
