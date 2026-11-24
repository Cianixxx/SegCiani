"""Short human-readable names for cluster slices, for the pages (not for the text the judges read).

A cluster slice is identified by an id like ``cluster_fn_3_r1`` (error type, mixture component,
peeling round). Its description already says what it is, from three facts the pipeline has
verified or measured: the Qwen tag its members carry, the class of its objects and the error type
that dominates it. The name is those three facts for the slice as a whole, short:

    occluded objects of class 'dog' not detected (missed)   ->  Occluded dogs · missed
    objects of class 'person' grouped by appearance only ... under-segmented
                                                             ->  People · merged

An attribute is named only when at least 40% of the members that carry a Qwen tag have it
(``normal`` counts as a tag: it is the VLM saying "nothing remarkable"), so a slice whose objects
share no verified attribute gets the class alone. A class is named when it holds 60% of the
members, two classes when they hold 70%, otherwise "objects (mostly A, B, C)". Slices that would
get the same name are numbered (``People · merged #1``, ``#2`` ...), the certified ones (KEEP)
first, then by size. Language slices keep their id: their card already shows the hypothesis that
names them.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .describe import _SKIP_ATTR

_PLURAL = {"aeroplane": "aeroplanes", "bicycle": "bicycles", "bird": "birds", "boat": "boats",
           "bottle": "bottles", "bus": "buses", "car": "cars", "cat": "cats", "chair": "chairs",
           "cow": "cows", "diningtable": "dining tables", "dog": "dogs", "horse": "horses",
           "motorbike": "motorbikes", "person": "people", "pottedplant": "potted plants",
           "sheep": "sheep", "sofa": "sofas", "train": "trains", "tvmonitor": "TV monitors"}
_ATTR = {"small": "small", "occluded": "occluded", "truncated": "cut-off", "cluttered": "cluttered",
         "blurry": "blurry"}
_TYPE = {"fn": "missed", "fp": "hallucinated", "merge": "merged", "split": "split",
         "boundary": "imprecise outlines", "extent": "wrong extent", "confusion": "wrong class"}
ATTR_MIN = 0.4
_pl = lambda c: _PLURAL.get(c, c + "s")


def _attr(members) -> str | None:
    tags = [m.metadata.get("attribute") for m in members if m.metadata.get("attribute") is not None]
    if not tags:
        return None
    for a, n in Counter(tags).most_common():
        if a not in _SKIP_ATTR:
            return _ATTR.get(a) if n / len(tags) >= ATTR_MIN else None
    return None


def _classes(members) -> str:
    vals = Counter((m.gt_class or m.pred_class) for m in members if (m.gt_class or m.pred_class))
    if not vals:
        return "objects"
    n = sum(vals.values())
    top = vals.most_common(3)
    if top[0][1] / n >= 0.6:
        return _pl(top[0][0])
    if len(top) > 1 and (top[0][1] + top[1][1]) / n >= 0.7:
        return f"{_pl(top[0][0])} and {_pl(top[1][0])}"
    return "objects (mostly " + ", ".join(_pl(c) for c, _n in top) + ")"


def cluster_name(members, dominant_type: str) -> str:
    attr = _attr(members)
    cls = _classes(members)
    base = f"{attr} {cls}" if attr else cls
    return f"{base[0].upper()}{base[1:]} · {_TYPE.get(dominant_type, dominant_type)}"


def cluster_names(pairs) -> dict[str, str]:
    """{slice_id: name} for the cluster slices among (slice, quality) pairs, duplicates numbered."""
    seen, raw = set(), []
    for s, q in pairs:
        if s.source != "cluster" or q.slice_id in seen:
            continue
        seen.add(q.slice_id)
        raw.append((q.slice_id, cluster_name(s.members, q.dominant_error_type.value), q.size,
                    q.verdict == "KEEP"))
    groups = defaultdict(list)
    for sid, name, size, keep in raw:
        groups[name].append((not keep, -size, sid))
    out = {}
    for name, items in groups.items():
        if len(items) == 1:
            out[items[0][2]] = name
        else:
            for k, (_k, _s, sid) in enumerate(sorted(items), 1):
                out[sid] = f"{name} #{k}"
    return out
