"""BLOCK 7 -- Refine cascade for slices flagged REFINE.

Source : new (closed-loop) + MCSD (residual peeling lives in pipeline.py).
Input  : a Slice whose SliceQuality.verdict == "REFINE".
Output : a list of sub-Slices (to be re-scored by BLOCK 6).

Cost-ordered cascade (bottleneck B3 -> do the cheap splits first):
  Case A  mixed error TYPES        -> split by seg_subtype          (free)
  Case B  mixed CONFUSION pairs    -> split by confusion pair       (free)
  Case C  mixed semantics, same type:
            - re-cluster member embeddings                          (free)
            - only if language-branch + still mixed: re-query LLM   (expensive)

Where to modify
---------------
* ``llm_refine_hook`` : wire your SliceLens LLM here for Case C (it is a no-op
                        stub by default so the pipeline runs without a backend).
* ``max_depth`` / ``min_split_size`` : config.RefineConfig.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

import numpy as np

from .config import RefineConfig
from .data_types import SegErrorType, Slice, TypedInstance
from .metrics.error_type_purity import error_type_purity
from .metrics.misprediction_coherence import misprediction_coherence

# Optional hook: (parent_slice, reason) -> list[sub-Slice]. Wire SliceLens here.
LLMRefineHook = Callable[[Slice, str], list[Slice]]


def _group(members: list[TypedInstance], key) -> list[list[TypedInstance]]:
    buckets: dict[str, list[TypedInstance]] = defaultdict(list)
    for m in members:
        buckets[str(key(m))].append(m)
    return list(buckets.values())


def _make_subslices(parent: Slice, groups: list[list[TypedInstance]], tag: str) -> list[Slice]:
    out = []
    for k, g in enumerate(groups):
        if not g:
            continue
        out.append(Slice(
            slice_id=f"{parent.slice_id}.{tag}{k}", members=g,
            source=parent.source, hypothesis=parent.hypothesis,
            membership={m.instance_id: parent.weight(m) for m in g},
        ))
    return out


def refine_slice(
    parent: Slice,
    features_by_id: Optional[dict[str, np.ndarray]] = None,
    cfg: RefineConfig | None = None,
    llm_refine_hook: Optional[LLMRefineHook] = None,
    depth: int = 0,
) -> list[Slice]:
    """Split an incoherent slice using the cheapest sufficient strategy."""
    cfg = cfg or RefineConfig()
    if depth >= cfg.max_depth or parent.size < cfg.min_split_size:
        return [parent]

    # Case A: mixed error types -> split by seg_subtype
    etp, _ = error_type_purity(parent.members)
    if etp < 0.6 and len({m.seg_subtype for m in parent.members}) > 1:
        return _make_subslices(parent, _group(parent.members, lambda m: m.seg_subtype), "T")

    # Case B: mixed confusion pairs -> split by confusion pair
    mpc = misprediction_coherence(parent.members)
    pairs = {m.confusion_pair for m in parent.members if m.confusion_pair}
    if mpc < 0.6 and len(pairs) > 1:
        return _make_subslices(
            parent, _group(parent.members, lambda m: m.confusion_pair or m.seg_subtype), "C")

    # Case C: same type/pair but mixed -> re-cluster embeddings (free)...
    if features_by_id is not None:
        sub = _recluster(parent, features_by_id)
        if len(sub) > 1:
            return sub
    # ...or, last resort, re-query the LLM (Case C, language branch only)
    if cfg.use_llm_refine and parent.source == "language" and llm_refine_hook is not None:
        return llm_refine_hook(parent, "semantic-mixed") or [parent]

    return [parent]


def _recluster(parent: Slice, features_by_id: dict[str, np.ndarray], k: int = 2) -> list[Slice]:
    """Sub-cluster the members' BLOCK-3 features into k groups (KMeans)."""
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return [parent]
    feats, members = [], []
    for m in parent.members:
        f = features_by_id.get(m.instance_id)
        if f is not None:
            feats.append(f)
            members.append(m)
    if len(members) < 2 * k:
        return [parent]
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(np.stack(feats))
    groups = [[members[i] for i in range(len(members)) if labels[i] == c] for c in range(k)]
    return _make_subslices(parent, groups, "E")
