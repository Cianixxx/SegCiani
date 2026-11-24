"""Cross-branch corroboration: name certified cluster slices with verified hypotheses.

When a validated LANGUAGE slice (SliceLens, slope-trend-verified) and a kept
CLUSTER slice share members, the visual pattern and the semantic situation
agree -- the strongest statement the pipeline can make: the two branches
discovered the same failure mode independently. This is post-hoc corroboration
between two already-certified objects; hypothesis generation is NOT conditioned
on clusters, so the open-world independence of the branches is untouched.

Metric: containment C = |A ∩ B_lang| / |B_lang| -- "how much of the situation
lies inside the visual cluster". Jaccard alone under-reports the (typical)
small-situation-inside-big-cluster case. Both are recorded.

The result is written into the CLUSTER slice's ``metadata["corroborated_by"]``
(list of {slice_id, hypothesis, containment, jaccard, n_overlap}, best first),
which the gallery and the slice index render. Call AFTER the describe gate so
``validated_lang_ids`` reflects the slices that actually feed descriptions.
"""

from __future__ import annotations


def cross_branch_corroboration(
    result,
    validated_lang_ids: set,
    *,
    containment_min: float = 0.5,
    n_overlap_min: int = 3,
) -> dict:
    """Annotate kept cluster slices with the verified hypotheses they contain.

    Returns {cluster_slice_id -> [corroboration records]} (also stored in each
    cluster slice's metadata).
    """
    lang = []
    seen = set()
    for s, q in result.scored:
        if (s.source == "language" and q.slice_id in validated_lang_ids
                and q.slice_id not in seen):
            seen.add(q.slice_id)
            lang.append((s, frozenset(m.instance_id for m in s.members)))

    out: dict = {}
    for ks in result.kept:
        s = ks.slice_
        if s.source != "cluster":
            continue
        members = frozenset(m.instance_id for m in s.members)
        recs = []
        for ls, lids in lang:
            inter = len(members & lids)
            if inter < n_overlap_min or not lids:
                continue
            containment = inter / len(lids)
            if containment < containment_min:
                continue
            recs.append({
                "slice_id": ls.slice_id,
                "hypothesis": ls.hypothesis,
                "containment": round(containment, 3),
                "jaccard": round(inter / len(members | lids), 3),
                "n_overlap": inter,
            })
        if recs:
            recs.sort(key=lambda r: (-r["containment"], -r["n_overlap"]))
            s.metadata["corroborated_by"] = recs
            out[s.slice_id] = recs
    return out
