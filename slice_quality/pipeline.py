"""BLOCK 0 -- End-to-end orchestrator (wiring).

Ties every block together and runs the discovery <-> validation closed loop with
residual peeling (MCSD). The SliceLens *language* branch is optional input
(``language_slices``): pass it when you have the LLM/VLM backends, or run
cluster-only without any model on the API side.

Pipeline
--------
  [1] typed_error      -> errors + reference
  [2] embeddings       -> fill .embedding   (needs the backbone; server-side)
  [3] features         -> feature matrix
  [4] clustering       -> cluster-branch slices                ┐
      language_slices  (from SliceLens, optional)              ┴-> candidate slices
  [6] slice_quality_vector -> score + verdict (KEEP/REFINE/DISCARD)
  [7] refine (REFINE)  -> re-score
  [8]/[9] explanation + recommendation (KEEP)
  peeling: remove KEPT instances, repeat on the residual until none pass.

Where to modify
---------------
* ``extractor`` / ``mask_lookup`` : your backbone + how to fetch a mask (BLOCK 2).
* ``language_slices``             : plug SliceLens grounding output here.
* peeling stop criterion          : ``while`` condition below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .clustering import cluster_slices
from .config import SliceQualityConfig
from .data_types import Slice, SliceQuality, TypedInstance
from .embeddings import FeatureExtractor, attach_embeddings
from .explanation import ExplanationCard, explain_slice
from .features import build_features
from .metrics.manifold_compactness import build_knn_graph
from .recommendation import RepairAdvice, recommend
from .refine import refine_slice
from .slice_quality_vector import score_slices


@dataclass
class KeptSlice:
    slice_: Slice
    quality: SliceQuality
    explanation: ExplanationCard
    advice: RepairAdvice


@dataclass
class PipelineResult:
    kept: list[KeptSlice] = field(default_factory=list)
    all_qualities: list[SliceQuality] = field(default_factory=list)
    # every scored candidate (Slice + its quality), both branches, all verdicts --
    # used by gallery.py to visualise intermediate results.
    scored: list[tuple[Slice, SliceQuality]] = field(default_factory=list)
    residual_error: float = 1.0          # fraction of error mass not yet explained


def _build_graph(instances: list[TypedInstance], cfg: SliceQualityConfig):
    """kNN graph + id->row map over the instances that have an embedding."""
    feats, ids = [], []
    for i in instances:
        if i.embedding is not None:
            feats.append(i.embedding)
            ids.append(i.instance_id)
    if not feats:
        return [], {}
    knn = build_knn_graph(np.stack(feats), cfg.manifold)
    id_to_index = {iid: k for k, iid in enumerate(ids)}
    return knn, id_to_index


def run_pipeline(
    samples: list[dict],
    extractor: FeatureExtractor,
    mask_lookup: Callable[[TypedInstance], Optional[np.ndarray]],
    language_slices: Optional[list[Slice]] = None,
    cfg: SliceQualityConfig | None = None,
    max_peeling_rounds: int = 5,
    residual_stop: float = 0.05,
) -> PipelineResult:
    """Full pipeline from raw masks: BLOCK 1 + 2, then ``run_from_instances``."""
    cfg = cfg or SliceQualityConfig()
    from .typed_error import decompose_dataset

    # [1] typed error (deterministic, once) --------------------------------
    errors, reference = decompose_dataset(samples, cfg.typed_error)
    # [2] embeddings (errors + reference live in the same space) -----------
    attach_embeddings(errors + reference, extractor, mask_lookup)

    return run_from_instances(errors, reference, language_slices, cfg,
                              max_peeling_rounds, residual_stop)


def run_from_instances(
    errors: list[TypedInstance],
    reference: list[TypedInstance],
    language_slices: Optional[list[Slice]] = None,
    cfg: SliceQualityConfig | None = None,
    max_peeling_rounds: int = 5,
    residual_stop: float = 0.05,
) -> PipelineResult:
    """Run the closed loop from instances that ALREADY have ``.embedding`` set.

    Use this to run from cached / past-run instances (e.g. a SliceLens run JSON)
    without re-doing BLOCK 1 (masks) or BLOCK 2 (backbone). ``run_pipeline`` is a
    thin wrapper that fills these two blocks first.
    """
    cfg = cfg or SliceQualityConfig()

    dataset_size = len(errors) + len(reference)
    total_error_mass = sum(1.0 - e.iou for e in errors) or 1.0

    result = PipelineResult()
    remaining = list(errors)
    language = list(language_slices or [])

    for _round in range(max_peeling_rounds):
        if not remaining:
            break
        # [3][4] cluster branch on the residual ----------------------------
        feats, kept_inst = build_features(remaining, cfg.feature)
        features_by_id = {kept_inst[k].instance_id: feats[k] for k in range(len(kept_inst))}
        cluster_branch = cluster_slices(feats, kept_inst, cfg.cluster)
        # Make ids unique across peeling rounds: cluster_slices restarts at
        # cluster_0 each round, so without this the same id names several
        # different slices (corrupts by_id / qmap / the report / the gallery).
        for _s in cluster_branch:
            _s.slice_id = f"{_s.slice_id}_r{_round}"

        candidates = cluster_branch + language
        if not candidates:
            break

        # graph over the residual instances --------------------------------
        knn, id_to_index = _build_graph(remaining, cfg)

        # [6] score (peers = ALL errors, so the axes keep the same meaning
        # across peeling rounds) --------------------------------------------
        qualities = score_slices(candidates, reference, id_to_index, knn,
                                 dataset_size, cfg, error_population=errors)
        result.all_qualities.extend(qualities)
        by_id = {s.slice_id: s for s in candidates}
        result.scored.extend((by_id[q.slice_id], q)
                             for q in qualities if q.slice_id in by_id)

        kept_now: list[Slice] = []
        for q in qualities:
            s = by_id.get(q.slice_id)
            if s is None:
                continue
            if q.verdict == "KEEP":
                kept_now.append(s)
            elif q.verdict == "REFINE":
                # [7] refine and re-score the sub-slices once ---------------
                subs = refine_slice(s, features_by_id, cfg.refine)
                if len(subs) > 1:
                    subq = score_slices(subs, reference, id_to_index, knn,
                                        dataset_size, cfg, error_population=errors)
                    sub_by_id = {x.slice_id: x for x in subs}
                    for sq in subq:
                        if sq.verdict == "KEEP" and sq.slice_id in sub_by_id:
                            kept_now.append(sub_by_id[sq.slice_id])
                    result.all_qualities.extend(subq)
                    qualities = qualities  # keep record

        if not kept_now:
            break

        # [8][9] explain + recommend, and peel -----------------------------
        peeled_ids: set[str] = set()
        qmap = {q.slice_id: q for q in result.all_qualities}
        for s in kept_now:
            q = qmap.get(s.slice_id)
            if q is None:
                continue
            card = explain_slice(s, reference)
            adv = recommend(q, card)
            result.kept.append(KeptSlice(s, q, card, adv))
            peeled_ids.update(m.instance_id for m in s.members)

        # residual peeling: drop kept instances, recompute residual --------
        remaining = [e for e in remaining if e.instance_id not in peeled_ids]
        language = []   # language hypotheses are tested once, on round 0
        residual_mass = sum(1.0 - e.iou for e in remaining)
        result.residual_error = residual_mass / total_error_mass
        if result.residual_error <= residual_stop:
            break

    return result
