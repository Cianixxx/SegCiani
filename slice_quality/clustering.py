"""BLOCK 4 -- Clustering branch: gamma-weighted GMM over emb (+) E (+) performance.

Source : SEG4SEG (the gamma-weighted GMM, Eq. 1) -- code unavailable, reimplemented.
Input  : feature matrix from BLOCK 3 + the kept TypedInstances.
Output : list[Slice] (one per GMM component), with soft membership = posterior.

SEG4SEG's Eq. 1 weights the *performance* likelihood by gamma so the slices are
both representation-coherent and performance-coherent. We implement this
pragmatically: append a gamma-scaled performance channel (1 - IoU) to the BLOCK 3
feature, then fit a diagonal GaussianMixture. gamma -> 0 recovers plain Domino.

Improvement #6 (defaults, see ClusterConfig):
* STRATIFIED clustering: one GMM per error subtype. A single global GMM lets the
  dominant FN/FP mass absorb boundary/extent instances into impure blobs; per
  subtype, every slice is error-type pure BY CONSTRUCTION and the embedding
  structure *within* a type (what KIND of objects get missed) emerges cleanly.
  Tiny strata (< min_stratum_fit) become one whole-stratum slice.
* BIC model selection: k per (stratum) chosen by minimising GaussianMixture.bic
  over cfg.k_grid, instead of an arbitrary fixed n_slices.

This is the embedding/data-driven branch. The language branch (SliceLens
hypotheses -> grounding) produces its own slices; both feed BLOCK 6, which scores
them identically.

Where to modify
---------------
* ``stratify_by_subtype`` / ``k_selection`` / ``k_grid`` / ``n_slices`` / ``gamma``
  : config.ClusterConfig. Legacy behaviour = (False, "fixed").
* swap GaussianMixture for the exact SEG4SEG joint objective if needed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    GaussianMixture = None

from .config import ClusterConfig
from .data_types import Slice, TypedInstance


def _candidate_ks(n: int, cfg: ClusterConfig) -> list[int]:
    """BIC candidate component counts, capped so components stay populated."""
    kmax = max(1, n // max(1, cfg.min_points_per_component))
    ks = [k for k in cfg.k_grid if 1 <= k <= kmax]
    return ks or [1]


def _fit_gmm(
    X: np.ndarray,
    instances: list[TypedInstance],
    cfg: ClusterConfig,
    prefix: str,
) -> list[Slice]:
    """Fit one GMM (fixed-k or BIC-selected) -> one Slice per component."""
    if cfg.k_selection == "bic":
        best_gmm, best_bic = None, None
        for k in _candidate_ks(len(instances), cfg):
            gmm = GaussianMixture(
                n_components=k, covariance_type=cfg.covariance_type,
                random_state=cfg.random_state,
            ).fit(X)
            b = gmm.bic(X)
            if best_bic is None or b < best_bic:
                best_gmm, best_bic = gmm, b
        gmm = best_gmm
    else:                                     # "fixed" (legacy)
        if len(instances) < cfg.n_slices:
            return []
        gmm = GaussianMixture(
            n_components=cfg.n_slices, covariance_type=cfg.covariance_type,
            random_state=cfg.random_state,
        ).fit(X)

    posterior = gmm.predict_proba(X)          # (N, k)  soft membership
    hard = posterior.argmax(axis=1)

    slices: list[Slice] = []
    for c in range(gmm.n_components):
        members = [inst for k, inst in enumerate(instances) if hard[k] == c]
        if not members:
            continue
        membership = {
            inst.instance_id: float(posterior[k, c])
            for k, inst in enumerate(instances) if hard[k] == c
        }
        slices.append(Slice(
            slice_id=f"{prefix}_{c}", members=members,
            source="cluster", membership=membership,
        ))
    return slices


def cluster_slices(
    features: np.ndarray,
    instances: list[TypedInstance],
    cfg: ClusterConfig | None = None,
) -> list[Slice]:
    """Cluster the instances into candidate slices (see module docstring)."""
    cfg = cfg or ClusterConfig()
    if GaussianMixture is None:
        raise ImportError("scikit-learn is required for clustering.cluster_slices")
    if not len(instances):
        return []

    perf = np.array([[cfg.gamma * (1.0 - i.iou)] for i in instances])   # performance channel
    X = np.concatenate([features, perf], axis=1)

    if not cfg.stratify_by_subtype:
        return _fit_gmm(X, instances, cfg, prefix="cluster")

    strata: dict[str, list[int]] = defaultdict(list)
    for k, inst in enumerate(instances):
        strata[inst.seg_subtype.value].append(k)

    slices: list[Slice] = []
    for subtype in sorted(strata):
        idx = strata[subtype]
        members = [instances[k] for k in idx]
        if len(idx) < cfg.min_stratum_fit:
            # too small to model: keep as ONE (already error-type pure) slice
            slices.append(Slice(
                slice_id=f"cluster_{subtype}_all", members=members,
                source="cluster",
                membership={m.instance_id: 1.0 for m in members},
            ))
            continue
        slices.extend(_fit_gmm(X[idx], members, cfg, prefix=f"cluster_{subtype}"))
    return slices
