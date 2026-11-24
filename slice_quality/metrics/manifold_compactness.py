"""Axis 2/6 -- manifold compactness.  Source: MCSD (Definition 3.1).

Average within-slice connectivity on a kNN graph over instance embeddings: for
each member, the fraction of its k nearest neighbours that also belong to the
slice. High = the slice is a tight region of the manifold.

NOTE (bottleneck B4): on cluster-branch slices this is near-tautological (they
were formed by embedding proximity). It is genuinely informative on
language-branch (SliceLens) slices, where grouping was NOT based on embeddings.

Input  : all embeddings (build the graph once) + the member indices of a slice.
Output : compactness in [0, 1].
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None

from ..config import ManifoldConfig


def build_knn_graph(embeddings: np.ndarray, cfg: ManifoldConfig | None = None) -> list[set[int]]:
    """Return, for each node, the set of its k nearest-neighbour indices."""
    cfg = cfg or ManifoldConfig()
    if NearestNeighbors is None:
        raise ImportError("scikit-learn is required for build_knn_graph")
    n = len(embeddings)
    k = min(cfg.k + 1, n)               # +1 because the first neighbour is itself
    nn = NearestNeighbors(n_neighbors=k, metric=cfg.metric).fit(embeddings)
    _, idx = nn.kneighbors(embeddings)
    return [set(row[1:]) for row in idx]   # drop self


def manifold_compactness(member_indices: list[int], knn: list[set[int]]) -> float:
    """Average fraction of each member's neighbours that are inside the slice."""
    members = set(member_indices)
    if not members:
        return 0.0
    total = 0.0
    for i in member_indices:
        neigh = knn[i]
        if not neigh:
            continue
        total += len(neigh & members) / len(neigh)
    return float(total / len(member_indices))
