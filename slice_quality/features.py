"""BLOCK 3 -- Per-instance clustering feature: emb (+) alpha*E (+) subtype.

Source : new (the typed-error feature is the differentiator vs SEG4SEG, which
         clusters on scalar performance only).
Input  : TypedInstances with ``.embedding`` and ``.error_vector`` set.
Output : an (N, D) feature matrix used by the clustering branch (BLOCK 4).

The two streams (geometry vs failure-signature) must be balanced, otherwise the
high-dim embedding swamps the 3-d error vector. We L2-normalise each block, then
scale the error block by ``alpha``.

Where to modify
---------------
* ``alpha``           : config.FeatureConfig.alpha  (alpha->0 == Domino/MCSD).
* ``normalize``       : per-block normalisation strategy.
* ``use_subtype_onehot`` : append a one-hot of the 8 SegErrorType values.
"""

from __future__ import annotations

import numpy as np

from .config import FeatureConfig
from .data_types import SegErrorType, TypedInstance


def _normalize_block(X: np.ndarray, how: str) -> np.ndarray:
    if how == "none" or X.size == 0:
        return X
    if how == "zscore":
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
        return (X - mu) / sd
    # default l2 per row
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / norm


def _subtype_onehot(insts: list[TypedInstance]) -> np.ndarray:
    types = list(SegErrorType)
    idx = {t: k for k, t in enumerate(types)}
    M = np.zeros((len(insts), len(types)))
    for r, inst in enumerate(insts):
        M[r, idx[inst.seg_subtype]] = 1.0
    return M


def build_features(
    instances: list[TypedInstance], cfg: FeatureConfig | None = None
) -> tuple[np.ndarray, list[TypedInstance]]:
    """Return (feature_matrix, instances_kept).

    Instances without an embedding are dropped (and returned list reflects that),
    so downstream indices line up with the matrix rows.
    """
    cfg = cfg or FeatureConfig()
    kept = [i for i in instances if i.embedding is not None]
    if not kept:
        return np.empty((0, 0)), kept

    emb = np.stack([i.embedding for i in kept])
    err = np.stack([i.error_vector for i in kept])

    emb = _normalize_block(emb, cfg.normalize)
    err = _normalize_block(err, cfg.normalize) * cfg.alpha

    blocks = [emb, err]
    if cfg.use_subtype_onehot:
        blocks.append(_subtype_onehot(kept) * cfg.alpha)

    return np.concatenate(blocks, axis=1), kept
