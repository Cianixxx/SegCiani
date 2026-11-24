"""BLOCK 2 -- Per-instance mask-pooled embeddings.

Source : MCSD / Domino (representations) -- interface only.
Input  : TypedInstances (from BLOCK 1) + a backbone that yields a feature map.
Output : the same TypedInstances with ``.embedding`` filled.

Key rule (design): for FN instances there is NO predicted mask, so we pool the
backbone features under the *GT* mask (metadata["from_gt_mask"] == True). Every
instance ends up in the SAME embedding space, so they cluster together.

This block needs the model backbone, which usually lives on the server. Keep the
embedding *extractor* pluggable: implement ``FeatureExtractor.feature_map`` for
your model (Mask2Former / Mask R-CNN / a frozen DINOv2) and the pooling is
shared below.

Where to modify
---------------
* ``FeatureExtractor`` : plug your backbone here (the ONLY model-dependent part).
* ``mask_pool``        : change the pooling (mean -> attention-weighted, etc.).
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import numpy as np

from .data_types import TypedInstance


class FeatureExtractor(Protocol):
    """Return a (C, H, W) feature map for a given image id.

    Implement this for your model on the server. A cheap, model-agnostic option
    is a frozen DINOv2 (matches SEG4SEG, which uses DINOv2_small as backbone).
    """

    def feature_map(self, image_id: str) -> np.ndarray: ...   # (C, H, W)


def mask_pool(feat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average-pool a (C, H, W) feature map over a HxW boolean mask -> (C,)."""
    C, Hf, Wf = feat.shape
    if mask.shape != (Hf, Wf):                # resize mask to the feature grid
        ys = (np.linspace(0, mask.shape[0] - 1, Hf)).astype(int)
        xs = (np.linspace(0, mask.shape[1] - 1, Wf)).astype(int)
        mask = mask[np.ix_(ys, xs)]
    m = mask.astype(np.float32)
    denom = m.sum()
    if denom == 0:
        return feat.reshape(C, -1).mean(axis=1)
    return (feat * m[None]).reshape(C, -1).sum(axis=1) / denom


def attach_embeddings(
    instances: list[TypedInstance],
    extractor: FeatureExtractor,
    mask_lookup: Callable[[TypedInstance], Optional[np.ndarray]],
) -> list[TypedInstance]:
    """Fill ``.embedding`` for every instance.

    ``mask_lookup`` returns the HxW mask to pool over for a given instance
    (the predicted mask normally, the GT mask for FN -- the caller decides using
    ``inst.metadata.get("from_gt_mask")``).
    """
    cache: dict[str, np.ndarray] = {}
    for inst in instances:
        if inst.image_id not in cache:
            cache[inst.image_id] = extractor.feature_map(inst.image_id)
        mask = mask_lookup(inst)
        if mask is None:
            continue
        inst.embedding = mask_pool(cache[inst.image_id], mask)
    return instances
