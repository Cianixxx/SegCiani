"""Core data structures for the slice_quality pipeline.

These extend the SliceLens vocabulary (``slicelens.data_types``) with the
instance-segmentation typed-error signal and the Slice-Quality Vector.

Glossary
--------
* ``TypedInstance`` -- a single erroneous instance enriched with its typed error
  (Beyond IoU), its scalar performance (IoU), its segment sub-type, its
  embedding, and its source image id (needed for the block bootstrap).
* ``Slice`` -- a set of TypedInstances grouped together, either by the SliceLens
  language branch or by the embedding clustering branch.
* ``SliceQuality`` -- the 6-axis Slice-Quality Vector + the Def.1 verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class SegErrorType(str, Enum):
    """Per-instance error type for instance segmentation (the 6-case taxonomy).

    BOUNDARY/EXTENT come from Beyond IoU's pixel decomposition on a matched
    pair; the others are segment-level and are unique to instance seg.
    """

    CORRECT = "correct"        # 1<->1, right class, IoU high  (population reference)
    BOUNDARY = "boundary"      # 1<->1, right class, mask edges off
    EXTENT = "extent"          # 1<->1, right class, mask under/over-extended
    CONFUSION = "confusion"    # 1<->1, wrong class
    FP = "fp"                  # 1<->0, hallucination
    FN = "fn"                  # 0<->1, miss
    SPLIT = "split"            # N<->1, over-segmentation
    MERGE = "merge"            # 1<->M, under-segmentation


# The error types that count as "segment" errors in the Beyond IoU sense
# (E_seg == 1). Used by error_type_purity and the repair router.
SEGMENT_SUBTYPES = (
    SegErrorType.CONFUSION,
    SegErrorType.FP,
    SegErrorType.FN,
    SegErrorType.SPLIT,
    SegErrorType.MERGE,
)


@dataclass
class TypedInstance:
    """An erroneous instance enriched with all signals the SQV needs.

    Built by ``typed_error.decompose`` (BLOCK 1). One per erroneous predicted
    instance, plus one per missed GT instance (FN).
    """

    instance_id: str
    image_id: str                       # source image -> block bootstrap unit
    # --- performance / error (from masks vs GT, deterministic) ---------------
    iou: float                          # scalar performance y_i in [0,1] (0 for FP/FN)
    error_vector: np.ndarray            # [E_bnd, E_ext, E_seg], sums to ~ 1-iou
    seg_subtype: SegErrorType
    pred_class: Optional[str] = None
    gt_class: Optional[str] = None
    # --- representation (from the model backbone) ----------------------------
    embedding: Optional[np.ndarray] = None   # mask-pooled feature vector
    # --- bookkeeping / grounding ---------------------------------------------
    bbox: Optional[tuple[float, float, float, float]] = None  # for VLM grounding
    metadata: dict[str, Any] = field(default_factory=dict)    # VLM tags, etc.

    @property
    def confusion_pair(self) -> Optional[tuple[str, str]]:
        """(gt_class, pred_class) for CONFUSION instances, else None."""
        if self.seg_subtype is SegErrorType.CONFUSION:
            return (self.gt_class or "?", self.pred_class or "?")
        return None

    @property
    def dominant_error_type(self) -> SegErrorType:
        """The single error type assigned to this instance (used by purity)."""
        return self.seg_subtype


@dataclass
class Slice:
    """A candidate slice: a set of instances + how it was formed."""

    slice_id: str
    members: list[TypedInstance]
    source: str = "cluster"             # "language" (SliceLens) | "cluster"
    hypothesis: Optional[str] = None    # NL description if source == "language"
    # Soft membership P(S_j | x) per member id; defaults to hard (1.0) if empty.
    membership: dict[str, float] = field(default_factory=dict)
    # Bookkeeping (e.g. "aliases": paraphrase hypotheses merged by dedup).
    metadata: dict[str, Any] = field(default_factory=dict)

    def weight(self, inst: TypedInstance) -> float:
        return self.membership.get(inst.instance_id, 1.0)

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class SliceQuality:
    """The 6-axis Slice-Quality Vector + the Def.1 success verdict.

    Produced by ``slice_quality_vector.score_slice`` (BLOCK 6).
    """

    slice_id: str
    # --- the 6 axes ----------------------------------------------------------
    omega_squared: float                # severity        (SEG4SEG)
    manifold_compactness: float         # geometric coh.  (MCSD)
    misprediction_coherence: float      # output coh.     (CB-SLICE)
    error_type_purity: float            # error-type coh. (new / Beyond IoU)
    slice_purity: float                 # attribute coh.  (SEG4SEG Def.1)
    significant: bool                   # reliability     (new: bootstrap + FDR)
    # --- derived -------------------------------------------------------------
    dominant_error_type: SegErrorType
    size: int
    passes_def1: bool                   # the binary success gate
    verdict: str = "DISCARD"            # "KEEP" | "REFINE" | "DISCARD"
    # raw extras for inspection (bootstrap CI, p-value, etc.)
    extras: dict[str, Any] = field(default_factory=dict)

    def as_vector(self) -> np.ndarray:
        """The 6 numeric axes as an array (significant -> 0/1) for ranking."""
        return np.array([
            self.omega_squared,
            self.manifold_compactness,
            self.misprediction_coherence,
            self.error_type_purity,
            self.slice_purity,
            1.0 if self.significant else 0.0,
        ])
