"""slice_quality -- error-slice discovery & validation for instance segmentation.

A modular pipeline that extends SliceLens with Beyond IoU (typed errors), MCSD
(manifold compactness), CB-SLICE (misprediction coherence / explanation) and
SEG4SEG (gamma-GMM clustering + Def.1 success criterion).

See README.md for the block map. Public entry points:

    from slice_quality import run_pipeline, SliceQualityConfig
    from slice_quality.typed_error import decompose_dataset
    from slice_quality.slice_quality_vector import score_slices
"""

from .config import SliceQualityConfig
from .data_types import SegErrorType, Slice, SliceQuality, TypedInstance
from .pipeline import KeptSlice, PipelineResult, run_from_instances, run_pipeline
from .slice_quality_vector import score_slices

__all__ = [
    "SliceQualityConfig",
    "SegErrorType",
    "Slice",
    "SliceQuality",
    "TypedInstance",
    "KeptSlice",
    "PipelineResult",
    "run_pipeline",
    "run_from_instances",
    "score_slices",
]
