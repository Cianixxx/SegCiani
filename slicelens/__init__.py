"""SliceLens -- Fine-Grained and Grounded Error Slice Discovery.

A faithful re-implementation of the method described in:

    Wei Zhang, Chaoqun Wang, Zixuan Guan, Sam Kao, Pengfei Zhao, Peng Wu,
    Sifeng He. "SliceLens: Fine-Grained and Grounded Error Slice Discovery for
    Multi-Instance Vision Tasks." arXiv:2512.24592v1, Apple, 2025.

Quick start (offline, no GPU / API key)::

    from slicelens import SliceLens, SliceLensConfig
    from slicelens.backends import MockLLM, MockVLM

    sl = SliceLens(MockLLM(), MockVLM(), SliceLensConfig(task_type="detection"))
    result = sl.run(instances, label="bicycle")

Real backends::

    from slicelens.backends import GeminiLLM, QwenVLM
    sl = SliceLens(GeminiLLM(), QwenVLM(grounding="bbox"))
"""

from .config import SliceLensConfig, SlopeTrendConfig
from .data_types import (
    CandidateSlice,
    ErrorSlice,
    ErrorType,
    Hypothesis,
    Instance,
    PromptType,
    Region,
    SliceLensResult,
    TaskType,
)
from .pipeline import SliceLens

__all__ = [
    "SliceLens",
    "SliceLensConfig",
    "SlopeTrendConfig",
    "Instance",
    "Region",
    "Hypothesis",
    "ErrorType",
    "TaskType",
    "PromptType",
    "CandidateSlice",
    "ErrorSlice",
    "SliceLensResult",
]

__version__ = "0.1.0"
