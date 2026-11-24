"""Core data structures for SliceLens.

The vocabulary follows the paper:

* An **instance** is a model prediction or ground-truth object in a multi-instance
  task (a detection box, a segmentation mask, ...). Each instance is the unit
  that gets scored and ranked -- this is the *instance-level* (grounded) analysis
  that distinguishes SliceLens from image-level methods (Section 3.2.1, Fig. 6).
* An **error region** ``r`` is the grounding signal of an instance: a bounding box
  (detection) or a box/mask (segmentation). It is what the VLM verification prompt
  points at.
* A **hypothesis** is a natural-language failure description ``q`` (e.g.
  "bicycle obscured by a person"), produced either knowledge-driven or
  data-driven (Section 3.1).
* A **candidate slice** is the set of instances retrieved for a hypothesis,
  ranked by slice confidence (Section 3.2.1).
* An **error slice** is a candidate slice that the slope-trend analysis flags as
  a *systematic* error (Section 3.2.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ErrorType(str, Enum):
    """Type of model error attached to an instance / region."""

    FALSE_POSITIVE = "FP"
    FALSE_NEGATIVE = "FN"
    MISCLASSIFICATION = "MIS"
    CORRECT = "OK"  # not an error; included so the slope analysis sees both labels


class TaskType(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


class PromptType(str, Enum):
    """The two prompt kinds emitted by the knowledge-driven generator (Fig. 8).

    ``SEARCH`` prompts are concrete and can directly retrieve images
    ("a white cat"); ``CLUSTER`` prompts are broad attribute seeds that feed the
    data-driven branch ("cat colors").
    """

    SEARCH = "search"
    CLUSTER = "cluster"


@dataclass
class Region:
    """An error region ``r`` -- the grounding signal for an instance.

    Either ``bbox`` (xmin, ymin, xmax, ymax) or ``point`` (x, y) must be set;
    ``mask`` is optional and used for segmentation tasks.
    """

    bbox: Optional[tuple[float, float, float, float]] = None
    point: Optional[tuple[float, float]] = None
    mask: Optional[Any] = None  # e.g. RLE dict or HxW boolean array

    def center_point(self) -> tuple[float, float]:
        """Return a representative 2D point inside the region (Section 3.2.1)."""
        if self.point is not None:
            return self.point
        if self.bbox is not None:
            xmin, ymin, xmax, ymax = self.bbox
            return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
        raise ValueError("Region has neither point nor bbox to derive a center.")


@dataclass
class Instance:
    """A single grounded model prediction / ground-truth object."""

    instance_id: str
    image_path: str
    region: Region
    label: Optional[str] = None
    # Whether the model actually erred on this instance. This is the *outcome*
    # signal used by the slope-trend analysis (it is the per-instance error label
    # that error rate is computed from). It comes from comparing model output to
    # ground truth during "Model Errors Collection" (Section 4.1).
    is_error: bool = False
    error_type: ErrorType = ErrorType.CORRECT
    # Optional model confidence for the prediction (detector score), kept for
    # bookkeeping; the slope analysis uses *slice* confidence, not this.
    model_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A failure hypothesis -- the natural-language query ``q``."""

    query: str
    prompt_type: PromptType = PromptType.SEARCH
    title: str = ""
    description: str = ""
    factor: str = ""  # e.g. "Object Attributes" (Fig. 8 grouping)
    source: str = "knowledge"  # "knowledge" or "data"

    def __hash__(self) -> int:  # allow de-duplication in a set
        return hash(self.query.strip().lower())

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Hypothesis)
            and self.query.strip().lower() == other.query.strip().lower()
        )


@dataclass
class ScoredInstance:
    """An instance together with its slice confidence P(yes | q, r) (Eq. 1)."""

    instance: Instance
    slice_confidence: float


@dataclass
class CandidateSlice:
    """Instances retrieved for one hypothesis, ranked by slice confidence."""

    hypothesis: Hypothesis
    scored_instances: list[ScoredInstance]  # sorted descending by confidence

    def top_k(self, k: int = 10) -> list[ScoredInstance]:
        return self.scored_instances[:k]

    def confidences(self) -> list[float]:
        return [s.slice_confidence for s in self.scored_instances]


@dataclass
class ErrorSlice:
    """A candidate slice classified as a *systematic* error (Section 3.2.2)."""

    candidate: CandidateSlice
    max_positive_slope: float
    is_systematic: bool

    @property
    def hypothesis(self) -> Hypothesis:
        return self.candidate.hypothesis


@dataclass
class SliceLensResult:
    """Everything produced by a full pipeline run."""

    task_description: str
    hypotheses: list[Hypothesis]
    candidate_slices: list[CandidateSlice]
    error_slices: list[ErrorSlice]
