"""Configuration objects for SliceLens.

Defaults mirror the paper where a value is given (Sections 5.1.2, 5.3.3, B.1):

* LLM      : Gemini-2.5-Pro
* VLM      : Qwen2.5-VL-7B-Instruct
* Eq. (2)  : Precision@k with k = 10
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlopeTrendConfig:
    """Parameters for the automated error-slice identification (Section 3.2.2).

    The analysis estimates the *error rate vs. confidence* curve by binning the
    confidences once, then slides a threshold to define overlapping
    high-confidence windows (the tails of that curve), fits a local line in each
    window, and records the maximum positive slope. A slice is flagged as
    systematic when that slope exceeds ``slope_threshold``.
    """

    # Number of (overlapping) sliding-threshold windows. Thresholds are placed
    # at evenly spaced quantiles of the confidence distribution so every window
    # is populated regardless of the absolute confidence scale.
    num_thresholds: int = 10
    # Highest threshold expressed as a quantile of the confidence distribution
    # (e.g. 0.8 -> the tightest window keeps the top ~20% of confidences).
    max_threshold_quantile: float = 0.8
    # Number of confidence bins used to estimate the error-rate curve (computed
    # once over the full confidence range, then sub-selected per window).
    num_bins: int = 10
    # Bins with fewer than this many instances give unreliable error-rate
    # estimates (e.g. 0/2) and are dropped before any line is fit.
    min_bin_count: int = 10
    # A window must keep at least this many instances for its slope to count.
    min_window_size: int = 20
    # A window's fit needs at least this many populated bins (>= 2) so the local
    # line is not driven by one or two noisy points.
    min_bins_per_window: int = 3
    # A window must span at least this much confidence; without x-variation a
    # slope is meaningless (and numerically unstable).
    min_confidence_spread: float = 0.05
    # Decision threshold on the maximum positive slope (the paper notes the
    # method is robust to this choice; F1 stays high for ~0.10-0.35, Fig. 7).
    slope_threshold: float = 0.20


@dataclass
class SliceLensConfig:
    """Top-level pipeline configuration."""

    task_type: str = "detection"  # "detection" | "segmentation" | "classification"
    # Grounding mode for the VLM verification prompt: "bbox" or "point"
    # (Section 3.2.1). Detection/segmentation default to bbox.
    grounding: str = "bbox"
    # Precision@k (Eq. 2).
    precision_k: int = 10
    # How many sampled instances to caption per cluster attribute in the
    # data-driven branch (Section 3.1, step (a)).
    data_driven_sample_size: int = 50
    # Whether to run the (more expensive) data-driven branch in addition to the
    # knowledge-driven one.
    use_data_driven: bool = True
    # Default model identifiers for the real backends.
    llm_model: str = "gemini-2.5-pro"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    slope: SlopeTrendConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.slope is None:
            self.slope = SlopeTrendConfig()
