"""End-to-end SliceLens pipeline (Figure 3).

Wires the three modules together:

    hypotheses = knowledge-driven  ∪  data-driven        (Section 3.1)
    candidate slices = rank instances by P(yes | q, r)    (Section 3.2.1)
    error slices = slope-trend analysis over candidates    (Section 3.2.2)
"""

from __future__ import annotations

from collections.abc import Sequence

from .backends.base import LLMBackend, VLMBackend
from .config import SliceLensConfig
from .data_types import (
    CandidateSlice,
    ErrorSlice,
    Hypothesis,
    Instance,
    PromptType,
    SliceLensResult,
)
from .hypothesis.data_driven import generate_data_driven_hypotheses
from .hypothesis.knowledge_driven import generate_knowledge_hypotheses
from .identification.slope_trend import analyze_slope_trend, identify_error_slices
from .verification.candidate_slices import discover_candidate_slices


class SliceLens:
    """The SliceLens framework.

    Parameters
    ----------
    llm:
        Text backend for hypothesis generation / refinement (e.g. ``GeminiLLM``).
    vlm:
        Vision-language backend for grounded captioning and slice confidence
        (e.g. ``QwenVLM``).
    config:
        Pipeline configuration (task type, grounding mode, k, slope params).
    """

    def __init__(
        self,
        llm: LLMBackend,
        vlm: VLMBackend,
        config: SliceLensConfig | None = None,
    ) -> None:
        self.llm = llm
        self.vlm = vlm
        self.config = config or SliceLensConfig()

    # -- Stage 1: hypothesis generation -----------------------------------
    def generate_hypotheses(
        self,
        instances: Sequence[Instance] | None = None,
        label: str | None = None,
        dataset: str | None = None,
        task_description: str | None = None,
    ) -> list[Hypothesis]:
        """Generate knowledge-driven (and optionally data-driven) hypotheses."""
        knowledge = generate_knowledge_hypotheses(
            self.llm,
            task_type=self.config.task_type,
            label=label,
            dataset=dataset,
            task_description=task_description,
        )

        hypotheses: list[Hypothesis] = list(knowledge)

        if self.config.use_data_driven and instances:
            # Cluster-type prompts are the attribute seeds for the data branch.
            attributes = [
                h.query for h in knowledge if h.prompt_type == PromptType.CLUSTER
            ]
            if attributes:
                data = generate_data_driven_hypotheses(
                    self.llm,
                    self.vlm,
                    instances,
                    attributes,
                    sample_size=self.config.data_driven_sample_size,
                )
                hypotheses.extend(data)

        # De-duplicate by query (Hypothesis hashes on its query).
        return list(dict.fromkeys(hypotheses))

    # -- Stage 2: candidate slice discovery -------------------------------
    def discover_candidate_slices(
        self,
        hypotheses: Sequence[Hypothesis],
        instances: Sequence[Instance],
        search_only: bool = True,
    ) -> list[CandidateSlice]:
        """Rank instances for each hypothesis by slice confidence (Eq. 1).

        ``search_only`` skips ``cluster`` prompts -- those are broad attribute
        seeds, not directly verifiable queries (Figure 8).
        """
        to_verify = [
            h
            for h in hypotheses
            if not search_only or h.prompt_type == PromptType.SEARCH
        ]
        return discover_candidate_slices(to_verify, instances, self.vlm)

    # -- Stage 3: error slice identification ------------------------------
    def identify_error_slices(
        self, candidate_slices: Sequence[CandidateSlice]
    ) -> list[ErrorSlice]:
        """Flag systematic error slices via slope-trend analysis (Section 3.2.2)."""
        return identify_error_slices(list(candidate_slices), self.config.slope)

    # -- Full pipeline ----------------------------------------------------
    def run(
        self,
        instances: Sequence[Instance],
        label: str | None = None,
        dataset: str | None = None,
        task_description: str | None = None,
    ) -> SliceLensResult:
        """Run hypothesis generation -> discovery -> identification."""
        hypotheses = self.generate_hypotheses(
            instances=instances,
            label=label,
            dataset=dataset,
            task_description=task_description,
        )
        candidates = self.discover_candidate_slices(hypotheses, instances)
        error_slices = self.identify_error_slices(candidates)
        return SliceLensResult(
            task_description=task_description or (label or dataset or ""),
            hypotheses=hypotheses,
            candidate_slices=candidates,
            error_slices=error_slices,
        )

    # Convenience re-export for callers that want the full trend diagnostics.
    analyze_slope_trend = staticmethod(analyze_slope_trend)
