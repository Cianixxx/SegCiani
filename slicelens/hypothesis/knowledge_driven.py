"""Knowledge-driven hypothesis generation (Section 3.1, part 1).

Top-down: the LLM uses broad world knowledge about the task to enumerate failure
modes that often align with real error modes ("background objects similar to
bicycles", "very small objects"). Driven by the system prompt of Figure 8 and a
task/dataset-specific user prompt (Figures 12-13, Section A.2).

The structured JSON output is flattened into :class:`Hypothesis` objects. Each
``prompt`` entry becomes one hypothesis; its ``type`` (search/cluster) is
preserved -- ``search`` prompts are concrete queries usable directly for
verification, while ``cluster`` prompts are attribute seeds that feed the
data-driven branch.
"""

from __future__ import annotations

from ..backends.base import LLMBackend
from ..data_types import Hypothesis, PromptType, TaskType
from ..prompts import (
    DATASET_USER_PROMPTS,
    FESD_DETECTION_USER_PROMPT,
    FESD_SEGMENTATION_USER_PROMPT,
    KNOWLEDGE_DRIVEN_SYSTEM_PROMPT,
)
from ..utils import extract_json


def build_user_prompt(
    task_type: str,
    label: str | None = None,
    dataset: str | None = None,
    task_description: str | None = None,
) -> str:
    """Construct the user prompt that frames the task for the LLM.

    Priority: explicit ``task_description`` > named ``dataset`` (Section A.2) >
    task-type template filled with ``label`` (Figures 12/13).
    """
    if task_description:
        return task_description
    if dataset:
        key = dataset.strip().lower()
        if key in DATASET_USER_PROMPTS:
            return DATASET_USER_PROMPTS[key]
    if label is None:
        raise ValueError("Provide task_description, a known dataset, or a label.")
    if task_type == TaskType.SEGMENTATION.value:
        return FESD_SEGMENTATION_USER_PROMPT.format(label=label)
    return FESD_DETECTION_USER_PROMPT.format(label=label)


def parse_hypotheses(raw_json: dict) -> list[Hypothesis]:
    """Flatten the Figure 8 JSON structure into Hypothesis objects."""
    hypotheses: list[Hypothesis] = []
    seen: set[str] = set()
    factors = raw_json.get("hypothesis", {}) if isinstance(raw_json, dict) else {}
    for factor_name, patterns in factors.items():
        if not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            title = str(pattern.get("title", ""))
            description = str(pattern.get("description", ""))
            for prompt in pattern.get("prompts", []) or []:
                if not isinstance(prompt, dict):
                    continue
                query = str(prompt.get("prompt", "")).strip()
                if not query:
                    continue
                key = query.lower()
                if key in seen:
                    continue
                seen.add(key)
                ptype = str(prompt.get("type", "search")).strip().lower()
                hypotheses.append(
                    Hypothesis(
                        query=query,
                        prompt_type=(
                            PromptType.CLUSTER
                            if ptype == "cluster"
                            else PromptType.SEARCH
                        ),
                        title=title,
                        description=description,
                        factor=str(factor_name),
                        source="knowledge",
                    )
                )
    return hypotheses


def generate_knowledge_hypotheses(
    llm: LLMBackend,
    task_type: str = "detection",
    label: str | None = None,
    dataset: str | None = None,
    task_description: str | None = None,
) -> list[Hypothesis]:
    """Generate knowledge-driven hypotheses for a task."""
    user = build_user_prompt(task_type, label, dataset, task_description)
    response = llm.chat(system=KNOWLEDGE_DRIVEN_SYSTEM_PROMPT, user=user)
    data = extract_json(response)
    return parse_hypotheses(data)
