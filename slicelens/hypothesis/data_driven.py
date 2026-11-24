"""Data-driven hypothesis generation (Section 3.1, part 2).

Bottom-up: uncover errors arising from the *dataset distribution*. Three steps,
seeded by the ``cluster``-type attributes from the knowledge-driven branch:

(a) **Grounded attribute-aware caption extraction** (VLM, Figure 9): for each
    attribute, caption a sampled set of grounded regions.
(b) **Attribute inference** (LLM, Figure 10): Step 1 extracts the relevant
    attribute value/keyword from each caption; Step 2 clusters the keywords into
    non-overlapping semantic categories.
(c) **Hypothesis generation** (LLM, Figure 11): refine the clustered
    attributes/values into well-formed natural-language query phrases that serve
    as concrete, data-driven hypotheses to be verified.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..backends.base import LLMBackend, VLMBackend
from ..data_types import Hypothesis, Instance, PromptType
from ..prompts import (
    ATTRIBUTE_EXTRACTION_PROMPT,
    KEYWORD_CLUSTERING_PROMPT,
    QUERY_REFINEMENT_PROMPT,
)
from ..utils import extract_json


def extract_attribute_value(llm: LLMBackend, attribute: str, caption: str) -> str:
    """Figure 10, Step 1: infer the most relevant attribute value from a caption."""
    prompt = ATTRIBUTE_EXTRACTION_PROMPT.format(attribute=attribute, caption=caption)
    value = llm.chat(system="", user=prompt).strip().strip(".").lower()
    return value


def cluster_keywords(
    llm: LLMBackend, attribute: str, keywords: Sequence[str]
) -> dict[str, list[str]]:
    """Figure 10, Step 2: cluster keywords into non-overlapping categories."""
    kept = [k for k in keywords if k and k != "none"]
    if not kept:
        return {}
    prompt = KEYWORD_CLUSTERING_PROMPT.format(attribute=attribute, keywords=kept)
    response = llm.chat(system="", user=prompt)
    try:
        data = extract_json(response)
    except ValueError:
        return {attribute: sorted(set(kept))}
    if not isinstance(data, dict):
        return {attribute: sorted(set(kept))}
    return {str(k): [str(x) for x in v] for k, v in data.items() if isinstance(v, list)}


def refine_to_queries(
    llm: LLMBackend, attribute: str, clusters: dict[str, list[str]]
) -> list[str]:
    """Figure 11: compose text-to-image retrieval query phrases from clusters."""
    if not clusters:
        return []
    prompt = QUERY_REFINEMENT_PROMPT.format(attribute=attribute, clusters=clusters)
    response = llm.chat(system="", user=prompt)
    try:
        data = extract_json(response)
    except ValueError:
        return []
    results = data.get("results", []) if isinstance(data, dict) else []
    return [str(r).strip() for r in results if str(r).strip()]


def generate_data_driven_hypotheses(
    llm: LLMBackend,
    vlm: VLMBackend,
    instances: Sequence[Instance],
    attributes: Sequence[str],
    sample_size: int = 50,
    seed: int = 0,
) -> list[Hypothesis]:
    """Run the full (a)->(b)->(c) data-driven pipeline for each attribute."""
    rng = random.Random(seed)
    pool = list(instances)
    if len(pool) > sample_size:
        sampled = rng.sample(pool, sample_size)
    else:
        sampled = pool

    hypotheses: list[Hypothesis] = []
    seen: set[str] = set()
    for attribute in attributes:
        # (a) grounded captions
        captions = [
            vlm.caption_region(inst.image_path, inst.region, attribute)
            for inst in sampled
        ]
        # (b) step 1: attribute value extraction
        values = [extract_attribute_value(llm, attribute, c) for c in captions]
        # (b) step 2: keyword clustering
        clusters = cluster_keywords(llm, attribute, values)
        # (c) refine to query phrases
        queries = refine_to_queries(llm, attribute, clusters)
        for q in queries:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            hypotheses.append(
                Hypothesis(
                    query=q,
                    prompt_type=PromptType.SEARCH,
                    title=f"Data-driven: {attribute}",
                    description=f"Derived from grounded captions about '{attribute}'.",
                    factor=attribute,
                    source="data",
                )
            )
    return hypotheses
