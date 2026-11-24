"""Small shared helpers."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from an LLM response.

    Handles ```json ... ``` fences and leading/trailing prose by falling back to
    the first balanced ``{...}`` or ``[...]`` block. Raises ``ValueError`` if no
    JSON can be parsed.
    """
    if text is None:
        raise ValueError("Cannot extract JSON from None.")
    candidate = text.strip()

    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced JSON block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            if candidate[i] == opener:
                depth += 1
            elif candidate[i] == closer:
                depth -= 1
                if depth == 0:
                    block = candidate[start : i + 1]
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No parsable JSON found in response: {text[:200]!r}")
