"""Slice confidence -- Equation (1) of the paper.

The VLM is asked whether an error region ``r`` matches a hypothesis query ``q``
(Section 3.2.1). Let ``z_yes`` and ``z_no`` be the logits of the "yes" and "no"
output tokens. The slice confidence is the softmax over just those two::

    P(yes | q, r) = exp(z_yes) / (exp(z_yes) + exp(z_no))

This module holds the (tiny, dependency-free) math so that *both* the real Qwen
backend and the mock backend share an identical Eq. (1) implementation.
"""

from __future__ import annotations

import math


def softmax_yes_no(z_yes: float, z_no: float) -> float:
    """Two-way softmax giving P(yes) from the yes/no logits (Eq. 1).

    Numerically stable (subtracts the max before exponentiating).
    """
    m = max(z_yes, z_no)
    e_yes = math.exp(z_yes - m)
    e_no = math.exp(z_no - m)
    return e_yes / (e_yes + e_no)
