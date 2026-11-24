"""Abstract backend interfaces.

SliceLens needs two model capabilities, kept behind small interfaces so the
real models (Gemini, Qwen2.5-VL) and the offline mock can be swapped freely:

* :class:`LLMBackend`  -- a text-in / text-out chat model used for hypothesis
  generation, attribute inference, query refinement and the semantic evaluator.
* :class:`VLMBackend`  -- a vision-language model that (a) produces grounded
  captions and (b) returns the **slice confidence** P(yes | q, r) by reading the
  logits of the "yes"/"no" output tokens (Equation 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..data_types import Region


class LLMBackend(ABC):
    """Text chat model (paper default: Gemini-2.5-Pro)."""

    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        """Return the model's text response for a (system, user) prompt pair."""
        raise NotImplementedError


class VLMBackend(ABC):
    """Vision-language model (paper default: Qwen2.5-VL-7B-Instruct)."""

    @abstractmethod
    def caption_region(self, image_path: str, region: Region, attribute: str) -> str:
        """Grounded attribute-aware caption for ``region`` (Figure 9)."""
        raise NotImplementedError

    @abstractmethod
    def slice_confidence(self, image_path: str, region: Region, query: str) -> float:
        """Return P(yes | q, r) for whether ``region`` matches ``query`` (Eq. 1).

        Implementations must compute a softmax over only the "yes" and "no"
        output-token logits::

            P(yes|q,r) = exp(z_yes) / (exp(z_yes) + exp(z_no))
        """
        raise NotImplementedError
