"""Model backends for SliceLens.

The real backends (:class:`GeminiLLM`, :class:`QwenVLM`) import heavy optional
dependencies lazily, so importing this package never requires torch / the SDKs.
The mock backends have no third-party dependencies.
"""

from .base import LLMBackend, VLMBackend
from .mock import MockLLM, MockVLM

__all__ = [
    "LLMBackend",
    "VLMBackend",
    "MockLLM",
    "MockVLM",
    "GeminiLLM",
    "QwenVLM",
    "GeminiVLM",
    "HFVLM",
    "HFLLM",
]


def __getattr__(name: str):  # lazy attribute access for optional backends
    if name == "GeminiLLM":
        from .gemini_llm import GeminiLLM

        return GeminiLLM
    if name == "QwenVLM":
        from .qwen_vlm import QwenVLM

        return QwenVLM
    if name == "GeminiVLM":
        from .gemini_vlm import GeminiVLM

        return GeminiVLM
    if name == "HFVLM":
        from .hf_vlm import HFVLM

        return HFVLM
    if name == "HFLLM":
        from .hf_llm import HFLLM

        return HFLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
