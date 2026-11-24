"""Gemini LLM backend (paper default: Gemini-2.5-Pro, Section 5.1.2).

Requires the ``google-genai`` SDK and the ``GOOGLE_API_KEY`` environment
variable::

    pip install google-genai
    export GOOGLE_API_KEY=...   # PowerShell: $env:GOOGLE_API_KEY="..."

All heavy imports are deferred to construction time so this module can be
imported in environments without the SDK.
"""

from __future__ import annotations

import os

from .base import LLMBackend


class GeminiLLM(LLMBackend):
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "google-genai is required for GeminiLLM. Install with "
                "`pip install google-genai`."
            ) from exc

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Set GOOGLE_API_KEY (env var) or pass api_key= to GeminiLLM."
            )
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        from google.genai import types  # type: ignore

        response = self.client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
            ),
        )
        return (response.text or "").strip()
