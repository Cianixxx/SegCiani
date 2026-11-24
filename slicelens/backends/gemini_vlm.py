"""Gemini VLM backend -- run the *whole* pipeline via API, no GPU required.

This implements :class:`VLMBackend` with Google's multimodal Gemini models:

* ``caption_region``     -> multimodal generation (Figure 9).
* ``slice_confidence``   -> Equation (1) via ``response_logprobs``: we read the
  log-probabilities of the "yes"/"no" output tokens and take a two-way softmax
  (``softmax`` over log-probs == renormalised P(yes) over {yes, no}).

If the model/endpoint does not return logprobs, it falls back to a
sampling estimate ``P(yes) ~= #yes / (#yes + #no)`` over ``n_samples`` draws.

Trade-off (cf. paper Table 5 / B.1): an API VLM matches or beats local Qwen on
quality, but the VLM is called once per (instance x hypothesis), so for large
runs API latency/cost dominate -- prefer local Qwen there.

Requires ``google-genai`` and ``GOOGLE_API_KEY``. Heavy imports are deferred.
"""

from __future__ import annotations

import mimetypes
import os
import re
from typing import Any, Optional

from ..data_types import Region
from ..prompts import (
    BBOX_VERIFICATION_PROMPT,
    GROUNDED_CAPTION_SYSTEM_PROMPT,
    GROUNDED_CAPTION_USER_PROMPT,
    POINT_VERIFICATION_PROMPT,
)
from .base import VLMBackend

_VERIFY_SYSTEM_PROMPT = (
    "You are a visual reasoning assistant. You are given an image and a region "
    "of interest. Decide whether the region matches the given description and "
    "answer with a single word: yes or no."
)

# Big log-prob gap used when only one of yes/no appears among the top tokens
# (the missing class is treated as essentially impossible).
_MISSING_LOGPROB_GAP = 12.0


def _get(obj: Any, name: str) -> Any:
    """Attribute- or dict-style access (the SDK returns objects; tests pass dicts)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _normalize_token(token: str) -> str:
    """Strip whitespace and common subword markers, lowercase."""
    return token.strip().strip("▁Ġ").strip().lower()


def _parse_yes_no(text: str) -> Optional[str]:
    """First standalone 'yes'/'no' anywhere in the text (lenient for verbose models)."""
    m = re.search(r"\b(yes|no)\b", text.lower())
    return m.group(1) if m else None


def extract_yes_no_logprobs(
    response: Any,
) -> tuple[Optional[float], Optional[float]]:
    """Pull (logprob_yes, logprob_no) from a Gemini ``response``.

    Scans the per-step top-candidate logprobs (``candidates[0].logprobs_result``)
    and returns the first output step that contains "yes" and/or "no". Either
    value may be ``None`` if that token was not among the returned top
    candidates. Pure function (no SDK dependency) so it is unit-testable.
    """
    candidates = _get(response, "candidates")
    if not candidates:
        return (None, None)
    lp_result = _get(candidates[0], "logprobs_result")
    if lp_result is None:
        return (None, None)
    top_steps = _get(lp_result, "top_candidates")
    if not top_steps:
        return (None, None)

    for step in top_steps:
        step_cands = _get(step, "candidates") or []
        lp_yes: Optional[float] = None
        lp_no: Optional[float] = None
        for cand in step_cands:
            tok = _normalize_token(str(_get(cand, "token") or ""))
            lp = _get(cand, "log_probability")
            if lp is None:
                continue
            if tok == "yes":
                lp_yes = lp if lp_yes is None else max(lp_yes, lp)
            elif tok == "no":
                lp_no = lp if lp_no is None else max(lp_no, lp)
        if lp_yes is not None or lp_no is not None:
            return (lp_yes, lp_no)
    return (None, None)


def confidence_from_logprobs(
    lp_yes: Optional[float], lp_no: Optional[float]
) -> float:
    """Equation (1) from yes/no log-probs (pure, SDK-free, unit-testable).

    Softmax over the two log-probs gives the renormalised P(yes). A one-sided
    result (only yes or only no among the top tokens) treats the missing class
    as essentially impossible.
    """
    from ..verification.slice_confidence import softmax_yes_no

    if lp_yes is None and lp_no is None:
        raise ValueError("Both yes and no log-probs are missing.")
    if lp_yes is None:
        lp_yes = lp_no - _MISSING_LOGPROB_GAP  # type: ignore[operator]
    if lp_no is None:
        lp_no = lp_yes - _MISSING_LOGPROB_GAP
    return softmax_yes_no(lp_yes, lp_no)


class GeminiVLM(VLMBackend):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        grounding: str = "bbox",
        top_logprobs: int = 5,
        fallback_to_sampling: bool = True,
        n_samples: int = 5,
        sampling_temperature: float = 1.0,
    ) -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "google-genai is required for GeminiVLM. Install with "
                "`pip install google-genai`."
            ) from exc
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GOOGLE_API_KEY or pass api_key= to GeminiVLM.")
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.model = model
        self.grounding = grounding
        self.top_logprobs = top_logprobs
        self.fallback_to_sampling = fallback_to_sampling
        self.n_samples = n_samples
        self.sampling_temperature = sampling_temperature

    # -- helpers ----------------------------------------------------------
    def _image_part(self, image_path: str):
        from google.genai import types  # type: ignore

        mime, _ = mimetypes.guess_type(image_path)
        with open(image_path, "rb") as f:
            data = f.read()
        return types.Part.from_bytes(data=data, mime_type=mime or "image/jpeg")

    def _verification_prompt(self, region: Region, query: str) -> str:
        if self.grounding == "point" or region.bbox is None:
            x, y = region.center_point()
            return POINT_VERIFICATION_PROMPT.format(
                x=round(x, 1), y=round(y, 1), query=query
            )
        xmin, ymin, xmax, ymax = region.bbox
        return BBOX_VERIFICATION_PROMPT.format(
            xmin=round(xmin, 1), ymin=round(ymin, 1),
            xmax=round(xmax, 1), ymax=round(ymax, 1), query=query,
        )

    # -- interface --------------------------------------------------------
    def caption_region(self, image_path: str, region: Region, attribute: str) -> str:
        from google.genai import types  # type: ignore

        box = region.bbox if region.bbox is not None else (
            (*region.center_point(), *region.center_point())
        )
        user = GROUNDED_CAPTION_USER_PROMPT.format(
            xmin=round(box[0], 1), ymin=round(box[1], 1),
            xmax=round(box[2], 1), ymax=round(box[3], 1), attribute=attribute,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=[self._image_part(image_path), user],
            config=types.GenerateContentConfig(
                system_instruction=GROUNDED_CAPTION_SYSTEM_PROMPT.format(attribute=attribute),
                temperature=0.0,
            ),
        )
        return (response.text or "").strip()

    def _gen_config(self, *, logprobs: bool, temperature: float,
                    max_tokens: int, disable_thinking: bool):
        from google.genai import types  # type: ignore

        kw: dict = dict(
            system_instruction=_VERIFY_SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if logprobs:
            kw["response_logprobs"] = True
            kw["logprobs"] = self.top_logprobs
        if disable_thinking:
            # "Thinking" models (e.g. 2.5/3.x flash) otherwise spend the output
            # budget reasoning and never emit the yes/no answer token.
            try:
                kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:  # noqa: BLE001 - SDK/model without ThinkingConfig
                pass
        return types.GenerateContentConfig(**kw)

    def _call(self, image_part, user: str, *, logprobs: bool,
              temperature: float, max_tokens: int):
        """generate_content with thinking disabled; retry plain if rejected."""
        last_exc: Exception | None = None
        for disable_thinking in (True, False):
            try:
                cfg = self._gen_config(
                    logprobs=logprobs, temperature=temperature,
                    max_tokens=max_tokens, disable_thinking=disable_thinking,
                )
                return self.client.models.generate_content(
                    model=self.model, contents=[image_part, user], config=cfg,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def slice_confidence(self, image_path: str, region: Region, query: str) -> float:
        user = self._verification_prompt(region, query)
        image_part = self._image_part(image_path)

        lp_yes = lp_no = None
        try:
            resp = self._call(image_part, user, logprobs=True,
                              temperature=0.0, max_tokens=8)
            lp_yes, lp_no = extract_yes_no_logprobs(resp)
        except Exception:  # noqa: BLE001 - model may reject the logprobs request
            if not self.fallback_to_sampling:
                raise

        if lp_yes is None and lp_no is None:
            if self.fallback_to_sampling:
                return self._sample_confidence(image_part, user)
            raise RuntimeError(
                "No logprobs returned. Use a model that supports response_logprobs, "
                "or set fallback_to_sampling=True."
            )
        return confidence_from_logprobs(lp_yes, lp_no)

    def _sample_confidence(self, image_part, user: str) -> float:
        yes = no = 0
        for _ in range(self.n_samples):
            try:
                resp = self._call(image_part, user, logprobs=False,
                                  temperature=self.sampling_temperature, max_tokens=16)
                text = resp.text or ""
            except Exception:  # noqa: BLE001
                continue
            ans = _parse_yes_no(text)
            if ans == "yes":
                yes += 1
            elif ans == "no":
                no += 1
        total = yes + no
        return yes / total if total else 0.5
