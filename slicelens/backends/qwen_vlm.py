"""Qwen2.5-VL backend (paper default: Qwen2.5-VL-7B-Instruct, Section 5.1.2 / B.1).

This is the faithful implementation of Equation (1): we run a *single forward
pass* over the verification prompt and read the next-token logits, then take a
softmax over only the "yes" and "no" token logits::

    P(yes | q, r) = exp(z_yes) / (exp(z_yes) + exp(z_no))

Requires (run on the GPU server)::

    pip install "transformers>=4.49" accelerate qwen-vl-utils pillow torch

All heavy imports are deferred to construction time.
"""

from __future__ import annotations

import os

from ..data_types import Region
from ..prompts import (
    BBOX_VERIFICATION_PROMPT,
    GROUNDED_CAPTION_SYSTEM_PROMPT,
    GROUNDED_CAPTION_USER_PROMPT,
    POINT_VERIFICATION_PROMPT,
)
from .base import VLMBackend

# Surface forms treated as the affirmative / negative answer. We take the max
# logit within each set, which is robust to tokenizer-specific leading spaces
# and capitalisation.
_YES_FORMS = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
_NO_FORMS = ["no", " no", "No", " No", "NO", " NO"]

# System prompt that nudges the model to answer with a single yes/no token.
_VERIFY_SYSTEM_PROMPT = (
    "You are a visual reasoning assistant. You are given an image and a region "
    "of interest. Decide whether the region matches the given description and "
    "answer with a single word: yes or no."
)


class QwenVLM(VLMBackend):
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        grounding: str = "bbox",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 64,
        load_in_4bit: bool = False,
    ) -> None:
        """Load Qwen2.5-VL.

        Set ``load_in_4bit=True`` to run the 7B model on an ~11 GB GPU (e.g.
        AIRLab's *westworld* 1080Ti nodes) via bitsandbytes NF4 quantization.
        On a 24 GB GPU (e.g. *elysium*'s RTX6000) leave it ``False`` for bf16.
        """
        try:
            import torch  # type: ignore  # noqa: F401
            from transformers import (  # type: ignore
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "transformers>=4.49, accelerate, qwen-vl-utils, pillow and torch "
                "are required for QwenVLM."
            ) from exc

        self._torch = torch
        self.grounding = grounding
        self.max_new_tokens = max_new_tokens

        # SQ_QWEN_DEVICE_MAP=cuda:0 pins every weight on the GPU: with "auto",
        # a few GB taken by another user's process at load time make HF offload
        # part of the model to the CPU ("offloaded to the cpu"), and a verify
        # pass then takes 8 s instead of 1.
        device_map = os.environ.get("SQ_QWEN_DEVICE_MAP", device_map)
        model_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "bitsandbytes is required for load_in_4bit. Install the "
                    "'quant' extra: pip install -e \".[quant]\"."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, **model_kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self._yes_ids = self._token_ids(_YES_FORMS)
        self._no_ids = self._token_ids(_NO_FORMS)
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("Could not resolve yes/no token ids for this tokenizer.")

    # -- helpers ----------------------------------------------------------
    def _token_ids(self, forms: list[str]) -> list[int]:
        tok = self.processor.tokenizer
        ids: set[int] = set()
        for form in forms:
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def _build_inputs(self, image_path: str, system: str, user: str):
        from qwen_vl_utils import process_vision_info  # type: ignore

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _verification_prompt(self, region: Region, query: str) -> str:
        if self.grounding == "point":
            x, y = region.center_point()
            return POINT_VERIFICATION_PROMPT.format(x=round(x, 1), y=round(y, 1), query=query)
        if region.bbox is None:
            x, y = region.center_point()
            return POINT_VERIFICATION_PROMPT.format(x=round(x, 1), y=round(y, 1), query=query)
        xmin, ymin, xmax, ymax = region.bbox
        return BBOX_VERIFICATION_PROMPT.format(
            xmin=round(xmin, 1), ymin=round(ymin, 1),
            xmax=round(xmax, 1), ymax=round(ymax, 1), query=query,
        )

    # -- interface --------------------------------------------------------
    def caption_region(self, image_path: str, region: Region, attribute: str) -> str:
        box = region.bbox if region.bbox is not None else (
            (*region.center_point(), *region.center_point())
        )
        system = GROUNDED_CAPTION_SYSTEM_PROMPT.format(attribute=attribute)
        user = GROUNDED_CAPTION_USER_PROMPT.format(
            xmin=round(box[0], 1), ymin=round(box[1], 1),
            xmax=round(box[2], 1), ymax=round(box[3], 1), attribute=attribute,
        )
        inputs = self._build_inputs(image_path, system, user)
        with self._torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def slice_confidence(self, image_path: str, region: Region, query: str) -> float:
        from ..verification.slice_confidence import softmax_yes_no

        user = self._verification_prompt(region, query)
        inputs = self._build_inputs(image_path, _VERIFY_SYSTEM_PROMPT, user)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        # Logits that predict the first answer token.
        next_logits = outputs.logits[0, -1, :]
        z_yes = float(next_logits[self._yes_ids].max())
        z_no = float(next_logits[self._no_ids].max())
        return softmax_yes_no(z_yes, z_no)
