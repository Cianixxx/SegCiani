"""Generic HuggingFace VLM backend -- run *any* open multimodal model locally.

Uses the unified ``AutoModelForImageTextToText`` + ``AutoProcessor`` API, so the
same backend works for Gemma 3/4, Qwen2.5-VL, LLaVA, InternVL, ... -- you just
pass the HF model id. The slice confidence (Eq. 1) is read from the logits of the
"yes"/"no" output tokens, exactly like :class:`QwenVLM`, so any of these models
gives a *continuous* confidence (unlike the API sampling fallback).

This makes "which VLM performs better?" an experiment: swap ``--vlm-model`` and
compare with the identical harness.

Requires (on the GPU server)::

    pip install "transformers>=4.49" accelerate pillow torch   # + the model's deps

Notes / caveats:
* The model must be **multimodal** (vision) -- a text-only model will fail.
* SliceLens grounds the question on a region (bbox/point in the prompt). Models
  that are *grounding-tuned* (e.g. Qwen2.5-VL, PaliGemma) reason about the region
  better than general multimodal models; verify grounding quality empirically.
"""

from __future__ import annotations

from ..data_types import Region
from ..prompts import (
    BBOX_VERIFICATION_PROMPT,
    CROP_VERIFICATION_PROMPT,
    GROUNDED_CAPTION_SYSTEM_PROMPT,
    GROUNDED_CAPTION_USER_PROMPT,
    POINT_VERIFICATION_PROMPT,
)
from .base import VLMBackend

_YES_FORMS = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
_NO_FORMS = ["no", " no", "No", " No", "NO", " NO"]

_VERIFY_SYSTEM_PROMPT = (
    "You are a visual reasoning assistant. You are given an image and a region "
    "of interest. Decide whether the region matches the given description and "
    "answer with a single word: yes or no."
)


class HFVLM(VLMBackend):
    def __init__(
        self,
        model: str,
        grounding: str = "bbox",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 64,
        load_in_4bit: bool = False,
        skip_quant_modules: list[str] | None = None,
    ) -> None:
        try:
            import torch  # type: ignore  # noqa: F401
            from transformers import (  # type: ignore
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "transformers>=4.49, accelerate, pillow and torch are required "
                "for HFVLM."
            ) from exc

        self._torch = torch
        self.grounding = grounding
        self.max_new_tokens = max_new_tokens

        model_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            # Keep the vision tower / projector / lm_head in full precision: 4-bit
            # quantizing a multimodal model's vision encoder makes pixel_values get
            # cast to the packed (uint8) weight dtype and breaks LayerNorm
            # ("LayerNormKernelImpl not implemented for 'Byte'").
            skip = skip_quant_modules or [
                # generic VLM vision/projector names ...
                "vision_tower", "vision_model", "embed_vision", "vision_embedder",
                "multi_modal_projector", "multimodal_embedder",
                # ... + the exact Gemma 4 vision leaf modules (from Google's QAT
                # config ignore list) that break under 4-bit (LayerNorm on Byte).
                "patch_dense", "patch_ln1", "patch_ln2", "pos_norm",
                "embedding_projection", "lm_head",
            ]
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=skip,
            )
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model)
        tok = getattr(self.processor, "tokenizer", self.processor)
        self._tok = tok
        self._yes_ids = self._token_ids(_YES_FORMS)
        self._no_ids = self._token_ids(_NO_FORMS)
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError("Could not resolve yes/no token ids for this tokenizer.")

    # -- helpers ----------------------------------------------------------
    def _token_ids(self, forms: list[str]) -> list[int]:
        ids: set[int] = set()
        for form in forms:
            enc = self._tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    def _build_inputs(self, image_path: str, system: str, user: str):
        from PIL import Image  # type: ignore

        image = (Image.open(image_path).convert("RGB")
                 if isinstance(image_path, str) else image_path.convert("RGB"))
        # Fold the system instruction into the user turn: some chat templates
        # (e.g. Gemma) do not support a separate "system" role.
        text = f"{system}\n\n{user}" if system else user
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image},
                        {"type": "text", "text": text}],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self.model.device)

    def _crop_region(self, image_path: str, region: Region, margin: float = 0.4):
        """Crop the image to the region's bbox, expanded by ``margin`` on each
        side so some surrounding context (background, occluders) is retained.

        This is the grounding strategy for VLMs that are *not* grounding-tuned:
        instead of trusting them to localise a text bbox, we hand them the
        cropped object directly.
        """
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        if region.bbox is None:
            return img
        w, h = img.size
        x0, y0, x1, y1 = region.bbox
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        box = (max(0, int(x0 - mx)), max(0, int(y0 - my)),
               min(w, int(x1 + mx)), min(h, int(y1 + my)))
        if box[2] <= box[0] or box[3] <= box[1]:
            return img
        return img.crop(box)

    def _verification_prompt(self, region: Region, query: str) -> str:
        if self.grounding == "point" or region.bbox is None:
            x, y = region.center_point()
            return POINT_VERIFICATION_PROMPT.format(
                x=round(x, 1), y=round(y, 1), query=query)
        xmin, ymin, xmax, ymax = region.bbox
        return BBOX_VERIFICATION_PROMPT.format(
            xmin=round(xmin, 1), ymin=round(ymin, 1),
            xmax=round(xmax, 1), ymax=round(ymax, 1), query=query)

    # -- interface --------------------------------------------------------
    def caption_region(self, image_path: str, region: Region, attribute: str) -> str:
        box = region.bbox if region.bbox is not None else (
            (*region.center_point(), *region.center_point()))
        system = GROUNDED_CAPTION_SYSTEM_PROMPT.format(attribute=attribute)
        user = GROUNDED_CAPTION_USER_PROMPT.format(
            xmin=round(box[0], 1), ymin=round(box[1], 1),
            xmax=round(box[2], 1), ymax=round(box[3], 1), attribute=attribute)
        inputs = self._build_inputs(image_path, system, user)
        in_len = inputs["input_ids"].shape[1]
        with self._torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        trimmed = generated[:, in_len:]
        return self._tok.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def describe(self, image_path: str, prompt: str, system: str | None = None,
                 max_new_tokens: int = 320) -> str:
        """Free-form: an image + a text prompt -> generated text.

        Not region-grounded -- used e.g. to caption/analyse a whole image (the
        Pascal-VOC 3-panel segmentation-error description task).
        """
        inputs = self._build_inputs(image_path, system or "", prompt)
        in_len = inputs["input_ids"].shape[1]
        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = generated[:, in_len:]
        return self._tok.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def slice_confidence(self, image_path: str, region: Region, query: str) -> float:
        from ..verification.slice_confidence import softmax_yes_no

        if self.grounding == "crop":
            image = self._crop_region(image_path, region)
            user = CROP_VERIFICATION_PROMPT.format(query=query)
        else:
            image = image_path
            user = self._verification_prompt(region, query)
        inputs = self._build_inputs(image, _VERIFY_SYSTEM_PROMPT, user)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        next_logits = outputs.logits[0, -1, :]
        z_yes = float(next_logits[self._yes_ids].max())
        z_no = float(next_logits[self._no_ids].max())
        return softmax_yes_no(z_yes, z_no)
