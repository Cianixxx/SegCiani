"""Local Hugging Face LLM backend for hypothesis generation (SliceLens stage 1).

Lets a local instruction model (e.g. Gemma 4 12B already on the GPU server) play
the LLM role, so the whole pipeline runs offline -- no Gemini / cloud needed.

The paper's ablation (Appendix B) shows the choice of LLM is nearly immaterial
for slice quality (0.69 / 0.68 / 0.67 across Gemini-2.5-Pro / GPT-4o /
Claude-3.5-Sonnet), so a strong *local* model is a faithful substitute for the
generation stage; the grounded verification (Qwen2.5-VL, Eq. 1) is unchanged.

Text-only chat: the multimodal checkpoint is driven with no image. Gemma has no
"system" role, so the system instruction is folded into the user turn.
"""

from __future__ import annotations

from .base import LLMBackend

# Same vision-leaf skip list as HFVLM: 4-bit quantizing a multimodal model's
# vision/LayerNorm leaves breaks ("LayerNormKernelImpl not implemented for Byte").
_SKIP_QUANT = [
    "vision_tower", "vision_model", "embed_vision", "vision_embedder",
    "multi_modal_projector", "multimodal_embedder",
    "patch_dense", "patch_ln1", "patch_ln2", "pos_norm",
    "embedding_projection", "lm_head",
]


class HFLLM(LLMBackend):
    def __init__(
        self,
        model: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 2048,
        load_in_4bit: bool = False,
    ) -> None:
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "transformers>=4.49, accelerate, pillow and torch are required "
                "for HFLLM."
            ) from exc

        self._torch = torch
        self.max_new_tokens = max_new_tokens

        model_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=list(_SKIP_QUANT),
            )
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForImageTextToText.from_pretrained(model, **model_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self._tok = getattr(self.processor, "tokenizer", self.processor)

    def chat(self, system: str, user: str) -> str:
        # Gemma's chat template has no system role -> fold it into the user turn.
        text = f"{system}\n\n{user}" if system else user
        messages = [{"role": "user", "content": text}]
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)
        in_len = inputs["input_ids"].shape[1]
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        return self._tok.batch_decode(
            out[:, in_len:], skip_special_tokens=True)[0].strip()
