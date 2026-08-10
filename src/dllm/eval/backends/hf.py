"""Local Transformers backend.

Used for the baseline run and for scoring checkpoints on the training box,
where standing up a vLLM server between every eval would be wasted GPU time.

Two things worth knowing:

* **Left padding.** Decoder-only models must be left-padded for batched
  generation, otherwise the padding sits between the prompt and the first
  generated token and the output is garbage. It is the single most common way
  to get a falsely terrible baseline number.
* **Greedy by default.** ``temperature=0`` maps to ``do_sample=False``. A
  sampled eval is not reproducible, and an irreproducible baseline makes the
  after-number meaningless.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from dllm.config import BackendConfig, GenerationConfig, ModelConfig, QuantConfig
from dllm.eval.backends.base import Backend
from dllm.prompts import Message, render_fallback_chat

log = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> Any:
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return _DTYPES[name]


class HFBackend(Backend):
    name = "hf"

    def __init__(
        self,
        model_cfg: ModelConfig,
        backend_cfg: BackendConfig,
        *,
        model_path: str | None = None,
        adapter_path: str | None = None,
        quant: QuantConfig | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_cfg = model_cfg
        self.cfg = backend_cfg
        self.model_path = model_path or model_cfg.id
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            revision=model_cfg.revision if model_path is None else None,
            trust_remote_code=model_cfg.trust_remote_code,
        )
        # Batched generation on a decoder-only model requires left padding.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "dtype": resolve_dtype(model_cfg.dtype),
            "trust_remote_code": model_cfg.trust_remote_code,
            "attn_implementation": model_cfg.attn_implementation,
        }
        if model_path is None:
            load_kwargs["revision"] = model_cfg.revision

        if quant and quant.enabled:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quant.bits == 4,
                load_in_8bit=quant.bits == 8,
                bnb_4bit_quant_type=quant.quant_type,
                bnb_4bit_use_double_quant=quant.double_quant,
                bnb_4bit_compute_dtype=_DTYPES[quant.compute_dtype],
            )
            load_kwargs["device_map"] = "auto"
        elif backend_cfg.device == "auto":
            load_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            log.info("attaching LoRA adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()

    # ---------------------------------------------------------------- #
    def _render(self, messages: list[Message]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return render_fallback_chat(messages)

    @torch.inference_mode()
    def generate(self, conversations: list[list[Message]], gen: GenerationConfig) -> list[str]:
        outputs: list[str] = []
        bs = max(1, self.cfg.batch_size)
        device = next(self.model.parameters()).device

        for start in range(0, len(conversations), bs):
            batch = conversations[start : start + bs]
            texts = [self._render(c) for c in batch]
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_cfg.max_seq_len,
                add_special_tokens=False,
            ).to(device)

            kwargs: dict[str, Any] = {
                "max_new_tokens": gen.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if gen.stop:
                # Stop early rather than always generating max_new_tokens and
                # truncating afterwards. A batch stops when every sequence in it
                # has hit a stop string, so _apply_stops still runs below.
                kwargs["stop_strings"] = list(gen.stop)
                kwargs["tokenizer"] = self.tokenizer
            if gen.temperature > 0:
                kwargs.update(do_sample=True, temperature=gen.temperature, top_p=gen.top_p)
            else:
                kwargs["do_sample"] = False

            generated = self.model.generate(**enc, **kwargs)
            # Slice off the prompt: with left padding every sequence in the
            # batch shares the same prompt width.
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend(_apply_stops(d, gen.stop) for d in decoded)

            log.info("generated %d/%d", min(start + bs, len(conversations)), len(conversations))

        return outputs

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_path": self.model_path,
            "base_model_id": self.model_cfg.id,
            "revision": self.model_cfg.revision,
            "adapter_path": self.adapter_path,
            "dtype": self.model_cfg.dtype,
            "device": str(next(self.model.parameters()).device),
            "torch": torch.__version__,
        }

    def close(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _apply_stops(text: str, stops: list[str]) -> str:
    """Truncate at the earliest stop string. Mirrors the server's stop handling."""
    cut = len(text)
    for s in stops:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()
