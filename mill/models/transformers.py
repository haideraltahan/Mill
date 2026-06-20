"""HuggingFace Transformers model backend.

Supports text-only and multimodal (vision/audio) models via AutoModelForCausalLM
and AutoProcessor. Multimodal requests carry ChatMessages in Instance.arguments[0].
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from mill.api.model import MillModel, ModelCapabilities
from mill.api.registry import register_model
from mill.models.base import is_multimodal_request

if TYPE_CHECKING:
    from mill.api.instance import Instance

logger = logging.getLogger(__name__)


@register_model("hf", "huggingface", "transformers")
class TransformersModel(MillModel):
    """HuggingFace AutoModel backend.

    Config dict fields (opencompass style):
        path (str): HuggingFace model ID or local path.
        modalities (list[str]): e.g. ["text", "image", "video"].
        dtype (str): "bfloat16" | "float16" | "float32". Default "bfloat16".
        device_map (str): "auto" | "cuda" | "cpu". Default "auto".
        max_context_length (int): Token budget. Default 4096.
        batch_size (int): Samples per forward pass. Default 8.
        attn_implementation (str | None): "flash_attention_2" | "sdpa" | None.
        trust_remote_code (bool): Default True.
        use_chat_template (bool): Wrap prompts with the tokenizer's chat template
            before generation. Default False (raw prompt passed as-is).
    """

    def __init__(
        self,
        path: str,
        modalities: list[str] | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        max_context_length: int = 4096,
        batch_size: int | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        use_chat_template: bool = False,
        **kwargs,
    ):
        self._path = path
        self._batch_size = batch_size  # None = auto; int = fixed
        self._use_chat_template = use_chat_template
        self.capabilities = ModelCapabilities(
            modalities=set(modalities or ["text"]),
            max_context_length=max_context_length,
            supports_logprobs=True,
            supports_chat_template=True,
        )

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        model_kwargs: dict = {
            "dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        is_multimodal = bool(set(modalities or []) - {"text"}) if modalities else False

        if is_multimodal:
            from transformers import AutoProcessor, AutoModelForCausalLM
            self._processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
            self._model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
            self._tokenizer = self._processor.tokenizer
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
            self._processor = None
            self._model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._tokenizer.truncation_side = "left"

        self._model.eval()
        logger.info(f"Loaded {path} ({dtype}, device_map={device_map})")

    @property
    def model_name(self) -> str:
        return self._path

    @property
    def batch_size(self) -> int | None:
        return self._batch_size

    @property
    def auto_batch_size(self) -> bool:
        return self._batch_size is None

    def _estimate_starting_batch_size(self) -> int:
        try:
            if not torch.cuda.is_available():
                from mill.constants import FALLBACK_STARTING_BS
                return FALLBACK_STARTING_BS
            free_bytes, _ = torch.cuda.mem_get_info()
            num_params = self._model.num_parameters()
            _BPP = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1}
            bpp = _BPP.get(next(self._model.parameters()).dtype, 2)
            model_bytes = num_params * bpp
            ctx_scale = max(1, self.max_length) / 2048
            per_sample_bytes = model_bytes * ctx_scale / 16
            bs = max(1, int(free_bytes * 0.4 / per_sample_bytes))
            bs = 1 << max(0, bs.bit_length() - 1)
            return min(bs, 512)
        except Exception:
            return super()._estimate_starting_batch_size()

    # ── MillModel hooks ───────────────────────────────────────────────────────

    def _generate_batch(self, batch: list["Instance"], gen_kwargs: dict) -> list[str]:
        max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
        stop_seqs = gen_kwargs.get("stop", [])
        with torch.inference_mode():
            if is_multimodal_request(batch[0]):
                return self._decode_multimodal(batch, max_new_tokens)
            return self._decode_text(batch, max_new_tokens, stop_seqs)

    def _loglikelihood_batch(self, batch: list["Instance"]) -> list[tuple[float, bool]]:
        out = []
        for req in batch:
            context, continuation = req.arguments[0], req.arguments[1]
            ctx_enc = self._tokenizer(context, return_tensors="pt", add_special_tokens=True)
            cont_enc = self._tokenizer(continuation, return_tensors="pt", add_special_tokens=False)

            inp = torch.cat([ctx_enc["input_ids"], cont_enc["input_ids"]], dim=1).to(self._model.device)
            ctx_len = ctx_enc["input_ids"].shape[1]

            with torch.inference_mode():
                logits = self._model(inp).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = inp[:, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            cont_log_probs = log_probs[:, ctx_len - 1:, :]
            cont_labels = shift_labels[:, ctx_len - 1:]

            gathered = cont_log_probs.gather(2, cont_labels.unsqueeze(-1)).squeeze(-1)
            total_log_prob = gathered.sum().item()
            is_greedy = (shift_logits[:, ctx_len - 1:, :].argmax(-1) == cont_labels).all().item()
            out.append((total_log_prob, bool(is_greedy)))
        return out

    def _loglikelihood_rolling_single(self, request: "Instance") -> float:
        text = request.arguments[0]
        enc = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            logits = self._model(**enc).logits
        shift = logits[:, :-1, :].float()
        labels = enc["input_ids"][:, 1:]
        log_probs = torch.nn.functional.log_softmax(shift, dim=-1)
        gathered = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        n_tokens = labels.shape[1]
        return (gathered.sum() / n_tokens).item()

    # ── Decode helpers ────────────────────────────────────────────────────────

    def _decode_text(self, batch: list["Instance"], max_new_tokens: int, stop_seqs: list[str] | None = None) -> list[str]:
        contexts = [req.arguments[0] for req in batch]
        if self._use_chat_template:
            contexts = [
                self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": ctx}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ctx in contexts
            ]
        enc = self._tokenizer(contexts, return_tensors="pt", padding=True, truncation=True,
                              max_length=self.max_length).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        decoded = self._tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
        if stop_seqs:
            decoded = [_truncate_at_stop(text, stop_seqs) for text in decoded]
        return decoded

    def _decode_multimodal(self, batch: list["Instance"], max_new_tokens: int) -> list[str]:
        results = []
        for req in batch:
            chat_msgs = req.arguments[0]
            hf_messages = chat_msgs.to_hf_messages()
            if self._use_chat_template:
                text = self._processor.apply_chat_template(hf_messages, add_generation_prompt=True, tokenize=False)
            else:
                text = "\n".join(
                    f"{m['role']}: {m['content']}" for m in hf_messages
                )
            images, videos, audios = chat_msgs.extract_media()

            proc_kwargs: dict = {"text": text, "return_tensors": "pt"}
            if images:
                proc_kwargs["images"] = images
            if videos:
                proc_kwargs["videos"] = videos
            if audios:
                proc_kwargs["audios"] = audios

            inputs = self._processor(**proc_kwargs).to(self._model.device)
            with torch.inference_mode():
                out = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            results.append(self._tokenizer.decode(out[0][input_len:], skip_special_tokens=True))
        return results

    def cleanup(self) -> None:
        del self._model
        if self._processor:
            del self._processor
        torch.cuda.empty_cache()


def _truncate_at_stop(text: str, stop_seqs: list[str]) -> str:
    for seq in stop_seqs:
        idx = text.find(seq)
        if idx != -1:
            text = text[:idx]
    return text
