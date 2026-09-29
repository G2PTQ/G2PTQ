"""Qwen3 — Llama-shaped dense model (extra q/k head norms are not quantized)."""

from utils.models.llama import LlamaSpec
from utils.models.registry import register_model


@register_model("qwen3")
class Qwen3Spec(LlamaSpec):
    DECODER_LAYER_CLASS = "Qwen3DecoderLayer"
