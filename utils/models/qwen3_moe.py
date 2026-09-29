"""Qwen3-MoE — Qwen3 attention with a routed expert MLP and no shared expert."""

from utils.models.mixins import MoEMixin, SoftmaxTopkGateMixin
from utils.models.qwen3 import Qwen3Spec
from utils.models.registry import register_model


@register_model("qwen3_moe")
class Qwen3MoeSpec(MoEMixin, SoftmaxTopkGateMixin, Qwen3Spec):
    DECODER_LAYER_CLASS = "Qwen3MoeDecoderLayer"
    # TODO. temporary: borrow qwen2_moe's pattern so experts load directly as 2D per-expert
    # weights (transformers==5.12.1 has no 2D mapping registered for this arch).
    CONVERSION_PATTERN = "qwen2_moe"
