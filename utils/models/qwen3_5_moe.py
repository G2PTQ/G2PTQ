"""Qwen3.5-MoE — Qwen3.5 attention with routed experts plus a gated shared expert."""

from transformers.models.qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from utils.models.mixins import MoEMixin, SoftmaxTopkGateMixin
from utils.models.qwen3_5 import Qwen3_5Spec
from utils.models.registry import register_model


@register_model("qwen3_5_moe")
class Qwen3_5MoeSpec(MoEMixin, SoftmaxTopkGateMixin, Qwen3_5Spec):
    DECODER_LAYER_CLASS = "Qwen3_5MoeDecoderLayer"
    # TODO. temporary: borrow qwen2_moe's pattern so experts load directly as 2D per-expert
    # weights (transformers==5.12.1 has no 2D mapping registered for this arch).
    CONVERSION_PATTERN = "qwen2_moe"
    SHARED_EXPERT_NAME = "mlp.shared_expert"
    # The shared expert's own sigmoid gate also reads the post-LN MLP input.
    EXTRA_MLP_INPUTS = ("mlp.shared_expert_gate",)

    @classmethod
    def load_model_class(cls, config):
        return Qwen3_5MoeForConditionalGeneration
