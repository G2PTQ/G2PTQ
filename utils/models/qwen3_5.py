"""Qwen3.5 — multimodal wrapper around a text decoder that interleaves full and linear attention."""

from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from utils.models.base import ModelSpec
from utils.models.mixins import LinearAttentionMixin
from utils.models.registry import register_model


@register_model("qwen3_5")
class Qwen3_5Spec(LinearAttentionMixin, ModelSpec):
    DECODER_LAYER_CLASS = "Qwen3_5DecoderLayer"
    # The decoder stack sits under the multimodal wrapper's language model.
    MODEL_PREFIX = "model.language_model"
    # Qwen3_5RMSNorm stores its weight as an offset from 1: `x * (1 + w)`.
    LN_ZERO_CENTERED = True
    IS_MULTIMODAL = True

    @classmethod
    def load_model_class(cls, config):
        return Qwen3_5ForConditionalGeneration
