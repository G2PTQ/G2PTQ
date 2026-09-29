"""Llama — the canonical dense spec. Other dense archs subclass it."""

from utils.models.base import ModelSpec
from utils.models.registry import register_model


@register_model("llama")
class LlamaSpec(ModelSpec):
    DECODER_LAYER_CLASS = "LlamaDecoderLayer"
    CAN_ROTATE_OV = True
