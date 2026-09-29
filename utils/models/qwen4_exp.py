"""Qwen4-Exp (Qwen3.8-Flash-Next) — MoE + QSA/GDN on a gated multi-stream residual with n-gram embed."""

import contextlib
from types import MethodType

import torch
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.models.qwen4_exp import Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextAttention
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from utils.models.base import ModelSpec, resolve
from utils.models.mixins import LinearAttentionMixin, MoEMixin, SoftmaxTopkGateMixin
from utils.models.registry import register_model


@register_model("qwen4_exp", "qwen4_exp_text")
class Qwen4ExpSpec(MoEMixin, SoftmaxTopkGateMixin, LinearAttentionMixin, ModelSpec):
    DECODER_LAYER_CLASS = "Qwen4ExpTextDecoderLayer"
    MODEL_PREFIX = "model.language_model"
    IS_MULTIMODAL = True
    CAN_ROTATE_GLOBAL = False
    #: ``Qwen4ExpPreTrainedModel._supports_flash_attn = False`` (the indexer needs the dense mask).
    SUPPORTS_FLASH_ATTN = False
    CONVERSION_PATTERN = "qwen2_moe"
    SHARED_EXPERT_NAME = "mlp.shared_expert"
    # The hashed n-gram table is kept on CPU and is not part of the quantization optimization.
    # Keep the rest of each PLE layer trainable for gradient-based calibration.
    NO_GRAD_PARAM_PREFIXES = ("ple.ple_embedding",)

    # ---- Layout ----------------------------------------------------------------------------
    @property
    def residual_width(self):
        """GR carries ``hc_count`` parallel residual streams between blocks."""
        return self.config.hc_count * self.config.hidden_size

    def get_layernorm_before_head(self, model):
        """No final norm exists; ``hyper_connection_mixer`` reduces 10240 -> 2560 in its place.

        Callers apply this to the final hidden states and run ``lm_head`` separately.
        """
        return resolve(f"{self.MODEL_PREFIX}.hyper_connection_mixer", model)

    # ---- Block internals -------------------------------------------------------------------
    def capture_block_internals(self, model):
        return capture_qwen4_exp_internals(resolve(self.MODEL_PREFIX, model))

    def build_block_kwargs(self, internals, kwargs, *, prev_topk_indices,
                           layer_idx, sample_idx, bsz, dev):
        # The decoder layer takes position_embeddings only.
        kwargs.pop("position_ids", None)

        kwargs["position_embeddings"] = internals["position_embeddings"]
        kwargs["conv_mask"] = internals["conv_mask"]
        # Both masks go to every layer exactly as Qwen4ExpTextModel.forward passes them; the
        # decoder layer itself dispatches on self.layer_type, so no per-layer selection here.
        kwargs["attention_mask"] = internals["causal_mask_mapping"]["full_attention"]

        ple_input_ids = internals["ple_input_ids"]
        ple_layer_index = self.config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in self.config.ple_layer_ids else None
        if ple_layer_index is not None and ple_input_ids is not None:
            kwargs["ple_input_ids"] = ple_input_ids[sample_idx // bsz].to(dev)

    @contextlib.contextmanager
    def _enter_quantization_context(self, args, model):
        """Temporarily bypass QSA indexers for an entirely within-budget calibration pass.
        """
        calibration_seq_len = args.seq_len
        if calibration_seq_len > self.config.indexer_budget:
            yield
            return

        patched = []
        try:
            for layer in self.get_layers(model):
                if layer.layer_type != "qwen_sparse_attention":
                    continue
                attention = layer.self_attn
                assert isinstance(attention, Qwen4ExpTextAttention)

                original_forward = attention.forward
                attention.forward = MethodType(
                    _qwen4_exp_text_attention_forward_without_indexer,
                    attention,
                )
                patched.append((attention, original_forward))
            yield
        finally:
            for attention, original_forward in reversed(patched):
                attention.forward = original_forward

    # ---- Loading ---------------------------------------------------------------------------
    @classmethod
    def load_model_class(cls, config):
        return Qwen4ExpForConditionalGeneration


# Mirrors Qwen4ExpTextAttention.forward from transformers==5.16.1,
# except that the QSA indexer and selected-mask overlay are intentionally omitted.
def _qwen4_exp_text_attention_forward_without_indexer(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    if past_key_values is not None:
        raise RuntimeError("Qwen4-Exp indexer bypass does not support cached attention calls")
    if attention_mask is None:
        raise RuntimeError("Qwen4-Exp indexer bypass requires a materialized attention mask")
    if attention_mask.shape[-1] > self.indexer.token_budget:
        raise RuntimeError(
            "Qwen4-Exp indexer bypass received an attention mask longer than indexer_budget: "
            f"{attention_mask.shape[-1]} > {self.indexer.token_budget}"
        )

    position_embeddings = tuple(
        x[:, -hidden_states.shape[1]:, :] for x in position_embeddings
    )
    return Qwen3_5Attention.forward(
        self,
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        **kwargs,
    )


@contextlib.contextmanager
def capture_qwen4_exp_internals(base_model):
    """Capture the per-forward internals ``Qwen4ExpTextModel.forward`` builds before its stack.

    Block-wise execution replays the decoder layers one at a time, so the position embeddings (built
    from the 3D position_ids the indexer needs), both masks, and the eos-rewritten ``ple_input_ids``
    have to be reproduced here. ``ple_input_ids`` is per-batch and kept on the host; the rest is
    shared across batches.

    ``base_model`` is the ``Qwen4ExpTextModel`` (resolved through the spec's ``MODEL_PREFIX``).

    The masks and position embeddings are batch-shaped and captured once, then reused for every
    batch — the same uniform-``bsz`` assumption the drivers already make for every other arch (they
    cache one ``attention_mask`` / ``position_embeddings`` from the first block's kwargs). Here the
    indexer additionally indexes ``cos``/``sin`` by batch row, so a batch-1 capture cannot be
    broadcast in their place. Keep ``nsamples`` a multiple of ``--bsz``.
    """
    assert base_model is not None, "qwen4_exp block internals need the text model"

    captured = {
        "position_embeddings": None,
        "causal_mask_mapping": {},
        "conv_mask": None,
        "ple_input_ids": [] if base_model.config.ple_layer_ids else None,
    }

    def pre_hook(self, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
        past_key_values = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        ple_input_ids = kwargs.get("ple_input_ids")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.config.ple_layer_ids and ple_input_ids is None:
            ple_input_ids = input_ids if input_ids is not None else self.reverse_embedding(inputs_embeds)

        # position_ids are the *full* positions, 3D, as the indexer needs full position embeddings.
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        elif position_ids.shape[0] == 1:
            text_position_ids = position_ids[0]
            position_ids = position_ids.expand(3, -1, -1)
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
                # The indexer asserts the mask is never None and overlays its own selection on it,
                # so the sdpa is-causal skip must stay disabled.
                "allow_is_causal_skip": False,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
        captured["causal_mask_mapping"] = {
            k: v.detach() if v is not None else None for k, v in causal_mask_mapping.items()
        }

        # Legitimately None for the dense, unpadded calibration batches.
        conv_mask = captured["causal_mask_mapping"].get("linear_attention")
        captured["conv_mask"] = conv_mask
        if self.config.ple_layer_ids and conv_mask is not None:
            eos_token_id = self.config.eos_token_id
            eos_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            ple_input_ids = torch.where(conv_mask.bool(), ple_input_ids, eos_token_id)
        if captured["ple_input_ids"] is not None:
            captured["ple_input_ids"].append(ple_input_ids.detach().cpu())

        pos_emb = self.rotary_emb(inputs_embeds, position_ids)
        captured["position_embeddings"] = (
            tuple(t.detach() for t in pos_emb) if isinstance(pos_emb, tuple) else pos_emb.detach()
        )
        return args, kwargs

    handle = base_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()
