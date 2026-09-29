"""Gemma4 — dense text decoder with per-layer inputs, KV-shared layers, and logit softcapping.

Block-wise execution can't just replay the decoder layers: ``Gemma4TextModel.forward`` computes
per-layer inputs, a per-layer-type causal mask, and per-layer-type position embeddings before the
stack runs, and KV-shared layers read a cache written by an earlier layer. Those internals are
captured once with a forward pre-hook (:meth:`Gemma4Spec.capture_block_internals`) and reassembled
into each layer's forward kwargs (:meth:`Gemma4Spec.build_block_kwargs`).
"""

import contextlib
from collections import UserDict, defaultdict
from typing import Dict

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM, Gemma4TextModel

from utils import memory_utils, offload_utils
from utils.models.base import ModelSpec, resolve
from utils.models.registry import register_model


# "gemma4" is the multimodal checkpoint's model_type, seen while loading; after `post_load` unwraps
# the text model the analyzer sees "gemma4_text". Both resolve to this spec.
@register_model("gemma4", "gemma4_text")
class Gemma4Spec(ModelSpec):
    DECODER_LAYER_CLASS = "Gemma4TextDecoderLayer"
    # Gemma4 has extra norms in the residual path (pre/post feedforward, per-layer input) that LN
    # fusion does not handle. Disable global rotation while leaving online MLP rotation available.
    CAN_ROTATE_GLOBAL = False
    #: Projections reading the residual stream on ordinary (non-KV-shared) layers.
    ATTN_GROUPS = (
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ("self_attn.o_proj",),
    )
    #: Extra pre-block modules present only when the model has per-layer inputs (E2B/E4B).
    PER_LAYER_INPUT_MODULES = (
        "embed_tokens_per_layer",
        "per_layer_model_projection",
        "per_layer_projection_norm",
    )

    @property
    def _has_per_layer_input(self):
        return bool(self.config.hidden_size_per_layer_input)

    def get_pre_block_modules(self, model):
        modules = super().get_pre_block_modules(model)
        if self._has_per_layer_input:
            modules += [
                resolve(f"{self.MODEL_PREFIX}.{name}", model)
                for name in self.PER_LAYER_INPUT_MODULES
            ]
        return modules

    def get_attn_groups(self, layer):
        if layer.self_attn.is_kv_shared_layer:
            # This layer reuses an earlier layer's K/V, so it has no k_proj / v_proj.
            return [["self_attn.q_proj"], ["self_attn.o_proj"]]
        if layer.self_attn.use_alternative_attention:
            return [["self_attn.q_proj", "self_attn.k_proj"], ["self_attn.o_proj"]]
        return super().get_attn_groups(layer)

    def _shared_input_groups_attn(self, layer):
        """Mirror get_attn_groups: KV-shared has no k/v; alternative attn groups (q,k)."""
        if layer.self_attn.is_kv_shared_layer:
            return []  # q_proj and o_proj are both singletons
        if layer.self_attn.use_alternative_attention:
            return [("self_attn.q_proj", "self_attn.k_proj")]
        return super()._shared_input_groups_attn(layer)

    def get_module_before_final_residual(self, layer):
        if self._has_per_layer_input:
            return resolve("post_per_layer_input_norm", layer)
        return resolve("post_feedforward_layernorm", layer)

    def get_layer_scalar(self, layer):
        """Gemma4 scales the block output before the final residual add."""
        return layer.layer_scalar.item()

    def post_process_logits(self, logits):
        if self.config.final_logit_softcapping is not None:
            logits = LogitSoftcapping.apply(logits, self.config.final_logit_softcapping)
        return logits

    # ---- Loading ---------------------------------------------------------------------------
    @classmethod
    def post_load(cls, model, config):
        """Keep the text model only. TODO. load multimodal model"""
        if not hasattr(config, "text_config"):
            return model  # already a text-only checkpoint
        language_model = Gemma4ForCausalLM(config=config.text_config)
        language_model.model = model.model.language_model
        language_model.lm_head = model.lm_head
        memory_utils.cleanup_memory()
        return language_model

    # ---- Block internals -------------------------------------------------------------------
    def capture_block_internals(self, model):
        return capture_gemma4_internals(model)

    def build_block_kwargs(self, internals, kwargs, *, prev_topk_indices,
                           layer_idx, sample_idx, bsz, dev):
        layer_type = self.config.layer_types[layer_idx]

        if internals["per_layer_inputs"] is not None:
            per_layer_input = (
                internals["per_layer_inputs"][sample_idx // bsz][:, :, layer_idx, :].to(dev)
            )
        else:
            per_layer_input = None

        kwargs["per_layer_input"] = per_layer_input
        kwargs["shared_kv_states"] = _move_shared_kv_states_to_dev(
            internals["shared_kv_states"][sample_idx // bsz], dev
        )
        kwargs["position_embeddings"] = internals["position_embeddings"][layer_type]
        kwargs["attention_mask"] = internals["causal_mask_mapping"][layer_type]

    def finalize_block(self, internals, index_buffer, extras, *, layer_idx, sample_idx, bsz):
        """Push this batch's shared KV cache back to host memory after the layer call."""
        batch_idx = sample_idx // bsz
        internals["shared_kv_states"][batch_idx] = _offload_shared_kv_states(
            internals["shared_kv_states"][batch_idx], internals["kv_staging"], batch_idx
        )


@contextlib.contextmanager
def capture_gemma4_internals(model: Gemma4ForCausalLM, offload_per_layer_inputs=True):
    """
    Capture per_layer_inputs, causal_mask_mapping, and position_embeddings in a Gemma4 model.
    """
    assert isinstance(model, Gemma4ForCausalLM), \
        f"Gemma4 block internals need a Gemma4ForCausalLM, got {type(model).__name__}"

    base_model = model.model

    captured_data = {
        "per_layer_inputs": [] if base_model.hidden_size_per_layer_input else None,
        "causal_mask_mapping": {},
        "position_embeddings": {},
        "shared_kv_states": defaultdict(UserDict),
        # Reusable pinned host buffers the shared KV cache is offloaded into, keyed by
        # `(batch_idx, layer_type)`.
        "kv_staging": {},
    }

    def pre_hook(self: Gemma4TextModel, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        position_ids = kwargs.get("position_ids", args[2] if len(args) > 2 else None)
        past_key_values = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
        inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        per_layer_inputs = kwargs.get("per_layer_inputs", args[5] if len(args) > 5 else None)

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Re-compute per_layer_inputs
        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs).detach()
            if offload_per_layer_inputs:
                per_layer_inputs = per_layer_inputs.cpu()
            captured_data["per_layer_inputs"].append(per_layer_inputs)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0)

        # Re-compute causal_mask_mapping
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        captured_data["causal_mask_mapping"] = {
            k: v.detach() if v is not None else None
            for k, v in causal_mask_mapping.items()
        }

        # Re-compute position_embeddings
        hidden_states = inputs_embeds
        for layer_type in self.unique_layer_types:
            pos_emb = self.rotary_emb(hidden_states, position_ids, layer_type)
            if isinstance(pos_emb, tuple):
                captured_data["position_embeddings"][layer_type] = tuple(t.detach() for t in pos_emb)
            else:
                captured_data["position_embeddings"][layer_type] = pos_emb.detach()

        return args, kwargs

    hook_handle = base_model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    try:
        yield captured_data
    finally:
        hook_handle.remove()


def _move_shared_kv_states_to_dev(shared_kv_states: Dict, dev):
    for layer_type, key_value_states in shared_kv_states.items():
        shared_kv_states[layer_type] = (
            key_value_states[0].to(dev),
            key_value_states[1].to(dev)
        )
    return shared_kv_states


def _offload_shared_kv_states(shared_kv_states: Dict, staging: Dict, batch_idx: int):
    """Push this batch's shared KV cache to host memory without stalling the block forward.

    The tensors are copied into pinned buffers cached in `staging` and reused across layers, so
    the copy can be handed to `offload_utils.current_d2h_queue()` and overlap the next batch's
    compute.
    """
    queue = offload_utils.current_d2h_queue()
    for layer_type, key_value_states in shared_kv_states.items():
        key_value_states = tuple(t.detach() for t in key_value_states)
        buffers = staging.get((batch_idx, layer_type))
        if buffers is None or any(
            buf.shape != src.shape for buf, src in zip(buffers, key_value_states)
        ):
            buffers = tuple(offload_utils.empty_pinned_like(t) for t in key_value_states)
            staging[(batch_idx, layer_type)] = buffers

        for buf, src in zip(buffers, key_value_states):
            queue.copy_(buf, src)
        shared_kv_states[layer_type] = buffers
    return shared_kv_states


class LogitSoftcapping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, softcap):
        out = (logits / softcap).tanh() * softcap

        ctx.save_for_backward(out)
        ctx.softcap = softcap

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        softcap = ctx.softcap

        grad_input = out / softcap
        grad_input.pow_(2)
        grad_input.neg_().add_(1.0)
        grad_input.mul_(grad_output)

        return grad_input, None
