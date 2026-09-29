"""GLM-5.2 — MLA attention with a DeepSeek-style sparse-attention indexer and routed experts.

Two features drive the overrides here:

- **Mixed dense/sparse MLP.** ``config.mlp_layer_types`` makes the first few blocks dense, so
  ``get_experts`` returns ``None`` for them and :class:`MoEMixin` falls through to the dense layout.
- **Cross-layer top-k sharing (DSA).** Layers with ``config.indexer_types[i] == "full"`` run their
  own indexer and return the top-k token indices as a second output; ``"shared"`` layers consume the
  previous full layer's indices via the ``prev_topk_indices`` forward kwarg. Block-wise execution
  therefore has to carry those indices between layers itself, in a per-sample buffer.
"""

import torch

from utils import offload_utils
from utils.models.base import ModelSpec, resolve
from utils.models.mixins import LogitsOnlyGateMixin, MoEMixin
from utils.models.registry import register_model


@register_model("glm_moe_dsa")
class GlmMoeDsaSpec(MoEMixin, LogitsOnlyGateMixin, ModelSpec):
    DECODER_LAYER_CLASS = "GlmMoeDsaDecoderLayer"
    USES_DSA_INDEXSHARE = True
    # GlmMoeDsaPreTrainedModel sets `_supports_flash_attn = False` — DSA attention wants flash-mla.
    SUPPORTS_FLASH_ATTN = False
    # MLA caches a joint low-rank latent, not separate K/V, and has no `v_proj` for the output
    # quantizer to attach to, so --k_bits / --v_bits cannot be expressed here.
    SUPPORTS_KV_QUANT = False
    SHARED_EXPERT_NAME = "mlp.shared_experts"
    # TODO. temporary: borrow qwen2_moe's pattern so experts load directly as 2D per-expert
    # weights (transformers==5.12.1 has no 2D mapping registered for this arch).
    CONVERSION_PATTERN = "qwen2_moe"

    # MLA: the low-rank q/kv projections and o_proj all quantize in one sequential group.
    ATTN_GROUPS = (
        (
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_b_proj",
            "self_attn.o_proj",
        ),
    )
    # Only the two projections reading the residual stream directly take the input rotation;
    # q_b_proj / kv_b_proj sit behind their own LayerNorms.
    ATTN_INPUTS = ("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa")
    #: Indexer projections that also read the post-LN residual stream, on "full" layers only.
    INDEXER_INPUTS = ("self_attn.indexer.wk", "self_attn.indexer.weights_proj")
    # MLA puts all five projections in one sequential group, but only the two reading the residual
    # stream directly share an input tensor.
    ATTN_SHARED_INPUTS = (("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"),)

    def get_attn_inputs(self, layer):
        names = list(self.ATTN_INPUTS)
        if getattr(layer.self_attn, "indexer", None) is not None:
            names += list(self.INDEXER_INPUTS)
        return [resolve(name, layer) for name in names]

    # ---- DSA cross-layer index sharing -----------------------------------------------------
    def alloc_index_buffer(self, nsamples, seqlen, device):
        return offload_utils.alloc_pinned(
            (nsamples, seqlen, self.config.index_topk), dtype=torch.int32, device=device, pin=False,
        )

    def build_block_kwargs(self, internals, kwargs, *, prev_topk_indices,
                           layer_idx, sample_idx, bsz, dev):
        """Feed a "shared" layer the indices the last "full" layer computed."""
        kwargs["prev_topk_indices"] = prev_topk_indices

    def finalize_block(self, internals, index_buffer, extras, *, layer_idx, sample_idx, bsz):
        """Record a "full" layer's fresh indices for the "shared" layers that follow."""
        if index_buffer is None or self.config.indexer_types[layer_idx] != "full":
            return
        topk_indices = extras[0]
        # `topk_indices` is `[B, S, min(index_topk, T)]`; with `seqlen > index_topk`
        # the last dim equals `index_topk`, otherwise only the leading slice is written.
        width = topk_indices.shape[-1]
        # Stream-ordered so the copy overlaps the next batch's compute; the enclosing
        # `prefetch_generator` loop drains the queue before anything reads the buffer back.
        offload_utils.current_d2h_queue().copy_(
            index_buffer[sample_idx: sample_idx + bsz, :, :width], topk_indices
        )
