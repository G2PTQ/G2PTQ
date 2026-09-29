"""Reusable spec fragments shared by more than one architecture."""

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models.base import resolve


class MoEMixin:
    """Expands the MLP-side module lists over a layer's experts.

    Layers whose MLP is dense (``get_experts`` returns ``None``, e.g. ``glm_moe_dsa``'s first few
    blocks) fall through to the dense layout on the base spec, so an arch with a mix of dense and
    sparse blocks needs no extra branching.
    """

    IS_MOE = True

    #: Layer-relative path to the experts container.
    EXPERTS_NAME = "mlp.experts"
    #: Layer-relative path to the router.
    GATE_NAME = "mlp.gate"
    #: Layer-relative path to the always-on shared expert, or ``None`` if the arch has none.
    SHARED_EXPERT_NAME = None
    #: Extra layer-relative modules reading the post-LN MLP input (e.g. a shared-expert gate).
    EXTRA_MLP_INPUTS = ()

    def get_experts(self, layer):
        return resolve(self.EXPERTS_NAME, layer)

    def get_gate(self, layer):
        return resolve(self.GATE_NAME, layer)

    def _expert_names(self, layer, proj):
        experts = self.get_experts(layer)
        return [f"{self.EXPERTS_NAME}.{i}.{proj}" for i in range(experts.num_experts)]

    def get_mlp_groups(self, layer):
        if self.get_experts(layer) is None:
            return super().get_mlp_groups(layer)
        return [
            self._expert_names(layer, "up_proj") + self._expert_names(layer, "gate_proj"),
            self._expert_names(layer, "down_proj"),
        ]

    def get_mlp_inputs(self, layer):
        if self.get_experts(layer) is None:
            return super().get_mlp_inputs(layer)
        names = self._expert_names(layer, "up_proj") + self._expert_names(layer, "gate_proj")
        names.append(self.GATE_NAME)
        if self.SHARED_EXPERT_NAME is not None:
            names += [
                f"{self.SHARED_EXPERT_NAME}.gate_proj",
                f"{self.SHARED_EXPERT_NAME}.up_proj",
            ]
        names += list(self.EXTRA_MLP_INPUTS)
        return [resolve(name, layer) for name in names]

    def get_mlp_outputs(self, layer):
        if self.get_experts(layer) is None:
            return super().get_mlp_outputs(layer)
        names = self._expert_names(layer, "down_proj")
        if self.SHARED_EXPERT_NAME is not None:
            names.append(f"{self.SHARED_EXPERT_NAME}.down_proj")
        return [resolve(name, layer) for name in names]

    def _shared_input_groups_mlp(self, layer):
        """One group per expert: an expert's up/gate see the same routed tokens.
        """
        if self.get_experts(layer) is None:
            return super()._shared_input_groups_mlp(layer)
        return [
            (f"{self.EXPERTS_NAME}.{i}.up_proj", f"{self.EXPERTS_NAME}.{i}.gate_proj")
            for i in range(self.get_experts(layer).num_experts)
        ]

    def get_down_proj(self, layer):
        if self.get_experts(layer) is None:
            return super().get_down_proj(layer)
        return [resolve(name, layer) for name in self._expert_names(layer, "down_proj")]

    def patch_experts_forward(self, experts):
        # Expert weights are already loaded as per-expert 2D modules.
        experts.forward = MethodType(_forward_with_split_experts, experts)
        if hasattr(experts, "_old_forward"):
            experts._old_forward = experts.forward
        if hasattr(experts, "_original_forward_func"):
            experts._original_forward_func = experts.forward

    def stack_experts(self, experts, weight_packed=False):
        num_experts = experts.num_experts
        gate_proj = [experts[i].gate_proj for i in range(num_experts)]
        up_proj = [experts[i].up_proj for i in range(num_experts)]
        down_proj = [experts[i].down_proj for i in range(num_experts)]

        if weight_packed:
            experts.gate_up_proj_packed = nn.Parameter(
                torch.stack([
                    torch.cat([gate_proj[i].weight_packed, up_proj[i].weight_packed], dim=0)
                    for i in range(num_experts)
                ], dim=0),
                requires_grad=False,
            )
            experts.down_proj_packed = nn.Parameter(
                torch.stack([down_proj[i].weight_packed for i in range(num_experts)], dim=0),
                requires_grad=False,
            )
            experts.gate_up_proj_scale = nn.Parameter(
                torch.stack([
                    torch.cat([gate_proj[i].weight_scale, up_proj[i].weight_scale], dim=0)
                    for i in range(num_experts)
                ], dim=0),
                requires_grad=False,
            )
            experts.down_proj_scale = nn.Parameter(
                torch.stack([down_proj[i].weight_scale for i in range(num_experts)], dim=0),
                requires_grad=False,
            )
            intermediate_dim = int(gate_proj[0].weight_shape[0])
            hidden_dim = int(gate_proj[0].weight_shape[1])
            experts.gate_up_proj_shape = torch.tensor((num_experts, 2 * intermediate_dim, hidden_dim))
            experts.down_proj_shape = torch.tensor((num_experts, hidden_dim, intermediate_dim))
        else:
            experts.gate_up_proj = nn.Parameter(
                torch.stack([
                    torch.cat([gate_proj[i].weight, up_proj[i].weight], dim=0)
                    for i in range(num_experts)
                ], dim=0),
                requires_grad=False,
            )
            experts.down_proj = nn.Parameter(
                torch.stack([down_proj[i].weight for i in range(num_experts)], dim=0),
                requires_grad=False,
            )

        # Drop the per-expert 2D modules
        for i in range(num_experts):
            del experts._modules[str(i)]

        # Monkey patch the forward function
        experts.forward = MethodType(_stack_experts_forward, experts)


class SoftmaxTopkGateMixin:
    """Router patch for gates returning ``(logits, scores, indices)`` with softmax top-k."""

    def wrap_gate_forward(self, gate_module, func):
        """transformers==5.12.1"""
        def new_forward(self, hidden_states):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
            router_logits = func(router_logits)
            router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            router_scores = router_top_value
            return router_logits, router_scores, router_indices

        gate_module.forward = MethodType(new_forward, gate_module)


class LogitsOnlyGateMixin:
    """Router patch for gates returning the raw fp32 logits, with top-k done by the caller."""

    def wrap_gate_forward(self, gate_module, func):
        """transformers==5.12.1"""
        def new_forward(self, hidden_states):
            hidden_states = hidden_states.view(-1, self.config.hidden_size)
            router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
            router_logits = func(router_logits)
            return router_logits

        gate_module.forward = MethodType(new_forward, gate_module)


class LinearAttentionMixin:
    """Per-layer dispatch for archs interleaving full attention with linear attention."""

    LINEAR_ATTN_GROUPS = (
        ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
        ("linear_attn.out_proj",),
    )
    LINEAR_ATTN_INPUTS = (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
    )
    LINEAR_ATTN_OUTPUTS = ("linear_attn.out_proj",)
    LINEAR_ATTN_SHARED_INPUTS = (("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),)

    @staticmethod
    def _is_linear_attn(layer):
        return layer.layer_type == "linear_attention"

    def get_attn_groups(self, layer):
        if self._is_linear_attn(layer):
            return [list(group) for group in self.LINEAR_ATTN_GROUPS]
        return super().get_attn_groups(layer)

    def get_attn_inputs(self, layer):
        if self._is_linear_attn(layer):
            return [resolve(name, layer) for name in self.LINEAR_ATTN_INPUTS]
        return super().get_attn_inputs(layer)

    def get_attn_outputs(self, layer):
        if self._is_linear_attn(layer):
            return [resolve(name, layer) for name in self.LINEAR_ATTN_OUTPUTS]
        return super().get_attn_outputs(layer)

    def _shared_input_groups_attn(self, layer):
        """Linear attention: (in_proj_qkv, in_proj_z) both read the post-LN input."""
        if self._is_linear_attn(layer):
            return [tuple(group) for group in self.LINEAR_ATTN_SHARED_INPUTS]
        return super()._shared_input_groups_attn(layer)

    def get_kv_attn_module(self, layer):
        """Skip linear-attention layers: they hold a recurrent state, not a quantizable KV cache.
        """
        if self._is_linear_attn(layer):
            return None
        return super().get_kv_attn_module(layer)


###########################################################################
"""MoE experts forward implementations for split and stacked weight layouts."""


def _forward_with_split_experts(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        expert = self[expert_idx]
        gate = expert.gate_proj(current_state)
        up = expert.up_proj(current_state)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = expert.down_proj(current_hidden_states)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states


def _stack_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx].weight).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx].weight)
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states
