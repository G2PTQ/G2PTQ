import contextlib
import re
import logging
from tqdm import tqdm
from typing import List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import PreTrainedModel, AutoTokenizer, AutoProcessor, \
                         PreTrainedTokenizerBase, ProcessorMixin, AutoConfig
from compressed_tensors.offload import disable_onloading
from llmcompressor.utils.dev import skip_weights_initialize

from utils.moe_linearize_utils import load_quantizable_moe
from utils.models import get_model_spec, resolve
from utils import offload_utils, dist_utils, quant_utils

LINEAR_LAYERS = (nn.Linear, _ConvNd)


def _matches_no_grad_param_prefix(param_name, no_grad_param_prefixes):
    return any(
        param_name == prefix
        or param_name.startswith(f"{prefix}.")
        or param_name.endswith(f".{prefix}")
        or f".{prefix}." in param_name
        for prefix in (no_grad_param_prefixes or ())
    )


def set_requires_grad(
    model: nn.Module,
    enable=True,
    no_grad_param_prefixes=(),
    onload_parameters=False,
):
    """Set parameter gradient flags, honoring architecture-specific exclusions.

    When ``enable`` is true, all parameters are enabled except paths listed in
    ``no_grad_param_prefixes``. By default, parameter inspection is protected
    by ``disable_onloading`` so a huge offloaded tensor is not materialized.
    Callers that already hold a module in ``disable_offloading`` should pass
    ``onload_parameters=True`` so flags are applied to the onloaded parameter
    objects used by its forward.
    """
    context = contextlib.nullcontext() if onload_parameters else disable_onloading()
    with context:
        for name, param in model.named_parameters():
            param.requires_grad_(
                enable and not _matches_no_grad_param_prefix(name, no_grad_param_prefixes)
            )


def _build_attn_impl_kwargs(attn_implementation, spec_cls, config):
    """Return the `attn_implementation` kwarg for `from_pretrained`, or `{}` to leave it unset.
    """
    if attn_implementation is None:
        return {}

    if "flash" in attn_implementation and not spec_cls.SUPPORTS_FLASH_ATTN:
        logging.warning(
            f"{config.model_type} does not support flash attention; falling back to sdpa "
            f"instead of {attn_implementation}."
        )
        attn_implementation = "sdpa"

    return {"attn_implementation": attn_implementation}


def load_model(model_str_or_model, offload_folder="./offload_weights", offload_extra_cpu_mem=100e9,
               attn_implementation=None):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace
    """
    process_word_embeddings = False
    if isinstance(model_str_or_model, str):
        config = AutoConfig.from_pretrained(model_str_or_model)
        # Resolve the spec from the config so it can drive loading, before the model exists.
        spec_cls = get_model_spec(config.model_type)
        spec_cls.register_conversion_mappings(config)
        if config.tie_word_embeddings:
            config.tie_word_embeddings = False
            process_word_embeddings = True
        model_class = spec_cls.load_model_class(config)
        load_kwargs = _build_attn_impl_kwargs(attn_implementation, spec_cls, config)
        with skip_weights_initialize():
            with (
                offload_utils.load_offloaded_model(
                    model_class=model_class,
                    extra_cpu_mem=int(offload_extra_cpu_mem),
                ),
                load_quantizable_moe(model_cls=model_class),
            ):
                model = model_class.from_pretrained(
                    model_str_or_model,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype='auto',
                    device_map='auto_offload',
                    offload_folder=offload_folder,
                    offload_buffers=True,
                    **load_kwargs,
                )
            model = spec_cls.post_load(model, config)
        model.config.tie_word_embeddings = False
        if process_word_embeddings:
            spec = spec_cls(getattr(model.config, "text_config", model.config))
            embed_tokens = spec.get_embed_layer(model)
            offload_utils.update_shared_offload_parameter(
                model.lm_head, "weight", embed_tokens.weight.data.clone()
            )
    else:
        assert isinstance(model_str_or_model, PreTrainedModel), "model must be a string or a PreTrainedModel"
        model = model_str_or_model
        spec_cls = get_model_spec(model.config.model_type)

    model.eval()

    set_requires_grad(
        model,
        True,
        no_grad_param_prefixes=spec_cls.NO_GRAD_PARAM_PREFIXES,
    )

    return model, process_word_embeddings


def load_tokenizer(model_str_or_model_or_tokenizer):
    """Returns a tokenizer from the model string or model object or tokenizer object"""
    if isinstance(model_str_or_model_or_tokenizer, str):
        model_str = model_str_or_model_or_tokenizer
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    elif isinstance(model_str_or_model_or_tokenizer, PreTrainedModel):
        model_str = model_str_or_model_or_tokenizer.name_or_path
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    else:
        assert isinstance(model_str_or_model_or_tokenizer, PreTrainedTokenizerBase), \
            f"Unsupported type for model_str_or_model_or_tokenizer: {type(model_str_or_model_or_tokenizer)}"
        return model_str_or_model_or_tokenizer


def load_processor(model_str_or_model_or_processor):
    """Returns a processor from the model string or model object or processor object"""
    if isinstance(model_str_or_model_or_processor, str):
        model_str = model_str_or_model_or_processor
        return AutoProcessor.from_pretrained(model_str, trust_remote_code=True)
    elif isinstance(model_str_or_model_or_processor, PreTrainedModel):
        model_str = model_str_or_model_or_processor.name_or_path
        return AutoProcessor.from_pretrained(model_str, trust_remote_code=True)
    else:
        assert isinstance(model_str_or_model_or_processor, ProcessorMixin), \
            f"Unsupported type for model_str_or_model_or_processor: {type(model_str_or_model_or_processor)}"
        return model_str_or_model_or_processor


def select_layers(
    model: nn.Module,
    layer_prefix: Optional[str] = "",
    layer_regex: str = ".*",
    layer_classes: Union[nn.Module, List[nn.Module]] = nn.Module,
) -> Dict[str, nn.Module]:
    layers = {}
    for layer_name, layer in model.named_modules():
        if (
            isinstance(layer, layer_classes)
            and re.search(layer_regex, layer_name)
            and layer_name.startswith(layer_prefix)
        ):
            layers[layer_name] = layer
    return layers


class ModelAnalyzer:
    """ModelAnalyzer is a class that provides an interface to access relevant model information for quantization.
    """

    def __init__(
        self,
        model_str_or_model,
        seq_len,
        ignore_attn=False,
        offload_folder="./offload_weights",
        offload_extra_cpu_mem=5e9,
        attn_implementation=None,
    ):
        self.model, self.process_word_embeddings = load_model(
            model_str_or_model,
            offload_folder=offload_folder,
            offload_extra_cpu_mem=offload_extra_cpu_mem,
            attn_implementation=attn_implementation,
        )
        self.model_arch = self.model.config.model_type
        self.config = getattr(self.model.config, "text_config", self.model.config)
        #: Per-architecture layout/behaviour description (see utils/models/).
        self.spec = get_model_spec(self.model_arch)(self.config)
        self.patch_experts_forward()
        self.tokenizer = load_tokenizer(model_str_or_model)
        self.processor = load_processor(model_str_or_model)
        self.ignore_attn = ignore_attn

        self.model.seqlen = seq_len
        self.num_layers = len(self.get_layers())
        self.hidden_size = self.config.hidden_size
        self.residual_width = self.spec.residual_width
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = getattr(self.config, "num_key_value_heads", self.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.hidden_size // self.num_attention_heads)

        self.onload_device = f"cuda:{dist_utils.get_rank()}"
        offload_utils.set_onload_device(self.model, torch.device(self.onload_device))
        offload_utils.assert_shared_offload_storage(self.model)

    def recover_tie_word_embeddings(self):
        self.model.config.tie_word_embeddings = self.process_word_embeddings

    def is_moe(self):
        return self.spec.IS_MOE

    def uses_dsa_indexshare(self):
        return self.spec.USES_DSA_INDEXSHARE

    def is_multimodal(self):
        return self.spec.IS_MULTIMODAL

    def get_lm_head(self):
        return self.spec.get_lm_head(self.model)

    def get_embed_layer(self):
        return self.spec.get_embed_layer(self.model)

    def get_layernorm_before_head(self):
        return self.spec.get_layernorm_before_head(self.model)

    def get_layers(self):
        """Return the layers of the model."""
        return self.spec.get_layers(self.model)

    def get_pre_block_modules(self):
        """Return pre-block modules of the model."""
        return self.spec.get_pre_block_modules(self.model)

    def get_quantizable_modules(self, layer, layer_classes=LINEAR_LAYERS, remove_wrapper=True):
        """Return the quantizable modules of the layer."""
        modules = {}
        for group in self.get_sequential_quantizable_module_names(layer):
            for name in group:
                module = self.get_model_attribute(name, layer)
                if remove_wrapper and isinstance(module, quant_utils.ActQuantWrapper):
                    module = module.module
                if isinstance(module, layer_classes):
                    modules[name] = module
                else:
                    raise NotImplementedError(f"Quantizable module {name} not found!")
        return modules

    def get_sequential_quantizable_module_names(self, layer):
        """Return the quantizable module names of the layer in sequential order."""
        attn_modules = self.spec.get_attn_groups(layer)
        mlp_modules = self.spec.get_mlp_groups(layer)

        if self.ignore_attn:
            sequential_quantizable_module_names = [
                *mlp_modules,
            ]
        else:
            sequential_quantizable_module_names = [
                *attn_modules,
                *mlp_modules,
            ]
        return sequential_quantizable_module_names

    def get_shared_input_groups(self, layer):
        """Return a full partition of quantizable module names into shared-input groups.
        """
        return self.spec.get_shared_input_groups(layer)

    # ---- Rotation interfaces ---------------------------------------------------------------
    # Global rotation accessors are valid only when can_rotate_global() is true.
    def can_rotate_global(self):
        return self.spec.CAN_ROTATE_GLOBAL

    def get_layernorms(self, layer):
        """Return a list of tuples (LN module: nn.Module, is_zero_centered: bool)"""
        return self.spec.get_layernorms(layer)

    def get_perlayer_input_modules(self, layer):
        return [
            self.spec.get_attn_inputs(layer),
            self.spec.get_mlp_inputs(layer),
        ]

    def get_perlayer_output_modules(self, layer):
        return [
            self.spec.get_attn_outputs(layer),
            self.spec.get_mlp_outputs(layer),
        ]

    # Online MLP rotation is independent of global and OV rotation support.
    def get_perlayer_down_proj(self, layer):
        return self.spec.get_down_proj(layer)

    # OV rotation accessors are valid only when can_rotate_ov() is true for the layer.
    def can_rotate_ov(self, layer):
        return self.spec.CAN_ROTATE_OV

    def get_perlayer_v_proj(self, layer):
        return self.spec.get_v_proj(layer)

    def get_perlayer_o_proj(self, layer):
        return self.spec.get_o_proj(layer)

    def get_perlayer_gate(self, layer):
        gate = self.spec.get_gate(layer)
        if gate is None:
            raise NotImplementedError(f"{self.model_arch} layer has no MoE router")
        return gate

    def get_module_before_final_residual(self, layer):
        return self.spec.get_module_before_final_residual(layer)

    def get_layer_scalar(self, layer):
        """Some models multiply the hidden states after final residual by a layer scalar"""
        return self.spec.get_layer_scalar(layer)

    def get_perlayer_experts(self, layer):
        return self.spec.get_experts(layer)

    def patch_experts_forward(self):
        """Install the experts forward implementation for per-expert 2D weights."""
        if self.is_moe():
            for layer in self.get_layers():
                if (experts := self.get_perlayer_experts(layer)) is not None:
                    self.spec.patch_experts_forward(experts)

    def stack_experts_weights(self, weight_packed=False):
        if self.is_moe():
            for layer in tqdm(self.get_layers(), ncols=120, desc="Stacking Experts"):
                if (experts := self.get_perlayer_experts(layer)) is not None:
                    self.spec.stack_experts(experts, weight_packed=weight_packed)

    def post_process_logits(self, logits):
        return self.spec.post_process_logits(logits)

    def capture_block_internals(self):
        """Capture the per-layer forward inputs block-wise execution must reproduce by hand."""
        return self.spec.capture_block_internals(self.model)

    @contextlib.contextmanager
    def enter_quantization_context(self, args):
        """Apply temporary architecture-specific state for quantization and restore it on exit."""
        with self.spec._enter_quantization_context(args, self.model):
            yield

    def alloc_index_buffer(self, nsamples, seqlen, device):
        """Allocate the per-sample cross-layer index buffer, or None when the arch has none."""
        return self.spec.alloc_index_buffer(nsamples, seqlen, device)

    def get_decoder_layer_class_name(self):
        return self.spec.DECODER_LAYER_CLASS

    def get_model_attribute(self, attr_str, module):
        return resolve(attr_str, module)


def run_block_layer(analyzer, layer, hidden_states, *, out_buffer=None, prev_topk_indices=None,
                    index_buffer=None, block_internals=None,
                    layer_idx=None, sample_idx=None, bsz=None, dev=None,
                    **kwargs):
    """Run a decoder layer in block-wise mode, returning the hidden-states tensor.

    A decoder layer returns either a bare hidden-states tensor or a tuple
    `(hidden_states, *extras)` (e.g. GLM-5.2 DSA returns the top-k indices as an extra).
    We always return just the hidden states; the extras are consumed here.

    The arch-specific work is delegated to the model spec (see `utils/models/`):

    Two spec hooks bracket the layer call:

    - `build_block_kwargs` adds this layer/batch's arch-specific forward kwargs — the internals
      captured by `analyzer.capture_block_internals()` (gemma4 onloads its shared KV cache here)
      and `prev_topk_indices`, the cross-layer indices an earlier layer produced.
    - `finalize_block` consumes the layer's outputs and releases its per-batch state — gemma4
      offloads the KV cache again, GLM-5.2 records fresh top-k indices into `index_buffer`.

    `index_buffer` is passed only on advance passes, so a read-only pass over the same layer
    cannot clobber the indices it is itself reading via `prev_topk_indices`.

    Pass `out_buffer` on an advance pass to have this batch's output written back to
    `out_buffer[sample_idx: sample_idx + bsz]` through the ambient write-back queue
    (`offload_utils.current_d2h_queue()`), which overlaps the copy with the next batch's compute
    when the buffer is offloaded to pinned host memory. The hidden states are still returned
    either way; callers that only need the buffer updated can ignore the return value.
    """
    spec = analyzer.spec

    spec.build_block_kwargs(block_internals, kwargs, prev_topk_indices=prev_topk_indices,
                            layer_idx=layer_idx, sample_idx=sample_idx, bsz=bsz, dev=dev)

    out = layer(hidden_states, **kwargs)
    hidden_states, *extras = (out[0], *out[1:]) if isinstance(out, tuple) else (out,)

    spec.finalize_block(block_internals, index_buffer, extras,
                        layer_idx=layer_idx, sample_idx=sample_idx, bsz=bsz)

    if out_buffer is not None:
        offload_utils.current_d2h_queue().copy_(
            out_buffer[sample_idx: sample_idx + bsz], hidden_states
        )

    return hidden_states
