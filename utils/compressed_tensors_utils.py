import os
import re
import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedConfig, PreTrainedTokenizer, ProcessorMixin
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme, QuantizationArgs, \
                                            QuantizationType, QuantizationStrategy
from compressed_tensors.compressors import pack_to_int32
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint


MERGE_PATTERNS = {
    "qkv_proj": r"(q_proj|k_proj|v_proj)",
    "in_proj_qkvz": r"(in_proj_qkv|in_proj_z)",
    "in_proj_ba": r"(in_proj_b|in_proj_a)"
}
Q_PARAMS = ("weight_packed", "weight_scale", "weight_shape")


def compress(
    layer: nn.Linear,
    Scale: torch.Tensor,
    W_int: torch.Tensor,
    bits: int,
    groupsize: int,
):
    weight = layer.weight
    rows, columns = weight.shape
    dtype, dev = weight.dtype, weight.device
    quantized_weight = W_int.reshape(weight.shape).to(torch.int8)
    group_scale = Scale.reshape(rows, columns // groupsize, groupsize)
    assert (group_scale == group_scale[..., 0:1]).all(), "Scales in the same quantization group should be the same."

    # Stash the quantized params as a plain dict attribute. They are turned into
    # real buffers later by ``register_qparam_buffers``.
    layer._export_qparams = {
        "weight_packed": pack_to_int32(quantized_weight, bits).contiguous(),
        "weight_scale": group_scale[:, :, 0].to(dtype).contiguous(),
        "weight_shape": torch.tensor(weight.shape).to(dev).contiguous(),
    }


def register_qparam_buffers(modules):
    """Register the qparams staged by ``compress`` as real buffers.
    """
    for module in modules:
        qparams = module.__dict__.pop("_export_qparams", None)
        if qparams is None:
            continue
        for name, tensor in qparams.items():
            module.register_buffer(name, tensor)


def expand_ignore_list(
    ignore_list: list[str], 
    merge_patterns: dict[str, str] = None
) -> list[str]:
    """
    Expands the ignore list to include merged layers for inference engine compatibility.
    
    Args:
        ignore_list: The original list of layers to ignore.
        merge_patterns: A dictionary mapping merged layer names to regex patterns.
    """
    if merge_patterns is None:
        merge_patterns = MERGE_PATTERNS

    expanded_ignores = set(ignore_list)
    
    for layer in ignore_list:
        for merged_name, pattern in merge_patterns.items():
            if re.search(pattern, layer):
                merged_layer = re.sub(pattern, merged_name, layer)
                expanded_ignores.add(merged_layer)
                
    return sorted(list(expanded_ignores))


def save_to_compressed_tensors(
    source_model: str,
    config: PreTrainedConfig,
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer,
    processor: ProcessorMixin,
    save_path: str, 
    bits: int = 4, 
    group_size: int = 128, 
    sym: bool = True,
    export_mtp: bool = False,
):
    """Save a GPTQ quantized model to the compressed-tensors format.
    """
    assert (bits in [4, 8]) and sym
    os.makedirs(save_path, exist_ok=True)

    # Remove the original float weight and construct the ignore list
    ignore = ["re:^mtp.*"]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or \
            isinstance(module, nn.modules.conv._ConvNd):
            if hasattr(module, "weight_packed"):
                del module.weight
            else:
                ignore.append(name)
    ignore = expand_ignore_list(ignore)

    # Create the compressed-tensors quantization config
    if group_size == -1:
        strategy = QuantizationStrategy.CHANNEL
    else:
        strategy = QuantizationStrategy.GROUP
    scheme_args = dict(
        weights=QuantizationArgs(
            num_bits=bits,
            type=QuantizationType.INT,
            strategy=strategy,
            group_size=group_size,
            symmetric=sym,
            dynamic=False,
        ),
    )
    scheme = QuantizationScheme(
        targets=["Linear"],
        format="pack-quantized",
        **scheme_args,
    )
    quant_config = QuantizationConfig(
        config_groups={
            "group_0": scheme,
        },
        ignore=ignore,
        format="pack-quantized",
        quant_method="compressed-tensors",
        quantization_status="compressed",
    )
    
    # Attach config to the model
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.quantization_config = quant_config.model_dump()

    # Save the model, tokenizer, and config
    logging.info(f"Saving compressed model to {save_path}...")
    config.save_pretrained(save_path)
    model.generation_config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    model.save_pretrained(
        save_path,
        safe_serialization=True,
        save_original_format=False,
    )
    if export_mtp:
        save_mtp_tensors_to_checkpoint(source_model=source_model, dest_dir=save_path)
