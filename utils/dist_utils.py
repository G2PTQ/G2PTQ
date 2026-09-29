import os
from typing import Optional, Any
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from utils import memory_utils, model_utils


def init_process_group():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=datetime.timedelta(days=1),
    )
    dist.barrier()


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def is_main():
    return get_rank() == 0


def broadcast_parameters(module: nn.Module, src: Any = 0, group: Optional[Any] = None):
    for param in module.parameters():
        dist.broadcast(param.data, src=src, group=group)


def gather_into_tensor(tensor, dim: int = 0):
    world_size = get_world_size()
    if is_main():
        gathered_shape = (*tensor.shape[:dim], world_size * tensor.shape[dim], *tensor.shape[dim + 1 :])
        gathered_tensor = torch.empty(gathered_shape, device=tensor.device, dtype=tensor.dtype)
        gathered_tensor_chunks = list(gathered_tensor.chunk(world_size, dim=dim))
    else:
        gathered_tensor = None
        gathered_tensor_chunks = None
    dist.gather(tensor, gathered_tensor_chunks)
    return gathered_tensor


def print_on_main(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def distribute_model(analyzer: model_utils.ModelAnalyzer) -> None:
    model = analyzer.model
    no_split_module_classes = [analyzer.get_decoder_layer_class_name()]
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )
    memory_utils.cleanup_memory()
