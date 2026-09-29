import inspect
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Union, List, Optional

import torch
import torch.nn as nn


def cleanup_memory(verbos=False) -> None:
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    gc.collect()

    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    # Preserve torch.compile caches when --enable_torch_compile is active.
    from utils import compile_utils
    if not compile_utils.is_enabled():
        torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    memory_after = total_reserved_mem()
    if verbos:
        logging.info(
            f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )


def get_param_size_in_bytes(dtype: torch.dtype) -> int:
    """
    Returns the size of a single element (in bytes) for a given PyTorch dtype.
    """
    # PyTorch Tensors have an element_size() method, but defining a map 
    # here ensures we can report on the calculation clearly.
    # Note: torch.float is usually float32 (4 bytes).
    dtype_map = {
        torch.float32: 4,
        torch.float: 4,
        torch.float64: 8,
        torch.double: 8,
        torch.float16: 2,
        torch.half: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.int: 4,
        torch.int64: 8,
        torch.long: 8,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
        torch.complex64: 8, # 2 * float32
        torch.complex128: 16 # 2 * float64
    }
    return dtype_map.get(dtype, 0) # Return 0 for unsupported/unknown types


def compute_trainable_param_memory(
    model: nn.Module, 
    unit: str = 'GB'
) -> Dict[str, Union[float, int]]:
    """
    Computes the total memory space occupied by the trainable parameters 
    in a PyTorch model and returns the result in the specified unit (MB, GB, or Bytes).

    Args:
        model (nn.Module): The PyTorch model instance.
        unit (str): The desired output unit ('B', 'MB', or 'GB'). Case-insensitive.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing total bytes, 
                                      total trainable parameters count, and 
                                      the calculated size in the desired unit.
    """
    total_bytes = 0
    total_trainable_params = 0

    # Iterate over all named parameters of the model
    for name, param in model.named_parameters():
        # Check if the parameter requires a gradient (i.e., is trainable)
        if param.requires_grad:
            # Get the number of elements in the tensor
            num_elements = param.numel()
            
            # Get the size of a single element in bytes based on its dtype
            element_bytes = get_param_size_in_bytes(param.dtype)
            
            # Calculate the total bytes for this parameter
            param_bytes = num_elements * element_bytes
            
            # Accumulate totals
            total_bytes += param_bytes
            total_trainable_params += num_elements

    # Conversion factors
    UNIT_FACTORS = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3
    }
    
    # Calculate the size in the requested unit
    unit_upper = unit.upper()
    factor = UNIT_FACTORS.get(unit_upper, UNIT_FACTORS['GB']) # Default to GB
    
    size_in_unit = total_bytes / factor

    return {
        "unit": unit_upper,
        "size": round(size_in_unit, 4),
        "total_bytes": total_bytes,
        "trainable_params": total_trainable_params,
    }


def get_tensor_size_gb(tensor: torch.Tensor) -> float:
    """
    Computes the size of a tensor in Gigabytes (GB).
    """
    num_elements = tensor.numel()
    element_size = tensor.element_size()
    
    total_bytes = num_elements * element_size
    size_gb = total_bytes / (1024 ** 3)
    
    return size_gb


def enable_memory_visualize(
    trace_alloc_max_entries: int = 200_000,
    stack_depth: int = 32,
    context: str = "all",
    stacks: str = "all",
    devices=None,
    record_context: bool = True,
):
    """
    Enables memory history recording for CUDA allocations. This function
    should be called before any large-scale CUDA allocations. For DDP or
    multi-process setups, it must be called on each rank.

    Args:
        trace_alloc_max_entries (int): Maximum number of allocation entries
            to record.
        stack_depth (int): The depth of the call stack to capture for each
            allocation. (Supported by some PyTorch versions).
        context (str): The type of memory events to record.
            'alloc': records only allocation events.
            'state': records memory state changes.
            'all': records both.
        stacks (str): The type of call stacks to record.
            'python': records Python stacks.
            'cpp': records C++ stacks (available in some versions).
            'all': records both.
        devices (Union[int, list[int], None]): The device for which to enable
            memory history. `None` enables it for the current default device.
        record_context (bool): Whether to record context information for
            allocations. Required by older PyTorch versions.
    """
    f = torch.cuda.memory._record_memory_history
    params = set(inspect.signature(f).parameters.keys())

    def _one_call(dev_kw=None):
        kwargs = {}
        if "context" in params:
            kwargs["context"] = context
        if "stacks" in params:
            kwargs["stacks"] = stacks
        if "max_entries" in params:
            kwargs["max_entries"] = trace_alloc_max_entries
        elif "trace_alloc_max_entries" in params:
            kwargs["trace_alloc_max_entries"] = trace_alloc_max_entries
        if "stack_depth" in params:
            kwargs["stack_depth"] = stack_depth
        if dev_kw is not None:
            if "device" in params:
                kwargs["device"] = dev_kw
            elif "devices" in params:
                kwargs["devices"] = dev_kw if isinstance(dev_kw, list) else [dev_kw]
        if "record_context" in params:
            kwargs["record_context"] = record_context

        try:
            f(**kwargs)
            return "native", kwargs
        except TypeError:
            try:
                if "trace_alloc_max_entries" in params and "record_context" in params:
                    f(enabled=True, trace_alloc_max_entries=trace_alloc_max_entries, record_context=True)
                    return "legacy", {
                        "enabled": True,
                        "trace_alloc_max_entries": trace_alloc_max_entries,
                        "record_context": True,
                    }
                else:
                    f(enabled=True)
                    return "legacy-min", {"enabled": True}
            except Exception:
                raise

    if devices is None or isinstance(devices, str | int | torch.device):
        mode, used = _one_call(devices if devices is not None else None)
    else:
        mode, used = "multi-device", {}
        for d in list(devices):
            _mode, _used = _one_call(d)
            used[f"dev{d}"] = _used

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    rank = int(os.environ.get("RANK", "0") or 0)
    logging.info(f"[memory_visualize][rank {rank}] recording enabled ({mode}); args={used}")


class MemorySnapshotSampler:
    """
    A utility class that dumps GPU memory snapshots.
    This is useful for monitoring memory usage over a long-running process.

    The dumped files can be visualized with https://docs.pytorch.org/memory_viz

    Args:
        out_dir (str): The directory where the snapshots will be saved.
        tag (str): A tag for the snapshot filenames.
    """

    def __init__(self, out_dir: str = "./mem_snapshots", tag: str = "periodic"):
        self.out_dir = out_dir
        self.tag = tag

    def dump_memory_snapshot(self, out_dir: str = "./outputs/mem_snapshots", tag: str = "snapshot", sub_dir: str = None):
        """
        Generates a memory snapshot and saves it as a pickle file in a specified directory.
        The files are organized by timestamp in subdirectories, with all ranks' files
        placed in the same timestamp subdirectory.

        Args:
            out_dir (str): The directory where the snapshot file will be saved.
                The directory is created if it does not exist.
            tag (str): A string tag to prepend to the filename for easier identification.
            sub_dir (str): A subdirectory to place the snapshot file in.
        """
        if sub_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            out_path = Path(out_dir) / timestamp
        else:
            out_path = Path(out_dir) / sub_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # get the GPU rank on the current process
        rank = os.environ.get("RANK", "0")
        pid = os.getpid()
        # todo(chenyang): check wether we need to sync all ranks before dump
        fname = f"{tag}_rank{rank}_pid{pid}.pickle"
        path = out_path / fname

        try:
            torch.cuda.synchronize()
            # Memory snapshot is CUDA-specific functionality
            torch.cuda.memory._dump_snapshot(str(path))
            logging.info(f"[memory_visualize] dumped: {path}")
        except Exception as e:
            logging.info(f"[memory_visualize][warn] dump failed: {e}")
