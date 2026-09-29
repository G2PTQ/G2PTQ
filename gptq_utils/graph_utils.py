"""Shared CUDA Graph capture and replay for compensation blocks.

Adapted from MoE-Quant.
"""

import torch


_GRAPH_CACHE = {}


def validate_tensors(description: str, *tensors: torch.Tensor) -> torch.device:
    if any(tensor is None or not tensor.is_cuda for tensor in tensors):
        raise RuntimeError(f"{description} requires every tensor to be on CUDA.")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError(f"{description} requires every tensor on the same device.")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(f"{description} requires contiguous tensors.")
    return device


def _copy_inputs(static_params, params, input_names) -> None:
    for name in input_names:
        static_params[name].copy_(params[name])


def _graph_key(name, run_eager, params, input_names, output_names, device):
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    tensors = tuple(
        (key, tuple(value.shape), value.dtype)
        for key, value in params.items()
        if isinstance(value, torch.Tensor)
    )
    constants = tuple(
        (key, value)
        for key, value in params.items()
        if not isinstance(value, torch.Tensor)
    )
    return (
        name,
        run_eager,
        device_index,
        tensors,
        constants,
        tuple(input_names),
        tuple(output_names),
    )


def run_graph(
    run_eager,
    params,
    *,
    input_names,
    output_names,
    name,
):
    """Capture ``run_eager(**params)`` once and replay it with current inputs."""
    tensor_params = [
        value for value in params.values() if isinstance(value, torch.Tensor)
    ]
    device = validate_tensors(f"CUDA Graph {name}", *tensor_params)
    key = _graph_key(
        name, run_eager, params, input_names, output_names, device
    )

    with torch.cuda.device(device):
        state = _GRAPH_CACHE.get(key)
        if state is None:
            static_params = {
                key: torch.empty_like(value)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in params.items()
            }
            _copy_inputs(static_params, params, input_names)

            current_stream = torch.cuda.current_stream(device)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                run_eager(**static_params)
            current_stream.wait_stream(warmup_stream)

            _copy_inputs(static_params, params, input_names)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_eager(**static_params)

            state = {"graph": graph, "params": static_params}
            _GRAPH_CACHE[key] = state

        _copy_inputs(state["params"], params, input_names)
        state["graph"].replay()
        for output_name in output_names:
            params[output_name].copy_(state["params"][output_name])

    return tuple(params[name] for name in output_names)


def clear_graph_cache() -> None:
    """Release process-local CUDA Graphs and their static tensor buffers."""
    _GRAPH_CACHE.clear()
