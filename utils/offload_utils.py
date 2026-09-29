import contextlib
import logging
from collections import deque
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, PreTrainedModel
from compressed_tensors.distributed import is_distributed, is_source_process
from compressed_tensors.offload import (
    disable_onloading,
    from_accelerate,
    update_offload_parameter,
)
from compressed_tensors.offload.cache import OffloadCache
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.utils import patch_attr
from compressed_tensors.offload.utils import as_single_threaded
from compressed_tensors.offload.load import _get_cpu_memory, _get_device_memory, _estimate_tensor_count


def alloc_pinned(*size, dtype=None, device="cpu", pin=False):
    """Allocate a zeroed tensor, page-locked when it lives on the host.
    """
    pin_memory = bool(pin) and torch.device(device).type == "cpu" and torch.cuda.is_available()
    return torch.zeros(*size, dtype=dtype, device=device, pin_memory=pin_memory)


def clone_pinned(t):
    """Copy of `t` that preserves page-locking (`Tensor.clone` does not)."""
    if not t.is_pinned():
        return t.clone()
    return torch.empty_like(t, pin_memory=True).copy_(t)


def empty_pinned_like(t):
    """Uninitialized host tensor shaped like `t`, page-locked so it can receive an async
    device-to-host copy (see `D2HQueue`).
    """
    return torch.empty(
        t.shape, dtype=t.dtype, device="cpu",
        pin_memory=torch.cuda.is_available(),
    )


class D2HQueue:
    """Stream-ordered device-to-host write-back with a bounded in-flight window.

    Copies are issued on a dedicated stream so a batch's write-back overlaps the
    next batch's compute. A copy only goes async when it can: the source must be on
    the accelerator and the destination must be pinned host memory. Otherwise it falls
    back to a plain blocking `copy_`.
    """

    #: Shared across queues, keyed by device index.
    _streams: dict[int, torch.Stream] = {}

    def __init__(self, max_inflight: int = 2):
        self.max_inflight = max_inflight
        self._events = deque()

    @classmethod
    def _get_stream(cls, device_index: int):
        stream = cls._streams.get(device_index)
        if stream is None:
            stream = torch.Stream(device=torch.device("cuda", device_index))
            cls._streams[device_index] = stream
        return stream

    @staticmethod
    def _can_overlap(dst, src) -> bool:
        """Whether this pair can use a true async memcpy.
        """
        return (
            torch.accelerator.is_available()
            and src.is_cuda
            and dst.device.type == "cpu"
            and dst.is_pinned()
            and dst.is_contiguous()
            and src.is_contiguous()
        )

    def copy_(self, dst, src) -> None:
        """Commit `src` into `dst`, overlapped when possible.
        """
        if not self._can_overlap(dst, src):
            dst.copy_(src)
            return

        # Derive both streams from `src`'s device rather than the ambient current device:
        # CUDA's current device is thread-local, so a write-back issued off the main thread
        # would otherwise fork from -- and copy on -- the wrong device's stream.
        device_index = src.device.index
        stream = self._get_stream(device_index)
        consumer = torch.accelerator.current_stream(device_index)

        # Order the copy behind the compute that produced `src`.
        fork = torch.Event()
        fork.record(consumer)
        stream.wait_event(fork)

        with stream:
            dst.copy_(src, non_blocking=True)
            # Keep `src` alive on the copy stream, so the caching allocator cannot recycle
            # its block while the copy is still in flight.
            src.record_stream(stream)

        done = torch.Event()
        done.record(stream)
        self._events.append(done)

        # Bound how many source tensors are pinned by in-flight copies.
        while len(self._events) > self.max_inflight:
            self._events.popleft().synchronize()

    def wait(self) -> None:
        """Block until every queued copy has landed in host memory."""
        while self._events:
            self._events.popleft().synchronize()


class _SyncD2HQueue:
    """Fallback used when no `prefetch_generator` loop is driving: copies immediately."""

    def copy_(self, dst, src) -> None:
        dst.copy_(src)

    def wait(self) -> None:
        pass


_SYNC_D2H_QUEUE = _SyncD2HQueue()
_active_d2h_queue = None


def current_d2h_queue():
    """The queue installed by the enclosing `prefetch_generator` loop.

    Returns a synchronous no-op queue when called outside such a loop, so any write-back is
    correct by default -- just not overlapped.
    """
    return _active_d2h_queue if _active_d2h_queue is not None else _SYNC_D2H_QUEUE


@contextlib.contextmanager
def _install_d2h_queue():
    """Publish a fresh `D2HQueue` as the ambient one, draining it on the way out.

    Ambient rather than threaded through call signatures so the `ModelSpec.finalize_block`
    hooks (which do their own per-batch write-back) need no signature change for what is
    purely a transport concern.
    """
    global _active_d2h_queue

    # A previous loop that was abandoned early (`break`) drains at collection time, which is
    # GC-dependent; draining here makes "no copies outstanding when a loop starts" hold
    # regardless.
    if _active_d2h_queue is not None:
        _active_d2h_queue.wait()

    previous = _active_d2h_queue
    queue = D2HQueue()
    _active_d2h_queue = queue
    try:
        yield queue
    finally:
        queue.wait()
        _active_d2h_queue = previous


@contextlib.contextmanager
def onload_attrs(owner, names, enabled=True, device=None):
    """Onload ``owner.<name>`` to device for each name on enter, write it back to the host on exit.
    """
    if not enabled:
        yield
        return

    with _install_d2h_queue():
        staged = {}
        for name in names:
            host = getattr(owner, name, None)
            if host is not None:
                staged[name] = host     # hold the page-locking memory
                setattr(owner, name, host.to(device=device, non_blocking=host.is_pinned()))

        try:
            yield
        finally:
            for name in names:
                value = getattr(owner, name, None)
                if value is None or value.device.type == "cpu":
                    continue

                host = staged.get(name)
                if host is None or host.shape != value.shape or host.dtype != value.dtype:
                    host = empty_pinned_like(value)
                current_d2h_queue().copy_(host, value)
                setattr(owner, name, host)


def prefetch_generator(tensors, num_samples, bsz, onload_device, use_prefetch=False):
    """Yield `(j, batch_tuple)` of onloaded per-sample inputs, batch by batch.

    `tensors` is a tuple whose entries are onloaded by type:
      - a `Tensor` is sliced `[j:j+bsz]` and moved to `onload_device`;
      - `None` passes through as `None` (e.g. an unallocated DSA index buffer on non-DSA models).
    `batch_tuple` always has the same arity as `tensors`.

    Stream + event handoff adapted from `llmcompressor.pipelines.cache
    .IntermediatesCache.iter_prefetch`.

    For the duration of the loop a `D2HQueue` is installed as the ambient write-back queue
    (see `current_d2h_queue`), so the download side overlaps too: `run_block_layer` writes its
    output through it, as do the `finalize_block` spec hooks. The queue is drained when the
    loop finishes, which is what makes a buffer safe to read after the loop.
    """
    def onload_value(t, j):
        if t is None:
            return None
        return t[j: j+bsz].to(onload_device, non_blocking=True)

    def fetch(j):
        return tuple(onload_value(t, j) for t in tensors)

    if not use_prefetch:
        with _install_d2h_queue():
            for j in range(0, num_samples, bsz):
                yield j, fetch(j)
        return

    h2d_stream = torch.Stream() if torch.accelerator.is_available() else None
    # CUDA's current device is *thread-local*, and only the main thread runs
    # `torch.cuda.set_device` (`dist_utils.init_process_group`). The prefetch worker therefore
    # defaults to `cuda:0`, not this rank's device, so any stream or device handle it needs must
    # be captured on the driving thread.
    consumer_stream = torch.accelerator.current_stream() if h2d_stream is not None else None
    worker_device_index = torch.accelerator.current_device_index() if h2d_stream is not None else None

    def init_worker():
        # Pin the worker to this rank's device, so its onload copies and any allocator work
        # land there instead of on cuda:0 (which would also leave a stray context behind).
        torch.accelerator.set_device_index(worker_device_index)

    def fetch_and_record(j):
        if h2d_stream is None:
            return fetch(j), None
        with h2d_stream:
            data = fetch(j)
            # Keep each onloaded tensor alive on the consumer stream, so the caching
            # allocator cannot recycle its block while the copy is still in flight. Issued
            # while the copy stream is current and before the event, so the copy, the
            # registration, and the event form one stream-ordered unit.
            for t in data:
                if t is not None:
                    t.record_stream(consumer_stream)
            event = torch.Event()
            event.record(h2d_stream)
        return data, event

    with (
        ThreadPoolExecutor(
            max_workers=1,
            initializer=init_worker if worker_device_index is not None else None,
        ) as executor,
        _install_d2h_queue(),
    ):
        future = None
        for j in range(0, num_samples, bsz):
            if future is not None:
                data, event = future.result()
            else:
                data, event = fetch_and_record(j)

            next_j = j + bsz
            if next_j < num_samples:
                future = executor.submit(fetch_and_record, next_j)
            else:
                future = None

            # Order the consumer behind the copy.
            if event is not None:
                torch.accelerator.current_stream().wait_event(event)

            yield j, data


@contextlib.contextmanager
def load_offloaded_model(
    model_class: type[PreTrainedModel] = AutoModelForCausalLM, extra_cpu_mem: int = 5e9
):
    """
    Context manager used to load a transformers model with offloading implemented by
    compressed-tensors.

    The model is first loaded with accelerate's offloading, then convereted into
    offloading implemented by compressed-tensors. If a distributed environment has been
    initialized, then rank 0 loads the weights while other ranks load on the meta
    device, then the offload is shared across ranks during conversion.

    In addition to the standard `device_map` options, this context also supports
    `device_map="auto_offload"`, which means that the model will load as many parameters
    can fit onto the cpu, and any extra parameters will be loaded on disk.

    :param model_class: model class to patch
    :param extra_cpu_mem: extra cpu memory to reserve for any operations not related to
        model loading (bytes). Defaults to 5Gb.
    """
    original_from_pretrained = model_class.from_pretrained
    patched_fn_called = False

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        nonlocal patched_fn_called
        patched_fn_called = True

        kwargs.setdefault("device_map", None)

        # Rank 0 does loading, other ranks init on meta device
        if not is_source_process():
            kwargs["device_map"] = "meta"
            # # Workaround: transformers v5 tie_weights() calls torch.equal() on
            # # meta tensors which is unsupported. Since rank 0 broadcasts the real
            # # weights, we can safely skip tying on non-rank workers.
            # kwargs.setdefault("tie_word_embeddings", False)

        # Intercept `auto_offload`: same as "auto", but only cpu/disk are visible
        elif kwargs["device_map"] == "auto_offload":
            kwargs["device_map"] = "auto"
            if "max_memory" not in kwargs:
                num_tensors, total_bytes = 0, 0
                if is_distributed():
                    num_tensors, total_bytes = _estimate_tensor_count(
                        original_from_pretrained, *args, **kwargs
                    )
                kwargs["max_memory"] = _get_cpu_memory(
                    extra_cpu_mem, num_tensors, total_bytes
                )

        # Unless the user specifies, use our memory estimates, which take into
        # account distributed setups and extra cpu reserved memory
        elif "max_memory" not in kwargs:
            num_tensors, total_bytes = 0, 0
            if is_distributed():
                num_tensors, total_bytes = _estimate_tensor_count(
                    original_from_pretrained, *args, **kwargs
                )
            kwargs["max_memory"] = _get_device_memory() | _get_cpu_memory(
                extra_cpu_mem, num_tensors, total_bytes
            )

        # Unless the user specifies, use `offload_buffers` to avoid accelerate weirdness
        if not kwargs.get("offload_buffers", True):
            logging.warning("Loading with `offload_buffers=False` is not supported")
        kwargs["offload_buffers"] = True

        with as_single_threaded():
            model = original_from_pretrained(*args, **kwargs)
        from_accelerate(model)  # rank 0 shares weights with ranks via offload/broadcast

        return model

    with patch_attr(model_class, "from_pretrained", patched):
        try:
            yield
        finally:
            if not patched_fn_called:
                logging.warning(
                    f"`{model_class.__name__}.from_pretrained` was never called. If "
                    "you are loading with a model class other than "
                    f"{model_class.__name__}, please pass as argument to "
                    "`load_offloaded_model`"
                )


def _matches_no_placement(param_name: str, no_placement_params) -> bool:
    """Whether `param_name` is one of the model's `_no_placement_params`.

    Entries are decoder-layer-relative (`"ple.ple_embedding.ngram_embedding.weight"`), so match on a
    dot-aligned suffix of the full parameter name rather than equality.
    """
    return any(
        param_name == entry or param_name.endswith(f".{entry}")
        for entry in no_placement_params
    )


def set_onload_device(
    model: torch.nn.Module,
    onload_device: torch.device | str,
) -> torch.nn.Module:
    """
    Modify the dispatch of a model to onload to the provided `onload_device`. Existing
    offloaded tensors will not be modified.

    Modules owning a parameter listed in the model's `_no_placement_params` are pinned to the CPU
    instead, *including* when `onload_device` is a GPU. transformers declares that attribute for
    tensors too large to place at all (`qwen4_exp`'s ~95 GB hashed n-gram table), keeping them out of
    the device map.

    :param model: model to dispatch
    :param onload_device: device to move weights to during forward pass
    :return: dispatched model
    """
    no_placement = getattr(model, "_no_placement_params", None) or ()
    cpu_device = torch.device("cpu")

    for name, module in model.named_modules():
        if not isinstance(module._parameters, OffloadCache):
            continue
        # Iterate keys only: indexing an OffloadCache onloads the tensor, and the one being kept
        # off the device is precisely the one too large to onload.
        holds_unplaceable = no_placement and any(
            _matches_no_placement(f"{name}.{key}" if name else key, no_placement)
            for key in (*module._parameters, *module._buffers)
        )
        # onload_device is per-cache, so a module holding both a listed and an unlisted tensor
        # pins both. Fine for the known case (`ngram_embedding`'s only tensor is `weight`).
        device = cpu_device if holds_unplaceable else onload_device
        module._parameters.onload_device = device
        module._buffers.onload_device = device

    return model


def update_shared_offload_parameter(
    module: torch.nn.Module,
    name: str,
    data: torch.Tensor,
    src: int | None = None,
    sync: bool = True,
) -> None:
    """
    Commit `data` to shared offload storage from exactly one rank, then synchronize.

    :param module: module containing the parameter/buffer to update
    :param name: name of the parameter/buffer to update
    :param data: tensor to update the parameter/buffer with
    :param src: rank designated to write. Defaults to the compressed-tensors source rank.
    :param sync: barrier on all ranks after the write. Set False only when this call is not
        reached uniformly by every rank; the caller then owns synchronization.
    """
    if src is None:
        is_writer = is_source_process()
    else:
        is_writer = (not is_distributed()) or dist.get_rank() == src

    if is_writer:
        update_offload_parameter(module, name, data)

    if sync and is_distributed():
        dist.barrier()


def zero_onloaded_grads() -> None:
    """
    Clear `.grad` on the currently-onloaded tensors.
    """
    onloaded = OffloadCache.keep_onloaded_values
    for tensor in onloaded.values():
        tensor.grad = None


def _offloaded_storage_id(cache: OffloadCache, offloaded: torch.Tensor | None):
    """
    Return a hashable identifier for the backing cpu/disk storage of an offloaded
    tensor, or None if the tensor is not offloaded.

    - DiskCache / DistributedDiskCache: tensors are meta tensors whose data lives in a
      safetensors file recorded in `cache.index`. The file path is the shared identity.
    - CPUCache / DistributedCPUCache: tensors are cpu tensors whose data lives in a
      shared-memory file. `untyped_storage()._share_filename_cpu_()[1]` is the filename
      that every rank reconstructs from, i.e. the shared identity.
    """
    if offloaded is None:
        return None

    if isinstance(cache, DiskCache):
        # meta tensor -> safetensors file on disk
        return ("disk", cache.index[offloaded]["safetensors_file"])

    if offloaded.device.type == "cpu":
        # cpu tensor -> shared-memory filename (index [1] of the share handle).
        # A storage that is not in shared memory is a private per-rank copy, which is
        # exactly the failure this check is meant to catch -- so probe is_shared()
        # first instead of calling _share_filename_cpu_(), which would *move* a private
        # storage into shared memory as a side effect.
        storage = offloaded.untyped_storage()
        if not storage.is_shared():
            return ("cpu_local", id(storage))
        return ("cpu_shared", storage._share_filename_cpu_()[1])

    return ("device", offloaded.device.type, offloaded.device.index)


def _unmanaged_storage_id(value: torch.Tensor | None):
    """
    Identity for a tensor living in a plain dict (not an OffloadCache).
    """
    if value is None:
        return None
    return ("unmanaged", str(value.device))


def _is_unmanaged(storage_id) -> bool:
    return isinstance(storage_id, tuple) and bool(storage_id) and storage_id[0] == "unmanaged"


def _collect_offload_storage_ids(model: nn.Module) -> dict[str, object]:
    """
    Map every parameter/buffer name to the identity of its backing storage, without
    onloading any tensor.

    Tensors managed by an OffloadCache get the identity of their shared cpu/disk
    storage. Tensors that live in a plain dict (i.e. the module was never offloaded)
    are reported with an "unmanaged" marker, which the checker treats as a violation.
    """
    storage_ids: dict[str, object] = {}
    with disable_onloading():
        for module_name, module in model.named_modules():
            for collection in (module._parameters, module._buffers):
                is_cache = isinstance(collection, OffloadCache)
                items = (
                    collection.offloaded_values.items()
                    if is_cache
                    else collection.items()
                )
                for name, value in items:
                    full_name = f"{module_name}.{name}" if module_name else name
                    storage_ids[full_name] = (
                        _offloaded_storage_id(collection, value)
                        if is_cache
                        else _unmanaged_storage_id(value)
                    )
    return storage_ids


def assert_shared_offload_storage(model: nn.Module, verbose: bool = True) -> bool:
    """
    Verify two invariants over `model`'s parameters and buffers:

    1. Every (non-None) tensor is managed by an OffloadCache. A tensor in a plain dict
       was never offloaded and is therefore a private per-rank copy.
    2. Every offloaded tensor is backed by storage that is *shared across distributed
       ranks* -- all ranks point at the same shared-memory file (DistributedCPUCache) or
       the same on-disk safetensors file (DistributedDiskCache).

    This catches the failure mode where a rank silently materializes a private copy of a
    tensor instead of reconstructing from the broadcast handle, which wastes cpu/disk and
    breaks in-place updates that are meant to be visible to all ranks.
    """
    local_ids = _collect_offload_storage_ids(model)

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return True

    world_size = dist.get_world_size()
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local_ids)

    # Rank 0 builds the report, then broadcasts it so every rank raises identically.
    mismatches: list[str] = []
    if is_source_process():
        all_names = set().union(*(ids.keys() for ids in gathered))

        for name in sorted(all_names):
            per_rank = [ids.get(name, "<absent>") for ids in gathered]

            # invariant 1: must be managed by OffloadCache on every rank
            bad_ranks = [r for r, sid in enumerate(per_rank) if _is_unmanaged(sid)]
            if bad_ranks:
                mismatches.append(f"{name}: not managed by OffloadCache on ranks {bad_ranks}")
                continue

            # invariant 2: backing storage identity must match across ranks
            if any(sid != per_rank[0] for sid in per_rank):
                mismatches.append(f"{name}: storage differs across ranks -> {per_rank}")

    payload = [mismatches]
    dist.broadcast_object_list(payload, src=0)
    mismatches = payload[0]

    if mismatches:
        report = "\n  ".join(mismatches)
        raise AssertionError(
            f"Offload storage invariants violated across {world_size} ranks:\n  {report}"
        )

    if verbose:
        logging.info(f"[assert_shared_offload_storage] OK: {len(local_ids)} tensors managed by "
                     f"OffloadCache and sharing storage across {world_size} ranks")
    return True
