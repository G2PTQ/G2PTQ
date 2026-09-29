import copy
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.distributed as dist
from compressed_tensors.distributed import greedy_bin_packing, wait_for_comms
from compressed_tensors.offload.dist_utils import as_broadcastable

from utils import dist_utils, compressed_tensors_utils


def make_group_quantizers(quantizer, columns, groupsize):
    """Return one independent quantizer for each weight group."""
    num_groups = 1 if groupsize == -1 else (columns + groupsize - 1) // groupsize
    return [copy.deepcopy(quantizer) for _ in range(num_groups)]


def merge_group_quantizers(quantizers, start, end, groupsize):
    """Merge one batched quantizer list into a per-layer ``WeightQuantizer``."""
    result = copy.deepcopy(quantizers[0])
    result.scale = torch.cat(
        [quantizer.scale[start:end] for quantizer in quantizers], dim=1
    ).clone()
    result.zero = torch.cat(
        [quantizer.zero[start:end] for quantizer in quantizers], dim=1
    ).clone()
    result.weight_groupsize = groupsize
    result._ready = all(quantizer.ready() for quantizer in quantizers)
    return result


def fake_quantize_grouped(quantizers, weight, groupsize):
    """Fake-quantize ``weight`` using its group-indexed quantizer list."""
    flat_rows = weight.shape[0] * weight.shape[1]
    width = weight.shape[-1] if groupsize == -1 else groupsize
    quantized = []
    for group, quantizer in enumerate(quantizers):
        start = group * width
        end = min(start + width, weight.shape[-1])
        values = weight[:, :, start:end]
        q = quantizer.fake_quantize(values.reshape(flat_rows, end - start))[0]
        quantized.append(q.reshape(*weight.shape[:2], end - start))
    return torch.cat(quantized, dim=-1)


@dataclass(eq=False)
class ChunkSpec:
    """A batch of same-shaped modules to be quantized by one quantizer instance.
    """
    chunk_names: tuple[str, ...]
    rows: int
    columns: int

    @property
    def num_layers(self):
        return len(self.chunk_names)


def build_chunk_specs(subset, layer_bsz, group_key_fn=None) -> list[ChunkSpec]:
    """Group ``subset``'s modules by weight shape, then split each group into chunks of
    at most ``layer_bsz`` modules.

    :param subset: mapping of module name -> module.
    :param layer_bsz: max number of modules per chunk.
    :param group_key_fn: optional ``name -> hashable`` returning an extra grouping key,
        so that only modules agreeing on both shape and key share a chunk.
    """
    groups = defaultdict(list)
    for name, module in subset.items():
        key = module.weight.shape
        if group_key_fn is not None:
            key = (key, group_key_fn(name))
        groups[key].append(name)

    specs = []
    for group_names in groups.values():
        rows, columns = subset[group_names[0]].weight.shape
        for chunk_start in range(0, len(group_names), layer_bsz):
            specs.append(ChunkSpec(
                chunk_names=tuple(group_names[chunk_start: chunk_start + layer_bsz]),
                rows=rows,
                columns=columns,
            ))
    return specs


def partition_layers(gptqs, world_size):
    return greedy_bin_packing(
        gptqs,
        world_size,
        item_weight_fn=lambda gptq: gptq.num_layers * gptq.rows * gptq.columns,
    )


def broadcast_quantized_param(gptqs, gptq2rank, export_compressed_tensors=False):
    shared_qparams = ["weight"]
    export_qparams = list(compressed_tensors_utils.Q_PARAMS) if export_compressed_tensors else []

    rank = dist_utils.get_rank()
    dist.barrier()

    # ---- 1. Refresh the in-cache params (``weight``) in place on every rank ----
    # The shared offloaded master was already updated by ``src_rank``; this only refreshes
    # the per-rank onloaded copies used by the subsequent propagation forward.
    pending_comms = []
    for gptq in gptqs:
        src_rank = gptq2rank[gptq]
        for b in range(gptq.num_layers):
            layer = gptq.layers[b]
            for attr in shared_qparams:
                pending_comms.append(
                    dist.broadcast(as_broadcastable(getattr(layer, attr)), src=src_rank, async_op=True)
                )

    if not export_qparams:
        dist.barrier()
        return {}

    # ---- 2. Broadcast the staged export qparams to every rank ----
    # ``compress`` stashes qparams in ``layer._export_qparams``. We broadcast that dict
    # from the owning rank to all ranks and set it on the non-owning ranks.
    broadcasted = {}  # (id(gptq), b)
    for gptq in gptqs:
        src_rank = gptq2rank[gptq]

        # Metadata so ranks that lack the qparams can allocate matching placeholders.
        meta_list = [None]
        if rank == src_rank:
            layer_metas = []
            for b in range(gptq.num_layers):
                qparams = gptq.layers[b].__dict__.get("_export_qparams", None)
                layer_meta = None if qparams is None else {
                    attr: {"shape": tuple(t.shape), "dtype": t.dtype}
                    for attr, t in qparams.items()
                }
                layer_metas.append(layer_meta)
            meta_list = [layer_metas]
        dist.broadcast_object_list(meta_list, src=src_rank)
        qparams_meta = meta_list[0]

        dev = torch.device("cuda")
        for b in range(gptq.num_layers):
            layer = gptq.layers[b]
            layer_meta = qparams_meta[b]
            if layer_meta is None:
                continue
            if rank == src_rank:
                qparams = layer._export_qparams
                qparams = {attr: t.detach().to(dev) for attr, t in qparams.items()}
            else:
                qparams = {
                    attr: torch.empty(m["shape"], dtype=m["dtype"], device=dev)
                    for attr, m in layer_meta.items()
                }
            for attr in export_qparams:
                pending_comms.append(
                    dist.broadcast(as_broadcastable(qparams[attr]), src=src_rank, async_op=True)
                )
            layer._export_qparams = qparams
            broadcasted[(id(gptq), b)] = qparams
    wait_for_comms(pending_comms)

    dist.barrier()
    return broadcasted
