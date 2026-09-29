import logging
import math
import functools
import copy
import pprint
import contextlib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import triton
from compressed_tensors.distributed import wait_for_comms
from compressed_tensors.offload import disable_offloading

from utils import quant_utils, memory_utils, model_utils, offload_utils, compressed_tensors_utils, dist_utils
from gptq_utils.common_utils import (
    broadcast_quantized_param,
    build_chunk_specs,
    make_group_quantizers,
    merge_group_quantizers,
    partition_layers,
)
from gptq_utils.graph_utils import run_graph, validate_tensors
from gptq_utils.triton_utils import gptaq_kernels


def _run_torch_compensation(
    weight, hinv, p, scale, zero, column_weight, int_weight, error, maxq,
):
    columns = weight.shape[1]
    for column in range(columns):
        column_weight.copy_(weight[:, column])
        quantized, quant_scale, quant_zero = quant_utils.asym_quant(
            column_weight, scale[:, column], zero[:, column], maxq
        )
        q = quant_utils.asym_dequant(quantized, quant_scale, quant_zero)
        column_error = (
            column_weight - q
        ) / hinv[:, column, column].unsqueeze(1)

        weight[:, column].copy_(q)
        int_weight[:, column].copy_(quantized)
        error[:, column].copy_(column_error)
        if column + 1 < columns:
            weight[:, column + 1 :].baddbmm_(
                hinv[:, column, column + 1 :].unsqueeze(2),
                column_error.unsqueeze(1),
                beta=1,
                alpha=-1,
            ).baddbmm_(
                p[:, column, column + 1 :].unsqueeze(2),
                column_weight.unsqueeze(1),
                beta=1,
                alpha=1,
            )


def _run_triton_compensation(
    weight, hinv, p, scale, zero, column_weight, int_weight, error, maxq,
):
    columns = weight.shape[1]
    tile_size = triton.next_power_of_2(columns)
    for column in range(columns):
        gptaq_kernels.launch_quantize(
            weight, hinv, scale, zero, column_weight, int_weight, error,
            column, maxq, tile_size,
        )
        gptaq_kernels.launch_update(
            weight, hinv, p, column_weight, error, column, tile_size
        )


def run_eager(
    weight: torch.Tensor,
    hinv: torch.Tensor,
    p: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
    backend: str,
    column_weight: torch.Tensor | None = None,
    int_weight: torch.Tensor | None = None,
    error: torch.Tensor | None = None,
):
    """Run one prepared GPTAQ compensation block eagerly."""
    validate_tensors(
        f"{backend} GPTAQ",
        weight, hinv, p, scale, zero, column_weight, int_weight, error,
    )
    _run_compensation = (
        _run_torch_compensation
        if backend == "torch"
        else _run_triton_compensation
    )
    _run_compensation(
        weight, hinv, p, scale, zero, column_weight, int_weight, error,
        maxq,
    )
    return weight, int_weight, error


class GPTAQ:
    def __init__(self, layers, offload_hessians: bool = False):
        self.layers = layers
        self.num_layers = len(layers)
        self.offload_hessians = offload_hessians
        weight = layers[0].weight
        self.dev = weight.device

        self.rows, self.columns = weight.shape

        H_device = "cpu" if offload_hessians else self.dev
        self.H = offload_utils.alloc_pinned(
            (self.num_layers, self.columns, self.columns), device=H_device, pin=False,
        )
        self.dXXT = offload_utils.alloc_pinned(
            (self.num_layers, self.columns, self.columns), device=H_device, pin=False,
        )
        self.nsamples = torch.tensor([0] * self.num_layers, device=self.dev)
        self.fp_inp = [[] for _ in range(self.num_layers)]

    def _maybe_onload_hessian(self):
        """Onload `self.H`/`self.dXXT` to the execution device on enter, offload back to
        CPU on exit.
        """
        return offload_utils.onload_attrs(self, ("H", "dXXT"), enabled=self.offload_hessians,
                                          device=self.dev)

    def add_batch(self, idx, inp, out):
        fp_inp = self.fp_inp[idx][0].to(inp, non_blocking=True).t()
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H[idx] *= self.nsamples[idx] / (self.nsamples[idx] + tmp)
        self.dXXT[idx] *= self.nsamples[idx] / (self.nsamples[idx] + tmp)
        self.nsamples[idx] += tmp
        inp = math.sqrt(2 / self.nsamples[idx]) * inp.float()
        self.H[idx] += inp.matmul(inp.t())

        dX = fp_inp.float() * math.sqrt(2 / self.nsamples[idx]) - inp
        self.dXXT[idx] += dX.matmul(inp.t())

        del self.fp_inp[idx][0]

    def _run_compensation_block(
        self,
        weight,
        hinv,
        p,
        scale,
        zero,
        backend,
        graph,
    ):
        quantizer = self.quantizer[0]
        weight = weight.transpose(1, 2).contiguous()
        params = {
            "weight": weight,
            "hinv": hinv.contiguous(),
            "p": p.contiguous(),
            "scale": scale.transpose(1, 2).contiguous(),
            "zero": zero.transpose(1, 2).contiguous(),
            "maxq": quantizer.maxq,
            "backend": backend,
            "column_weight": torch.empty(
                weight.shape[0], weight.shape[2],
                dtype=weight.dtype, device=weight.device,
            ),
            "int_weight": torch.empty_like(weight),
            "error": torch.empty_like(weight),
        }
        if graph:
            qweight, int_weight, error = run_graph(
                run_eager,
                params,
                input_names=("weight", "hinv", "p", "scale", "zero"),
                output_names=("weight", "int_weight", "error"),
                name="GPTAQ compensation block",
            )
        else:
            qweight, int_weight, error = run_eager(**params)
        qweight, int_weight, error = (
            tensor.transpose(1, 2).contiguous()
            for tensor in (qweight, int_weight, error)
        )
        return qweight, int_weight, error

    def _run_compensation(
        self,
        W,
        Hinv,
        P,
        blocksize,
        groupsize,
        static_groups,
        qparam_perm,
        backend,
        graph,
        H_diag=None,
    ):
        Q = torch.zeros_like(W)
        W_int = torch.zeros_like(W)
        Scale = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W1 = W[:, :, i1:i2].clone()
            Hinv1 = Hinv[:, i1:i2, i1:i2]
            P1 = P[:, i1:i2, i1:i2]

            if groupsize == -1:
                column_quantizers = [self.quantizer[0]] * (i2 - i1)
            elif static_groups:
                indices = range(i1, i2)
                if qparam_perm is not None:
                    indices = qparam_perm[i1:i2].detach().cpu().tolist()
                column_quantizers = [
                    self.quantizer[index // groupsize] for index in indices
                ]
            else:
                column_quantizers = []
                for i in range(i1, i2):
                    quantizer = self.quantizer[i // groupsize]
                    if i % groupsize == 0:
                        quantizer.find_params(
                            W[:, :, i : i + groupsize].reshape(
                                self.num_layers * self.rows, -1
                            ),
                            None if H_diag is None else H_diag[:, i : i + groupsize],
                        )
                    column_quantizers.append(quantizer)

            Scale1 = torch.stack(
                [quantizer.scale for quantizer in column_quantizers], dim=1
            ).reshape(W1.shape)
            Zero1 = torch.stack(
                [quantizer.zero for quantizer in column_quantizers], dim=1
            ).reshape(W1.shape)
            Q1, W_int1, Err1 = self._run_compensation_block(
                W1, Hinv1, P1, Scale1, Zero1, backend, graph
            )
            Q[:, :, i1:i2] = Q1
            W_int[:, :, i1:i2] = W_int1
            Scale[:, :, i1:i2] = Scale1
            W[:, :, i2:].baddbmm_(
                Err1,
                Hinv[:, i1:i2, i2:],
                beta=1,
                alpha=-1,
            ).baddbmm_(
                Q1,
                P[:, i1:i2, i2:],
                beta=1,
                alpha=1,
            )

        return Q, W_int, Scale

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        alpha=0.25,
        export_compressed_tensors=False,
        backend="torch",
        graph=False,
    ):
        W = torch.stack([layer.weight.data.clone() for layer in self.layers]).float()   # (B, rows, columns)

        W_flat = W.view(self.num_layers * self.rows, self.columns)  # (B * rows, columns)
        hclip = self.quantizer[0].hclip

        with self._maybe_onload_hessian():
            H = self.H
            self.H = None
            H_diag = (
                torch.diagonal(H, dim1=-2, dim2=-1).clone()
                if hclip
                else None
            )

            if groupsize == -1 or static_groups:
                for i, quantizer in enumerate(self.quantizer):
                    if not quantizer.ready():
                        if groupsize == -1:
                            quantizer.find_params(W_flat, H_diag)
                        else:
                            quantizer.find_params(
                                W_flat[:, i * groupsize : (i + 1) * groupsize],
                                None
                                if H_diag is None
                                else H_diag[:, i * groupsize : (i + 1) * groupsize],
                            )

            for b in range(self.num_layers):
                if (H[b] == 0).all():  # Fall back to RTN if no calibration data is provided
                    H[b] = torch.eye(self.columns).to(H[b])
                dead = torch.diag(H[b]) == 0
                H[b, dead, dead] = 1
                W[b, :, dead] = 0
                self.dXXT[b, :, dead] = 0

            if actorder:
                # We use a shared permutation across all layers in the batch
                perm = torch.argsort(torch.mean(torch.diagonal(H, dim1=1, dim2=2), dim=0), descending=True)
                invperm = torch.argsort(perm)

            # Iterate over linear layers to avoid exessive memory allocation
            Hinv = torch.empty_like(H)
            for b in range(self.num_layers):
                if actorder:
                    W[b] = W[b][:, perm]
                    H[b] = H[b][perm][:, perm]
                    self.dXXT[b] = self.dXXT[b][perm][:, perm]

                damp_percent = percdamp
                damp_auto_increment = 0.0015
                while 1 > damp_percent > 0:
                    try:
                        damp = damp_percent * torch.mean(torch.diag(H[b]))
                        diag = torch.arange(self.columns, device=self.dev)
                        H_tmp = H[b].clone()
                        H_tmp[diag, diag] += damp

                        H_chol = torch.linalg.cholesky(H_tmp)
                        Hinv[b] = torch.cholesky_inverse(H_chol)
                        Hinv[b] = torch.linalg.cholesky(Hinv[b], upper=True)
                        if torch.isnan(Hinv[b]).any().item():
                            raise torch._C._LinAlgError
                        break
                    except torch._C._LinAlgError as e:
                        logging.warning(f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{damp_auto_increment:.5f}`")
                        damp_percent += damp_auto_increment

                if not (0 < damp_percent < 1):
                    raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")
                # memory_utils.cleanup_memory()

            if H_diag is not None and actorder:
                H_diag = H_diag[:, perm].contiguous()

            # scale it by alpha due to collection of dXXT and H
            dXXT_HinvT = torch.bmm(self.dXXT, Hinv.transpose(1, 2))
            dXXT_HinvT_triu = torch.triu(dXXT_HinvT, diagonal=1)
            P = alpha * torch.bmm(dXXT_HinvT_triu, Hinv)

        Q, W_int, Scale = self._run_compensation(
            W,
            Hinv,
            P,
            blocksize,
            groupsize,
            static_groups,
            perm if static_groups and actorder else None,
            backend,
            graph,
            H_diag=H_diag,
        )

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, :, invperm]
            W_int = W_int[:, :, invperm]
            Scale = Scale[:, :, invperm]

        for b, layer in enumerate(self.layers):
            if export_compressed_tensors:
                compressed_tensors_utils.compress(
                    layer=layer,
                    Scale=Scale[b],
                    W_int=W_int[b],
                    bits=self.quantizer[0].bits,
                    groupsize=self.columns if groupsize == -1 else groupsize,
                )

            weight = layer.weight
            q_weight = Q[b].reshape(weight.shape).to(weight.dtype)
            if torch.any(torch.isnan(q_weight)):
                logging.warning(f"NaN in weights of layer at batch index {b}")
                raise ValueError("NaN in weights")
            offload_utils.update_shared_offload_parameter(
                layer, "weight", q_weight, src=dist_utils.get_rank(), sync=False
            )

    def free(self):
        self.H = None
        self.Losses = None
        self.dXXT = None
        self.nsamples = None
        self.fp_inp = None
        memory_utils.cleanup_memory()


def reduce_to_target_rank(gptqs: list[GPTAQ], gptq2rank):
    rank = dist_utils.get_rank()
    dist.barrier()
    for gptq in gptqs:
        with gptq._maybe_onload_hessian():
            pending_comms = []
            target_rank = gptq2rank[gptq]
            global_nsamples = gptq.nsamples.clone()
            dist.all_reduce(
                global_nsamples,
                op=dist.ReduceOp.SUM,
            )
            for b in range(gptq.num_layers):
                if global_nsamples[b].item() == 0:
                    gptq.H[b].zero_()
                    gptq.dXXT[b].zero_()
                    gptq.nsamples[b].zero_()
                else:
                    gptq.H[b] *= gptq.nsamples[b] / global_nsamples[b]
                    gptq.dXXT[b] *= gptq.nsamples[b] / global_nsamples[b]
                    pending_comms.extend([
                        dist.reduce(
                            gptq.H[b],
                            op=dist.ReduceOp.SUM,
                            dst=target_rank,
                            async_op=True,
                        ),
                        dist.reduce(
                            gptq.dXXT[b],
                            op=dist.ReduceOp.SUM,
                            dst=target_rank,
                            async_op=True,
                        )
                    ])
            if rank == target_rank:
                gptq.nsamples.copy_(global_nsamples)
            wait_for_comms(pending_comms)
    dist.barrier()


class FPInputsCache:
    """Saves the full-precision input of each quantizable module in a layer.

    Modules receiving the *identical* input tensor share one host buffer, as declared by the spec's
    ``get_shared_input_groups``.
    """
    def __init__(self, sequential, modules, nsamples, seqlen, dtype, offload_inps=False,
                 shared_input_groups=None):
        self.fp_cache = {}
        self.fp_buffers = {}
        self.cursor = {}
        self.names = []
        for names in sequential:
            self.names += names
        self.handles = []
        self.offload_inps = offload_inps
        #: group_key -> [remaining_members, dst_slice, src_data_ptr], reset once a group completes.
        self._pending = {}

        for name in self.names:
            self.fp_cache[name] = []

        self.groups, self.group_of = self._build_shared_input_groups(shared_input_groups)
        if not self.offload_inps:
            return

        total_tokens = nsamples * seqlen
        for group_key, group in self.groups.items():
            in_features = {
                modules.get(n, modules.get(n + ".module", None)).weight.shape[1] for n in group
            }
            assert len(in_features) == 1, (
                f"shared-input group {group} disagrees on in_features {in_features}; "
                f"get_shared_input_groups grouped modules that cannot share an input tensor"
            )
            self.fp_buffers[group_key] = offload_utils.alloc_pinned(
                total_tokens, in_features.pop(), dtype=dtype, device="cpu", pin=False,
            )
            self.cursor[group_key] = 0

    def _build_shared_input_groups(self, shared_input_groups):
        """Filter the layer's partition to just this cache's names (e.g. drops attention if --ignore_attn)."""
        in_cache = set(self.names)
        groups, group_of = {}, {}
        for group in (shared_input_groups or ()):
            members = tuple(n for n in group if n in in_cache)
            if not members:
                continue
            groups[members[0]] = members
            for name in members:
                group_of[name] = members[0]
        return groups, group_of

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach()
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        if not self.offload_inps:
            self.fp_cache[name].append(inp)
            return

        group_key = self.group_of[name]
        group = self.groups[group_key]
        pending = self._pending.get(group_key)

        identity = (inp.data_ptr(), tuple(inp.shape))
        if pending is not None:
            # A group-mate already copied this batch's tensor. The declaration says every member
            # sees it, so a mismatch is a spec bug.
            dst = pending[1]
            assert pending[2] == identity, (
                f"shared-input group {group} member {name!r} received a different tensor "
                f"{identity} than the member that fired first {pending[2]}; "
                f"get_shared_input_groups declared them as sharing an input but they do not"
            )
            pending[0] -= 1
        else:
            # First member of this group to fire this batch; do the copy.
            n = inp.shape[0]
            cursor = self.cursor[group_key]
            buffer = self.fp_buffers[group_key]
            assert cursor + n <= buffer.shape[0], (
                f"FP input buffer for group {group} overflowed ({cursor + n} > {buffer.shape[0]})"
            )
            dst = buffer[cursor: cursor + n]
            offload_utils.current_d2h_queue().copy_(dst, inp.contiguous())
            self.cursor[group_key] = cursor + n
            pending = self._pending[group_key] = [len(group) - 1, dst, identity]

        self.fp_cache[name].append(dst)
        if pending[0] <= 0:
            # Every member has fired; drop the entry so the next batch cannot alias a stale slice
            # if the allocator hands back the same address.
            del self._pending[group_key]

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full.get(name, full.get(name + ".module", None)).register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear_cache(self):
        self.fp_cache = {}
        self.fp_buffers = {}
        self.cursor = {}
        self._pending = {}
        memory_utils.cleanup_memory()


@torch.no_grad()
def gptq_fwrd(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    logging.info('-----GPTAQ Quantization-----')

    model = analyzer.model
    use_cache = analyzer.config.use_cache
    analyzer.config.use_cache = False
    layers = analyzer.get_layers()
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    nsamples = len(dataloader)

    dtype = next(iter(model.parameters())).dtype
    inps = offload_utils.alloc_pinned(
        (nsamples, model.seqlen, analyzer.residual_width), dtype=dtype,
        device="cpu" if args.offload_inps else dev, pin=False,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module, bsz):
            super().__init__()
            self.module = module
            self.bsz = bsz
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, *args, **kwargs):
            inps[cache["i"]: cache["i"] + self.bsz] = inp.to(inps.device)
            cache["i"] += self.bsz
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs.get("position_ids")
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0], args.bsz)
    with (
        disable_offloading(),
        analyzer.capture_block_internals() as block_internals
    ):
        for i in range(0, nsamples, args.bsz):
            try:
                input_ids = torch.cat([data[0] for data in dataloader[i: i + args.bsz]], dim=0)
                model(input_ids.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    memory_utils.cleanup_memory()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    kwargs = {}

    quantizers = {}

    fp_inps = offload_utils.clone_pinned(inps)
    if block_internals is not None:
        block_internals_fp = copy.deepcopy(block_internals)
    else:
        block_internals_fp = None

    topk_buffer_fp = analyzer.alloc_index_buffer(nsamples, model.seqlen, fp_inps.device)
    topk_buffer_q = analyzer.alloc_index_buffer(nsamples, model.seqlen, inps.device)

    pbar = tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers")
    for i in pbar:
        layer = layers[i]
        with disable_offloading():
            sequential = analyzer.get_sequential_quantizable_module_names(layer)
            full = analyzer.get_quantizable_modules(layer)
            fp_inputs_cache = FPInputsCache(
                sequential, full, nsamples, model.seqlen, dtype, offload_inps=args.offload_inps,
                shared_input_groups=analyzer.get_shared_input_groups(layer),
            )

            bits_config = quant_utils.disable_act_quant(layer)
            fp_inputs_cache.add_hook(full)
            for j, (fp_inps_batch, topk_batch) in offload_utils.prefetch_generator(
                (fp_inps, topk_buffer_fp), nsamples, args.bsz, dev, args.offload_inps
            ):
                model_utils.run_block_layer(
                    analyzer, layer, fp_inps_batch, out_buffer=fp_inps,
                    prev_topk_indices=topk_batch, index_buffer=topk_buffer_fp,
                    attention_mask=attention_mask, position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    block_internals=block_internals_fp,
                    layer_idx=i, sample_idx=j, bsz=args.bsz, dev=dev,
                    **kwargs,
                )
            fp_inputs_cache.clear_hook()
            quant_utils.enable_act_quant(layer, bits_config)
            memory_utils.cleanup_memory()

            for names in sequential:
                subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}

                specs = build_chunk_specs(subset, args.layer_bsz)
                if world_size > 1:
                    specs, rank2spec, spec2rank = partition_layers(specs, world_size)
                onload_bsz = len(specs) if args.onload_gptq_bsz <= 0 else args.onload_gptq_bsz

                gptqs, gptqs_partition = [], []
                gptq2rank, gptq2chunk_names = {}, {}

                # Accumulate Hessians in batches so at most `onload_bsz` gptq instances are resident
                # on the execution device at once. Each batch costs one full forward over the block.
                for batch_start in range(0, len(specs), onload_bsz):
                    specs_batch = specs[batch_start: batch_start + onload_bsz]

                    # Build this batch's instances.
                    gptqs_batch = []
                    for spec in specs_batch:
                        gptq = GPTAQ([subset[n] for n in spec.chunk_names],
                                     offload_hessians=args.offload_hessians)
                        quantizer = quant_utils.WeightQuantizer()
                        quantizer.configure(
                            args.w_bits,
                            perchannel=True,
                            sym=not (args.w_asym),
                            mse=args.w_clip,
                            hclip=args.w_hclip,
                        )
                        gptq.quantizer = make_group_quantizers(
                            quantizer, spec.columns, args.w_groupsize
                        )
                        for idx, name in enumerate(spec.chunk_names):
                            gptq.fp_inp[idx] = fp_inputs_cache.fp_cache[name]

                        gptqs_batch.append(gptq)
                        gptq2chunk_names[gptq] = spec.chunk_names
                        if world_size > 1:
                            gptq2rank[gptq] = spec2rank[spec]

                    handles = []
                    for gptq in gptqs_batch:
                        chunk_names = gptq2chunk_names[gptq]
                        for idx, name in enumerate(chunk_names):
                            def add_batch(inst, current_idx):
                                def tmp(_, inp, out):
                                    inst.add_batch(current_idx, inp[0].data, out.data)
                                return tmp
                            handles.append(subset[name].register_forward_hook(add_batch(gptq, idx)))

                    with contextlib.ExitStack() as onload_stack:
                        for gptq in gptqs_batch:
                            onload_stack.enter_context(gptq._maybe_onload_hessian())
                        for j, (inps_batch, topk_batch) in offload_utils.prefetch_generator(
                            (inps, topk_buffer_q), nsamples, args.bsz, dev, args.offload_inps
                        ):
                            _ = model_utils.run_block_layer(
                                analyzer, layer, inps_batch,
                                prev_topk_indices=topk_batch,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                                block_internals=block_internals,
                                layer_idx=i, sample_idx=j, bsz=args.bsz, dev=dev,
                                **kwargs,
                            )
                    for h in handles:
                        h.remove()

                    # Reduce this batch to its owner ranks now, then free the calibration
                    # stats for gptqs this rank will not quantize.
                    if world_size > 1:
                        reduce_to_target_rank(gptqs_batch, gptq2rank)
                    for spec, gptq in zip(specs_batch, gptqs_batch):
                        if world_size > 1 and spec2rank[spec] != rank:
                            gptq.free()
                        else:
                            gptqs_partition.append(gptq)
                    gptqs.extend(gptqs_batch)
                    memory_utils.cleanup_memory()

                for gptq in gptqs_partition:
                    chunk_names = gptq2chunk_names[gptq]
                    display_name = chunk_names[0] + (f" (+{len(chunk_names)-1} batched)" if len(chunk_names) > 1 else "")
                    pbar.set_postfix(module=f"layers.{i}.{display_name}")
                    layer_w_groupsize = args.w_groupsize
                    gptq.fasterquant(
                        percdamp=args.percdamp,
                        groupsize=layer_w_groupsize,
                        actorder=args.act_order,
                        static_groups=args.act_order,
                        export_compressed_tensors=args.export_compressed_tensors,
                        backend=args.gptq_backend,
                        graph=args.gptq_graph,
                    )

                    # Merge the group list back into one saved quantizer per layer.
                    rows = subset[chunk_names[0]].weight.shape[0]
                    for idx, name in enumerate(chunk_names):
                        quantizers["model.layers.%d.%s" % (i, name)] = merge_group_quantizers(
                            gptq.quantizer,
                            idx * rows,
                            (idx + 1) * rows,
                            args.w_groupsize,
                        )

                    gptq.free()

                if world_size > 1:
                    broadcast_quantized_param(gptqs, gptq2rank, export_compressed_tensors=args.export_compressed_tensors)

                # Register the staged qparams as buffers
                if args.export_compressed_tensors:
                    compressed_tensors_utils.register_qparam_buffers(
                        layer for gptq in gptqs for layer in gptq.layers
                    )

            for j, (inps_batch, topk_batch) in offload_utils.prefetch_generator(
                (inps, topk_buffer_q), nsamples, args.bsz, dev, args.offload_inps
            ):
                model_utils.run_block_layer(
                    analyzer, layer, inps_batch, out_buffer=inps,
                    prev_topk_indices=topk_batch, index_buffer=topk_buffer_q,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    block_internals=block_internals,
                    layer_idx=i, sample_idx=j, bsz=args.bsz, dev=dev,
                    **kwargs,
                )

        fp_inputs_cache.clear_cache()
        del layer
        del gptqs, specs
        memory_utils.cleanup_memory()

    analyzer.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info('-----GPTAQ Quantization Done-----\n')
    return quantizers
