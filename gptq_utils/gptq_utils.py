import copy
import logging
import math
import pprint
import contextlib
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import triton
from compressed_tensors.distributed import wait_for_comms
from compressed_tensors.offload import disable_offloading

from utils import quant_utils, memory_utils, model_utils, offload_utils, compressed_tensors_utils, dist_utils
from gptq_utils.common_utils import (
    build_chunk_specs,
    partition_layers,
    broadcast_quantized_param,
    make_group_quantizers,
    merge_group_quantizers,
)
from gptq_utils.graph_utils import run_graph, validate_tensors
from gptq_utils.triton_utils import gptq_kernels


def _run_torch_compensation(
    weight, hinv, scale, zero, int_weight, error, maxq,
):
    columns = weight.shape[1]
    for column in range(columns):
        w = weight[:, column]
        # Use the shared asymmetric primitives for both modes. Symmetric
        # quantizers provide a zero-filled zero-point tensor.
        quantized, quant_scale, quant_zero = quant_utils.asym_quant(
            w, scale[:, column], zero[:, column], maxq
        )
        q = quant_utils.asym_dequant(quantized, quant_scale, quant_zero)
        column_error = (w - q) / hinv[:, column, column].unsqueeze(1)

        weight[:, column].copy_(q)
        int_weight[:, column].copy_(quantized)
        error[:, column].copy_(column_error)
        if column + 1 < columns:
            weight[:, column + 1 :].baddbmm_(
                hinv[:, column, column + 1 :].unsqueeze(2),
                column_error.unsqueeze(1),
                beta=1,
                alpha=-1,
            )


def _run_triton_compensation(
    weight, hinv, scale, zero, int_weight, error, maxq,
):
    columns = weight.shape[1]
    tile_size = triton.next_power_of_2(columns)
    for column in range(columns):
        gptq_kernels.launch_quantize(
            weight, hinv, scale, zero, int_weight, error, column,
            maxq, tile_size,
        )
        gptq_kernels.launch_update(weight, hinv, error, column, tile_size)


def run_eager(
    weight: torch.Tensor,
    hinv: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
    backend: str,
    int_weight: torch.Tensor | None = None,
    error: torch.Tensor | None = None,
):
    """Run one prepared GPTQ compensation block eagerly."""
    validate_tensors(
        f"{backend} GPTQ", weight, hinv, scale, zero, int_weight, error
    )

    _run_compensation = (
        _run_torch_compensation
        if backend == "torch"
        else _run_triton_compensation
    )
    _run_compensation(
        weight, hinv, scale, zero, int_weight, error, maxq
    )
    return weight, int_weight, error


class GPTQ:
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
        self.nsamples = torch.tensor([0] * self.num_layers, device=self.dev)

    def _maybe_onload_hessian(self, idx=None):
        """Onload the Hessian to the execution device on enter, offload on exit.
        """
        return offload_utils.onload_attrs(self, ("H",), enabled=self.offload_hessians,
                                          device=self.dev)

    def add_batch(self, idx, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H[idx] *= self.nsamples[idx] / (self.nsamples[idx] + tmp)
        self.nsamples[idx] += tmp
        inp = math.sqrt(2 / self.nsamples[idx]) * inp.float()
        self.H[idx] += inp.matmul(inp.t())

    def _run_compensation_block(
        self,
        weight,
        hinv,
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
            "scale": scale.transpose(1, 2).contiguous(),
            "zero": zero.transpose(1, 2).contiguous(),
            "maxq": quantizer.maxq,
            "backend": backend,
            "int_weight": torch.empty_like(weight),
            "error": torch.empty_like(weight),
        }
        if graph:
            qweight, int_weight, error = run_graph(
                run_eager,
                params,
                input_names=("weight", "hinv", "scale", "zero"),
                output_names=("weight", "int_weight", "error"),
                name="GPTQ compensation block",
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
                W1,
                Hinv1,
                Scale1,
                Zero1,
                backend,
                graph,
            )
            Q[:, :, i1:i2] = Q1
            W_int[:, :, i1:i2] = W_int1
            Scale[:, :, i1:i2] = Scale1

            W[:, :, i2:].baddbmm_(
                Err1,
                Hinv[:, i1:i2, i2:],
                beta=1,
                alpha=-1,
            )

        return Q, W_int, Scale

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
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

        Q, W_int, Scale = self._run_compensation(
            W,
            Hinv,
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
        self.nsamples = None
        memory_utils.cleanup_memory()


def reduce_to_target_rank(gptqs: list[GPTQ], gptq2rank):
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
                    gptq.nsamples[b].zero_()
                else:
                    gptq.H[b] *= gptq.nsamples[b] / global_nsamples[b]
                    pending_comms.append(
                        dist.reduce(
                            gptq.H[b],
                            op=dist.ReduceOp.SUM,
                            dst=target_rank,
                            async_op=True,
                        )
                    )
            if rank == target_rank:
                gptq.nsamples.copy_(global_nsamples)
            wait_for_comms(pending_comms)
    dist.barrier()


@torch.no_grad()
def gptq_fwrd(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    """
    From GPTQ repo
    """
    logging.info("-----GPTQ Quantization-----")

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

    topk_buffer = analyzer.alloc_index_buffer(nsamples, model.seqlen, inps.device)

    quantizers = {}
    pbar = tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers")
    for i in pbar:
        layer = layers[i]
        with disable_offloading():
            sequential = analyzer.get_sequential_quantizable_module_names(layer)
            full = analyzer.get_quantizable_modules(layer)
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
                        gptq = GPTQ([subset[n] for n in spec.chunk_names],
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
                            (inps, topk_buffer), nsamples, args.bsz, dev, args.offload_inps
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
                (inps, topk_buffer), nsamples, args.bsz, dev, args.offload_inps
            ):
                model_utils.run_block_layer(
                    analyzer, layer, inps_batch, out_buffer=inps,
                    prev_topk_indices=topk_batch, index_buffer=topk_buffer,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    block_internals=block_internals,
                    layer_idx=i, sample_idx=j, bsz=args.bsz, dev=dev,
                    **kwargs,
                )

        del layer
        del gptqs, specs
        memory_utils.cleanup_memory()

    analyzer.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(args, analyzer: model_utils.ModelAnalyzer, dev):
    layers = analyzer.get_layers()

    quantizers = {}
    for i in tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers"):
        layer = layers[i]

        with disable_offloading():
            full = analyzer.get_quantizable_modules(layer)
            sequential = analyzer.get_sequential_quantizable_module_names(layer)
            names = [n for ns in sequential for n in ns]
            subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}

            groups = defaultdict(list)
            for name in subset:
                shape = subset[name].weight.shape
                groups[shape].append(name)

            for shape, group_names in groups.items():
                layer_weight_bits = args.w_bits
                w_groupsize = args.w_groupsize
                # Split the group into batches
                for chunk_start in range(0, len(group_names), args.layer_bsz):
                    chunk_names = group_names[chunk_start: chunk_start + args.layer_bsz]

                    # Stack weights: (B, rows, columns)
                    W_stacked = torch.stack([subset[n].weight.data for n in chunk_names])
                    weight_dtype = W_stacked.dtype
                    B, rows, columns = W_stacked.shape
                    W_flat = W_stacked.view(B * rows, columns)

                    quantizer = quant_utils.WeightQuantizer()
                    quantizer.configure(
                        layer_weight_bits,
                        perchannel=True,
                        sym=(not (args.w_asym)),
                        mse=args.w_clip,
                        weight_groupsize=w_groupsize,
                    )

                    quantizer.find_params(W_flat)
                    q_flat, int_weight_flat, scale_flat = quantizer.fake_quantize(W_flat)

                    q_stacked = q_flat.view(B, rows, columns)
                    int_weight_stacked = int_weight_flat.view(B, rows, columns)
                    scale_stacked = scale_flat.view(B, rows, -1)

                    for idx, name in enumerate(chunk_names):
                        W_orig_shape = subset[name].weight.shape
                        q = q_stacked[idx]

                        if args.export_compressed_tensors:
                            groupsize = columns if w_groupsize == -1 else w_groupsize
                            scale_expanded = scale_stacked[idx].unsqueeze(-1).expand(-1, -1, groupsize).reshape(W_orig_shape)
                            compressed_tensors_utils.compress(
                                layer=subset[name],
                                Scale=scale_expanded,
                                W_int=int_weight_stacked[idx],
                                bits=quantizer.bits,
                                groupsize=groupsize,
                            )
                            compressed_tensors_utils.register_qparam_buffers([subset[name]])

                        offload_utils.update_shared_offload_parameter(
                            subset[name], "weight", q.to(weight_dtype), sync=False
                        )

                        q_inst = copy.deepcopy(quantizer)
                        for attr_name in dir(q_inst):
                            attr = getattr(q_inst, attr_name)
                            if isinstance(attr, torch.Tensor) and attr.shape and attr.shape[0] == B * rows:
                                # Slice flattened tensors (like scale or zero_point) for this specific layer
                                setattr(q_inst, attr_name, attr[idx * rows : (idx + 1) * rows].clone())

                        quantizers["model.layers.%d.%s" % (i, name)] = q_inst.cpu()

        dist.barrier()
        torch.cuda.empty_cache()
        del layer

    memory_utils.cleanup_memory(verbos=True)
    return quantizers
