import logging
import os
import math
import pprint
import contextlib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
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
from gptq_utils.gptq_utils import GPTQ


class GPTQGuided(GPTQ):
    def __init__(self, layers, saliencies, num_groups: int, offload_hessians: bool = False):
        self.layers = layers
        self.num_layers = len(layers)
        self.offload_hessians = offload_hessians
        weight = layers[0].weight
        self.dev = weight.device

        self.rows, self.columns = weight.shape
        self.num_groups = num_groups

        # saliencies is a list of tensors of shape (N, seq_len, G)
        self.saliencies = [saliency.float() for saliency in saliencies]
        H_device = "cpu" if offload_hessians else self.dev
        self.H = offload_utils.alloc_pinned(
            (self.num_layers, self.num_groups, self.columns, self.columns),
            device=H_device, pin=False,
        )
        self.act_square = torch.zeros(
            (self.num_layers, self.columns), device=self.dev
        )
        self.index = torch.tensor([0] * self.num_layers, device=self.dev)

        assert self.rows % self.num_groups == 0, (
            f"Number of rows ({self.rows}) must be divisible "
            f"by num_groups ({self.num_groups})"
        )

    def _maybe_onload_hessian(self):
        """Onload `self.H` to the execution device on enter, offload back to CPU on exit.
        """
        return offload_utils.onload_attrs(self, ("H",), enabled=self.offload_hessians,
                                          device=self.dev)

    @torch.no_grad()
    def add_batch(self, idx, inp, out):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        else:
            assert inp.dim() == 3, "Input must be 2D or 3D. Got %dD." % inp.dim()

        bsz = inp.shape[0]
        sal_batch = self.saliencies[idx][self.index[idx]: self.index[idx] + bsz].to(self.dev)
        self.index[idx] += bsz

        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
            sal_batch = sal_batch.reshape(-1, sal_batch.shape[-1])

        inp = inp.float()
        sal_batch = sal_batch.float()
        n_tokens = inp.shape[0]

        sal_weighted_inp = torch.einsum("nj,ng->njg", inp, sal_batch)
        block = torch.einsum("ni,njg->gij", inp, sal_weighted_inp)
        self.H[idx].add_(block)
        self.act_square[idx].add_((inp ** 2).sum(0), alpha=1 / n_tokens)

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

        rows_per_sub = self.rows // self.num_groups

        W_sub = W.contiguous().view(self.num_layers * self.num_groups, rows_per_sub, self.columns)

        with self._maybe_onload_hessian():
            H = self.H
            self.H = None
            H_sub = H.view(self.num_layers * self.num_groups, self.columns, self.columns)
            H_diag = (
                torch.diagonal(H_sub, dim1=-2, dim2=-1).clone()
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

            for b in range(self.num_layers * self.num_groups):
                if (H_sub[b] == 0).all():  # Fall back to RTN if no calibration data is provided
                    H_sub[b] = torch.eye(self.columns, device=self.dev)
                dead = torch.diag(H_sub[b]) == 0
                H_sub[b, dead, dead] = 1
                W_sub[b, :, dead] = 0

            if actorder:
                # We use a shared permutation across all layers in the batch
                perm = torch.argsort(torch.mean(self.act_square, dim=0), descending=True)
                invperm = torch.argsort(perm)

            # Iterate over linear layers to avoid exessive memory allocation
            Hinv = torch.empty_like(H_sub)
            for b in range(self.num_layers * self.num_groups):
                if actorder:
                    W_sub[b] = W_sub[b][:, perm]
                    H_sub[b] = H_sub[b][perm][:, perm]

                damp_percent = percdamp
                damp_auto_increment = 0.0015
                while 1 > damp_percent > 0:
                    try:
                        damp = damp_percent * torch.mean(torch.diag(H_sub[b]))
                        diag = torch.arange(self.columns, device=self.dev)
                        H_tmp = H_sub[b].clone()
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
            W_sub,
            Hinv,
            blocksize,
            groupsize,
            static_groups,
            perm if static_groups and actorder else None,
            backend,
            graph,
            H_diag=H_diag,
        )

        if actorder:
            Q = Q[:, :, invperm]
            W_int = W_int[:, :, invperm]
            Scale = Scale[:, :, invperm]

        Q = Q.view(self.num_layers, self.rows, self.columns)
        W_int = W_int.view(self.num_layers, self.rows, self.columns)
        Scale = Scale.view(self.num_layers, self.rows, self.columns)

        torch.cuda.synchronize()

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
        self.act_square = None
        self.saliencies = None
        self.index = None
        memory_utils.cleanup_memory()


def reduce_to_target_rank(gptqs: list[GPTQGuided], gptq2rank):
    dist.barrier()
    for gptq in gptqs:
        with gptq._maybe_onload_hessian():
            pending_comms = []
            target_rank = gptq2rank[gptq]
            pending_comms.extend([
                dist.reduce(
                    gptq.H,
                    op=dist.ReduceOp.SUM,
                    dst=target_rank,
                    async_op=True,
                ),
                dist.reduce(
                    gptq.act_square,
                    op=dist.ReduceOp.SUM,
                    dst=target_rank,
                    async_op=True,
                ),
                dist.reduce(
                    gptq.index,
                    op=dist.ReduceOp.SUM,
                    dst=target_rank,
                    async_op=True,
                )
            ])
            wait_for_comms(pending_comms)
    dist.barrier()


@torch.no_grad()
def gptq_fwrd(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    """
    From GPTQ repo
    """
    logging.info("-----GuidedQuant Quantization-----")

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
    topk_buffer = analyzer.alloc_index_buffer(nsamples, model.seqlen, inps.device)

    pbar = tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers")
    for i in pbar:
        saliency_dict = torch.load(os.path.join(args.saliency_cache_path, f"l{i}.pt"))
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
                        gptq = GPTQGuided(
                            [subset[n] for n in spec.chunk_names],
                            [saliency_dict.pop(n) for n in spec.chunk_names],
                            num_groups=args.num_groups,
                            offload_hessians=args.offload_hessians,
                        )
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
    logging.info("-----GuidedQuant Quantization Done-----\n")
    return quantizers
