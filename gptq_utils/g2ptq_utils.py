import copy
import logging
import os
import math
import pprint
import functools
import contextlib
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
from compressed_tensors.distributed import wait_for_comms
from compressed_tensors.offload import disable_offloading
import numpy as np

from utils import quant_utils, memory_utils, model_utils, offload_utils, compressed_tensors_utils, dist_utils
from gptq_utils.common_utils import (
    broadcast_quantized_param,
    build_chunk_specs,
    fake_quantize_grouped,
    make_group_quantizers,
    merge_group_quantizers,
    partition_layers,
)
from gptq_utils.graph_utils import run_graph, validate_tensors
from gptq_utils.triton_utils import g2ptq_kernels


def _run_torch_compensation(
    weight, hinv, scale, zero, z, ghinv, int_weight, error, maxq,
    weight_raw=None, approx_beta=0.0,
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
        column_error = w - q - ghinv[:, column]
        if approx_beta:
            # ``--approx_grad`` ablation: shrink the compensation toward the
            # original weight instead of using the true gradient term.
            column_error = column_error - (w - weight_raw[:, column]) * approx_beta
        column_error = column_error / hinv[:, column, column].unsqueeze(1)

        weight[:, column].copy_(q)
        int_weight[:, column].copy_(quantized)
        error[:, column].copy_(column_error)
        if column + 1 < columns:
            hinv_row = hinv[:, column, column + 1 :].unsqueeze(2)
            weight[:, column + 1 :].sub_(ghinv[:, column + 1 :]).baddbmm_(
                hinv_row,
                column_error.unsqueeze(1),
                beta=1,
                alpha=-1,
            )
            ghinv[:, column + 1 :].baddbmm_(
                hinv_row,
                z[:, column].unsqueeze(1),
                beta=1,
                alpha=-1,
            )
            if approx_beta:
                weight[:, column + 1].sub_(
                    (weight[:, column + 1] - weight_raw[:, column + 1]) * approx_beta
                )


def _run_triton_compensation(
    weight, hinv, scale, zero, z, ghinv, int_weight, error, maxq,
    weight_raw=None, approx_beta=0.0,
):
    if approx_beta:
        raise NotImplementedError(
            "--approx_grad is only implemented for --gptq_backend torch"
        )
    columns = weight.shape[1]
    tile_size = triton.next_power_of_2(columns)
    for column in range(columns):
        g2ptq_kernels.launch_quantize(
            weight, hinv, scale, zero, ghinv, int_weight, error, column,
            maxq, tile_size,
        )
        g2ptq_kernels.launch_update(
            weight, hinv, error, ghinv, z, column, tile_size
        )


def run_eager(
    weight: torch.Tensor,
    hinv: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    z: torch.Tensor,
    ghinv: torch.Tensor,
    maxq: int,
    backend: str,
    int_weight: torch.Tensor | None = None,
    error: torch.Tensor | None = None,
    weight_raw: torch.Tensor | None = None,
    approx_beta: float = 0.0,
):
    """Run one prepared G2PTQ compensation block eagerly."""
    tensors = [weight, hinv, scale, zero, z, ghinv, int_weight, error]
    if weight_raw is not None:
        tensors.append(weight_raw)
    validate_tensors(f"{backend} G2PTQ", *tensors)

    _run_compensation = (
        _run_torch_compensation
        if backend == "torch"
        else _run_triton_compensation
    )
    _run_compensation(
        weight, hinv, scale, zero, z, ghinv, int_weight, error, maxq,
        weight_raw=weight_raw, approx_beta=approx_beta,
    )
    return weight, int_weight, error


class G2PTQ:
    def __init__(self,
        layers,
        saliencies,
        gradients,
        num_groups: int,
        offload_hessians: bool = False,
        approx_grad_beta: float = 0.0,
    ):
        self.layers = layers
        self.num_layers = len(layers)
        self.offload_hessians = offload_hessians
        self.approx_grad_beta = approx_grad_beta
        weight = layers[0].weight
        self.dev = weight.device

        self.rows, self.columns = weight.shape

        self.num_groups = num_groups
        self.saliencies = []
        for saliency in saliencies:
            self.saliencies.append(
                [s.float() for s in saliency] if saliency is not None else None
            )
        H_device = "cpu" if offload_hessians else self.dev
        self.gradients = offload_utils.alloc_pinned(
            (self.num_layers, self.rows, self.columns), dtype=torch.float32, device=H_device, pin=False,
        )
        for b, gradient in enumerate(gradients):
            if gradient is not None and isinstance(gradient, torch.Tensor):
                self.gradients[b].copy_(gradient, non_blocking=True)
        self.H = offload_utils.alloc_pinned(
            (self.num_layers, self.num_groups, self.columns, self.columns), device=H_device, pin=False,
        )
        self.act_square = torch.zeros((self.num_layers, self.columns), device=self.dev)
        self.index = torch.tensor([0] * self.num_layers, device=self.dev)
        self.total_tokens = torch.tensor([0] * self.num_layers, device=self.dev)

        assert self.rows % self.num_groups == 0, (
            f"Number of rows ({self.rows}) must be divisible "
            f"by num_groups ({self.num_groups})"
        )

    def _maybe_onload_hessian_grad(self, hessian=True, gradients=False):
        """Onload `self.H` and/or `self.gradients` to the execution device on enter, offload
        back to CPU on exit.
        """
        names = ("H",) if hessian else ()
        if gradients:
            names += ("gradients",)
        return offload_utils.onload_attrs(self, names, enabled=self.offload_hessians,
                                          device=self.dev)

    @torch.no_grad()
    def add_batch(self, idx, inp: torch.Tensor, out):
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float()
        n_tokens = inp.shape[0]

        sal_batch = self.saliencies[idx][0].to(self.dev)
        sal_batch = sal_batch.float()
        del self.saliencies[idx][0]

        self.H[idx] *= self.index[idx] / (self.index[idx] + 1)
        self.index[idx] += 1
        self.act_square[idx] *= self.total_tokens[idx] / (self.total_tokens[idx] + n_tokens)
        self.total_tokens[idx] += n_tokens

        sal_weighted_inp = torch.einsum("nj,ng->njg", inp, sal_batch)
        block = torch.einsum("ni,njg->gij", inp, sal_weighted_inp)
        self.H[idx].add_(block, alpha=1 / (n_tokens * self.index[idx]))
        self.act_square[idx].add_((inp ** 2).sum(0), alpha=1 / self.total_tokens[idx])

    def _run_compensation_block(
        self,
        weight,
        hinv,
        scale,
        zero,
        z,
        ghinv,
        backend,
        graph,
        weight_raw=None,
    ):
        quantizer = self.quantizer[0]
        weight = weight.transpose(1, 2).contiguous()
        params = {
            "weight": weight,
            "hinv": hinv.contiguous(),
            "scale": scale.transpose(1, 2).contiguous(),
            "zero": zero.transpose(1, 2).contiguous(),
            "z": z.transpose(1, 2).contiguous(),
            "ghinv": ghinv.transpose(1, 2).contiguous(),
            "maxq": quantizer.maxq,
            "backend": backend,
            "int_weight": torch.empty_like(weight),
            "error": torch.empty_like(weight),
        }
        input_names = ["weight", "hinv", "scale", "zero", "z", "ghinv"]
        # Keep the default path's params (and hence its CUDA Graph key) untouched.
        approx_grad_beta = getattr(self, "approx_grad_beta", 0.0)
        if approx_grad_beta:
            params["weight_raw"] = weight_raw.transpose(1, 2).contiguous()
            params["approx_beta"] = approx_grad_beta
            input_names.append("weight_raw")
        if graph:
            qweight, int_weight, error = run_graph(
                run_eager,
                params,
                input_names=tuple(input_names),
                output_names=("weight", "int_weight", "error"),
                name="G2PTQ compensation block",
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
        Z,
        GHinv,
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
        D = torch.arange(blocksize - 1, -1, -1).to(GHinv)
        # Snapshot the original weights once, after the caller's act-order
        # permutation and dead-column cleanup, so ``W - W_raw`` measures the
        # total drift rather than just the drift within the current block.
        W_raw = W.clone() if getattr(self, "approx_grad_beta", 0.0) else None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            W1 = W[:, :, i1:i2].clone()
            Hinv1 = Hinv[:, i1:i2, i1:i2]
            Z1 = Z[:, :, i1:i2]
            GHinv1 = GHinv[:, :, i1:i2].clone()

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
                Z1,
                GHinv1,
                backend,
                graph,
                weight_raw=W_raw[:, :, i1:i2] if W_raw is not None else None,
            )
            Q[:, :, i1:i2] = Q1
            W_int[:, :, i1:i2] = W_int1
            Scale[:, :, i1:i2] = Scale1
            if i2 == self.columns:
                continue

            gradient_update = blocksize * GHinv[:, :, i2:]
            gradient_update.baddbmm_(
                Z1 * D.reshape(1, 1, -1),
                Hinv[:, i1:i2, i2:],
                beta=1,
                alpha=-1,
            )
            W[:, :, i2:].sub_(gradient_update).baddbmm_(
                Err1,
                Hinv[:, i1:i2, i2:],
                beta=1,
                alpha=-1,
            )
            GHinv[:, :, i2:].baddbmm_(
                Z[:, :, i1:i2],
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
        enable_gradient_update=True,
        alpha=0.0,
        export_compressed_tensors=False,
        backend="torch",
        graph=False,
    ):
        W = torch.stack([layer.weight.data.clone() for layer in self.layers]).float()
        W_flat = W.view(self.num_layers * self.rows, self.columns)
        hclip = self.quantizer[0].hclip

        rows_per_sub = self.rows // self.num_groups

        W_sub = W.contiguous().view(self.num_layers * self.num_groups, rows_per_sub, self.columns)
        gradients_sub = self.gradients.view(
            self.num_layers * self.num_groups, rows_per_sub, self.columns
        ).to(self.dev, copy=True)

        with self._maybe_onload_hessian_grad():
            H_sub = self.H.view(self.num_layers * self.num_groups, self.columns, self.columns)
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
                if (H_sub[b] == 0).all():   # Fall back to RTN if no calibration data is provided
                    H_sub[b] = torch.eye(self.columns, device=self.dev)
                    gradients_sub[b].zero_()

                dead = torch.diag(H_sub[b]) == 0
                H_sub[b, dead, dead] = 1
                W_sub[b, :, dead] = 0

            Q_sub = None
            if enable_gradient_update and alpha > 0:
                if groupsize != -1 and not static_groups:
                    for i, quantizer in enumerate(self.quantizer):
                        group = W_sub[
                            :, :, i * groupsize : (i + 1) * groupsize
                        ]
                        quantizer.find_params(
                            group.reshape(self.num_layers * self.rows, -1)
                        )
                Q_sub = fake_quantize_grouped(
                    self.quantizer, W_sub, groupsize
                )

            if actorder:
                perm = torch.argsort(torch.mean(self.act_square, dim=0), descending=True)
                invperm = torch.argsort(perm)

            # Iterate over linear layers to avoid exessive memory allocation
            Hinv_init = torch.empty_like(H_sub)
            Hinv = torch.empty_like(H_sub)
            for b in range(self.num_layers * self.num_groups):
                if actorder:
                    W_sub[b] = W_sub[b][:, perm]
                    H_sub[b] = H_sub[b][perm][:, perm]
                    gradients_sub[b] = gradients_sub[b][:, perm]

                damp_percent = percdamp
                damp_auto_increment = 0.0015
                while 1 > damp_percent > 0:
                    try:
                        damp = damp_percent * torch.mean(torch.diag(H_sub[b]))
                        diag = torch.arange(self.columns, device=self.dev)
                        H_sub_tmp = H_sub[b].clone()
                        H_sub_tmp[diag, diag] += damp

                        H_chol = torch.linalg.cholesky(H_sub_tmp)
                        Hinv_init[b] = torch.cholesky_inverse(H_chol)
                        Hinv[b] = torch.linalg.cholesky(Hinv_init[b], upper=True)
                        if torch.isnan(Hinv[b]).any().item():
                            raise torch._C._LinAlgError
                        break
                    except torch._C._LinAlgError as e:
                        logging.warning(f"Quantization: Current `damp_percent = {damp_percent:.5f}` is too low, auto-incrementing by `{damp_auto_increment:.5f}`")
                        damp_percent += damp_auto_increment

                if not (0 < damp_percent < 1):
                    raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp_percent}")
                # memory_utils.cleanup_memory()

            if Q_sub is not None and actorder:
                Q_sub = Q_sub[:, :, perm]

            if H_diag is not None and actorder:
                H_diag = H_diag[:, perm].contiguous()

        # Dynamic scaling for gradients
        if enable_gradient_update and alpha > 0:
            epsilon = torch.finfo(torch.float32).tiny
            GHinv = torch.bmm(gradients_sub, Hinv_init)

            quant_err = Q_sub - W_sub

            diag_Hinv = torch.diagonal(Hinv_init, dim1=1, dim2=2).unsqueeze(1)
            loss_gptq = ((quant_err * GHinv + (quant_err ** 2) / 2) / diag_Hinv).abs()
            c = (gradients_sub * GHinv).sum(dim=2, keepdim=True) - ((GHinv ** 2) / diag_Hinv)
            c = c.clamp(min=2 * alpha * loss_gptq).clamp(min=epsilon)
            beta = (1 - torch.sqrt(torch.clamp(1 - (2 * alpha * loss_gptq) / c, min=0.0))).mean(-1)
        else:
            beta = torch.zeros((self.num_layers * self.num_groups, 1)).to(gradients_sub)
        del Hinv_init

        Z = torch.bmm(gradients_sub, Hinv.transpose(1, 2)) * beta.unsqueeze(2)
        GHinv = torch.bmm(Z, Hinv)
        Q, W_int, Scale = self._run_compensation(
            W_sub,
            Hinv,
            Z,
            GHinv,
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
        self.gradients = None
        self.saliencies = None
        self.act_square = None
        memory_utils.cleanup_memory()


def reduce_to_target_rank(g2ptqs: list['G2PTQ'], g2ptq2rank: dict):
    rank = dist.get_rank()
    dist.barrier()
    for g2ptq in g2ptqs:
        with g2ptq._maybe_onload_hessian_grad(gradients=True):
            pending_comms = []
            target_rank = g2ptq2rank[g2ptq]
            global_index = g2ptq.index.clone()
            global_total_tokens = g2ptq.total_tokens.clone()
            dist.all_reduce(global_index, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_total_tokens, op=dist.ReduceOp.SUM)
            for b in range(g2ptq.num_layers):
                # Hessian
                if global_index[b].item() == 0:
                    g2ptq.H[b].zero_()
                    g2ptq.gradients[b].zero_()
                    g2ptq.act_square[b].zero_()
                else:
                    g2ptq.H[b] *= (g2ptq.index[b].item() / global_index[b].item())
                    g2ptq.gradients[b] *= (g2ptq.index[b].item() / global_index[b].item())
                    g2ptq.act_square[b] *= (g2ptq.total_tokens[b].item() / global_total_tokens[b].item())
                    pending_comms.extend([
                        dist.reduce(
                            g2ptq.H[b],
                            op=dist.ReduceOp.SUM,
                            dst=target_rank,
                            async_op=True,
                        ),
                        dist.reduce(
                            g2ptq.act_square[b],
                            op=dist.ReduceOp.SUM,
                            dst=target_rank,
                            async_op=True,
                        ),
                        dist.reduce(
                            g2ptq.gradients[b],
                            op=dist.ReduceOp.SUM,
                            dst=target_rank,
                            async_op=True,
                        )
                    ])
            if rank == target_rank:
                g2ptq.index.copy_(global_index)
                g2ptq.total_tokens.copy_(global_total_tokens)
            wait_for_comms(pending_comms)
    dist.barrier()


def normalize_loss_weights(loss_weights):
    """Linearly normalize a {loss_type: weight} dict so the weights sum to 1."""
    total = sum(loss_weights.values())
    if total == 0:
        return {k: 0.0 for k in loss_weights}
    return {k: w / total for k, w in loss_weights.items()}


class SaliencyCache:
    """
    class for saving the output activation gradients in each layer.
    """
    def __init__(self, names, num_groups):
        self.num_groups = num_groups
        self.saliency_cache = {}
        self.outs = {}
        self.names = names
        for name in self.names:
            self.saliency_cache[name] = {}   # loss -> list of per-batch saliency tensors
            self.outs[name] = 0
        self.handles = []
        self.hooks_enabled = False

    def cache_saliency(self, module, inp, out, name):
        self.outs[name] = out

    def grad_hook(self, grad, name, loss_type):
        """
        grad shape typically [bsz, seq_len, hidden_dim]. For expert layers, shape is [seq_len, hidden_dim]
        We group the channels, take abs, then average.
        """
        if not self.hooks_enabled:
            return
        grad = grad.reshape(-1, grad.shape[-1])
        n_tokens, hidden_dim = grad.shape
        group_size = hidden_dim // self.num_groups

        grad_squared = grad.float().pow(2).view(n_tokens, self.num_groups, group_size)
        mean_squared_grad = grad_squared.mean(dim=-1)  # -> [n_tokens, num_groups]

        self.saliency_cache[name].setdefault(loss_type, []).append(mean_squared_grad)

    def summarize_saliency(self, loss_weights):
        """Collapse the per-loss saliency dict for each name back into a single list of
        per-batch tensors, as a weighted sum over the losses in `loss_weights` ({loss: weight}).
        """
        for name in self.names:
            per_loss = self.saliency_cache[name]
            present = [(l, w) for l, w in loss_weights.items() if l in per_loss]
            if not present:
                self.saliency_cache[name] = []
                continue
            lengths = [len(per_loss[l]) for l, _ in present]
            assert len(set(lengths)) == 1, (
                f"Saliency for '{name}' has mismatched per-loss batch counts: "
                f"{ {l: len(per_loss[l]) for l, _ in present} }"
            )
            n_batches = lengths[0]
            summarized = []
            for i in range(n_batches):
                acc = sum(w * per_loss[l][i] for l, w in present)
                summarized.append(acc)
            self.saliency_cache[name] = summarized

    def set_forward_hook(self, full):
        for name in self.names:
            module: nn.Linear = full.get(name, full.get(name + ".module", None))
            self.handles.append(
                module.register_forward_hook(
                    functools.partial(self.cache_saliency, name=name)
                )
            )

    def backward(self, loss_tensor, loss_type="block", retain_graph=True):
        names = [name for name in self.names if isinstance(self.outs[name], torch.Tensor)]
        if not names:
            return

        self.hooks_enabled = True
        try:
            grads = torch.autograd.grad(
                loss_tensor,
                [self.outs[name] for name in names],
                retain_graph=retain_graph,
                allow_unused=True,
            )
            for name, grad in zip(names, grads):
                if grad is not None:
                    self.grad_hook(grad, name=name, loss_type=loss_type)
        finally:
            self.hooks_enabled = False

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hooks_enabled = False

    def clear_cache(self):
        for name in self.names:
            self.saliency_cache[name] = {}
            self.outs[name] = 0
        memory_utils.cleanup_memory()


class GradientCache:
    """
    class for saving the weight gradients in each layer.
    """
    def __init__(self, names, num_groups):
        self.num_groups = num_groups
        self.gradients_cache = {}
        self.index = {}
        self.current_batch_tokens = {}
        self.names = names
        for name in self.names:
            self.gradients_cache[name] = {}   # loss -> running-mean gradient tensor
            self.index[name] = {}             # loss -> accumulation count
        self.handles = []
        self.hooks_enabled = False

    def forward_hook(self, module, inp, out, name):
        x = inp[0]  # (num_routed_tokens, hidden_dim) in MoEs, or (batch, seq_len, hidden_dim)
        n_tokens = x.numel() // x.shape[-1]
        self.current_batch_tokens[name] = n_tokens

    def cache_gradient(self, grad, name, loss_type):
        if not self.hooks_enabled:
            return
        n_tokens = self.current_batch_tokens[name]
        index = self.index[name].get(loss_type, 0)
        cache = self.gradients_cache[name].get(loss_type, 0)
        cache = cache * (index / (index + 1))
        index += 1
        cache = cache + grad.float() / (n_tokens * index)
        self.gradients_cache[name][loss_type] = cache
        self.index[name][loss_type] = index

    def summarize_gradients(self, loss_weights):
        """Collapse the per-loss gradient dict for each name back into a single tensor, as a
        weighted sum over the losses in `loss_weights` ({loss: weight}). Names with no cached
        gradient collapse to 0.
        """
        for name in self.names:
            per_loss = self.gradients_cache[name]
            present = [(l, w) for l, w in loss_weights.items() if l in per_loss]
            self.gradients_cache[name] = sum(w * per_loss[l] for l, w in present) if present else 0

    def set_forward_hook(self, full):
        for name in self.names:
            module: nn.Linear = full.get(name, full.get(name + ".module", None))
            self.handles.append(
                module.register_forward_hook(
                    functools.partial(self.forward_hook, name=name)
                )
            )

    @contextlib.contextmanager
    def set_backward_hook(self, full, loss_type="block"):
        self.hooks_enabled = True
        handles = []

        for name in self.names:
            module: nn.Linear = full.get(name, full.get(name + ".module", None))
            handles.append(
                module.weight.register_hook(
                    functools.partial(self.cache_gradient, name=name, loss_type=loss_type)
                )
            )
        
        try:
            yield
        finally:
            for h in handles:
                h.remove()
            handles = []
            self.hooks_enabled = False

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.hooks_enabled = False

    def clear_cache(self):
        for name in self.names:
            self.gradients_cache[name] = {}
            self.index[name] = {}
        memory_utils.cleanup_memory()


class MoEGateCache:
    """
    Context manager class for caching MoE gating logits in full-precision 
    and forcing them during quantized model forward passes.
    """
    def __init__(self, analyzer: model_utils.ModelAnalyzer, layer, gate_forcing=False):
        self.gate_forcing = gate_forcing
        # Dense MLP layers (e.g. glm_moe_dsa's first few layers) have no router `mlp.gate`;
        # treat them as non-MoE so gate caching/forcing short-circuits.
        self.is_moe = analyzer.is_moe() and (analyzer.get_perlayer_experts(layer) is not None)
        self.spec = analyzer.spec
        self.gate_module: nn.Linear = analyzer.get_perlayer_gate(layer) if self.is_moe else None
        self.cached_logits = []
        self.q_logits = None
        self._handle = None
        self._force_idx = 0

    def _cache_hook(self, router_logits):
        self.cached_logits.append(router_logits.detach())
        return router_logits

    def _force_hook(self, router_logits):
        self.q_logits = router_logits
        if self.gate_forcing:
            router_logits = self.cached_logits[self._force_idx].to(router_logits)
            self._force_idx += 1
        return router_logits

    @contextlib.contextmanager
    def cache_mode(self):
        """Context manager to cache gating logits."""
        if not self.is_moe:
            yield
            return

        self.clear_cache()
        self.wrap_gate_module(self._cache_hook)
        try:
            yield
        finally:
            del self.gate_module.forward

    @contextlib.contextmanager
    def force_mode(self):
        """Context manager to apply cached gating logits via teacher forcing."""
        if not self.is_moe:
            yield
            return

        self._force_idx = 0  # Reset index at the start of forcing
        self.wrap_gate_module(self._force_hook)
        try:
            yield
        finally:
            del self.gate_module.forward

    def wrap_gate_module(self, func):
        self.spec.wrap_gate_forward(self.gate_module, func)

    def clear_cache(self):
        self.cached_logits = []
        self.q_logits = None
        memory_utils.cleanup_memory()


def hidden2logits(hidden_states, analyzer: model_utils.ModelAnalyzer):
    norm = analyzer.get_layernorm_before_head()
    lm_head = analyzer.get_lm_head()

    logits = lm_head(norm(hidden_states))

    logits = analyzer.post_process_logits(logits)

    return logits


@contextlib.contextmanager
def get_hidden_states_before_residual(args, analyzer: model_utils.ModelAnalyzer, layer,
                                      enable: bool = True, offload: bool = True):
    """Collect the block output just before the final residual add, one entry per batch.
    """
    module = analyzer.get_module_before_final_residual(layer)
    outs = []

    if not enable:
        yield outs
        return

    def hook_fn(module, inp, out):
        if args.offload_inps and offload:
            dst = offload_utils.empty_pinned_like(out)
            offload_utils.current_d2h_queue().copy_(dst, out)
            out = dst
        outs.append(out)

    handle = module.register_forward_hook(hook_fn)
    try:
        yield outs
    finally:
        handle.remove()


def get_final_hidden_states(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    logging.info("Getting Final Hidden States...")
    model = analyzer.model
    use_cache = analyzer.config.use_cache
    analyzer.config.use_cache = False
    layers = analyzer.get_layers()
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

    for i in tqdm(range(len(layers)), ncols=80, desc="Forwarding Layers"):
        layer = layers[i]
        bits_config = quant_utils.disable_act_quant(layer)
        with (
            disable_offloading(),
            get_hidden_states_before_residual(args, analyzer, layer, enable=(i == len(layers) - 1)) as hidden_states_before_residual,
        ):
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
        quant_utils.enable_act_quant(layer, bits_config)

        del layer
        memory_utils.cleanup_memory()

        hidden_states = inps

    analyzer.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)

    hidden_states_before_residual = torch.cat(hidden_states_before_residual, dim=0)
    residual = hidden_states / analyzer.get_layer_scalar(analyzer.get_layers()[-1]) - hidden_states_before_residual

    return hidden_states, hidden_states_before_residual, residual


def get_linear_patch(source, target, n_samples=64, dev="cuda"):
    # Fit on the first `n_samples` samples so the lstsq is cheap enough to run on CUDA.
    source = source[:n_samples].reshape(-1, source.size(-1)).to(dev)
    target = target[:n_samples].reshape(-1, target.size(-1)).to(dev)
    linear_patch = torch.linalg.lstsq(source.float(), target.float(),
                                      driver="gels").solution
    return linear_patch.to(source.dtype)


def compute_kl_grad_hessian(logits, logits_fp, saliency_cache, gradients_cache, full, args,
                            loss_type="block", retain_graph=False, compute_gradient=True):
    """Accumulate the saliency (hessian) and weight gradient for a KL-matching loss between
    `logits` (quantized) and `logits_fp` (reference) under the cache key `loss_type`. Returns the
    detached (un-normalized) KL loss scalar.
    """
    if args.kl_topk > 0:
        logits_fp, indices = logits_fp.topk(args.kl_topk, dim=-1, sorted=False)
        logits = logits.gather(-1, indices)

    # NLL loss -> saliency (hessian)
    labels = torch.distributions.Categorical(logits=logits_fp).sample()
    nll_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="sum",
    )
    offload_utils.zero_onloaded_grads()
    saliency_cache.backward(nll_loss, loss_type=loss_type, retain_graph=True)

    # KL loss -> weight gradient
    kl_loss = F.kl_div(
        F.log_softmax(logits, dim=-1),
        F.softmax(logits_fp, dim=-1),
        reduction="none",
    )
    kl_loss = kl_loss.sum()
    if compute_gradient:
        offload_utils.zero_onloaded_grads()
        with gradients_cache.set_backward_hook(full, loss_type=loss_type):
            kl_loss.backward(retain_graph=retain_graph)

    return kl_loss.detach().float()


def compute_mse_grad_hessian(hidden_states, hidden_states_fp, saliency_cache, gradients_cache,
                             full, args, loss_type="block", retain_graph=False,
                             compute_gradient=True):
    """Accumulate the saliency (hessian) and weight gradient for an MSE loss between
    `hidden_states` (quantized) and `hidden_states_fp` (reference) under the cache key `loss_type`.
    Returns the detached (un-normalized) MSE loss scalar.
    """
    # MSE loss Hessian (noise target)
    hidden_states_noise = torch.distributions.Normal(
        loc=torch.zeros_like(hidden_states),
        scale=math.sqrt(hidden_states.size(-1))
    ).sample()
    mse_loss = F.mse_loss(
        hidden_states.view(-1, hidden_states.size(-1)),
        (hidden_states + hidden_states_noise).detach().view(-1, hidden_states.size(-1)),
        reduction="none",
    ).mean(-1).sum() / 2
    offload_utils.zero_onloaded_grads()
    saliency_cache.backward(mse_loss, loss_type=loss_type, retain_graph=True)

    # MSE loss gradient
    mse_loss = F.mse_loss(
        hidden_states.view(-1, hidden_states.size(-1)),
        hidden_states_fp.view(-1, hidden_states.size(-1)),
        reduction="none",
    ).mean(-1).sum() / 2
    if compute_gradient:
        offload_utils.zero_onloaded_grads()
        with gradients_cache.set_backward_hook(full, loss_type=loss_type):
            mse_loss.backward(retain_graph=retain_graph)

    return mse_loss.detach().float()


def load_autotune_config(path, num_layers):
    """Load an autotune cache as {layer index: [alpha per sequential stage]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    best_alpha_dict = {}
    for key, value in raw.items():
        alphas = list(value) if isinstance(value, (list, tuple)) else [value]
        best_alpha_dict[int(key)] = [float(alpha) for alpha in alphas]

    missing = [i for i in range(num_layers) if i not in best_alpha_dict]
    if missing:
        raise ValueError(
            f"Autotune config {path} is missing layer indices {missing} "
            f"(the model has {num_layers} layers)"
        )
    return best_alpha_dict


def resolve_stage_alphas(alphas, num_stages, layer_idx):
    """Fit a loaded per-stage alpha list to this run's number of sequential stages."""
    if len(alphas) == num_stages:
        return list(alphas)
    if num_stages == 1:
        # Tuned with true sequential quantization, replayed without it.
        mean_alpha = sum(alphas) / len(alphas)
        logging.warning(
            f"Layer {layer_idx}: autotune config has {len(alphas)} stages but this run uses a "
            f"single stage; averaging the loaded alphas into {mean_alpha:.2e}"
        )
        return [mean_alpha]
    raise ValueError(
        f"Layer {layer_idx}: autotune config has {len(alphas)} alpha(s) but this run has "
        f"{num_stages} sequential stages. The config was tuned at a different "
        f"--true_sequential_ratio (or predates per-stage autotuning, in which case it holds one "
        f"scalar per layer and can only be replayed with a single stage)."
    )


def _snapshot_stage_weights(analyzer, layer, stage_names):
    """Clone only the weight state for the quantizable modules in one stage.
    """
    snapshot = {}
    for name in stage_names:
        module = analyzer.get_model_attribute(name, layer)

        weight_entries = [
            (local_key, value)
            for local_key, value in module.state_dict().items()
            if local_key.rsplit(".", 1)[-1] == "weight"
        ]

        for local_key, value in weight_entries:
            snapshot[f"{name}.{local_key}"] = value.detach().cpu().clone()

    return snapshot


@torch.no_grad()
def gptq_fwrd(args, analyzer: model_utils.ModelAnalyzer, dataloader, dev):
    """
    From GPTQ repo
    """
    logging.info("-----G2PTQ Quantization-----")

    model = analyzer.model
    use_cache = analyzer.config.use_cache
    analyzer.config.use_cache = False
    layers = analyzer.get_layers()
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    nsamples = len(dataloader)

    if args.load_autotune:
        try:
            best_alpha_dict = load_autotune_config(args.load_autotune, len(layers))
        except Exception as e:
            logging.warning(f"Loading autotuned config at {args.load_autotune} failed: {e}")
            args.autotune = True
            args.load_autotune = None
    if args.autotune:
        best_alpha_dict = {}
        logging.info(f"Autotuning enabled")

    if args.enable_linear_patch:
        final_hidden_states, final_hidden_states_before_residual, final_residual = \
            get_final_hidden_states(args, analyzer, dataloader, dev)

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

    fp_inps = offload_utils.clone_pinned(inps)
    if block_internals is not None:
        block_internals_fp = copy.deepcopy(block_internals)
    else:
        block_internals_fp = None

    topk_buffer_fp = analyzer.alloc_index_buffer(nsamples, model.seqlen, fp_inps.device)
    topk_buffer_q = analyzer.alloc_index_buffer(nsamples, model.seqlen, inps.device)

    quantizers = {}
    pbar = tqdm(range(len(layers)), ncols=120, desc="Quantizing Layers", position=0)
    for i in pbar:
        layer = layers[i]
        with disable_offloading():
            true_sequential = analyzer.get_sequential_quantizable_module_names(layer)

            # Map each module to its true sequential stage index
            name2stage = {}
            for stage_idx, stage_names in enumerate(true_sequential):
                for name in stage_names:
                    name2stage[name] = stage_idx

            if len(layers) - i <= int(len(layers) * args.true_sequential_ratio):
                sequential = true_sequential
            else:
                sequential = [[n for ns in true_sequential for n in ns]]
            full = analyzer.get_quantizable_modules(layer)
            use_kl = (not args.disable_kl) and (
                (i == len(layers) - 1) or ((i + 1) > int(len(layers) * (1 - args.kl_ratio)))
            )
            enable_linear_patch = (
                args.enable_linear_patch
                and (i < len(layers) - 1)
                and ((i + 1) > int(len(layers) * (1 - args.linear_patch_ratio)))
            )
            is_residual_forcing = ((i + 1) <= int(len(layers) * args.residual_forcing_ratio))

            if args.load_autotune:
                layer_alphas = resolve_stage_alphas(best_alpha_dict[i], len(sequential), i)
            else:
                alpha = args.alpha_kl if use_kl else args.alpha_mse
                alpha_warmup_layers = int(len(layers) * args.alpha_warmup_ratio)
                if i < alpha_warmup_layers:
                    progress = i / alpha_warmup_layers
                    # alpha = alpha * progress        # linear warmup
                    alpha = alpha * 0.5 * (1 - math.cos(math.pi * (progress ** 2)))    # cosine warmup
                layer_alphas = [alpha] * len(sequential)
            if args.autotune:
                best_alpha_dict[i] = []

            gate_cache = MoEGateCache(analyzer, layer, gate_forcing=args.moe_gate_forcing)
            bits_config = quant_utils.disable_act_quant(layer)
            with (
                gate_cache.cache_mode(),   # Cache gating logits
                get_hidden_states_before_residual(args, analyzer, layer, enable=enable_linear_patch and is_residual_forcing) as hidden_states_before_residual_fp,
            ):
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
            quant_utils.enable_act_quant(layer, bits_config)
            memory_utils.cleanup_memory()

            if enable_linear_patch:
                if is_residual_forcing:
                    hidden_states_before_residual_fp = torch.cat(hidden_states_before_residual_fp, dim=0)
                    linear_patch = get_linear_patch(
                        hidden_states_before_residual_fp,
                        final_hidden_states_before_residual,
                        dev=dev,
                    )
                else:
                    linear_patch = get_linear_patch(
                        fp_inps,
                        final_hidden_states,
                        dev=dev,
                    )
                memory_utils.cleanup_memory()

            for sequential_stage, names in enumerate(sequential):
                alpha = layer_alphas[sequential_stage]

                #############################################################################

                # Get per-block gradients and hessians
                with torch.enable_grad():
                    # Set requires_grad of onloaded params to True
                    model_utils.set_requires_grad(
                        layer,
                        True,
                        no_grad_param_prefixes=analyzer.spec.NO_GRAD_PARAM_PREFIXES,
                        onload_parameters=True,
                    )

                    # Add forward hooks
                    saliency_cache = SaliencyCache(names, args.num_groups)
                    saliency_cache.set_forward_hook(full)
                    gradients_cache = GradientCache(names, args.num_groups)
                    gradients_cache.set_forward_hook(full)

                    losses = {}   # loss_type -> list of per-batch (unweighted) loss scalars

                    with (
                        gate_cache.force_mode(),
                        get_hidden_states_before_residual(
                            args, analyzer, layer,
                            enable=enable_linear_patch and is_residual_forcing, offload=False,
                        ) as hidden_states_before_residual,
                    ):
                        for j, (inps_batch, fp_inps_batch, topk_batch) in tqdm(
                            offload_utils.prefetch_generator((inps, fp_inps, topk_buffer_q),
                                                                nsamples, args.bsz, dev, args.offload_inps),
                            total=math.ceil(nsamples / args.bsz),
                            ncols=120, desc=f"Layer {i} Computing Gradients and Hessians",
                            position=1, leave=False
                        ):
                            out = model_utils.run_block_layer(
                                analyzer, layer, inps_batch,
                                prev_topk_indices=topk_batch,
                                attention_mask=attention_mask, position_ids=position_ids,
                                position_embeddings=position_embeddings,
                                block_internals=block_internals,
                                layer_idx=i, sample_idx=j, bsz=args.bsz, dev=dev,
                                **kwargs,
                            )
                            
                            # Local block loss: raw block output vs full-precision output
                            block_retain_graph = enable_linear_patch or args.moe_gate_align
                            if use_kl:
                                logits, logits_fp = hidden2logits(out, analyzer), hidden2logits(fp_inps_batch, analyzer)
                                block_loss = compute_kl_grad_hessian(
                                    logits, logits_fp, saliency_cache, gradients_cache, full, args,
                                    loss_type="block", retain_graph=block_retain_graph,
                                    compute_gradient=not args.disable_grad,
                                )
                                del logits, logits_fp
                            else:
                                block_loss = compute_mse_grad_hessian(
                                    out, fp_inps_batch, saliency_cache, gradients_cache,
                                    full, args, loss_type="block", retain_graph=block_retain_graph,
                                    compute_gradient=not args.disable_grad,
                                )
                            losses.setdefault("block", []).append(block_loss / (model.seqlen * args.bsz))

                            # Additional linear-patch loss: patched hidden states as a proxy for
                            # the model's final output.
                            if enable_linear_patch:
                                if is_residual_forcing:
                                    layer_scalar = analyzer.get_layer_scalar(layer)
                                    hidden_states = (hidden_states_before_residual[0].to(dev) @ linear_patch + \
                                                    final_residual[j: j + args.bsz].to(dev)) * layer_scalar
                                    hidden_states_fp = (hidden_states_before_residual_fp[j: j + args.bsz].to(dev) @ linear_patch + \
                                                        final_residual[j: j + args.bsz].to(dev)) * layer_scalar
                                    del hidden_states_before_residual[0]
                                else:
                                    hidden_states = out @ linear_patch
                                    hidden_states_fp = fp_inps_batch @ linear_patch

                                if use_kl:
                                    logits, logits_fp = hidden2logits(hidden_states, analyzer), hidden2logits(hidden_states_fp, analyzer)
                                    patch_loss = compute_kl_grad_hessian(
                                        logits, logits_fp, saliency_cache, gradients_cache, full, args,
                                        loss_type="patch", retain_graph=args.moe_gate_align,
                                        compute_gradient=not args.disable_grad,
                                    )
                                    del logits, logits_fp
                                else:
                                    patch_loss = compute_mse_grad_hessian(
                                        hidden_states, hidden_states_fp, saliency_cache, gradients_cache,
                                        full, args, loss_type="patch", retain_graph=args.moe_gate_align,
                                        compute_gradient=not args.disable_grad,
                                    )
                                losses.setdefault("patch", []).append(patch_loss / (model.seqlen * args.bsz))

                            if analyzer.is_moe() and args.moe_gate_align:   # TODO. some models use sigmoid instead of softmax (e.g. GLM-5.2), we should use CE loss instead of KL
                                logits, logits_fp = gate_cache.q_logits, gate_cache.cached_logits[j // args.bsz]
                                compute_kl_grad_hessian(
                                    logits, logits_fp,
                                    saliency_cache, gradients_cache, full, args,
                                    loss_type="gate", retain_graph=False,
                                    compute_gradient=not args.disable_grad,
                                )
                                del logits, logits_fp
                            # memory_utils.cleanup_memory()     # reduce memory but slow down the quantization

                    offload_utils.zero_onloaded_grads()
                    memory_utils.cleanup_memory()
                    mean_losses = {lt: sum(vals).item() / len(vals) for lt, vals in losses.items()}
                    mean_loss = mean_losses["block"]
                    loss_str = ", ".join(f"{lt}: {v:.2e}" for lt, v in mean_losses.items())
                    logging.info(f"Layer {i} Stage {sequential_stage} {'KL' if use_kl else 'MSE'} Loss: {loss_str}")

                saliency_cache.clear_hook()
                gradients_cache.clear_hook()

                # Aggregate grads and saliencies of different losses
                loss_weights = {"block": 1.0}
                if enable_linear_patch:
                    loss_weights["patch"] = args.linear_patch_strength
                if args.moe_gate_align:
                    loss_weights["gate"] = args.moe_gate_align_strength
                loss_weights = normalize_loss_weights(loss_weights)
                saliency_cache.summarize_saliency(loss_weights)
                gradients_cache.summarize_gradients(loss_weights)
            
                # Layer 0 has zero gradient; --disable_grad drops it everywhere
                if (i == 0 and sequential_stage == 0) or args.disable_grad:
                    gradients_cache.clear_cache()
                saliency_dict = saliency_cache.saliency_cache
                gradients_dict = gradients_cache.gradients_cache

                #############################################################################

                subset = {n: full.get(n, full.get(n + ".module", None)) for n in names}

                specs = build_chunk_specs(subset, args.layer_bsz, group_key_fn=name2stage.__getitem__)
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
                        gptq = G2PTQ(
                            [subset[n] for n in spec.chunk_names],
                            saliencies=[saliency_dict.pop(n) for n in spec.chunk_names],
                            gradients=[gradients_dict.pop(n) for n in spec.chunk_names],
                            num_groups=args.num_groups,
                            offload_hessians=args.offload_hessians,
                            approx_grad_beta=(
                                args.approx_grad_beta if args.approx_grad else 0.0
                            ),
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

                    with gate_cache.force_mode(), contextlib.ExitStack() as onload_stack:
                        for gptq in gptqs_batch:
                            onload_stack.enter_context(gptq._maybe_onload_hessian_grad())
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

                saliency_cache.clear_cache()
                gradients_cache.clear_cache()
                del saliency_dict, gradients_dict
                memory_utils.cleanup_memory()

                if args.autotune and (i > 0):
                    best_mean_loss = float("inf")
                    best_alpha = 0
                    # Cache original weights for the current quantizable sequential stage
                    state_dict = _snapshot_stage_weights(analyzer, layer, names)
                    saved_H = {gptq: gptq.H.cpu().clone() for gptq in gptqs_partition}

                    for alpha in tqdm(
                        np.arange(args.autotune_min, args.autotune_max + 1e-3, args.autotune_stepsize),
                        ncols=80, desc="Searching for best alpha"
                    ):
                        alpha = float(f"{alpha.item():.2f}")
                        for gptq in gptqs_partition:
                            layer_w_groupsize = args.w_groupsize
                            gptq.fasterquant(
                                percdamp=args.percdamp,
                                groupsize=layer_w_groupsize,
                                actorder=args.act_order,
                                static_groups=args.act_order,
                                enable_gradient_update=(i > 0) and not args.disable_grad,
                                alpha=alpha,
                                export_compressed_tensors=False,
                                backend=args.gptq_backend,
                                graph=args.gptq_graph,
                            )
                        if world_size > 1:
                            broadcast_quantized_param(gptqs, gptq2rank, export_compressed_tensors=False)
                        # Compute loss
                        losses = []
                        for j, (inps_batch, fp_inps_batch, topk_batch) in offload_utils.prefetch_generator(
                            (inps, fp_inps, topk_buffer_q), args.autotune_nsamples, args.bsz, dev, args.offload_inps
                        ):
                            out = model_utils.run_block_layer(
                                analyzer, layer, inps_batch,
                                prev_topk_indices=topk_batch,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings,
                                block_internals=block_internals,
                                layer_idx=i, sample_idx=j, bsz=args.bsz, dev=dev,
                                **kwargs,
                            )
                            hidden_states = out
                            hidden_states_fp = fp_inps_batch
                            if use_kl:
                                logits, logits_fp = hidden2logits(hidden_states, analyzer), hidden2logits(hidden_states_fp, analyzer)
                                if args.kl_topk > 0:
                                    logits_fp, indices = logits_fp.topk(args.kl_topk, dim=-1, sorted=False)
                                    logits = logits.gather(-1, indices)
                                kl_loss = F.kl_div(
                                    F.log_softmax(logits, dim=-1),
                                    F.softmax(logits_fp, dim=-1),
                                    reduction="none",
                                )
                                kl_loss = kl_loss.sum()
                                losses.append(kl_loss.detach().float() / (model.seqlen * args.bsz))
                            else:
                                # Compute loss on non-outliers
                                hidden_states = hidden_states.detach().clone()
                                hidden_states_fp = hidden_states_fp.detach().clone()
                                if args.autotune_outlier_thresh > 0:
                                    hidden_states_fp_abs = hidden_states_fp.abs()
                                    smooth_mask = (hidden_states_fp_abs > hidden_states_fp_abs.mean() 
                                                    + hidden_states_fp_abs.std() * args.autotune_outlier_thresh)
                                    hidden_states[smooth_mask] = 0
                                    hidden_states_fp[smooth_mask] = 0
                                mse_loss = F.mse_loss(
                                    hidden_states.view(-1, hidden_states.size(-1)),
                                    hidden_states_fp.view(-1, hidden_states.size(-1)),
                                    reduction="none",
                                ).mean(-1).sum() / 2
                                losses.append(mse_loss.detach().float() / (model.seqlen * args.bsz))
                        mean_loss_t = sum(losses) / len(losses)
                        # Average the per-rank losses to determine the best alpha
                        if world_size > 1:
                            dist.all_reduce(mean_loss_t, op=dist.ReduceOp.SUM)
                            mean_loss_t = mean_loss_t / world_size
                        mean_loss = mean_loss_t.item()
                        if mean_loss < best_mean_loss:
                            best_mean_loss = mean_loss
                            best_alpha = alpha
                            logging.info(f"Layer {i} Stage {sequential_stage} {'KL' if use_kl else 'MSE'} best_alpha: {alpha:.2e}, best_Loss: {mean_loss:.2e}")

                        # Restore layer weights and Hessians (fasterquant mutates both in place)
                        layer.load_state_dict(state_dict, strict=False)
                        for gptq in gptqs_partition:
                            gptq.H.copy_(saved_H[gptq], non_blocking=True)

                        # Early stop
                        if alpha - best_alpha >= args.autotune_early_stop_margin:
                            break

                    alpha = best_alpha
                    del state_dict

                if args.autotune:
                    best_alpha_dict[i].append(alpha)

                for gptq in gptqs_partition:
                    chunk_names = gptq2chunk_names[gptq]
                    display_name = chunk_names[0] + (f" (+{len(chunk_names)-1} batched)" if len(chunk_names) > 1 else "")
                    pbar.set_postfix(module=f"layers.{i}.{display_name}", alpha=f"{alpha:.2e}", loss=f"{mean_loss:.2e}")
                    layer_w_groupsize = args.w_groupsize
                    gptq.fasterquant(
                        percdamp=args.percdamp,
                        groupsize=layer_w_groupsize,
                        actorder=args.act_order,
                        static_groups=args.act_order,
                        enable_gradient_update=(i > 0) and not args.disable_grad,
                        alpha=alpha,
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
                    analyzer, layer, inps_batch,
                    out_buffer=None if args.disable_refresh else inps,
                    prev_topk_indices=topk_batch, index_buffer=topk_buffer_q,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    block_internals=block_internals,
                    layer_idx=i, sample_idx=j, bsz=args.bsz, dev=dev,
                    **kwargs,
                )

            if args.disable_refresh:
                # Ablation: the next block sees full-precision inputs instead of this
                # block's quantized outputs.
                inps.copy_(fp_inps)
                if topk_buffer_q is not None and topk_buffer_fp is not None:
                    topk_buffer_q.copy_(topk_buffer_fp)

            gate_cache.clear_cache()
            del layer
            del gptqs, specs
            logging.info(f"Peak GPU Memory Usage during Model Quantization: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        memory_utils.cleanup_memory()

    if args.autotune and dist_utils.is_main():
        logging.info(f"Autotune Results saved at {args.autotune_cache_path}:\n {best_alpha_dict}")
        os.makedirs(os.path.dirname(args.autotune_cache_path), exist_ok=True)
        with open(args.autotune_cache_path, "w", encoding="utf-8") as f:
            json.dump(best_alpha_dict, f, ensure_ascii=False, indent=4, sort_keys=True)
    analyzer.config.use_cache = use_cache
    memory_utils.cleanup_memory(verbos=True)
    logging.info("-----G2PTQ Quantization Done-----\n")
    return quantizers
