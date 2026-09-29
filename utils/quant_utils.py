import math

import torch

from utils import compile_utils, hadamard_utils


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, -(maxq + 1), maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


# ---------------------------------------------------------------------------
# Pure quant kernels.
#
# Everything below is tensor-in/tensor-out with no side effects, no host syncs
# and no shape metadata juggling, so `compile_utils.maybe_compile` can wrap it.
# Buffer writes, asserts, reshapes and `.to(device)` stay in the Module methods
# that call these.
# ---------------------------------------------------------------------------


def _minmax(x, clip_ratio: float = 1.0):
    """Per-group min/max over the last dim, clamped to include zero."""
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    if clip_ratio != 1.0:
        xmin = xmin * clip_ratio
        xmax = xmax * clip_ratio
    return (
        torch.minimum(xmin, torch.zeros_like(xmin)),
        torch.maximum(xmax, torch.zeros_like(xmax)),
    )


def _params_from_range_sym(xmin, xmax, bit_range):
    scale = torch.maximum(torch.abs(xmin), xmax) / (bit_range / 2)
    return scale, torch.zeros_like(scale)


def _params_from_range_asym(xmin, xmax, bit_range, minq, maxq):
    scale = (xmax - xmin) / bit_range
    zero = torch.clamp(torch.round(minq - xmin / scale), minq, maxq)
    return scale, zero


def _find_params_sym(x, bit_range, clip_ratio: float = 1.0):
    """Returns `(scale, zero, xmin, xmax)`; the clip search reuses xmin/xmax."""
    xmin, xmax = _minmax(x, clip_ratio)
    scale, zero = _params_from_range_sym(xmin, xmax, bit_range)
    return scale, zero, xmin, xmax


def _find_params_asym(x, bit_range, minq, maxq, clip_ratio: float = 1.0):
    xmin, xmax = _minmax(x, clip_ratio)
    scale, zero = _params_from_range_asym(xmin, xmax, bit_range, minq, maxq)
    return scale, zero, xmin, xmax


def _quant_err(fq, x, norm: float, H_diag=None):
    """Reduce the quantization error over the last dimension.

    Mutates `fq` in place; callers pass a tensor they just created and never
    reuse. Keeping the in-place chain (rather than allocating fresh temporaries)
    keeps eager peak memory at the pre-refactor level -- `x` here can be several
    GB -- while inductor fuses the whole chain away when compiled.

    With no Hessian, the objective is ``sum(abs(fq - x) ** norm)``. With a
    block-Hessian diagonal, rows in ``x`` are grouped by Hessian block and the
    objective is ``sum((fq - x) ** 2 * H_diag)`` for each row.
    """
    if H_diag is None:
        return fq.sub_(x).abs_().pow_(norm).sum(-1)

    error_shape = fq.shape[:-1]
    error = fq.sub_(x).pow_(2)
    error = error.reshape(H_diag.shape[0], -1, *H_diag.shape[1:])
    error.mul_(H_diag.unsqueeze(1))
    return error.sum(-1).reshape(error_shape)


def _mse_step_sym(x, xmin1, xmax1, scale, zero, best, maxq, minq, bit_range, norm: float, H_diag=None):
    """One grid point of the clip search: keep the (scale, zero) that lowers err.

    `xmin1`/`xmax1` are *tensors* precomputed by the caller. Passing the loop
    indices instead would specialize the graph on each of the ~625 grid values
    and trigger one recompile per point.
    """
    scale1, zero1 = _params_from_range_sym(xmin1, xmax1, bit_range)
    err = _quant_err(sym_quant_dequant(x, scale1, maxq), x, norm, H_diag)
    better = err < best
    better_k = better.unsqueeze(-1)
    return (
        torch.where(better_k, scale1, scale),
        torch.where(better_k, zero1, zero),
        torch.where(better, err, best),
    )


def _mse_step_asym(x, xmin1, xmax1, scale, zero, best, maxq, minq, bit_range, norm: float, H_diag=None):
    scale1, zero1 = _params_from_range_asym(xmin1, xmax1, bit_range, minq, maxq)
    err = _quant_err(asym_quant_dequant(x, scale1, zero1, maxq), x, norm, H_diag)
    better = err < best
    better_k = better.unsqueeze(-1)
    return (
        torch.where(better_k, scale1, scale),
        torch.where(better_k, zero1, zero),
        torch.where(better, err, best),
    )


def _fake_quant_sym(x, scale, maxq):
    q, scale = sym_quant(x, scale, maxq)
    return sym_dequant(q, scale), q, scale


def _fake_quant_asym(x, scale, zero, maxq):
    q, scale, zero = asym_quant(x, scale, zero, maxq)
    return asym_dequant(q, scale, zero), q, scale


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        return sym_quant_dequant(x, scale, maxq)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        return asym_quant_dequant(x, scale, zero, maxq)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16

    def free(self) -> None:
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        self.maxq = 2 ** (bits - 1) - 1
        self.minq = -(self.maxq + 1)
        self.bit_range = self.maxq - self.minq
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params(self, x) -> None:
        if self.bits == 16:
            return

        init_shape = x.shape

        # Reshape to a canonical per-token view whose last dim is reduced over.
        if self.groupsize > 0:
            # (..., num_tokens, num_groups, groupsize)
            x = x.reshape(*x.shape[:-1], x.shape[-1] // self.groupsize, self.groupsize)
        else:
            x = x.reshape(-1, x.shape[-1])  # (num_tokens, features)

        # Deliberately NOT compiled. This is a single pass over a modest tensor called
        # once per linear per forward, so dynamo's per-call guard overhead
        # outweighs the fusion win.
        if self.sym:
            scale, zero, _, _ = _find_params_sym(x, self.bit_range, self.clip_ratio)
        else:
            scale, zero, _, _ = _find_params_asym(
                x, self.bit_range, self.minq, self.maxq, self.clip_ratio
            )

        self.scale = scale
        self.zero = torch.round(zero)

        self.scale = self.scale.expand_as(x).reshape(init_shape)
        self.zero = self.zero.expand_as(x).reshape(init_shape)
        assert self.scale.min().item() > 0


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    @property
    def weight(self):
        return self.module.weight

    @property
    def bias(self):
        return self.module.bias

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def forward(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    hadamard_utils.HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self._ready = False

    def configure(
        self,
        bits,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 50,
        maxshrink: float = 0.5,
        weight_groupsize: int = -1,
        hclip: bool = False,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.hclip = hclip
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize

        self.maxq = 2 ** (bits - 1) - 1
        self.minq = -(self.maxq + 1)
        self.bit_range = self.maxq - self.minq
        self._ready = False

    def find_params(self, x, H_diag=None) -> None:
        if self.bits == 16:
            return
        dev = x.device

        init_shape = x.shape

        # Reshape to a canonical (row_groups, col_groups, groupsize) view.
        if self.weight_groupsize > 0:
            rows, columns = init_shape
            x = x.reshape(rows, columns // self.weight_groupsize, self.weight_groupsize)    # (row_groups, col_groups, groupsize)
        elif self.perchannel:
            x = x.flatten(1).unsqueeze(1)  # (rows, 1, columns)
        else:
            x = x.reshape(1, 1, -1)  # per-tensor: a single group

        if self.hclip:
            H_diag = H_diag.reshape(H_diag.shape[0], *x.shape[1:])
            H_diag = H_diag.to(device=dev, dtype=x.dtype).contiguous()
            if (H_diag.min() == H_diag.max()).item():
                H_diag = None   # Fall back to mse-based clipping

        if self.sym:
            find_params = compile_utils.maybe_compile(_find_params_sym, dynamic=False)
            scale, zero, xmin, xmax = find_params(x, self.bit_range)
        else:
            find_params = compile_utils.maybe_compile(_find_params_asym, dynamic=False)
            scale, zero, xmin, xmax = find_params(x, self.bit_range, self.minq, self.maxq)

        if self.mse:
            best = torch.full(x.shape[:-1], float("inf"), device=dev).type_as(x)
            # Compile the *body* of one grid point, not the whole search: the
            # latter would unroll into a ~625-step graph with pathological
            # compile time.
            if self.sym:
                mse_step = compile_utils.maybe_compile(_mse_step_sym, dynamic=False)
            else:
                mse_step = compile_utils.maybe_compile(_mse_step_asym, dynamic=False)

            for i in range(int(self.maxshrink * self.grid)):
                for j in range(int(self.maxshrink * self.grid)):
                    xmin1 = (1 - i / self.grid) * xmin
                    xmax1 = (1 - j / self.grid) * xmax

                    scale, zero, best = mse_step(
                        x, xmin1, xmax1, scale, zero, best,
                        self.maxq, self.minq, self.bit_range, self.norm,
                        H_diag=H_diag if self.hclip else None,
                    )

        self.scale = scale
        self.zero = torch.round(zero)

        self.scale = self.scale.squeeze(-1)
        self.zero = self.zero.squeeze(-1)
        torch._assert_async(self.scale.min() > 0)
        self._ready = True

    def fake_quantize(self, x):
        x_dtype = x.dtype
        if not (self.ready() and self.bits < 16):
            return None, None, None

        if self.weight_groupsize > 0:
            in_shape = x.shape
            x = x.reshape(in_shape[0], in_shape[1] // self.weight_groupsize, self.weight_groupsize)
            scale, zero = self.scale.unsqueeze(-1), self.zero.unsqueeze(-1)
        else:
            scale, zero = self.scale, self.zero

        # Deliberately NOT compiled. The dominant caller is the GPTQ inner loop,
        # which runs this once per column on a `(num_layers*rows, 1)` slice.
        # Dynamo's ~140 us/call guard overhead dwarfs the fusion win at that size.
        if self.sym:
            fq, q, scale = _fake_quant_sym(x, scale, self.maxq)
        else:
            fq, q, scale = _fake_quant_asym(x, scale, zero, self.maxq)

        if self.weight_groupsize > 0:
            fq = fq.reshape(in_shape)
            q = q.reshape(in_shape)
            scale = self.scale
        return fq.to(x_dtype), q, scale

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return self._ready


def add_actquant(analyzer) -> None:
    """
    Replaces specific quantizable layers in the model with ActQuantWrapper 
    based on the ModelAnalyzer's selection criteria.
    """
    for layer in analyzer.get_layers():
        quant_modules = analyzer.get_quantizable_modules(layer)
        
        for name, module in quant_modules.items():
            if isinstance(module, ActQuantWrapper):
                continue

            # Navigate to the parent module for nested names (e.g., 'self_attn.q_proj')
            parts = name.split('.')
            parent_module = layer
            
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            # Replace the target module
            target_name = parts[-1]
            setattr(parent_module, target_name, ActQuantWrapper(module))


def remove_actquant(analyzer) -> None:
    """
    Restores the original modules by removing the ActQuantWrapper 
    from specific layers in the model.
    """
    for layer in analyzer.get_layers():
        quant_modules = analyzer.get_quantizable_modules(layer, layer_classes=(ActQuantWrapper), remove_wrapper=False)
        
        for name, module in quant_modules.items():
            if not isinstance(module, ActQuantWrapper):
                continue

            # Navigate to the parent module for nested names (e.g., 'self_attn.q_proj')
            parts = name.split('.')
            parent_module = layer
            
            for part in parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            # Replace the target module with the original module
            target_name = parts[-1]
            setattr(parent_module, target_name, module.module)


def find_qlayers(
    module,
    layers=[ActQuantWrapper],
    name: str = "",
):
    # fix for llama embedding layer
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def disable_act_quant(module):
    bits_config = {}
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            bits_config[name] = m.quantizer.bits
            m.quantizer.bits = 16
            if m.out_quantizer.bits != 16:
                bits_config[name+'out'] = m.out_quantizer.bits
                m.out_quantizer.bits = 16

    return bits_config


def enable_act_quant(module, bits_config):
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            m.quantizer.bits = bits_config[name]
            if name+'out' in bits_config.keys():
                m.out_quantizer.bits = bits_config[name+'out']
