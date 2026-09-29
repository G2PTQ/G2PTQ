import functools
import typing
import torch
import torch.distributed as dist
import math
from tqdm import tqdm

from utils import memory_utils, dist_utils, model_utils, hadamard_utils, quant_utils, \
                  monkeypatch
from utils.offload_utils import update_shared_offload_parameter


@torch.inference_mode()
def fuse_ln_linears(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear], is_zero_centered: bool) -> None:
    ln_weight = layernorm.weight
    ln_bias = getattr(layernorm, "bias", None)
    layernorm_weight = (1.0 + ln_weight.double()) if is_zero_centered else ln_weight.double()
    ln_bias = ln_bias.double() if ln_bias is not None else None
    for linear in linear_layers:
        weight = linear.weight
        linear_dtype = weight.dtype
        W_ = weight.data.double()
        update_shared_offload_parameter(
            linear, "weight", (W_ * layernorm_weight).to(linear_dtype), sync=False
        )

        if ln_bias is not None:
            bias = linear.bias
            fused_bias = torch.matmul(W_, ln_bias)
            if bias is not None:
                fused_bias = fused_bias + bias.data.double()
            fused_bias = fused_bias.to(linear_dtype)
            if bias is None:
                linear.bias = torch.nn.Parameter(fused_bias)
            else:
                update_shared_offload_parameter(linear, "bias", fused_bias, sync=False)
    update_shared_offload_parameter(
        layernorm, "weight", torch.full_like(ln_weight, 0. if is_zero_centered else 1.)
    )


@torch.inference_mode()
def fuse_layer_norms(analyzer: model_utils.ModelAnalyzer) -> None:
    if analyzer.process_word_embeddings:
        return
    layers = analyzer.get_layers()
    for layer in tqdm(layers, desc="Fusing LN"):
        for (ln, is_zero_centered), linears in zip(
            analyzer.get_layernorms(layer),
            analyzer.get_perlayer_input_modules(layer)
        ):
            fuse_ln_linears(ln, linears, is_zero_centered)
    fuse_ln_linears(analyzer.get_layernorm_before_head(), [analyzer.get_lm_head()], is_zero_centered)

    memory_utils.cleanup_memory()


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return hadamard_utils.random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return hadamard_utils.random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(analyzer: model_utils.ModelAnalyzer, R1: torch.Tensor) -> None:
    if analyzer.process_word_embeddings:
        return
    # Rotate the embeddings.
    for W in [analyzer.get_embed_layer()]:
        weight = W.weight
        dtype = weight.dtype
        W_ = weight.data.to(device=R1.device, dtype=torch.float64)
        update_shared_offload_parameter(W, "weight", torch.matmul(W_, R1).to(dtype))


def rotate_head(analyzer: model_utils.ModelAnalyzer, R1: torch.Tensor) -> None:
    if analyzer.process_word_embeddings:
        return
    # Rotate the head.
    W = analyzer.get_lm_head()
    weight = W.weight
    dtype = weight.dtype
    W_ = weight.data.to(device=R1.device, dtype=torch.float64)
    update_shared_offload_parameter(W, "weight", torch.matmul(W_, R1).to(dtype))


def rotate_attention_mlp_inputs(analyzer: model_utils.ModelAnalyzer, layer, R1) -> None:
    if analyzer.process_word_embeddings:
        return
    for layers in analyzer.get_perlayer_input_modules(layer):
        for W in layers:
            weight = W.weight
            dtype = weight.dtype
            W_ = weight.to(device=R1.device, dtype=torch.float64)
            update_shared_offload_parameter(W, "weight", torch.matmul(W_, R1).to(dtype), sync=False)
    dist.barrier()


def rotate_attention_mlp_output(analyzer: model_utils.ModelAnalyzer, layer, R1) -> None:
    if analyzer.process_word_embeddings:
        return
    for layers in analyzer.get_perlayer_output_modules(layer):
        for W in layers:
            weight = W.weight
            dtype = weight.dtype
            W_ = weight.data.to(device=R1.device, dtype=torch.float64)
            update_shared_offload_parameter(W, "weight", torch.matmul(R1.T, W_).to(dtype), sync=False)
            bias = W.bias
            if bias is not None:
                b = bias.data.to(device=R1.device, dtype=torch.float64)
                update_shared_offload_parameter(W, "bias", torch.matmul(R1.T, b).to(dtype), sync=False)
    dist.barrier()


def rotate_down_proj(analyzer: model_utils.ModelAnalyzer, layer):
    # apply exact (inverse) hadamard on the weights of mlp output
    for W in analyzer.get_perlayer_down_proj(layer):
        hadamard_utils.apply_exact_had_to_linear(
            W, had_dim=-1, output=False, sync=False
        )
    dist.barrier()


def rotate_ov_proj(analyzer: model_utils.ModelAnalyzer, layer, R2=None):
    v_proj = analyzer.get_perlayer_v_proj(layer)
    o_proj = analyzer.get_perlayer_o_proj(layer)

    hadamard_utils.apply_exact_had_to_linear(v_proj, had_dim=analyzer.head_dim, output=True, R2=R2)
    hadamard_utils.apply_exact_had_to_linear(o_proj, had_dim=analyzer.head_dim, output=False, R2=R2)


@torch.inference_mode()
def rotate_model(args, analyzer: model_utils.ModelAnalyzer):
    dev = analyzer.onload_device
    R1 = get_orthogonal_matrix(analyzer.hidden_size, "hadamard", device=dev)
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        R1 = torch.load(R_cpk)["R1"].to(dev).to(torch.float64)
    if dist_utils.is_dist_available_and_initialized():
        dist.broadcast(R1, src=0)

    if analyzer.can_rotate_global():
        fuse_layer_norms(analyzer)
        rotate_embeddings(analyzer, R1)
        rotate_head(analyzer, R1)
    memory_utils.cleanup_memory()
    layers = analyzer.get_layers()
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
        if analyzer.can_rotate_global():
            rotate_attention_mlp_inputs(analyzer, layer, R1)
            rotate_attention_mlp_output(analyzer, layer, R1)
        if not args.disable_online_rot:
            rotate_down_proj(analyzer, layer)
        if analyzer.can_rotate_ov(layer):
            if args.optimized_rotation_path is not None:
                key = f"model.layers.{idx}.self_attn.R2"
                R2 = torch.load(R_cpk)[key].to(dev).to(torch.float64)
            else:
                R2 = get_orthogonal_matrix(analyzer.head_dim, "hadamard", device=dev)
            if dist_utils.is_dist_available_and_initialized():
                dist.broadcast(R2, src=0)
            rotate_ov_proj(analyzer, layer, R2=R2)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, head_dim, *args, **kwargs):
        super().__init__()
        assert hadamard_utils.is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (hadamard_utils.HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (hadamard_utils.HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
