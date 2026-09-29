"""CUDA parity tests for the eager Torch and Triton compensation loops.

Run from the repository root with:

    python -m unittest discover -s tests -p 'test_triton_compensation.py' -v
"""

import contextlib
import copy
from types import SimpleNamespace
import unittest

import torch

# Match the application's import order. Importing dist_utils first lets it
# finish importing model_utils before its ModelAnalyzer annotation is resolved.
from utils import dist_utils as _dist_utils  # noqa: F401
from utils import quant_utils
from gptq_utils.g2ptq_utils import (
    _run_torch_compensation as run_g2ptq_torch,
)
from gptq_utils.g2ptq_utils import G2PTQ
from gptq_utils.g2ptq_utils import (
    _run_triton_compensation as run_g2ptq_triton,
)
from gptq_utils.g2ptq_utils import run_eager as run_g2ptq_eager
from gptq_utils.gptaq_utils import (
    _run_torch_compensation as run_gptaq_torch,
)
from gptq_utils.gptaq_utils import GPTAQ
from gptq_utils.gptaq_utils import (
    _run_triton_compensation as run_gptaq_triton,
)
from gptq_utils.gptaq_utils import run_eager as run_gptaq_eager
from gptq_utils.gptq_utils import (
    _run_torch_compensation as run_gptq_torch,
)
from gptq_utils.gptq_utils import GPTQ
from gptq_utils.gptq_utils import (
    _run_triton_compensation as run_gptq_triton,
)
from gptq_utils.gptq_guided_utils import GPTQGuided
from gptq_utils.graph_utils import clear_graph_cache, run_graph


FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-5
CASES = (
    # name, batch, columns, rows, maxq, asymmetric, seed
    ("symmetric_2bit_tail", 1, 7, 37, 1, False, 101),
    ("asymmetric_2bit_tail", 1, 7, 37, 1, True, 102),
    ("symmetric_4bit_batched", 2, 16, 193, 7, False, 201),
    ("asymmetric_4bit_batched", 2, 16, 193, 7, True, 202),
    ("symmetric_8bit_small", 3, 5, 11, 127, False, 301),
    ("asymmetric_8bit_small", 3, 5, 11, 127, True, 302),
)


@contextlib.contextmanager
def _tf32_disabled():
    """Prevent TF32 in Torch baddbmm while preserving the caller's setting."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _make_inputs(batch, columns, rows, maxq, asymmetric, seed):
    """Build stable qparams and an upper-triangular inverse-Hessian block."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, columns, rows)
    minq = -(maxq + 1)

    scale = torch.rand(shape, device=device, generator=generator)
    scale = scale.mul_(0.15).add_(0.05)
    if asymmetric:
        zero = torch.randint(
            minq,
            maxq + 1,
            shape,
            device=device,
            generator=generator,
        ).float()
    else:
        zero = torch.zeros(shape, device=device)

    # Keep values away from half-integer rounding boundaries. This ensures
    # that an exact integer mismatch identifies a quantization discrepancy,
    # rather than amplification of an earlier sub-ULP update difference.
    target_q = torch.randint(
        minq,
        maxq + 1,
        shape,
        device=device,
        generator=generator,
    ).float()
    fraction = torch.randint(
        0, 2, shape, device=device, generator=generator
    ).float()
    fraction = fraction.mul_(0.5).sub_(0.25)
    weight = (target_q - zero + fraction) * scale

    hinv = torch.randn(
        batch, columns, columns, device=device, generator=generator
    ).mul_(0.002)
    hinv = torch.triu(hinv, diagonal=1)
    diagonal = torch.rand(
        batch, columns, device=device, generator=generator
    ).mul_(0.5).add_(0.75)
    torch.diagonal(hinv, dim1=1, dim2=2).copy_(diagonal)

    return tuple(
        tensor.contiguous() for tensor in (weight, hinv, scale, zero)
    )


def _make_g2_inputs(weight, scale, seed):
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    z = torch.randn(
        weight.shape,
        device=weight.device,
        generator=generator,
    ).mul_(scale * 0.002)
    ghinv = torch.randn(
        weight.shape,
        device=weight.device,
        generator=generator,
    ).mul_(scale * 0.002)
    return z.contiguous(), ghinv.contiguous()


def _make_gptaq_p(weight, seed):
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    batch, columns, _ = weight.shape
    p = torch.randn(
        batch, columns, columns,
        device=weight.device,
        generator=generator,
    ).mul_(0.002)
    return torch.triu(p, diagonal=1).contiguous()


def _run_gptaq_reference(weight, hinv, p, scale, zero, maxq):
    qweight = torch.empty_like(weight)
    int_weight = torch.empty_like(weight)
    error = torch.empty_like(weight)
    for column in range(weight.shape[1]):
        column_weight = weight[:, column].clone()
        quantized, quant_scale, quant_zero = quant_utils.asym_quant(
            column_weight, scale[:, column], zero[:, column], maxq
        )
        q = quant_utils.asym_dequant(quantized, quant_scale, quant_zero)
        column_error = (
            column_weight - q
        ) / hinv[:, column, column].unsqueeze(1)
        qweight[:, column].copy_(q)
        int_weight[:, column].copy_(quantized)
        error[:, column].copy_(column_error)
        weight[:, column:].baddbmm_(
            hinv[:, column, column:].unsqueeze(2),
            column_error.unsqueeze(1),
            beta=1,
            alpha=-1,
        ).baddbmm_(
            p[:, column, column:].unsqueeze(2),
            column_weight.unsqueeze(1),
            beta=1,
            alpha=1,
        )
    return qweight, int_weight, error


def _run_original_gptq_loop(
    weight, hinv, quantizer, groups, blocksize, groupsize, perm,
):
    """Execute GPTQ's complete pre-refactor ``i1``/``i2`` loop from HEAD."""
    batch, rows, columns = weight.shape
    scale = torch.zeros_like(weight)
    int_weight = torch.zeros_like(weight)
    qweight = torch.zeros_like(weight)

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        weight1 = weight[:, :, i1:i2].clone()
        qweight1 = torch.zeros_like(weight1)
        int_weight1 = torch.zeros_like(weight1)
        scale1 = torch.zeros_like(weight1)
        error1 = torch.zeros_like(weight1)
        hinv1 = hinv[:, i1:i2, i1:i2]

        for i in range(count):
            w = weight1[:, :, i]
            diagonal = hinv1[:, i, i]
            if groupsize != -1:
                if groups is None:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(
                            weight[:, :, (i1 + i) : (i1 + i + groupsize)]
                            .reshape(batch * rows, -1)
                        )
                else:
                    index = i1 + i
                    if perm is not None:
                        index = perm[index]
                    quantizer = groups[index // groupsize]

            q, quantized, quant_scale = quantizer.fake_quantize(
                w.view(batch * rows, 1)
            )
            q = q.view(batch, rows)
            qweight1[:, :, i] = q
            int_weight1[:, :, i] = quantized.view(batch, rows)
            scale1[:, :, i] = quant_scale.view(batch, rows)
            column_error = (w - q) / diagonal.unsqueeze(1)
            weight1[:, :, i:] -= torch.bmm(
                column_error.unsqueeze(2),
                hinv1[:, i, i:].unsqueeze(1),
            )
            error1[:, :, i] = column_error

        qweight[:, :, i1:i2] = qweight1
        int_weight[:, :, i1:i2] = int_weight1
        scale[:, :, i1:i2] = scale1
        weight[:, :, i2:] -= torch.bmm(
            error1, hinv[:, i1:i2, i2:]
        )

    return qweight, int_weight, scale, weight


def _run_original_g2ptq_loop(
    weight, hinv, z, ghinv, quantizer, groups, blocksize, groupsize, perm,
):
    """Execute G2PTQ's complete pre-refactor ``i1``/``i2`` loop from HEAD."""
    batch, rows, columns = weight.shape
    scale = torch.zeros_like(weight)
    int_weight = torch.zeros_like(weight)
    qweight = torch.zeros_like(weight)
    distance = torch.arange(blocksize - 1, -1, -1).to(ghinv)

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        weight1 = weight[:, :, i1:i2].clone()
        qweight1 = torch.zeros_like(weight1)
        int_weight1 = torch.zeros_like(weight1)
        scale1 = torch.zeros_like(weight1)
        error1 = torch.zeros_like(weight1)
        hinv1 = hinv[:, i1:i2, i1:i2]
        ghinv1 = ghinv[:, :, i1:i2].clone()
        z1 = z[:, :, i1:i2]

        for i in range(count):
            w = weight1[:, :, i]
            diagonal = hinv1[:, i, i]
            if groupsize != -1:
                if groups is None:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(
                            weight[:, :, (i1 + i) : (i1 + i + groupsize)]
                            .reshape(batch * rows, -1)
                        )
                else:
                    index = i1 + i
                    if perm is not None:
                        index = perm[index]
                    quantizer = groups[index // groupsize]

            q, quantized, quant_scale = quantizer.fake_quantize(
                w.view(batch * rows, 1)
            )
            q = q.view(batch, rows)
            qweight1[:, :, i] = q
            int_weight1[:, :, i] = quantized.view(batch, rows)
            scale1[:, :, i] = quant_scale.view(batch, rows)
            column_error = (
                w - q - ghinv1[:, :, i]
            ) / diagonal.unsqueeze(1)
            weight1[:, :, i:] -= (
                torch.bmm(
                    column_error.unsqueeze(2),
                    hinv1[:, i, i:].unsqueeze(1),
                )
                + ghinv1[:, :, i:]
            )
            error1[:, :, i] = column_error
            ghinv1[:, :, i:] -= torch.bmm(
                z1[:, :, i].unsqueeze(2),
                hinv1[:, i, i:].unsqueeze(1),
            )

        qweight[:, :, i1:i2] = qweight1
        int_weight[:, :, i1:i2] = int_weight1
        scale[:, :, i1:i2] = scale1
        gradient_update = blocksize * ghinv[:, :, i2:] - torch.bmm(
            z1 * distance.reshape(1, 1, -1),
            hinv[:, i1:i2, i2:],
        )
        weight[:, :, i2:] -= (
            torch.bmm(error1, hinv[:, i1:i2, i2:]) + gradient_update
        )
        ghinv[:, :, i2:] -= torch.bmm(
            z[:, :, i1:i2], hinv[:, i1:i2, i2:]
        )

    return qweight, int_weight, scale, weight, ghinv


def _run_original_gptaq_loop(
    weight, hinv, p, quantizer, groups, blocksize, groupsize, perm,
):
    """Execute GPTAQ's complete pre-refactor ``i1``/``i2`` loop from HEAD."""
    batch, rows, columns = weight.shape
    scale = torch.zeros_like(weight)
    int_weight = torch.zeros_like(weight)
    qweight = torch.zeros_like(weight)

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1
        weight1 = weight[:, :, i1:i2].clone()
        qweight1 = torch.zeros_like(weight1)
        int_weight1 = torch.zeros_like(weight1)
        scale1 = torch.zeros_like(weight1)
        error1 = torch.zeros_like(weight1)
        hinv1 = hinv[:, i1:i2, i1:i2]
        p1 = p[:, i1:i2, i1:i2]

        for i in range(count):
            w = weight1[:, :, i]
            diagonal = hinv1[:, i, i]
            if groupsize != -1:
                if groups is None:
                    if (i1 + i) % groupsize == 0:
                        quantizer.find_params(
                            weight[:, :, (i1 + i) : (i1 + i + groupsize)]
                            .reshape(batch * rows, -1)
                        )
                else:
                    index = i1 + i
                    if perm is not None:
                        index = perm[index]
                    quantizer = groups[index // groupsize]

            q, quantized, quant_scale = quantizer.fake_quantize(
                w.view(batch * rows, 1)
            )
            q = q.view(batch, rows)
            qweight1[:, :, i] = q
            int_weight1[:, :, i] = quantized.view(batch, rows)
            scale1[:, :, i] = quant_scale.view(batch, rows)
            column_error = (w - q) / diagonal.unsqueeze(1)
            weight1[:, :, i:] -= (
                torch.bmm(
                    column_error.unsqueeze(2),
                    hinv1[:, i, i:].unsqueeze(1),
                )
                - torch.bmm(w.unsqueeze(2), p1[:, i, i:].unsqueeze(1))
            )
            error1[:, :, i] = column_error

        qweight[:, :, i1:i2] = qweight1
        int_weight[:, :, i1:i2] = int_weight1
        scale[:, :, i1:i2] = scale1
        weight[:, :, i2:] -= (
            torch.bmm(error1, hinv[:, i1:i2, i2:])
            - torch.bmm(weight1, p[:, i1:i2, i2:])
        )

    return qweight, int_weight, scale, weight


def _prepare_loop_quantizers(
    weight, bits, symmetric, groupsize, static_groups,
):
    """Build the single original quantizer and current group-quantizer list."""
    batch, rows, columns = weight.shape
    template = quant_utils.WeightQuantizer()
    template.configure(
        bits, perchannel=True, sym=symmetric, mse=False
    )

    original_quantizer = copy.deepcopy(template)
    original_quantizer.find_params(weight.reshape(batch * rows, columns))
    groups = None
    if static_groups:
        groups = []
        for start in range(0, columns, groupsize):
            quantizer = copy.deepcopy(original_quantizer)
            quantizer.find_params(weight[:, :, start : start + groupsize].reshape(
                batch * rows, -1
            ))
            groups.append(quantizer)
        current_quantizers = copy.deepcopy(groups)
    elif groupsize == -1:
        current_quantizers = [copy.deepcopy(original_quantizer)]
    else:
        num_groups = (columns + groupsize - 1) // groupsize
        current_quantizers = [
            copy.deepcopy(template) for _ in range(num_groups)
        ]

    return original_quantizer, groups, current_quantizers


def _make_g2_algorithm_inputs(
    batch, columns, rows, maxq, asymmetric, seed,
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    minq = -(maxq + 1)
    row_shape = (batch, rows)
    column_shape = (batch, columns, rows)

    row_scale = torch.rand(
        row_shape, device=device, generator=generator
    ).mul_(0.15).add_(0.05)
    if asymmetric:
        row_zero = torch.randint(
            minq,
            maxq + 1,
            row_shape,
            device=device,
            generator=generator,
        ).float()
    else:
        row_zero = torch.zeros(row_shape, device=device)
    scale = row_scale.unsqueeze(1).expand(column_shape)
    zero = row_zero.unsqueeze(1).expand(column_shape)

    target_q = torch.randint(
        minq,
        maxq + 1,
        column_shape,
        device=device,
        generator=generator,
    ).float()
    fraction = torch.randint(
        0, 2, column_shape, device=device, generator=generator
    ).float()
    fraction = fraction.mul_(0.5).sub_(0.25)
    weight = (target_q - zero + fraction) * scale

    hinv = torch.randn(
        batch, columns, columns, device=device, generator=generator
    ).mul_(0.002)
    hinv = torch.triu(hinv, diagonal=1)
    diagonal = torch.rand(
        batch, columns, device=device, generator=generator
    ).mul_(0.5).add_(0.75)
    torch.diagonal(hinv, dim1=1, dim2=2).copy_(diagonal)
    z, ghinv = _make_g2_inputs(weight, scale, seed + 1_000)

    quantizer = SimpleNamespace(
        maxq=maxq,
        scale=row_scale.reshape(batch * rows, 1),
        zero=row_zero.reshape(batch * rows, 1),
    )
    return (
        weight.transpose(1, 2).contiguous(),
        hinv.contiguous(),
        z.transpose(1, 2).contiguous(),
        ghinv.transpose(1, 2).contiguous(),
        quantizer,
    )


def _assert_float_parity(actual, expected):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
    )


def _run_g2ptq_graph(
    weight, hinv, scale, zero, z, ghinv, maxq, backend,
):
    params = {
        "weight": weight.clone(),
        "hinv": hinv.clone(),
        "scale": scale.clone(),
        "zero": zero.clone(),
        "z": z.clone(),
        "ghinv": ghinv.clone(),
        "maxq": maxq,
        "backend": backend,
        "int_weight": torch.empty_like(weight),
        "error": torch.empty_like(weight),
    }
    return run_graph(
        run_g2ptq_eager,
        params,
        input_names=(
            "weight", "hinv", "scale", "zero", "z", "ghinv"
        ),
        output_names=("weight", "int_weight", "error"),
        name="G2PTQ compensation block test",
    )


def _run_gptaq_graph(weight, hinv, p, scale, zero, maxq, backend):
    params = {
        "weight": weight.clone(),
        "hinv": hinv.clone(),
        "p": p.clone(),
        "scale": scale.clone(),
        "zero": zero.clone(),
        "maxq": maxq,
        "backend": backend,
        "column_weight": torch.empty(
            weight.shape[0], weight.shape[2],
            dtype=weight.dtype, device=weight.device,
        ),
        "int_weight": torch.empty_like(weight),
        "error": torch.empty_like(weight),
    }
    return run_graph(
        run_gptaq_eager,
        params,
        input_names=("weight", "hinv", "p", "scale", "zero"),
        output_names=("weight", "int_weight", "error"),
        name="GPTAQ compensation block test",
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonCompensationParityTest(unittest.TestCase):
    def tearDown(self):
        clear_graph_cache()

    def test_unified_asym_quantization_matches_original_fake_quantize(self):
        for bits in (2, 4, 8):
            for symmetric in (True, False):
                with self.subTest(bits=bits, symmetric=symmetric):
                    generator = torch.Generator(device="cuda").manual_seed(
                        5_000 + bits * 10 + int(symmetric)
                    )
                    batch, rows, columns = 2, 37, 13
                    weight = torch.randn(
                        batch, rows, columns,
                        device="cuda", generator=generator,
                    )
                    quantizer = quant_utils.WeightQuantizer()
                    quantizer.configure(
                        bits, perchannel=True, sym=symmetric, mse=False
                    )
                    quantizer.find_params(
                        weight.reshape(batch * rows, columns)
                    )

                    original_qweight, original_int, _ = (
                        quantizer.fake_quantize(
                            weight[:, :, 0].reshape(batch * rows, 1)
                        )
                    )
                    original_qweight = original_qweight.reshape(batch, rows)
                    original_int = original_int.reshape(batch, rows)
                    quantized, scale, zero = quant_utils.asym_quant(
                        weight[:, :, 0],
                        quantizer.scale.reshape(batch, rows),
                        quantizer.zero.reshape(batch, rows),
                        quantizer.maxq,
                    )
                    qweight = quant_utils.asym_dequant(
                        quantized, scale, zero
                    )

                    torch.testing.assert_close(
                        quantized, original_int, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        qweight, original_qweight, rtol=0, atol=0
                    )

    def test_torch_compensation_matches_original_full_loops(self):
        batch, rows, columns = 2, 37, 15
        blocksize = 5
        bits = 4
        configurations = (
            ("ungrouped", -1, False, False),
            ("dynamic_groups", 4, False, False),
            ("static_groups_actorder", 4, True, True),
        )

        def assert_results_equal(algorithm, actual, expected):
            with self.subTest(algorithm=algorithm):
                torch.testing.assert_close(
                    actual[1], expected[1], rtol=0, atol=0
                )
                _assert_float_parity(actual[2], expected[2])
                for actual_tensor, expected_tensor in zip(
                    actual[::3], expected[::3]
                ):
                    _assert_float_parity(actual_tensor, expected_tensor)
                if len(actual) == 5:
                    _assert_float_parity(actual[4], expected[4])

        for config_index, (
            name, groupsize, static_groups, actorder,
        ) in enumerate(configurations):
            for symmetric in (True, False):
                with self.subTest(configuration=name, symmetric=symmetric):
                    generator = torch.Generator(device="cuda").manual_seed(
                        8_000 + config_index * 100 + int(symmetric)
                    )
                    original_weight = torch.randn(
                        batch, rows, columns,
                        device="cuda", generator=generator,
                    )
                    perm = None
                    weight = original_weight.clone()
                    if actorder:
                        perm = torch.randperm(
                            columns, device="cuda", generator=generator
                        )
                        weight = weight[:, :, perm]

                    hinv = torch.randn(
                        batch, columns, columns,
                        device="cuda", generator=generator,
                    ).mul_(0.002)
                    hinv = torch.triu(hinv, diagonal=1)
                    torch.diagonal(hinv, dim1=1, dim2=2).copy_(
                        torch.rand(
                            batch, columns,
                            device="cuda", generator=generator,
                        ).mul_(0.5).add_(0.75)
                    )
                    z = torch.randn(
                        weight.shape, device="cuda", generator=generator
                    ).mul_(0.002)
                    ghinv = torch.randn(
                        weight.shape, device="cuda", generator=generator
                    ).mul_(0.002)
                    p = torch.randn(
                        batch, columns, columns,
                        device="cuda", generator=generator,
                    ).mul_(0.002)
                    p = torch.triu(p, diagonal=1).contiguous()
                    qparam_perm = perm if static_groups and actorder else None

                    def prepare_quantizers():
                        return _prepare_loop_quantizers(
                            original_weight,
                            bits,
                            symmetric,
                            groupsize,
                            static_groups,
                        )

                    original_quantizer, groups, current_quantizers = (
                        prepare_quantizers()
                    )
                    expected_gptq = _run_original_gptq_loop(
                        weight.clone(),
                        hinv.clone(),
                        original_quantizer,
                        groups,
                        blocksize,
                        groupsize,
                        qparam_perm,
                    )

                    for algorithm, runner_type in (
                        ("gptq", GPTQ),
                        ("gptq_guided", GPTQGuided),
                    ):
                        if algorithm == "gptq_guided":
                            _, _, current_quantizers = prepare_quantizers()
                        runner = object.__new__(runner_type)
                        runner.columns = columns
                        runner.num_layers = batch
                        runner.rows = rows
                        runner.quantizer = current_quantizers
                        current_weight = weight.clone()
                        current_result = runner._run_compensation(
                            current_weight,
                            hinv.clone(),
                            blocksize,
                            groupsize,
                            static_groups,
                            qparam_perm,
                            backend="torch",
                            graph=False,
                        )
                        assert_results_equal(
                            algorithm,
                            (*current_result, current_weight),
                            expected_gptq,
                        )

                    original_quantizer, groups, current_quantizers = (
                        prepare_quantizers()
                    )
                    expected_g2ptq = _run_original_g2ptq_loop(
                        weight.clone(),
                        hinv.clone(),
                        z.clone(),
                        ghinv.clone(),
                        original_quantizer,
                        groups,
                        blocksize,
                        groupsize,
                        qparam_perm,
                    )
                    g2ptq = object.__new__(G2PTQ)
                    g2ptq.columns = columns
                    g2ptq.num_layers = batch
                    g2ptq.rows = rows
                    g2ptq.quantizer = current_quantizers
                    current_weight = weight.clone()
                    current_ghinv = ghinv.clone()
                    current_result = g2ptq._run_compensation(
                        current_weight,
                        hinv.clone(),
                        z.clone(),
                        current_ghinv,
                        blocksize,
                        groupsize,
                        static_groups,
                        qparam_perm,
                        backend="torch",
                        graph=False,
                    )
                    assert_results_equal(
                        "g2ptq",
                        (*current_result, current_weight, current_ghinv),
                        expected_g2ptq,
                    )

                    original_quantizer, groups, current_quantizers = (
                        prepare_quantizers()
                    )
                    expected_gptaq = _run_original_gptaq_loop(
                        weight.clone(),
                        hinv.clone(),
                        p.clone(),
                        original_quantizer,
                        groups,
                        blocksize,
                        groupsize,
                        qparam_perm,
                    )
                    gptaq = object.__new__(GPTAQ)
                    gptaq.columns = columns
                    gptaq.num_layers = batch
                    gptaq.rows = rows
                    gptaq.quantizer = current_quantizers
                    current_weight = weight.clone()
                    current_result = gptaq._run_compensation(
                        current_weight,
                        hinv.clone(),
                        p.clone(),
                        blocksize,
                        groupsize,
                        static_groups,
                        qparam_perm,
                        backend="torch",
                        graph=False,
                    )
                    assert_results_equal(
                        "gptaq",
                        (*current_result, current_weight),
                        expected_gptaq,
                    )
                    torch.cuda.synchronize()

    def test_gptq_torch_matches_triton(self):
        for name, batch, columns, rows, maxq, asymmetric, seed in CASES:
            with self.subTest(case=name):
                weight, hinv, scale, zero = _make_inputs(
                    batch, columns, rows, maxq, asymmetric, seed
                )
                torch_weight = weight.clone()
                triton_weight = weight.clone()
                torch_int = torch.empty_like(weight)
                triton_int = torch.empty_like(weight)
                torch_error = torch.empty_like(weight)
                triton_error = torch.empty_like(weight)

                with _tf32_disabled():
                    run_gptq_torch(
                        torch_weight,
                        hinv.clone(),
                        scale.clone(),
                        zero.clone(),
                        torch_int,
                        torch_error,
                        maxq,
                    )
                    run_gptq_triton(
                        triton_weight,
                        hinv.clone(),
                        scale.clone(),
                        zero.clone(),
                        triton_int,
                        triton_error,
                        maxq,
                    )
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    triton_int, torch_int, rtol=0, atol=0
                )
                _assert_float_parity(triton_weight, torch_weight)
                _assert_float_parity(triton_error, torch_error)

    def test_g2ptq_torch_matches_triton(self):
        for name, batch, columns, rows, maxq, asymmetric, seed in CASES:
            with self.subTest(case=name):
                weight, hinv, scale, zero = _make_inputs(
                    batch, columns, rows, maxq, asymmetric, seed
                )
                z, ghinv = _make_g2_inputs(weight, scale, seed + 1_000)
                torch_weight = weight.clone()
                triton_weight = weight.clone()
                torch_ghinv = ghinv.clone()
                triton_ghinv = ghinv.clone()
                torch_int = torch.empty_like(weight)
                triton_int = torch.empty_like(weight)
                torch_error = torch.empty_like(weight)
                triton_error = torch.empty_like(weight)

                with _tf32_disabled():
                    run_g2ptq_torch(
                        torch_weight,
                        hinv.clone(),
                        scale.clone(),
                        zero.clone(),
                        z.clone(),
                        torch_ghinv,
                        torch_int,
                        torch_error,
                        maxq,
                    )
                    run_g2ptq_triton(
                        triton_weight,
                        hinv.clone(),
                        scale.clone(),
                        zero.clone(),
                        z.clone(),
                        triton_ghinv,
                        triton_int,
                        triton_error,
                        maxq,
                    )
                torch.cuda.synchronize()

                torch.testing.assert_close(
                    triton_int, torch_int, rtol=0, atol=0
                )
                _assert_float_parity(triton_weight, torch_weight)
                _assert_float_parity(triton_error, torch_error)
                _assert_float_parity(triton_ghinv, torch_ghinv)

    def test_gptaq_torch_and_triton_match_reference(self):
        for name, batch, columns, rows, maxq, asymmetric, seed in CASES:
            with self.subTest(case=name):
                weight, hinv, scale, zero = _make_inputs(
                    batch, columns, rows, maxq, asymmetric, seed
                )
                p = _make_gptaq_p(weight, seed + 3_000)
                reference = _run_gptaq_reference(
                    weight.clone(), hinv.clone(), p.clone(),
                    scale.clone(), zero.clone(), maxq,
                )
                actual_runs = []
                for backend, run_compensation in (
                    ("torch", run_gptaq_torch),
                    ("triton", run_gptaq_triton),
                ):
                    current_weight = weight.clone()
                    column_weight = torch.empty(
                        batch, rows, dtype=weight.dtype, device=weight.device
                    )
                    int_weight = torch.empty_like(weight)
                    error = torch.empty_like(weight)
                    run_compensation(
                        current_weight,
                        hinv.clone(),
                        p.clone(),
                        scale.clone(),
                        zero.clone(),
                        column_weight,
                        int_weight,
                        error,
                        maxq,
                    )
                    actual_runs.append(
                        (backend, (current_weight, int_weight, error))
                    )
                torch.cuda.synchronize()

                for backend, actual in actual_runs:
                    with self.subTest(case=name, backend=backend):
                        torch.testing.assert_close(
                            actual[1], reference[1], rtol=0, atol=0
                        )
                        _assert_float_parity(actual[0], reference[0])
                        _assert_float_parity(actual[2], reference[2])

    def test_guided_gptq_reuses_gptq_compensation(self):
        self.assertIs(
            GPTQGuided._run_compensation_block,
            GPTQ._run_compensation_block,
        )
        self.assertIs(
            GPTQGuided._run_compensation,
            GPTQ._run_compensation,
        )

    def test_gptaq_block_traversal_matches_across_backends(self):
        weight, hinv, _, _, quantizer = _make_g2_algorithm_inputs(
            batch=2,
            columns=13,
            rows=37,
            maxq=7,
            asymmetric=True,
            seed=701,
        )
        generator = torch.Generator(device=weight.device).manual_seed(702)
        weight_raw = weight.add(
            torch.randn(
                weight.shape, device=weight.device, generator=generator
            ).mul_(0.002)
        )
        p = torch.randn(
            weight.shape[0], weight.shape[2], weight.shape[2],
            device=weight.device,
            generator=generator,
        ).mul_(0.002)
        p = torch.triu(p, diagonal=1).contiguous()

        gptaq = object.__new__(GPTAQ)
        for runner in (gptaq):
            runner.columns = weight.shape[-1]
            runner.num_layers = weight.shape[0]
            runner.rows = weight.shape[1]
            runner.quantizer = [quantizer]

        def execute_gptaq(backend, graph):
            current_weight = weight.clone()
            outputs = gptaq._run_compensation(
                current_weight,
                hinv.clone(),
                p.clone(),
                blocksize=5,
                groupsize=-1,
                static_groups=False,
                qparam_perm=None,
                backend=backend,
                graph=graph,
            )
            return (*outputs, current_weight)

        with _tf32_disabled():
            for algorithm, execute in (
                ("gptaq", execute_gptaq),
            ):
                expected = execute("torch", graph=False)
                actual_runs = (
                    ("torch_graph", execute("torch", graph=True)),
                    ("triton", execute("triton", graph=False)),
                    ("triton_graph", execute("triton", graph=True)),
                )
                torch.cuda.synchronize()
                for mode, actual in actual_runs:
                    with self.subTest(algorithm=algorithm, mode=mode):
                        torch.testing.assert_close(
                            actual[1], expected[1], rtol=0, atol=0
                        )
                        for actual_tensor, expected_tensor in zip(
                            (actual[0], actual[2], actual[3]),
                            (expected[0], expected[2], expected[3]),
                        ):
                            _assert_float_parity(actual_tensor, expected_tensor)

    def test_g2ptq_graph_matches_eager_and_owns_outputs(self):
        case = (2, 16, 193, 7, True)
        for backend in ("torch", "triton"):
            with self.subTest(backend=backend):
                clear_graph_cache()
                retained = None
                for seed in (401, 402):
                    weight, hinv, scale, zero = _make_inputs(
                        *case, seed
                    )
                    z, ghinv = _make_g2_inputs(weight, scale, seed + 1_000)
                    eager_weight = weight.clone()
                    eager_int = torch.empty_like(weight)
                    eager_error = torch.empty_like(weight)
                    eager_ghinv = ghinv.clone()

                    with _tf32_disabled():
                        run_g2ptq_eager(
                            eager_weight,
                            hinv.clone(),
                            scale.clone(),
                            zero.clone(),
                            z.clone(),
                            eager_ghinv,
                            maxq=case[3],
                            backend=backend,
                            int_weight=eager_int,
                            error=eager_error,
                        )
                        graph_outputs = _run_g2ptq_graph(
                            weight,
                            hinv,
                            scale,
                            zero,
                            z,
                            ghinv,
                            maxq=case[3],
                            backend=backend,
                        )
                    torch.cuda.synchronize()

                    graph_weight, graph_int, graph_error = graph_outputs
                    torch.testing.assert_close(
                        graph_int, eager_int, rtol=0, atol=0
                    )
                    _assert_float_parity(graph_weight, eager_weight)
                    _assert_float_parity(graph_error, eager_error)

                    if retained is not None:
                        for output, snapshot in zip(retained[0], retained[1]):
                            torch.testing.assert_close(
                                output, snapshot, rtol=0, atol=0
                            )
                    retained = (
                        graph_outputs,
                        tuple(output.clone() for output in graph_outputs),
                    )

    def test_g2ptq_block_traversal_matches_across_backends(self):
        weight, hinv, z, ghinv, quantizer = _make_g2_algorithm_inputs(
            batch=2,
            columns=13,
            rows=37,
            maxq=7,
            asymmetric=True,
            seed=501,
        )
        runner = object.__new__(G2PTQ)
        runner.columns = weight.shape[-1]
        runner.num_layers = 1
        runner.rows = weight.shape[0] * weight.shape[1]
        runner.quantizer = [quantizer]

        def execute(backend, graph):
            current_weight = weight.clone()
            current_ghinv = ghinv.clone()
            outputs = runner._run_compensation(
                current_weight,
                hinv.clone(),
                z.clone(),
                current_ghinv,
                blocksize=5,
                groupsize=-1,
                static_groups=False,
                qparam_perm=None,
                backend=backend,
                graph=graph,
            )
            return (*outputs, current_weight, current_ghinv)

        with _tf32_disabled():
            expected = execute("torch", graph=False)
            actual_runs = (
                ("torch_graph", execute("torch", graph=True)),
                ("triton", execute("triton", graph=False)),
                ("triton_graph", execute("triton", graph=True)),
            )
        torch.cuda.synchronize()

        for name, actual in actual_runs:
            with self.subTest(mode=name):
                torch.testing.assert_close(
                    actual[1], expected[1], rtol=0, atol=0
                )
                for actual_tensor, expected_tensor in zip(
                    (actual[0], actual[2], actual[3], actual[4]),
                    (expected[0], expected[2], expected[3], expected[4]),
                ):
                    _assert_float_parity(actual_tensor, expected_tensor)

if __name__ == "__main__":
    unittest.main()
