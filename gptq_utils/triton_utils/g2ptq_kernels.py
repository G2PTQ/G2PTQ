"""Triton kernels for G2PTQ compensation.

Adapted from MoE-Quant.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=[
    "batch_size", "columns", "rows", "column", "maxq",
])
def _quantize_column_kernel(
    weight_ptr, hinv_ptr, scale_ptr, zero_ptr, ghinv_ptr, int_weight_ptr,
    error_ptr, batch_size, columns, rows, column, maxq,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    elements = batch_size * rows
    mask = offsets < elements
    batch = offsets // rows
    row = offsets % rows
    value_offsets = (batch * columns + column) * rows + row
    diagonal_offsets = (batch * columns + column) * columns + column

    weight = tl.load(weight_ptr + value_offsets, mask=mask)
    ghinv = tl.load(ghinv_ptr + value_offsets, mask=mask)
    scale = tl.load(scale_ptr + value_offsets, mask=mask)
    zero = tl.load(zero_ptr + value_offsets, mask=mask)
    diagonal = tl.load(hinv_ptr + diagonal_offsets, mask=mask)
    rounded = tl.floor(weight / scale + 0.5)
    # Use the same zero-point formulation for both modes. Symmetric quantizers
    # provide zero-filled zero-point tensors.
    quantized = tl.minimum(tl.maximum(rounded + zero, -(maxq + 1)), maxq)
    dequantized = (quantized - zero) * scale
    error = (weight - dequantized - ghinv) / diagonal

    tl.store(weight_ptr + value_offsets, dequantized, mask=mask)
    tl.store(int_weight_ptr + value_offsets, quantized, mask=mask)
    tl.store(error_ptr + value_offsets, error, mask=mask)


@triton.jit(do_not_specialize=[
    "batch_size", "columns", "rows", "column", "block_end",
])
def _g2ptq_update_kernel(
    weight_ptr, hinv_ptr, error_ptr, ghinv_ptr, z_ptr, batch_size, columns,
    rows, column, block_end, BLOCK_SIZE: tl.constexpr,
):
    future_columns = block_end - column - 1
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    elements = batch_size * future_columns * rows
    mask = offsets < elements
    row = offsets % rows
    tmp = offsets // rows
    future_column = column + 1 + (tmp % future_columns)
    batch = tmp // future_columns

    value_offsets = (batch * columns + future_column) * rows + row
    column_offsets = (batch * columns + column) * rows + row
    hinv_offsets = (batch * columns + column) * columns + future_column
    weight = tl.load(weight_ptr + value_offsets, mask=mask)
    ghinv = tl.load(ghinv_ptr + value_offsets, mask=mask)
    z = tl.load(z_ptr + column_offsets, mask=mask)
    hinv = tl.load(hinv_ptr + hinv_offsets, mask=mask)
    error = tl.load(error_ptr + column_offsets, mask=mask)

    tl.store(weight_ptr + value_offsets, weight - (error * hinv + ghinv), mask=mask)
    tl.store(ghinv_ptr + value_offsets, ghinv - z * hinv, mask=mask)


def launch_quantize(
    weight, hinv, scale, zero, ghinv, int_weight, error, column,
    maxq, tile_size,
) -> None:
    batch_size, columns, rows = weight.shape
    elements = batch_size * rows
    _quantize_column_kernel[(triton.cdiv(elements, tile_size),)](
        weight, hinv, scale, zero, ghinv, int_weight, error,
        batch_size, columns, rows, column, maxq, BLOCK_SIZE=tile_size,
    )


def launch_update(weight, hinv, error, ghinv, z, column, tile_size) -> None:
    block_end = weight.shape[1]
    if column + 1 >= block_end:
        return
    batch_size, columns, rows = weight.shape
    elements = batch_size * (block_end - column - 1) * rows
    _g2ptq_update_kernel[(triton.cdiv(elements, tile_size),)](
        weight, hinv, error, ghinv, z, batch_size, columns, rows,
        column, block_end, BLOCK_SIZE=tile_size,
    )
