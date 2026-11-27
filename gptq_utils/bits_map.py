"""Per-module weight bit widths and group sizes (layer-type-aware quantization).

``--w_bits_map`` takes a JSON object (inline, or a path to a .json file) mapping regex
patterns to either a bit width or ``{"bits": b, "groupsize": g}``. Patterns are matched with
``re.search`` against module names relative to the decoder layer, e.g.
``linear_attn.in_proj_qkv``, ``self_attn.o_proj``, ``mlp.down_proj``. The first matching
pattern wins; unmatched modules use ``--w_bits`` / ``--w_groupsize``.

Examples (Qwen3.5)::

    # DeltaNet 4-bit, full attention 4-bit, FFN 3-bit, all at --w_groupsize
    --w_bits_map '{"^linear_attn\\\\.": 4, "^self_attn\\\\.": 4, "^mlp\\\\.": 3}'

    # FFN at 3-bit with group size 64, everything else at the global setting
    --w_bits_map '{"^mlp\\\\.": {"bits": 3, "groupsize": 64}}'
"""

import json
import os
import re

QWEN35_TYPES = {
    "deltanet": r"^linear_attn\.",
    "attention": r"^self_attn\.",
    "ffn": r"^mlp\.",
}


def _parse_value(pattern, value):
    if isinstance(value, bool):
        raise ValueError(f"--w_bits_map: invalid value for {pattern!r}: {value!r}")
    if isinstance(value, int):
        bits, groupsize = value, None
    elif isinstance(value, dict):
        unknown = set(value) - {"bits", "groupsize"}
        if unknown or "bits" not in value:
            raise ValueError(f"--w_bits_map: {pattern!r} must be an int or {{'bits', 'groupsize'}}, got {value!r}")
        bits, groupsize = value["bits"], value.get("groupsize")
    else:
        raise ValueError(f"--w_bits_map: {pattern!r} must be an int or an object, got {value!r}")
    if not isinstance(bits, int) or isinstance(bits, bool) or not 2 <= bits <= 16:
        raise ValueError(f"--w_bits_map: bits for {pattern!r} must be an int in [2, 16], got {bits!r}")
    if groupsize is not None and (not isinstance(groupsize, int) or isinstance(groupsize, bool)
                                  or not (groupsize == -1 or groupsize > 0)):
        raise ValueError(f"--w_bits_map: groupsize for {pattern!r} must be -1 or a positive int, got {groupsize!r}")
    return bits, groupsize


def parse_bits_map(value):
    """Return an ordered list of (compiled_regex, bits, groupsize_or_None), or None."""
    if value is None:
        return None
    if isinstance(value, list):  # already parsed
        return value
    if os.path.isfile(value):
        with open(value) as f:
            raw = json.load(f)
    else:
        raw = json.loads(value)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("--w_bits_map must be a non-empty JSON object {pattern: bits}")
    return [(re.compile(p), *_parse_value(p, v)) for p, v in raw.items()]


def qconf_for(name, args):
    """(bits, groupsize) for the module ``name`` (relative to its decoder layer)."""
    for regex, bits, groupsize in getattr(args, "w_bits_map", None) or ():
        if regex.search(name):
            return bits, (args.w_groupsize if groupsize is None else groupsize)
    return args.w_bits, args.w_groupsize


def bits_for(name, args):
    return qconf_for(name, args)[0]


def groupsize_for(name, args):
    return qconf_for(name, args)[1]


def drop_unquantized(subset, args):
    """Remove modules assigned 16 (or more) bits: they stay in their original precision."""
    return {n: m for n, m in subset.items() if bits_for(n, args) < 16}


def all_bit_widths(args):
    """Every bit width that can be assigned."""
    widths = {args.w_bits}
    widths.update(bits for _, bits, _ in (getattr(args, "w_bits_map", None) or ()))
    return widths


def bits_key_fn(args, base_key_fn=None):
    """Chunk-grouping key so modules with different (bits, groupsize) never share a quantizer."""
    if base_key_fn is None:
        return lambda name: qconf_for(name, args)
    return lambda name: (base_key_fn(name), qconf_for(name, args))


def chunk_qconf(chunk_names, args):
    """(bits, groupsize) shared by all modules in a chunk (asserts they agree)."""
    confs = {qconf_for(n, args) for n in chunk_names}
    assert len(confs) == 1, f"Chunk mixes quantization settings {confs}: {chunk_names}"
    return confs.pop()


def spec_bits(spec, args):
    return chunk_qconf(spec.chunk_names, args)[0]


def spec_groupsize(spec, args):
    return chunk_qconf(spec.chunk_names, args)[1]


def describe(args):
    """Human-readable summary for logs."""
    if not getattr(args, "w_bits_map", None):
        return f"uniform: {args.w_bits} bits, groupsize {args.w_groupsize}"
    parts = [f"{r.pattern} -> {b} bits, groupsize {args.w_groupsize if g is None else g}"
             for r, b, g in args.w_bits_map]
    parts.append(f"(default) -> {args.w_bits} bits, groupsize {args.w_groupsize}")
    return "; ".join(parts)
