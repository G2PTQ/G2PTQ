"""Stdlib-only tests for --w_bits_map (run: python3 -m pytest tests/test_bits_map.py, or python3 tests/test_bits_map.py)."""
import json, os, sys, tempfile
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptq_utils.bits_map import parse_bits_map, bits_for, groupsize_for, bits_key_fn, spec_bits, spec_groupsize, all_bit_widths

QWEN35 = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
          "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]
MAP = '{"^linear_attn\\\\.": 4, "^self_attn\\\\.": 4, "^mlp\\\\.": 3}'

def args_with(m, w_bits=4, w_groupsize=128):
    return SimpleNamespace(w_bits=w_bits, w_groupsize=w_groupsize, w_bits_map=parse_bits_map(m))

def test_type_map():
    a = args_with(MAP)
    got = {n: bits_for(n, a) for n in QWEN35}
    assert all(got[n] == 3 for n in QWEN35 if n.startswith("mlp."))
    assert all(got[n] == 4 for n in QWEN35 if not n.startswith("mlp."))

def test_no_map_is_uniform():
    a = args_with(None, w_bits=3)
    assert {bits_for(n, a) for n in QWEN35} == {3}

def test_first_match_wins_and_fallback():
    a = args_with('{"down_proj": 8, "^mlp\\\\.": 3}', w_bits=4)
    assert bits_for("mlp.down_proj", a) == 8
    assert bits_for("mlp.up_proj", a) == 3
    assert bits_for("self_attn.q_proj", a) == 4
    assert all_bit_widths(a) == {3, 4, 8}

def test_json_file():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"^mlp\\.": 3}, f)
    try:
        assert bits_for("mlp.down_proj", args_with(f.name)) == 3
    finally:
        os.unlink(f.name)

def test_key_fn_separates_bits():
    a = args_with('{"down_proj": 3}')
    key = bits_key_fn(a, base_key_fn=lambda n: "stage0")
    assert key("mlp.down_proj") != key("mlp.up_proj")
    assert key("mlp.up_proj") == key("mlp.gate_proj")

def test_spec_bits_rejects_mixed_chunk():
    a = args_with('{"down_proj": 3}')
    assert spec_bits(SimpleNamespace(chunk_names=("mlp.up_proj", "mlp.gate_proj")), a) == 4
    try:
        spec_bits(SimpleNamespace(chunk_names=("mlp.up_proj", "mlp.down_proj")), a)
    except AssertionError:
        pass
    else:
        raise AssertionError("mixed chunk not rejected")

def test_bad_values():
    for bad in ['{"mlp": 1}', '{"mlp": "4"}', '{}', '[]', '{"mlp": true}',
                '{"mlp": {"groupsize": 32}}', '{"mlp": {"bits": 3, "groupsize": 0}}',
                '{"mlp": {"bits": 3, "gs": 32}}']:
        try:
            parse_bits_map(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

def test_groupsize_override():
    a = args_with('{"^mlp\\\\.": {"bits": 3, "groupsize": 32}, "^self_attn\\\\.": 4}', w_bits=4, w_groupsize=128)
    assert (bits_for("mlp.down_proj", a), groupsize_for("mlp.down_proj", a)) == (3, 32)
    assert (bits_for("self_attn.q_proj", a), groupsize_for("self_attn.q_proj", a)) == (4, 128)
    assert (bits_for("linear_attn.out_proj", a), groupsize_for("linear_attn.out_proj", a)) == (4, 128)
    key = bits_key_fn(a)
    assert key("mlp.up_proj") != key("self_attn.q_proj")
    assert spec_groupsize(SimpleNamespace(chunk_names=("mlp.up_proj", "mlp.gate_proj")), a) == 32


def test_same_bits_different_groupsize_not_batched():
    a = args_with('{"down_proj": {"bits": 4, "groupsize": 32}}', w_bits=4, w_groupsize=128)
    try:
        spec_bits(SimpleNamespace(chunk_names=("mlp.up_proj", "mlp.down_proj")), a)
    except AssertionError:
        pass
    else:
        raise AssertionError("mixed group sizes not rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
