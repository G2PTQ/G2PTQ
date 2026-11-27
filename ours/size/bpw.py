"""Storage size of a quantized checkpoint: bits per weight and bytes, from safetensors headers only.

Counts the decoder-layer Linear weights that ptq.py quantizes (for Qwen3.5: DeltaNet
in_proj_qkv / in_proj_z / out_proj, attention q/k/v/o_proj, FFN up/gate/down_proj) at
``bits + scale_bits / groupsize`` bits per weight (one FP16 scale per group; add a zero point
for asymmetric quantization), and everything else in the text model (embeddings, lm_head,
norms, DeltaNet gate projections and convolutions, ...) at 16 bits. The vision tower and MTP
head are reported separately and excluded from the totals.

Examples:
    # one configuration
    python ours/size/bpw.py --model ./modelzoo/Qwen3.5/Qwen3.5-4B --w_bits 4 --w_groupsize 128 \
        --w_bits_map '{"^mlp\\.": 3}'
    # every DeltaNet/attention/FFN assignment from {3, 4} bits, against a budget
    python ours/size/bpw.py --model ... --w_groupsize 128 --enumerate --bits_choices 3 4 --budget 3.5
    # uniform settings whose size is closest to a target
    python ours/size/bpw.py --model ... --match 3.5
"""

import argparse
import itertools
import json
import os
import re
import struct
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gptq_utils.bits_map import parse_bits_map, qconf_for, QWEN35_TYPES  # noqa: E402

QUANTIZABLE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.((?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))|"
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj))|(?:mlp\.(?:up_proj|gate_proj|down_proj)))\.weight$"
)
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1}


def read_headers(model_dir):
    """{tensor_name: (dtype, shape)} for every tensor in the checkpoint."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    files = (sorted(set(json.load(open(index))["weight_map"].values())) if os.path.isfile(index)
             else ["model.safetensors"])
    out = {}
    for fname in files:
        with open(os.path.join(model_dir, fname), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name != "__metadata__":
                out[name] = (info["dtype"], info["shape"])
    return out


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def module_type(rel_name):
    for t, pat in QWEN35_TYPES.items():
        if re.search(pat, rel_name):
            return t
    return "other"


def size_report(headers, args, scale_bits=16, asym=False):
    per_type = defaultdict(lambda: {"params": 0, "bits": 0.0})
    other_bytes, excluded = 0, defaultdict(int)
    for name, (dtype, shape) in headers.items():
        if name.startswith(("model.visual", "visual.", "mtp.", "model.mtp")) or ".visual." in name:
            excluded["vision" if "visual" in name else "mtp"] += numel(shape) * DTYPE_BYTES.get(dtype, 2)
            continue
        m = QUANTIZABLE.search(name)
        if m and len(shape) == 2:
            rel = m.group(2)
            bits, groupsize = qconf_for(rel, args)
            cols = shape[1]
            gs = cols if groupsize == -1 else min(groupsize, cols)
            overhead = (scale_bits + (bits if asym else 0)) / gs  # zero points stored at weight precision
            if bits >= 16:
                bpw = 16.0
            else:
                bpw = bits + overhead
            t = module_type(rel)
            per_type[t]["params"] += numel(shape)
            per_type[t]["bits"] += numel(shape) * bpw
        else:
            other_bytes += numel(shape) * 2  # kept at 16 bits
    q_params = sum(v["params"] for v in per_type.values())
    q_bits = sum(v["bits"] for v in per_type.values())
    return {
        "quantized_params": q_params,
        "avg_bpw_quantized": q_bits / q_params if q_params else float("nan"),
        "per_type": {t: {"params": v["params"], "share": v["params"] / q_params,
                         "bpw": v["bits"] / v["params"]} for t, v in sorted(per_type.items())},
        "quantized_gb": q_bits / 8 / 1e9,
        "unquantized_gb": other_bytes / 1e9,
        "text_model_gb": (q_bits / 8 + other_bytes) / 1e9,
        "bf16_text_model_gb": (q_params * 2 + other_bytes) / 1e9,
        "excluded_gb": {k: v / 1e9 for k, v in excluded.items()},
    }


def make_args(w_bits, w_groupsize, bits_map):
    return argparse.Namespace(w_bits=w_bits, w_groupsize=w_groupsize, w_bits_map=parse_bits_map(bits_map))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Local checkpoint directory (safetensors)")
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--w_groupsize", type=int, default=128)
    ap.add_argument("--w_bits_map", default=None)
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--scale_bits", type=int, default=16)
    ap.add_argument("--enumerate", action="store_true", help="All per-type bit assignments from --bits_choices")
    ap.add_argument("--bits_choices", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--budget", type=float, default=None, help="Max average bits per quantized weight")
    ap.add_argument("--match", type=float, default=None, help="List uniform settings near this bpw")
    ap.add_argument("--json", action="store_true", help="Print JSON")
    a = ap.parse_args()

    headers = read_headers(a.model)

    if a.enumerate:
        rows = []
        for combo in itertools.product(a.bits_choices, repeat=3):
            m = {QWEN35_TYPES[t]: b for t, b in zip(("deltanet", "attention", "ffn"), combo)}
            r = size_report(headers, make_args(a.w_bits, a.w_groupsize, json.dumps(m)), a.scale_bits, a.asym)
            rows.append({"deltanet": combo[0], "attention": combo[1], "ffn": combo[2], "bits_map": m,
                         "avg_bpw": r["avg_bpw_quantized"], "text_model_gb": r["text_model_gb"],
                         "within_budget": a.budget is None or r["avg_bpw_quantized"] <= a.budget + 1e-9})
        rows.sort(key=lambda r: r["avg_bpw"])
        if a.json:
            print(json.dumps(rows, indent=2))
        else:
            print(f"groupsize {a.w_groupsize}" + (f", budget {a.budget} bpw" if a.budget else ""))
            print(f"{'DeltaNet':>8} {'Attn':>5} {'FFN':>4} {'bpw':>7} {'GB':>7}  fits")
            for r in rows:
                print(f"{r['deltanet']:>8} {r['attention']:>5} {r['ffn']:>4} {r['avg_bpw']:7.3f} "
                      f"{r['text_model_gb']:7.3f}  {'yes' if r['within_budget'] else 'no'}")
        return

    if a.match is not None:
        rows = []
        for bits in (2, 3, 4, 5, 6, 8):
            for gs in (-1, 256, 128, 64, 32, 16):
                r = size_report(headers, make_args(bits, gs, None), a.scale_bits, a.asym)
                rows.append((abs(r["avg_bpw_quantized"] - a.match), bits, gs, r["avg_bpw_quantized"], r["text_model_gb"]))
        rows.sort()
        print(f"Uniform settings closest to {a.match} bpw:")
        for _, bits, gs, bpw, gb in rows[:8]:
            print(f"  {bits}-bit, groupsize {gs:>4}: {bpw:.3f} bpw, {gb:.3f} GB")
        return

    r = size_report(headers, make_args(a.w_bits, a.w_groupsize, a.w_bits_map), a.scale_bits, a.asym)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    print(f"Quantized Linear params: {r['quantized_params'] / 1e9:.3f} B, "
          f"average {r['avg_bpw_quantized']:.3f} bits/weight")
    for t, v in r["per_type"].items():
        print(f"  {t:<10} {v['params'] / 1e9:7.3f} B params ({100 * v['share']:5.1f}%), {v['bpw']:.3f} bpw")
    print(f"Text model: {r['text_model_gb']:.3f} GB (BF16: {r['bf16_text_model_gb']:.3f} GB; "
          f"unquantized parts {r['unquantized_gb']:.3f} GB)")
    if r["excluded_gb"]:
        print("Excluded: " + ", ".join(f"{k} {v:.3f} GB" for k, v in r["excluded_gb"].items()))


if __name__ == "__main__":
    main()
