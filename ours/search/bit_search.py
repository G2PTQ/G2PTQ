"""Choose per-layer-type bit widths under a size budget (Idea 1).

Step 1, sensitivity: quantize one layer type at a time (DeltaNet, attention, FFN) to the lowest
bit width, leaving everything else in BF16, and measure KL against the BF16 model on held-out data.
Step 2, search: quantize every per-type assignment from --bits_choices that fits --budget and
keep the one with the lowest held-out KL.
Step 3 (optional), confirm: rerun the best --confirm_top candidates with --confirm_method (GPTQ)
to check that the ranking holds beyond round-to-nearest.

Writes <out_dir>/search_results.json and <out_dir>/best_bits_map.json (pass the latter to
ptq.py --w_bits_map). Finished runs are reused, so the script can be re-run after interruption.

Example (GPU):
    python ours/search/bit_search.py --model ./modelzoo/Qwen3.5/Qwen3.5-4B --budget 3.5 \
        --w_groupsize 128 --dataset calib_ifchat --eval_datasets calib_heldout_if wikitext2 \
        --confirm_top 2
"""

import argparse
import itertools
import json
import math
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "ours", "size"))
from bpw import read_headers, size_report, make_args  # noqa: E402
from gptq_utils.bits_map import QWEN35_TYPES  # noqa: E402

TYPES = ("deltanet", "attention", "ffn")


def tag_of(bits_map_types, method, gs):
    parts = [f"{t[:2]}{bits_map_types[t]}" for t in TYPES if t in bits_map_types]
    return f"search_{method}_g{gs}_" + "_".join(parts)


def to_bits_map(types_to_bits):
    return {QWEN35_TYPES[t]: b for t, b in types_to_bits.items()}


def run_ptq(a, exp, w_bits, bits_map, method):
    metrics_path = os.path.join(a.output_dir, os.path.basename(a.model.rstrip("/")), exp, "metrics.json")
    if os.path.isfile(metrics_path) and not a.rerun:
        return json.load(open(metrics_path))
    cmd = shlex.split(a.launcher) + [
        "--model", a.model, "--exp", exp, "--w_method", method,
        "--w_bits", str(w_bits), "--w_groupsize", str(a.w_groupsize),
        "--w_bits_map", json.dumps(bits_map),
        "--dataset", a.dataset, "--nsamples", str(a.nsamples), "--seq_len", str(a.seq_len),
        "--eval_datasets", *a.eval_datasets,
        "--output_dir", a.output_dir, "--cache_dir", a.cache_dir, "--seed", str(a.seed),
    ] + shlex.split(a.extra_args)
    if method != "rtn":
        cmd += shlex.split(a.gptq_args)
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if a.dry_run:
        return None
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.load(open(metrics_path))


def score(metrics, select_on):
    """Selection score: mean log-KL over the selection datasets (lower is better)."""
    vals = [metrics["metrics"][f"KL-{d}"] for d in select_on]
    return sum(math.log(max(v, 1e-12)) for v in vals) / len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=float, required=True, help="Max average bits per quantized weight")
    ap.add_argument("--bits_choices", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--w_groupsize", type=int, default=128)
    ap.add_argument("--method", default="rtn", choices=["rtn", "gptq"])
    ap.add_argument("--dataset", default="calib_ifchat")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--eval_datasets", nargs="+", default=["calib_heldout_if", "wikitext2"])
    ap.add_argument("--select_on", nargs="+", default=None, help="Datasets used to rank (default: all eval)")
    ap.add_argument("--confirm_top", type=int, default=0)
    ap.add_argument("--confirm_method", default="gptq", choices=["gptq", "g2ptq"])
    ap.add_argument("--skip_sensitivity", action="store_true")
    ap.add_argument("--out_dir", default="./outputs/bit_search")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--launcher", default="python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 "
                                          "--rdzv_endpoint=localhost:29421 ./ptq.py")
    ap.add_argument("--extra_args", default="--deterministic --w_clip --bsz 4 --layer_bsz 64",
                    help="Extra ptq.py args for every run")
    ap.add_argument("--gptq_args", default="--act_order --gptq_backend triton --num_groups 4",
                    help="Extra ptq.py args for GPTQ/G2PTQ runs")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="Ignore finished runs")
    a = ap.parse_args()
    select_on = a.select_on or a.eval_datasets
    os.makedirs(a.out_dir, exist_ok=True)
    headers = read_headers(a.model)
    lo = min(a.bits_choices)
    results = {"settings": vars(a), "sensitivity": {}, "candidates": [], "confirm": []}

    if not a.skip_sensitivity:
        print(f"== Sensitivity: one type at {lo} bits, the rest BF16")
        for t in TYPES:
            m = run_ptq(a, tag_of({t: lo}, a.method, a.w_groupsize), 16, to_bits_map({t: lo}), a.method)
            if m:
                results["sensitivity"][t] = {"metrics": m["metrics"], "score": score(m, select_on)}

    print(f"== Search: every assignment from {a.bits_choices} within {a.budget} bpw")
    for combo in itertools.product(a.bits_choices, repeat=len(TYPES)):
        types_to_bits = dict(zip(TYPES, combo))
        bmap = to_bits_map(types_to_bits)
        bpw = size_report(headers, make_args(16, a.w_groupsize, json.dumps(bmap)))["avg_bpw_quantized"]
        if bpw > a.budget + 1e-9:
            continue
        m = run_ptq(a, tag_of(types_to_bits, a.method, a.w_groupsize), max(combo), bmap, a.method)
        results["candidates"].append({"bits": types_to_bits, "bits_map": bmap, "bpw": bpw,
                                      **({"metrics": m["metrics"], "score": score(m, select_on)} if m else {})})
    if a.dry_run:
        return
    if not results["candidates"]:
        sys.exit(f"No assignment from {a.bits_choices} fits {a.budget} bpw at groupsize {a.w_groupsize}")
    results["candidates"].sort(key=lambda c: c["score"])

    for c in results["candidates"][: a.confirm_top]:
        m = run_ptq(a, tag_of(c["bits"], a.confirm_method, a.w_groupsize), max(c["bits"].values()),
                    c["bits_map"], a.confirm_method)
        results["confirm"].append({"bits": c["bits"], "metrics": m["metrics"], "score": score(m, select_on)})
    results["confirm"].sort(key=lambda c: c["score"])

    best = (results["confirm"] or results["candidates"])[0]
    best_map = to_bits_map(best["bits"])
    results["best"] = {"bits": best["bits"], "bits_map": best_map}
    with open(os.path.join(a.out_dir, "search_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(a.out_dir, "best_bits_map.json"), "w") as f:
        json.dump(best_map, f, indent=2)

    print("\nSensitivity (KL with only this type quantized):")
    for t, r in results["sensitivity"].items():
        print(f"  {t:<10} " + ", ".join(f"{k}={v:.3g}" for k, v in r["metrics"].items() if k.startswith("KL")))
    print("\nCandidates (best first):")
    for c in results["candidates"]:
        print(f"  {c['bits']}  {c['bpw']:.3f} bpw  " +
              ", ".join(f"{k}={v:.3g}" for k, v in c["metrics"].items() if k.startswith("KL")))
    if results["confirm"]:
        print(f"\nConfirmed with {a.confirm_method}:")
        for c in results["confirm"]:
            print(f"  {c['bits']}  " + ", ".join(f"{k}={v:.3g}" for k, v in c["metrics"].items() if k.startswith("KL")))
    print(f"\nBest: {best['bits']} -> {os.path.join(a.out_dir, 'best_bits_map.json')}")


if __name__ == "__main__":
    main()
