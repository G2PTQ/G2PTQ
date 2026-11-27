"""Collect every run into one CSV: settings, size, KL/PPL, IFEval, time and memory.

Reads <output_dir>/<model_name>/<exp>/metrics.json (written by ptq.py) and, if present,
<exp>/ifeval/summary.json (written by ours/eval/ifeval_vllm.py). Bits per weight are computed
from the base model's safetensors headers.

    python ours/eval/collect_results.py --model ./modelzoo/Qwen3.5/Qwen3.5-4B --out results.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "ours", "size"))
sys.path.insert(0, REPO)
from bpw import read_headers, size_report, make_args  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base (BF16) model directory, for size accounting")
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    headers = read_headers(a.model)
    model_name = os.path.basename(a.model.rstrip("/"))
    rows = []
    for path in sorted(glob.glob(os.path.join(a.output_dir, model_name, "*", "metrics.json"))):
        exp_dir = os.path.dirname(path)
        m = json.load(open(path))
        bmap = json.dumps(m["w_bits_map_json"]) if m.get("w_bits_map_json") else None
        size = size_report(headers, make_args(m["w_bits"], m["w_groupsize"], bmap))
        row = {
            "exp": m["exp"], "method": m["w_method"], "bits": m["w_bits"], "groupsize": m["w_groupsize"],
            "bits_map": m["w_bits_map"], "calib": m["dataset"], "nsamples": m["nsamples"],
            "token_weights": os.path.basename(m["token_weights"]) if m.get("token_weights") else "",
            "avg_bpw": round(size["avg_bpw_quantized"], 4), "text_model_gb": round(size["text_model_gb"], 4),
            **{k: v for k, v in m["metrics"].items()},
            "quant_hours": round(m.get("quant_time_hours", float("nan")), 3),
            "peak_gpu_gb": round(m.get("peak_gpu_mem_gb", float("nan")), 2),
        }
        s_path = os.path.join(exp_dir, "ifeval", "summary.json")
        if os.path.isfile(s_path):
            s = json.load(open(s_path))
            row.update({"ifeval_prompt_strict": round(100 * s["prompt_level_strict_acc"], 2),
                        "ifeval_inst_strict": round(100 * s["inst_level_strict_acc"], 2),
                        "ifeval_prompt_loose": round(100 * s["prompt_level_loose_acc"], 2),
                        "ifeval_hit_max_tokens": round(100 * s["frac_hit_max_tokens"], 2)})
        rows.append(row)
    if not rows:
        raise SystemExit(f"No metrics.json under {os.path.join(a.output_dir, model_name)}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} runs to {a.out}")


if __name__ == "__main__":
    main()
