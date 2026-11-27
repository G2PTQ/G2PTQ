"""Paired bootstrap for IFEval: is method B better than method A on the same prompts?

Each argument after --pairs is "A_responses.jsonl:B_responses.jsonl" (from ifeval_vllm.py).
With several pairs (e.g. one per calibration set) the per-prompt differences are averaged across
pairs and prompts are resampled jointly, so the pooled interval respects that all pairs share
the same 541 prompts.

Reports the accuracy difference (B - A, percentage points), its 95% bootstrap interval, a
one-sided bootstrap p-value for B <= A, and the discordant counts (prompts only one side got
right).

Example:
    python ours/eval/bootstrap.py --metric prompt_strict --pairs \
        outputs/Qwen3.5-4B/g2ptq_ifchat/ifeval/responses.jsonl:outputs/Qwen3.5-4B/g2ptq_tw_ifchat/ifeval/responses.jsonl
"""

import argparse
import json
import random


def load(path, metric):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r["key"]: float(r[metric]) for r in rows}


def bootstrap(diffs, n_boot, seed):
    """diffs: list over prompts of the pair-averaged (B - A) difference."""
    rng = random.Random(seed)
    n = len(diffs)
    stats = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    lo, hi = stats[int(0.025 * n_boot)], stats[int(0.975 * n_boot) - 1]
    p = sum(x <= 0 for x in stats) / n_boot
    return lo, hi, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True, help="A.jsonl:B.jsonl [A2.jsonl:B2.jsonl ...]")
    ap.add_argument("--metric", default="prompt_strict", choices=["prompt_strict", "prompt_loose"])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="Also write results to this file")
    a = ap.parse_args()

    per_pair, keys = [], None
    for spec in a.pairs:
        pa, pb = spec.split(":")
        A, B = load(pa, a.metric), load(pb, a.metric)
        common = set(A) & set(B)
        if len(common) != len(A) or len(common) != len(B):
            raise SystemExit(f"{spec}: prompt sets differ ({len(A)} vs {len(B)}, {len(common)} shared)")
        keys = common if keys is None else keys & common
        per_pair.append((spec, A, B))
    keys = sorted(keys)

    out = {"metric": a.metric, "n_prompts": len(keys), "pairs": []}
    for spec, A, B in per_pair:
        acc_a = sum(A[k] for k in keys) / len(keys)
        acc_b = sum(B[k] for k in keys) / len(keys)
        b_only = sum(B[k] > A[k] for k in keys)
        a_only = sum(A[k] > B[k] for k in keys)
        lo, hi, p = bootstrap([B[k] - A[k] for k in keys], a.n_boot, a.seed)
        out["pairs"].append({"pair": spec, "acc_A": acc_a, "acc_B": acc_b, "diff_pp": 100 * (acc_b - acc_a),
                             "ci95_pp": [100 * lo, 100 * hi], "p_B_not_better": p,
                             "B_only_correct": b_only, "A_only_correct": a_only})
    if len(per_pair) > 1:
        diffs = [sum(B[k] - A[k] for _, A, B in per_pair) / len(per_pair) for k in keys]
        lo, hi, p = bootstrap(diffs, a.n_boot, a.seed)
        out["pooled"] = {"diff_pp": 100 * sum(diffs) / len(diffs), "ci95_pp": [100 * lo, 100 * hi],
                         "p_B_not_better": p,
                         "pairs_with_gain": sum(r["diff_pp"] > 0 for r in out["pairs"])}

    for r in out["pairs"]:
        print(f"{r['pair']}\n  A {100 * r['acc_A']:.2f}%  B {100 * r['acc_B']:.2f}%  "
              f"diff {r['diff_pp']:+.2f} pp  95% CI [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]  "
              f"p(B<=A) {r['p_B_not_better']:.3f}  discordant B-only {r['B_only_correct']} / A-only {r['A_only_correct']}")
    if "pooled" in out:
        r = out["pooled"]
        print(f"Pooled over {len(per_pair)} pairs: diff {r['diff_pp']:+.2f} pp  95% CI "
              f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]  p(B<=A) {r['p_B_not_better']:.3f}  "
              f"gain in {r['pairs_with_gain']}/{len(per_pair)} pairs")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
