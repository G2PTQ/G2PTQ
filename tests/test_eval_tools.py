"""Offline tests for the IFEval scoring wrapper, the paired bootstrap and the results collector."""
import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ours", "eval"))
import ifeval_vllm  # noqa: E402

DOCS = [
    {"key": 1, "prompt": "Describe cats. Do not use any commas.",
     "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
    {"key": 2, "prompt": "Describe dogs in all capital letters.",
     "instruction_id_list": ["change_case:english_capital"], "kwargs": [{}]},
    {"key": 3, "prompt": "List three fruits as exactly 3 markdown bullet points. No commas.",
     "instruction_id_list": ["detectable_format:number_bullet_lists", "punctuation:no_comma"],
     "kwargs": [{"num_bullets": 3}, {}]},
]


def test_strip_thinking():
    assert ifeval_vllm.strip_thinking("<think>\nhmm, ok\n</think>\n\nHELLO") == "HELLO"
    assert ifeval_vllm.strip_thinking("partial reasoning</think> answer") == "answer"
    assert ifeval_vllm.strip_thinking("plain") == "plain"


def test_scoring_and_summary():
    responses = ["Cats are soft and quiet.", "dogs are loyal", "* apple\n* pear\n* plum"]
    rows = []
    for d, r in zip(DOCS, responses):
        rows.append({"key": d["key"], **ifeval_vllm.score_ifeval(d, r)})
    assert rows[0]["prompt_strict"] is True
    assert rows[1]["prompt_strict"] is False  # not capitalised
    assert rows[2]["prompt_strict"] is True and rows[2]["inst_strict"] == [True, True]
    s = ifeval_vllm.summarize(rows)
    assert abs(s["prompt_level_strict_acc"] - 2 / 3) < 1e-9
    assert abs(s["inst_level_strict_acc"] - 3 / 4) < 1e-9


def _write(path, vals):
    with open(path, "w") as f:
        for k, v in enumerate(vals):
            f.write(json.dumps({"key": k, "prompt_strict": v, "prompt_loose": v}) + "\n")


def test_bootstrap_detects_gain_and_null():
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as d:
        base = [rng.random() < 0.7 for _ in range(541)]
        better = [b or (rng.random() < 0.15) for b in base]  # strictly better on ~4-5% of prompts
        same = list(base)
        _write(f"{d}/a.jsonl", base); _write(f"{d}/b.jsonl", better); _write(f"{d}/c.jsonl", same)
        out = f"{d}/r.json"
        cmd = [sys.executable, os.path.join(ROOT, "ours/eval/bootstrap.py"), "--n_boot", "2000", "--json", out,
               "--pairs", f"{d}/a.jsonl:{d}/b.jsonl", f"{d}/a.jsonl:{d}/c.jsonl"]
        subprocess.run(cmd, check=True, capture_output=True)
        r = json.load(open(out))
        gain, null = r["pairs"]
        assert gain["diff_pp"] > 2 and gain["ci95_pp"][0] > 0 and gain["p_B_not_better"] < 0.01
        assert null["diff_pp"] == 0 and null["A_only_correct"] == 0
        assert r["pooled"]["pairs_with_gain"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
