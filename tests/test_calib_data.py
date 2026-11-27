"""Offline tests for the calibration-data builder and packing sampler.

Run: python tests/test_calib_data.py [path/to/tokenizer]   (tokenizer test is skipped without one)
"""
import json
import os
import sys
import tempfile
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ours", "data"))

import build_calib  # noqa: E402

IFEVAL = [{"prompt": "Write a 300+ word summary of the wikipedia page about the history of the "
                     "printing press. Do not use any commas and highlight at least 3 sections."}]
IFBENCH = [{"prompt": "Describe a rainy day in Paris and include keyword kaleidoscope once in your response."}]


def _tulu(n):
    rows = []
    for i in range(n):
        rows.append({"id": f"t{i}", "messages": [
            {"role": "user", "content": f"Persona task number {i}: explain topic {i} in exactly {i % 5 + 2} sentences."},
            {"role": "assistant", "content": f"Answer {i}. " * 5}]})
    # one leaked benchmark prompt and one duplicate
    rows.append({"id": "leak", "messages": [
        {"role": "user", "content": "Please write a 300+ word summary of the wikipedia page about the history of "
                                    "the printing press, thanks"},
        {"role": "assistant", "content": "..."}]})
    rows.append(dict(rows[0], id="dup"))
    return rows


def _conifer(n):
    return [{"conversations": [{"from": "human", "value": f"Conifer question {i} with constraint {i}."},
                               {"from": "gpt", "value": f"Conifer answer {i}."},
                               {"from": "human", "value": "Now make it shorter."},
                               {"from": "gpt", "value": "Short."}]} for i in range(n)]


class FakeDS(list):
    pass


def fake_load_dataset(name, split=None, **kw):
    return FakeDS({"google/IFEval": IFEVAL, "allenai/IFBench_test": IFBENCH,
                   build_calib.TULU_ID: _tulu(60), build_calib.CONIFER_ID: _conifer(30)}[name])


def test_builder_decontaminates_and_splits():
    with tempfile.TemporaryDirectory() as d, \
            mock.patch.dict(sys.modules, {"datasets": types.SimpleNamespace(load_dataset=fake_load_dataset)}):
        argv = ["build_calib.py", "--out_dir", d, "--pool_size", "40", "--heldout_size", "10",
                "--conifer_frac", "0.2", "--ngram", "8"]
        with mock.patch.object(sys, "argv", argv):
            build_calib.main()
        read = lambda f: [json.loads(l) for l in open(os.path.join(d, f))]
        pool, held, prompts = read("calib_ifchat.jsonl"), read("calib_heldout_if.jsonl"), read("selfgen_prompts.jsonl")
        report = json.load(open(os.path.join(d, "build_report.json")))
        assert len(pool) == 40 and len(held) == 10 and len(prompts) == 40
        ids = {r["id"] for r in pool + held}
        assert len(ids) == 50, "pool and held-out must not overlap"
        assert not any("leak" in i for i in ids), "benchmark overlap not removed"
        assert report["removed"].get("overlap_tulu_if") == 1 and report["removed"].get("duplicate") == 1
        assert sum(r["source"] == "conifer" for r in pool + held) == 10
        for p in prompts:
            assert p["messages"][-1]["role"] == "user" and all(m["role"] != "assistant" for m in p["messages"])
        conifer = [r for r in pool + held if r["source"] == "conifer"]
        assert conifer and len(conifer[0]["messages"]) == 4  # multi-turn kept for calibration


def test_ngram_overlap_detection():
    idx = build_calib.build_index(["one two three four five six seven eight nine"], 8)
    assert build_calib.is_contaminated([{"role": "user", "content": "x One two three four five six seven eight!"}], idx, 8)
    assert not build_calib.is_contaminated([{"role": "user", "content": "one two three four five six seven"}], idx, 8)


def test_packing(tokenizer_path):
    from transformers import AutoTokenizer
    sys.modules.setdefault("fast_hadamard_transform", types.ModuleType("fast_hadamard_transform"))  # CUDA-only ext
    from utils import dist_utils  # noqa: F401  (repo import order)
    from utils.datasets.calib_chat import sample_packed_conversations, render_conversation
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    convs = [[{"role": "user", "content": f"question {i} " * (i % 7 + 1)},
              {"role": "assistant", "content": f"answer {i} " * (i % 5 + 1)}] for i in range(200)]
    texts = [render_conversation(tok, m) for m in convs]
    a = sample_packed_conversations(texts, tok, 64, 8, seed=1)
    b = sample_packed_conversations(texts, tok, 64, 8, seed=1)
    c = sample_packed_conversations(texts, tok, 64, 8, seed=2)
    assert len(a) == 8 and all(t.shape == (64,) for t in a)
    assert all((x == y).all() for x, y in zip(a, b)), "packing must be deterministic"
    assert any((x != y).any() for x, y in zip(a, c))
    start = tok.convert_tokens_to_ids("<|im_start|>")
    assert all(int(t[0]) == start for t in a), "every sequence starts at a conversation boundary"
    try:
        sample_packed_conversations(texts[:3], tok, 64, 8, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected an error when the pool is too small")


if __name__ == "__main__":
    test_ngram_overlap_detection(); print("PASS test_ngram_overlap_detection")
    test_builder_decontaminates_and_splits(); print("PASS test_builder_decontaminates_and_splits")
    if len(sys.argv) > 1:
        test_packing(sys.argv[1]); print("PASS test_packing")
