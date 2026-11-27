"""IFEval (and IFBench generation) with vLLM, scored per prompt.

Generates one greedy reply per prompt with the model's chat template (thinking disabled by
default), strips any <think> block, and scores with lm-eval's IFEval checkers. Writes:

    <out_dir>/responses.jsonl   one line per prompt: key, prompt, response, finish_reason,
                                prompt/inst-level strict & loose results
    <out_dir>/summary.json      prompt_level_strict_acc etc. (these are what you report)

Works on a BF16 model or on a quantized model saved with ptq.py --save_dequant_model.

Usage (GPU):
    python ours/eval/ifeval_vllm.py --model outputs/Qwen3.5-4B/<exp>/dequant_model \
        --out_dir outputs/Qwen3.5-4B/<exp>/ifeval
    # re-score saved responses without a GPU
    python ours/eval/ifeval_vllm.py --rescore outputs/.../ifeval/responses.jsonl --out_dir ...
    # IFBench: generate only (score with the official IFBench repo's run_eval on <out_dir>/ifbench_input.jsonl)
    python ours/eval/ifeval_vllm.py --benchmark ifbench --model ... --out_dir ...
"""

import argparse
import json
import os
import re
import time

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
BENCHMARKS = {"ifeval": ("google/IFEval", "train"), "ifbench": ("allenai/IFBench_test", "train")}


def strip_thinking(text):
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def load_docs(benchmark, limit=None):
    from datasets import load_dataset
    name, split = BENCHMARKS[benchmark]
    docs = [dict(d) for d in load_dataset(name, split=split)]
    for i, d in enumerate(docs):
        d.setdefault("key", d.get("key", i))
        if "prompt" not in d and "messages" in d:
            d["prompt"] = next(m["content"] for m in d["messages"] if m["role"] == "user")
    return docs[:limit] if limit else docs


def score_ifeval(doc, response):
    from lm_eval.tasks.ifeval.utils import process_results
    r = process_results(doc, [response])
    return {
        "prompt_strict": bool(r["prompt_level_strict_acc"]),
        "prompt_loose": bool(r["prompt_level_loose_acc"]),
        "inst_strict": [bool(x) for x in r["inst_level_strict_acc"]],
        "inst_loose": [bool(x) for x in r["inst_level_loose_acc"]],
    }


def summarize(rows):
    n = len(rows)
    inst_s = [x for r in rows for x in r["inst_strict"]]
    inst_l = [x for r in rows for x in r["inst_loose"]]
    return {
        "n_prompts": n,
        "prompt_level_strict_acc": sum(r["prompt_strict"] for r in rows) / n,
        "inst_level_strict_acc": sum(inst_s) / len(inst_s),
        "prompt_level_loose_acc": sum(r["prompt_loose"] for r in rows) / n,
        "inst_level_loose_acc": sum(inst_l) / len(inst_l),
        "frac_hit_max_tokens": sum(r.get("finish_reason") == "length" for r in rows) / n,
    }


def generate(args, docs):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp, seed=0,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              **({"tokenizer": args.tokenizer} if args.tokenizer else {}))
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0)
    convs = [[{"role": "user", "content": d["prompt"]}] for d in docs]
    kwargs = {"chat_template_kwargs": {"enable_thinking": args.thinking}}
    t0 = time.time()
    outs = llm.chat(convs, params, use_tqdm=True, **kwargs)
    elapsed = time.time() - t0
    return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs], elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="Model directory or Hub id (BF16 or dequantized)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), default="ifeval")
    ap.add_argument("--max_tokens", type=int, default=1280, help="lm-eval's IFEval default")
    ap.add_argument("--thinking", action="store_true", help="Enable thinking mode (default: off)")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="Only the first N prompts (debugging)")
    ap.add_argument("--rescore", default=None, help="Score an existing responses.jsonl instead of generating")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.rescore:
        rows = [json.loads(l) for l in open(args.rescore) if l.strip()]
        docs = load_docs("ifeval")
        by_key = {d["key"]: d for d in docs}
        for r in rows:
            r.update(score_ifeval(by_key[r["key"]], r["response"]))
        meta = {"rescored_from": args.rescore}
    else:
        if not args.model:
            ap.error("--model is required unless --rescore is given")
        docs = load_docs(args.benchmark, args.limit)
        gens, elapsed = generate(args, docs)
        rows = []
        for d, (text, finish) in zip(docs, gens):
            response = strip_thinking(text)  # graders see only the final answer
            row = {"key": d["key"], "prompt": d["prompt"], "response": response, "finish_reason": finish}
            if args.benchmark == "ifeval":
                row.update(score_ifeval(d, response))
            rows.append(row)
        meta = {"model": args.model, "benchmark": args.benchmark, "max_tokens": args.max_tokens,
                "thinking": args.thinking, "generation_seconds": elapsed}

    with open(os.path.join(args.out_dir, "responses.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.benchmark == "ifbench" and not args.rescore:
        with open(os.path.join(args.out_dir, "ifbench_input.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} IFBench responses; score them with the IFBench repo's run_eval.")
        return
    summary = {**summarize(rows), **meta}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
