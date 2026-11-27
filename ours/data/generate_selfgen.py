"""Self-generated calibration set: the BF16 model answers the calib_ifchat prompts itself.

Reads datasets/calib/selfgen_prompts.jsonl (from build_calib.py), generates one reply per prompt
with vLLM (thinking disabled, greedy by default), and writes datasets/calib/calib_selfgen.jsonl.

Usage (GPU):
    python ours/data/generate_selfgen.py --model ./modelzoo/Qwen3.5/Qwen3.5-4B --num_prompts 1500
"""

import argparse
import json
import os
import re

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text):
    text = THINK_RE.sub("", text)
    if "</think>" in text:  # unterminated block: keep what follows the last close tag
        text = text.split("</think>")[-1]
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib_dir", default="./datasets/calib")
    ap.add_argument("--num_prompts", type=int, default=1500,
                    help="How many prompts to answer (128 x 2048 tokens needs roughly 500-1000)")
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    with open(os.path.join(args.calib_dir, "selfgen_prompts.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()][: args.num_prompts]

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=0)
    outputs = llm.chat([r["messages"] for r in rows], params,
                       chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)

    kept, truncated, empty = [], 0, 0
    for r, out in zip(rows, outputs):
        o = out.outputs[0]
        reply = strip_thinking(o.text)
        if not reply:
            empty += 1
            continue
        truncated += o.finish_reason == "length"
        kept.append({"id": r["id"], "source": f"selfgen:{r['source']}",
                     "messages": r["messages"] + [{"role": "assistant", "content": reply}]})

    path = os.path.join(args.calib_dir, "calib_selfgen.jsonl")
    with open(path, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} conversations to {path} (dropped {empty} empty, {truncated} hit max_tokens)")


if __name__ == "__main__":
    main()
