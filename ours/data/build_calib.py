"""Build the instruction-following calibration and held-out sets.

Downloads Tulu 3 Persona IF and Conifer from the Hugging Face Hub, removes anything that
overlaps the IFEval or IFBench test prompts (word n-gram overlap), and writes:

    datasets/calib/calib_ifchat.jsonl        calibration conversations (original responses)
    datasets/calib/calib_heldout_if.jsonl    held-out conversations for selection (KL/PPL)
    datasets/calib/selfgen_prompts.jsonl     the calib_ifchat user prompts, for generate_selfgen.py
    datasets/calib/build_report.json         counts, removals, settings

Usage (from the repo root, on a machine with Hub access):
    python ours/data/build_calib.py
    python ours/data/build_calib.py --conifer_frac 0 --pool_size 3000   # Tulu only
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

TULU_ID = "allenai/tulu-3-sft-personas-instruction-following"
CONIFER_ID = "ConiferLM/Conifer"
IFEVAL_ID = "google/IFEval"
IFBENCH_ID = "allenai/IFBench_test"

_WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return _WORD.findall(text.lower())


def ngrams(text, n):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def normalize_messages(example):
    """Return [{"role", "content"}, ...] from common chat schemas, or None."""
    if "messages" in example and example["messages"]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    elif "conversations" in example and example["conversations"]:
        role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                    "system": "system"}
        msgs = [{"role": role_map.get(m.get("from", m.get("role")), None),
                 "content": m.get("value", m.get("content"))} for m in example["conversations"]]
    else:
        return None
    msgs = [m for m in msgs if m["role"] in ("system", "user", "assistant") and isinstance(m["content"], str)]
    if not any(m["role"] == "user" for m in msgs) or msgs[-1]["role"] != "assistant":
        return None
    if any(not m["content"].strip() for m in msgs):
        return None
    return msgs


def benchmark_prompts(dataset_id, split="train"):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    prompts = []
    for ex in ds:
        if "prompt" in ex and ex["prompt"]:
            prompts.append(ex["prompt"])
        elif "messages" in ex:
            prompts.extend(m["content"] for m in ex["messages"] if m["role"] == "user")
    if not prompts:
        raise RuntimeError(f"No prompts found in {dataset_id}")
    return prompts


def build_index(prompts, n):
    index = set()
    for p in prompts:
        index |= ngrams(p, n)
    return index


def is_contaminated(msgs, index, n):
    text = " ".join(m["content"] for m in msgs if m["role"] in ("user", "system"))
    return not ngrams(text, n).isdisjoint(index)


def load_source(dataset_id, split, tag, limit, rng):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        msgs = normalize_messages(ds[i])
        if msgs is None:
            continue
        out.append({"id": f"{tag}-{ds[i].get('id', i)}", "source": tag, "messages": msgs})
        if len(out) >= limit:
            break
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="./datasets/calib")
    ap.add_argument("--pool_size", type=int, default=4000,
                    help="Conversations in calib_ifchat (128 x 2048 tokens needs roughly 400-1000)")
    ap.add_argument("--heldout_size", type=int, default=400)
    ap.add_argument("--conifer_frac", type=float, default=0.25, help="Share of the pool taken from Conifer")
    ap.add_argument("--ngram", type=int, default=8, help="Word n-gram length for decontamination")
    ap.add_argument("--conifer_split", default="train_sft")
    ap.add_argument("--skip_ifbench", action="store_true", help="Do not decontaminate against IFBench")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    bench = {"ifeval": benchmark_prompts(IFEVAL_ID)}
    if not args.skip_ifbench:
        bench["ifbench"] = benchmark_prompts(IFBENCH_ID)
    index = build_index([p for ps in bench.values() for p in ps], args.ngram)
    print(f"Decontamination index: {len(index)} {args.ngram}-grams from "
          + ", ".join(f"{k} ({len(v)} prompts)" for k, v in bench.items()))

    total = args.pool_size + args.heldout_size
    n_conifer = round(total * args.conifer_frac)
    n_tulu = total - n_conifer
    over = 1.3  # over-fetch to survive filtering
    rows = load_source(TULU_ID, "train", "tulu_if", int(n_tulu * over), rng)
    if n_conifer:
        try:
            rows += load_source(CONIFER_ID, args.conifer_split, "conifer", int(n_conifer * over), rng)
        except Exception as e:  # split names differ between dataset versions
            sys.exit(f"Could not load {CONIFER_ID} split {args.conifer_split!r}: {e}\n"
                     "Pass --conifer_split <name> (see the dataset card) or --conifer_frac 0.")

    removed = Counter()
    clean, seen = [], set()
    for r in rows:
        key = r["messages"][0]["content"].strip().lower()
        if key in seen:
            removed["duplicate"] += 1
            continue
        if is_contaminated(r["messages"], index, args.ngram):
            removed[f"overlap_{r['source']}"] += 1
            continue
        seen.add(key)
        clean.append(r)

    by_src = {"tulu_if": [r for r in clean if r["source"] == "tulu_if"],
              "conifer": [r for r in clean if r["source"] == "conifer"]}
    if len(by_src["tulu_if"]) < n_tulu or len(by_src["conifer"]) < n_conifer:
        sys.exit(f"Not enough clean examples: tulu {len(by_src['tulu_if'])}/{n_tulu}, "
                 f"conifer {len(by_src['conifer'])}/{n_conifer}. Lower --pool_size.")
    chosen = by_src["tulu_if"][:n_tulu] + by_src["conifer"][:n_conifer]
    rng.shuffle(chosen)
    heldout, pool = chosen[:args.heldout_size], chosen[args.heldout_size:]

    write_jsonl(os.path.join(args.out_dir, "calib_ifchat.jsonl"), pool)
    write_jsonl(os.path.join(args.out_dir, "calib_heldout_if.jsonl"), heldout)
    def first_turn(msgs):  # everything before the first assistant reply (system + first user turn)
        end = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        return msgs[:end]

    prompts = [{"id": r["id"], "source": r["source"], "messages": first_turn(r["messages"])} for r in pool]
    write_jsonl(os.path.join(args.out_dir, "selfgen_prompts.jsonl"), prompts)

    report = {
        "settings": vars(args),
        "sources": {"tulu_if": TULU_ID, "conifer": CONIFER_ID},
        "decontaminated_against": {k: len(v) for k, v in bench.items()},
        "removed": dict(removed),
        "calib_ifchat": dict(Counter(r["source"] for r in pool)),
        "calib_heldout_if": dict(Counter(r["source"] for r in heldout)),
    }
    with open(os.path.join(args.out_dir, "build_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
