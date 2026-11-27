"""Instruction-following calibration sets built by ``ours/data/build_calib.py``.

Each set is a JSONL file under ``./datasets/calib/`` with one conversation per line:
``{"id": ..., "source": ..., "messages": [{"role": "user", "content": ...}, ...]}``.

Registered names (use with ``--dataset`` / ``--eval_datasets``):

- ``calib_ifchat``      Tulu 3 Persona IF + Conifer conversations (decontaminated vs IFEval/IFBench)
- ``calib_selfgen``     the same prompts, answered by the BF16 model itself
- ``calib_heldout_if``  held-out instruction-following conversations for selection (KL/PPL);
                        never used for calibration

Conversations are rendered with the model's chat template (thinking disabled) and packed
whole, separated by EOS, into fixed-length sequences, so every calibration sequence starts at
a conversation boundary. Packing is deterministic for a given seed.
"""

import json
import logging
import os
import random

import torch
from tqdm import tqdm

from utils.datasets.registry import register_dataset

CALIB_DIR = os.environ.get("G2PTQ_CALIB_DIR", "./datasets/calib")


def render_conversation(tokenizer, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_conversations(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Build it with: python ours/data/build_calib.py (see ours/README.md)"
        )
    with open(path) as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def _texts(name, tokenizer, split):
    assert split in ("train", "test"), f"Unknown split {split} for {name}"
    convs = load_conversations(os.path.join(CALIB_DIR, f"{name}.jsonl"))
    return [render_conversation(tokenizer, m) for m in convs]


def sample_packed_conversations(texts, tokenizer, seq_len, num_samples, seed=None):
    """Pack whole conversations (EOS-separated) into ``num_samples`` sequences of ``seq_len`` tokens.

    Conversations are shuffled with ``seed`` and used at most once. Each sequence starts at a
    conversation boundary; the last conversation in a sequence is truncated to fit.
    """
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    samples, current, it = [], [], iter(order)
    pbar = tqdm(total=num_samples, desc=f"Packing {num_samples} x {seq_len} tokens")
    while len(samples) < num_samples:
        try:
            idx = next(it)
        except StopIteration:
            raise ValueError(
                f"Not enough conversations ({len(texts)}) to fill {num_samples} x {seq_len} tokens; "
                "build a larger pool with ours/data/build_calib.py --pool_size"
            )
        ids = tokenizer(texts[idx], add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids = ids + [eos]
        current.extend(ids)
        if len(current) >= seq_len:
            samples.append(torch.tensor(current[:seq_len], dtype=torch.long))
            current = []
            pbar.update(1)
    pbar.close()
    logging.info(f"Packed {num_samples} sequences of {seq_len} tokens")
    return samples


@register_dataset("calib_ifchat", sampler=sample_packed_conversations)
def get_calib_ifchat(tokenizer, split):
    return _texts("calib_ifchat", tokenizer, split)


@register_dataset("calib_selfgen", sampler=sample_packed_conversations)
def get_calib_selfgen(tokenizer, split):
    return _texts("calib_selfgen", tokenizer, split)


@register_dataset("calib_heldout_if", sampler=sample_packed_conversations)
def get_calib_heldout_if(tokenizer, split):
    return _texts("calib_heldout_if", tokenizer, split)
