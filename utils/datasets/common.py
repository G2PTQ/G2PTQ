"""Shared helpers for dataset loaders and calibration sampling."""

import random
import logging

import numpy as np
from tqdm import tqdm


def format_messages(messages: list[dict]) -> str:
    chunks = []
    system_done = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()

        assert role in ["system", "user", "assistant"]

        if role == "system" and not system_done:
            chunks.append(f"### Instruction:\n{content}")
            system_done = True
        elif role == "user":
            chunks.append(f"### Instruction:\n{content}")
        elif role == "assistant":
            chunks.append(f"### Response:\n{content}")

    text = "\n\n".join(chunks).strip()
    return text


def sample_concat_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    logging.info(f"Tokenizing {len(texts)} texts")
    trainenc = tokenizer("\n\n".join(texts), return_tensors='pt')
    samples = []
    pbar = tqdm(total=num_samples, desc=f"Sampling {num_samples} samples of length {seq_len}")
    while len(samples) < num_samples:
        idx = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)

        # if selected_indices:
        #     closest_idx = min(selected_indices, key=lambda x: abs(x - idx), default=idx)
        #     if idx <= closest_idx + seq_len and idx >= closest_idx - seq_len:
        #         continue

        j = idx + seq_len
        inp = trainenc.input_ids[:, idx:j]
        tokens = inp.clone()
        tokens = tokens.squeeze(0)

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples


def sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    logging.info(f"Tokenizing {len(texts)} texts")
    samples = []
    pbar = tqdm(total=num_samples, desc=f"Sampling {num_samples} samples of length {seq_len}")
    while len(samples) < num_samples:
        current_texts = []
        current_indices = []
        
        # Keep sampling texts until the combined encoded length exceeds seq_len
        while True:
            # Safety check to prevent infinite loops if the dataset is exhausted
            if len(selected_indices) + len(current_indices) >= len(texts):
                raise ValueError("Dataset exhausted: not enough texts left to reach seq_len.")
                
            idx = random.randint(0, len(texts) - 1)
            
            # We don't want to sample the same text twice (either globally or in the current sample)
            if idx in selected_indices or idx in current_indices:  
                continue
            
            current_indices.append(idx)
            current_texts.append(texts[idx])
            
            # Join the currently sampled texts and encode them
            combined_text = "\n\n".join(current_texts)
            tokens = tokenizer(combined_text, return_tensors='pt')['input_ids'][0]
            
            # If we have enough tokens, break out of the inner loop
            if len(tokens) >= seq_len:
                break

        # Truncate to exactly seq_len
        tokens = tokens[:seq_len]

        # Mark all texts used in this sequence as selected
        selected_indices.update(current_indices)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()

    return samples


def sample_and_tokenize_from_middle(texts, tokenizer, seq_len, num_samples, seed=None):
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"

    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()
    samples = []
    pbar = tqdm(total=num_samples, desc="Sampling and tokenizing")
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt')['input_ids'][0]
        if len(tokens) < seq_len:  # if the text is too short, we skip it
            continue

        seq_start = random.randint(0, len(tokens) - seq_len)

        tokens = tokens[seq_start:seq_start + seq_len]
        assert tokens.shape[-1] == seq_len, f"Token length {len(tokens)} != seq_len {seq_len}"

        selected_indices.add(idx)
        samples.append(tokens)
        pbar.update(1)
    pbar.close()
    return samples
