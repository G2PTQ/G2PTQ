"""Public dataset entry points: turn a registered dataset into calibration tokens / eval loaders."""

import os
import random
import logging

import torch
import numpy as np

from utils.datasets.registry import get_dataset, get_dataset_spec
from utils import dist_utils


def get_tokens(dataset_name, split, tokenizer, seq_len, num_samples, save_path=None, seed=0):
    if save_path is not None and os.path.isfile(save_path):
        logging.info(f"Loading tokens from {save_path}")
        return torch.load(save_path)

    logging.info(f"Fetching dataset: {dataset_name}")
    dataset_spec = get_dataset_spec(dataset_name)
    texts = dataset_spec.loader(tokenizer, split)
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")

    tokens = dataset_spec.sampler(texts, tokenizer, seq_len, num_samples, seed)

    if save_path is not None and dist_utils.is_main():
        logging.info(f"Saving tokens to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tokens, save_path)

    return tokens


def get_loaders(dataset_name, split, tokenizer, seq_len, num_samples, seed=0):
    logging.info(f"Fetching dataset: {dataset_name}")
    texts = get_dataset(dataset_name, tokenizer, split)
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")

    enc = tokenizer("\n\n".join(texts), return_tensors='pt')
    assert split in ["train", "test"]
    if split == "train":
        np.random.seed(seed)
        random.seed(seed)
        trainloader = []
        for _ in range(num_samples):
            i = random.randint(0, enc.input_ids.shape[1] - seq_len - 1)
            j = i + seq_len
            inp = enc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    elif split == "test":
        return enc
