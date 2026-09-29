"""Dataset registry.

Each dataset lives in its own module under ``utils/datasets/`` and registers a loader via the
``@register_dataset("<name>")`` decorator. A loader has the uniform signature
``(tokenizer, split) -> list[str]`` and returns the raw texts for that split; sampling and
tokenization are handled downstream in ``loaders.py``. Registry entries also specify the sampler
used to turn those texts into calibration tokens.
"""

from dataclasses import dataclass
from typing import Callable

from utils.datasets.common import sample_concat_and_tokenize


@dataclass(frozen=True)
class DatasetSpec:
    """Registered dataset loader and its calibration-token sampler."""

    loader: Callable
    sampler: Callable


_REGISTRY = {}


def register_dataset(name, sampler=sample_concat_and_tokenize):
    """Register a dataset loader and its calibration-token sampler under ``name``.

    ``sample_concat_and_tokenize`` is used unless a dataset explicitly supplies another sampler.
    """

    def deco(fn):
        if name in _REGISTRY:
            raise ValueError(f"Dataset '{name}' already registered")
        _REGISTRY[name] = DatasetSpec(loader=fn, sampler=sampler)
        return fn

    return deco


def get_dataset_spec(dataset_name):
    """Return the registered loader and sampler for ``dataset_name``."""
    if dataset_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {get_dataset_names()}")
    return _REGISTRY[dataset_name]


def get_dataset(dataset_name, tokenizer, split):
    """Return the raw texts for ``dataset_name``/``split`` via the registered loader."""
    return get_dataset_spec(dataset_name).loader(tokenizer, split)


def get_dataset_names():
    """Return the sorted list of registered dataset names."""
    return sorted(_REGISTRY)
