import contextlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Optional, Type

import torch
from compressed_tensors.offload import get_cache_init_kwargs, offload_module
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.conversion_mapping import (
    _MODEL_TO_CONVERSION_PATTERN,
    register_checkpoint_conversion_mapping,
)
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from llmcompressor.modeling.moe.conversion_mappings import (
    ARCH_TO_2D_MAPPINGS,
    get_linearize_load_mappings,
    has_linearize_load_mappings,
    set_save_conversion_mapping,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.utils.dev import skip_weights_initialize

__all__ = ["load_quantizable_moe"]

# Number of threads used to copy per-expert weights within one expert layer.
_COPY_WORKERS = min(32, (os.cpu_count() or 8))


def _get_3d_expert_targets(model_type: str) -> Optional[list]:
    """The 3D fused expert tensor suffixes for ``model_type`` (the ``remove_targets``
    of its 2D mapping), or None if the arch has no 2D load mappings."""
    remapped = _MODEL_TO_CONVERSION_PATTERN.get(model_type, model_type)
    if remapped not in ARCH_TO_2D_MAPPINGS:
        return None
    remove_targets, _new_mappings = ARCH_TO_2D_MAPPINGS[remapped]
    return list(remove_targets)


def _iter_checkpoint_keys(path: str):
    """Yield checkpoint tensor names for a *local* model dir without loading weights.

    Returns None (not a generator) when keys cannot be cheaply determined — remote
    repos, single ``.bin`` files with no index, or any read error — so callers can
    treat "unknown" distinctly from "read and found nothing".
    """
    if not isinstance(path, (str, os.PathLike)) or not os.path.isdir(path):
        return None

    # sharded checkpoints (safetensors or bin): read key list from the index json
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = os.path.join(path, index_name)
        if os.path.isfile(index_path):
            try:
                with open(index_path, "r") as f:
                    weight_map = json.load(f).get("weight_map", {})
                return list(weight_map.keys())
            except (OSError, ValueError):
                return None

    # single-file safetensors: header carries the keys, no weight load needed
    safetensors_path = os.path.join(path, "model.safetensors")
    if os.path.isfile(safetensors_path):
        try:
            from safetensors import safe_open

            with safe_open(safetensors_path, framework="pt") as f:
                return list(f.keys())
        except Exception:
            return None

    # single-file .bin (no index) is too expensive to probe cheaply -> unknown
    return None


def _checkpoint_has_3d_experts(path: str, model_type: str) -> Optional[bool]:
    """True if the checkpoint stores 3D fused expert tensors, False if it stores 2D
    per-expert weights, None if it could not be determined (keep upstream behavior)."""
    targets = _get_3d_expert_targets(model_type)
    if not targets:
        return None

    keys = _iter_checkpoint_keys(path)
    if keys is None:
        return None

    return any(target in key for key in keys for target in targets)


def _extract_model_path(args, kwargs):
    """The pretrained path passed to ``from_pretrained`` (first positional or kwarg)."""
    if args:
        return args[0]
    return kwargs.get("pretrained_model_name_or_path")


@contextlib.contextmanager
def _skip_linearized_moe_initialization(model_cls, config):
    """Skip the native fused-expert initializer for a linearized MoE load.
    """
    # ``load_model`` passes AutoModelForCausalLM, while the actual implementation
    # class is selected from its lazy config-to-model mapping inside from_pretrained.
    # Resolve that concrete class so its inherited ``_init_weights`` is intercepted.
    init_cls = model_cls
    mapping = getattr(model_cls, "_model_mapping", None)
    if mapping is not None:
        try:
            init_cls = mapping[type(config)]
        except (AttributeError, KeyError, TypeError, ValueError, ImportError):
            init_cls = model_cls

    # ``initialize_weights`` dynamically dispatches nested ``PreTrainedModel``
    # instances, so patch the class that actually defines the method rather than
    # only the outer class.
    init_owner = next(
        (base for base in getattr(init_cls, "__mro__", ()) if "_init_weights" in base.__dict__),
        init_cls,
    )
    original_init_weights = getattr(init_owner, "_init_weights", None)
    if original_init_weights is None:
        yield
        return

    @wraps(original_init_weights)
    def patched_init_weights(self, module, *args, **kwargs):
        if isinstance(module, LinearExperts2D):
            return
        return original_init_weights(self, module, *args, **kwargs)

    with patch_attr(init_owner, "_init_weights", patched_init_weights):
        yield


@contextlib.contextmanager
def load_quantizable_moe(model_cls: Type[PreTrainedModel] = AutoModelForCausalLM):
    """Drop-in replacement for ``llmcompressor.modeling.moe.linearize.load_quantizable_moe``.

    Identical to upstream except the load-path decision also consults the actual
    checkpoint: if the arch has 2D load mappings but the checkpoint stores 3D fused
    expert tensors, we fall back to post-load ``linearize_moe`` instead of the
    (failing) 2D linearized-load path.
    """
    original_from_pretrained = model_cls.from_pretrained
    patched_fn_called = False

    @classmethod
    @wraps(original_from_pretrained)
    def patched(cls, *args, **kwargs):
        nonlocal patched_fn_called
        patched_fn_called = True

        config = AutoConfig.from_pretrained(*args, **kwargs)
        model_type = config.model_type

        # model is 3d (or otherwise doesn't have mappings)
        # fall back to post-load conversion
        model_path = _extract_model_path(args, kwargs)
        is_3d_ckpt = _checkpoint_has_3d_experts(model_path, model_type)
        if not has_linearize_load_mappings(model_type) or is_3d_ckpt:
            if is_3d_ckpt:
                logger.info(
                    f"Checkpoint for `{model_type}` stores 3D fused experts despite "
                    "having 2D load mappings; falling back to post-load linearization."
                )
            model = original_from_pretrained(*args, **kwargs)
            # split expert weights with multi-thread
            if (
                getattr(LinearExperts2D.from_experts_module, "__func__", None)
                is not _patched_from_experts_module
            ):
                LinearExperts2D.from_experts_module = classmethod(_patched_from_experts_module)
            linearize_moe(model)
            return model

        # prepare to load linearized weights
        experts_cls, load_map, save_map = get_linearize_load_mappings(model_type)
        linear_experts_2d_cls = LinearExperts2D.get_linear_experts_cls(experts_cls)
        register_patch_mapping({experts_cls.__name__: linear_experts_2d_cls})
        register_checkpoint_conversion_mapping(model_type, load_map, overwrite=True)

        # load model
        with _skip_linearized_moe_initialization(model_cls, config):
            model: PreTrainedModel = original_from_pretrained(*args, **kwargs)

        # prepare for saving to be called later
        clear_patch_mapping()
        set_save_conversion_mapping(model, save_map)
        register_checkpoint_conversion_mapping(model_type, save_map, overwrite=True)

        return model

    with patch_attr(model_cls, "from_pretrained", patched):
        try:
            yield
        finally:
            if not patched_fn_called:
                logger.warning(
                    f"`{model_cls.__name__}.from_pretrained` was never called. If you "
                    f"are loading with a model class other than {model_cls.__name__}, "
                    "please pass as argument to `load_quantizable_moe`"
                )


@torch.no_grad()
def _patched_from_experts_module(cls, experts, config):
    """Threaded replacement for ``LinearExperts2D.from_experts_module``.
    """
    from utils import dist_utils

    with skip_weights_initialize():
        self = cls(config)

    num_experts = self.num_experts

    def _copy(index):
        with torch.no_grad():  # torch.no_grad is thread-local; re-enter in each worker
            self[index].copy_from_experts_module(experts, index)

    copy_workers = min(max(num_experts, 1), _COPY_WORKERS)
    with ThreadPoolExecutor(max_workers=copy_workers) as ex:
        list(ex.map(_copy, range(num_experts)))

    # copy offloading from original
    offload_kwargs = get_cache_init_kwargs(experts)
    modules = list(self.modules())
    if dist_utils.get_world_size() == 1:
        with ThreadPoolExecutor(max_workers=_COPY_WORKERS) as ex:
            list(ex.map(lambda module: offload_module(module, **offload_kwargs), modules))
    else:
        # `offload_module` issues collectives (broadcast/barrier) that must run in the
        # same order on every rank, so it can't be threaded
        for module in modules:
            offload_module(module, **offload_kwargs)

    return self
