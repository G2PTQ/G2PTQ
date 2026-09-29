"""`torch.compile` support for the quantization kernels.

Ported from AutoRound's `--enable_torch_compile`. Only the *pure*
tensor-in/tensor-out quant kernels in `utils/quant_utils.py` are compiled.

The flag is process-wide: the kernels are module-level and stateless, so a
single cache shared by every `WeightQuantizer`/`ActQuantizer` is the natural
fit. When disabled, `maybe_compile` hands back the original function, which
makes the flag-off path byte-identical to the uncompiled implementation.
"""

import logging
import os
from typing import Callable, Optional

import torch

# Minimum value to which torch._dynamo cache_size_limit /
# accumulated_cache_size_limit / recompile_limit are bumped. Dynamo keys its
# compile cache on the *code object*, which a single quant kernel shares across
# every linear in a block (q/k/v/o_proj, gate/up/down_proj, ...), each with a
# different weight shape. Per-shape static recompiles therefore blow past the
# default limit of 8 and silently fall back to eager with a noisy warning. We
# keep static-shape compilation (best perf) and just allow more cache entries.
DEFAULT_DYNAMO_CACHE_SIZE_LIMIT = 16

_ENABLED = False
_CACHE: dict = {}
_FAILED: set = set()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    return _ENABLED


def bump_dynamo_cache_limit(min_size: Optional[int] = None) -> None:
    """Raise the torch._dynamo cache/recompile limits to `min_size`.

    Overridable with `G2PTQ_DYNAMO_CACHE_SIZE_LIMIT`. Best-effort: the config
    attribute names have moved between torch versions.
    """
    if min_size is None:
        min_size = int(
            os.environ.get("G2PTQ_DYNAMO_CACHE_SIZE_LIMIT", DEFAULT_DYNAMO_CACHE_SIZE_LIMIT)
        )
    try:
        from torch._dynamo import config as dynamo_config

        for attr in ("cache_size_limit", "accumulated_cache_size_limit", "recompile_limit"):
            if hasattr(dynamo_config, attr) and getattr(dynamo_config, attr) < min_size:
                setattr(dynamo_config, attr, min_size)
    except Exception as e:  # pragma: no cover - best effort
        logging.warning(f"Could not raise torch._dynamo cache limits: {e}")


def configure(enabled: bool, cache_size_limit: Optional[int] = None) -> None:
    """Enable/disable kernel compilation for this process.
    """
    global _ENABLED

    enabled = bool(enabled) or _env_flag("G2PTQ_TORCH_COMPILE")
    if enabled and not hasattr(torch, "compile"):
        logging.warning(
            "torch.compile is unavailable in this torch build; running eager."
        )
        enabled = False

    _ENABLED = enabled
    _CACHE.clear()
    _FAILED.clear()
    if enabled:
        bump_dynamo_cache_limit(cache_size_limit)
        logging.info("torch.compile enabled for quantization kernels.")


def _guarded(fn: Callable, compiled: Callable, key) -> Callable:
    """Run `compiled`, falling back to `fn` for good if the *first* call fails.

    Compilation errors surface at call time, not at `torch.compile` time. Only
    the first call is guarded: after that a raised exception is a real error and
    must propagate instead of being retried. The kernels are pure, so re-running
    eagerly after a failed compile is side-effect free.
    """
    state = {"verified": False}

    def wrapper(*a, **kw):
        if key in _FAILED:
            return fn(*a, **kw)
        if state["verified"]:
            return compiled(*a, **kw)
        try:
            out = compiled(*a, **kw)
        except Exception as e:
            _FAILED.add(key)
            logging.warning(
                f"torch.compile failed for {getattr(fn, '__name__', fn)} ({e}); running eager.",
            )
            return fn(*a, **kw)
        state["verified"] = True
        return out

    return wrapper


def maybe_compile(fn: Callable, *, dynamic: Optional[bool] = None) -> Callable:
    """Return a compiled `fn`, or `fn` itself when compilation is disabled.

    `dynamic=False` suits large tensors with few distinct shapes (best kernels);
    `dynamic=True` suits small tensors whose leading dim varies a lot, where
    specializing buys nothing and only causes recompile churn.
    """
    if not _ENABLED:
        return fn

    key = (fn, dynamic)
    if key not in _CACHE:
        try:
            _CACHE[key] = _guarded(fn, torch.compile(fn, dynamic=dynamic), key)
        except Exception as e:  # pragma: no cover - torch.compile() itself refused
            _FAILED.add(key)
            logging.warning(
                f"torch.compile unavailable for {getattr(fn, '__name__', fn)} ({e}); running eager.",
            )
            _CACHE[key] = fn
    return _CACHE[key]
