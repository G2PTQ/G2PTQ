import importlib
import logging
import pkgutil

from utils.models.registry import (
    register_model,
    get_model_spec,
    get_model_spec_names,
    get_import_failures,
    record_import_failure,
)
from utils.models.base import ModelSpec, resolve

# Auto-import every model module so its @register_model runs on package import.
_NON_MODEL_MODULES = {"registry", "base", "mixins"}
for _m in pkgutil.iter_modules(__path__):
    if _m.name in _NON_MODEL_MODULES:
        continue
    try:
        importlib.import_module(f"{__name__}.{_m.name}")
    except Exception as _e:
        # One unimportable spec must not take down every other arch.
        record_import_failure(_m.name, _e)
        logging.warning(
            f"Skipping model spec '{_m.name}': failed to import "
            f"({type(_e).__name__}: {_e}). Architectures it registers will be unavailable."
        )

__all__ = [
    "register_model",
    "get_model_spec",
    "get_model_spec_names",
    "get_import_failures",
    "ModelSpec",
    "resolve",
]
