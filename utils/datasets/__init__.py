import importlib
import pkgutil

from utils.datasets.registry import register_dataset, get_dataset, get_dataset_spec, get_dataset_names
from utils.datasets.common import format_messages

# Auto-import every dataset module so its @register_dataset runs on package import.
_NON_DATASET_MODULES = {"registry", "common", "loaders"}
for _m in pkgutil.iter_modules(__path__):
    if _m.name not in _NON_DATASET_MODULES:
        importlib.import_module(f"{__name__}.{_m.name}")

from utils.datasets.loaders import get_tokens, get_loaders

__all__ = [
    "register_dataset",
    "get_dataset",
    "get_dataset_spec",
    "get_dataset_names",
    "format_messages",
    "get_tokens",
    "get_loaders",
]
