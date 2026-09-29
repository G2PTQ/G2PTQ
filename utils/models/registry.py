"""Model registry.

Each supported architecture lives in its own module under ``utils/models/`` and registers a
:class:`~utils.models.base.ModelSpec` subclass via the ``@register_model("<model_type>")`` decorator,
keyed on HF's ``config.model_type``. A spec is a stateless descriptor of the model's module 
layout plus a handful of arch-specific hooks; ``ModelAnalyzer`` binds one and delegates to it.
"""

_REGISTRY = {}

#: Spec modules that failed to import, ``{module_name: exception}``. A spec whose module raises (a
#: missing transformers model class, an unreleased symbol) is skipped with a warning.
_FAILED_IMPORTS = {}


def record_import_failure(module_name, exc):
    """Note that a spec module could not be imported."""
    _FAILED_IMPORTS[module_name] = exc


def get_import_failures():
    """Return ``{module_name: exception}`` for spec modules that failed to import."""
    return dict(_FAILED_IMPORTS)


def register_model(*names):
    """Decorator registering a ``ModelSpec`` subclass under one or more ``model_type`` strings."""

    def deco(cls):
        for name in names:
            if name in _REGISTRY:
                raise ValueError(f"Model '{name}' already registered")
            _REGISTRY[name] = cls
        return cls

    return deco


def get_model_spec(model_type):
    """Return the ``ModelSpec`` subclass registered for ``model_type``."""
    if model_type not in _REGISTRY:
        msg = (
            f"Unsupported model_type '{model_type}'. Registered: {get_model_spec_names()}. "
            f"Add a spec module under utils/models/ to support it."
        )
        # A spec module that failed to import never ran its @register_model, so the arch looks
        # unsupported. Surface the real cause.
        for mod, exc in _FAILED_IMPORTS.items():
            if model_type.startswith(mod) or mod.startswith(model_type):
                raise NotImplementedError(
                    f"{msg} Note: spec module '{mod}' failed to import and may be the one that "
                    f"supports '{model_type}' — fix that import first: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        if _FAILED_IMPORTS:
            msg += f" Spec modules skipped due to import errors: {sorted(_FAILED_IMPORTS)}."
        raise NotImplementedError(msg)
    return _REGISTRY[model_type]


def get_model_spec_names():
    """Return the sorted list of registered ``model_type`` strings."""
    return sorted(_REGISTRY)
