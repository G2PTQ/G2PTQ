"""Developer smoke test: run ptq.py end to end on CPU (tiny model), stubbing CUDA-only pieces.

    python tests/cpu_smoke_ptq.py <ptq.py args...>

Not for real runs: it fakes CUDA memory statistics and skips the multimodal processor.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("fast_hadamard_transform", types.ModuleType("fast_hadamard_transform"))

import torch  # noqa: E402

if not torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0

import torch.distributed as _dist  # noqa: E402

_barrier = _dist.barrier
_dist.barrier = lambda *a, **k: _barrier(*a, **k) if _dist.is_initialized() else None

from utils import dist_utils, model_utils  # noqa: E402

_orig_load_processor = model_utils.load_processor


def _load_processor(x):
    try:
        return _orig_load_processor(x)
    except Exception:
        return None


model_utils.load_processor = _load_processor

if not torch.cuda.is_available():
    import contextlib
    from utils import offload_utils

    # Load the tiny model plainly on CPU instead of through the GPU offload machinery.
    offload_utils.load_offloaded_model = lambda **kw: contextlib.nullcontext()
    _from_pretrained = {}

    def _patch_cls(cls):
        if cls in _from_pretrained:
            return
        orig = cls.from_pretrained.__func__
        _from_pretrained[cls] = orig

        @classmethod
        def fp(c, *a, **kw):
            for k in ("device_map", "offload_folder", "offload_buffers"):
                kw.pop(k, None)
            kw["torch_dtype"] = torch.float32
            return orig(c, *a, **kw)
        cls.from_pretrained = fp

    _orig_get_spec = model_utils.get_model_spec

    def _get_spec(name):
        spec = _orig_get_spec(name)
        _orig_lmc = spec.load_model_class

        def load_model_class(config, _o=_orig_lmc):
            cls = _o(config)
            _patch_cls(cls)
            return cls
        spec.load_model_class = staticmethod(load_model_class)
        return spec
    model_utils.get_model_spec = _get_spec
    _orig_init = model_utils.ModelAnalyzer.__init__
    dist_utils.get_rank = lambda: 0
    _orig_set_onload = offload_utils.set_onload_device
    offload_utils.set_onload_device = lambda model, device, *a, **k: _orig_set_onload(model, torch.device("cpu"), *a, **k)
    offload_utils.assert_shared_offload_storage = lambda *a, **k: None

    def _init(self, *a, **k):
        _orig_init(self, *a, **k)
        self.onload_device = "cpu"
    model_utils.ModelAnalyzer.__init__ = _init

    # The eager GPTQ kernels only *check* for CUDA tensors; the math runs fine on CPU.
    from gptq_utils import graph_utils, g2ptq_utils, gptq_utils as _gq
    for _m in (graph_utils, g2ptq_utils, _gq):
        if hasattr(_m, "validate_tensors"):
            _m.validate_tensors = lambda *a, **k: None

import ptq  # noqa: E402
from process_args import parse_gen  # noqa: E402

if __name__ == "__main__":
    pass  # single process, no torch.distributed on CPU
    ptq.main(parse_gen())
