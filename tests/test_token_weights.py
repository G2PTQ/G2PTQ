"""CPU tests for token-weighted G2PTQ (run: python tests/test_token_weights.py).

Checks that (1) loading no weights or all-ones weights reproduces plain G2PTQ exactly, and
(2) non-uniform weights scale each token's saliency (Hessian) and loss term (gradient) as intended.
"""
import contextlib
import math
import os
import sys
import tempfile
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("fast_hadamard_transform", types.ModuleType("fast_hadamard_transform"))  # CUDA-only ext

import torch
import torch.nn as nn

from utils import dist_utils  # noqa: F401  (repo import order avoids a circular import)
from gptq_utils import g2ptq_utils as g2
from gptq_utils import token_weights as tw

B, T, D, V, GROUPS = 2, 5, 8, 11, 2


class NullGradCache:
    @contextlib.contextmanager
    def set_backward_hook(self, full, loss_type="block"):
        yield


def _toy(seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(D, D)
    head = nn.Linear(D, V)
    x = torch.randn(B, T, D)
    return lin, head, x


def _saliency(loss_kind, weights, seed=0):
    """Saliency the G2PTQ Hessian is built from, for one toy batch."""
    tw.reset()
    if weights is not None:
        tw._STATE["weights"] = weights
        tw.set_batch(0, B, "cpu")
    lin, head, x = _toy(seed)
    cache = g2.SaliencyCache(["lin"], GROUPS)
    cache.set_forward_hook({"lin": lin})
    with torch.enable_grad():
        torch.manual_seed(123)  # same label / noise samples in every call
        h = lin(x)
        args = SimpleNamespace(kl_topk=0)
        if loss_kind == "kl":
            logits = head(h)
            logits_fp = head(x)
            g2.compute_kl_grad_hessian(logits, logits_fp, cache, NullGradCache(), {}, args, compute_gradient=False)
        else:
            g2.compute_mse_grad_hessian(h, x, cache, NullGradCache(), {}, args, compute_gradient=False)
    cache.clear_hook()
    tw.reset()
    return cache.saliency_cache["lin"]["block"][0]


def _gradient(loss_kind, weights, topk=0, seed=0):
    """Weight gradient G2PTQ uses for first-order compensation, for one toy batch."""
    tw.reset()
    if weights is not None:
        tw._STATE["weights"] = weights
        tw.set_batch(0, B, "cpu")
    lin, head, x = _toy(seed)
    cache = g2.SaliencyCache(["lin"], GROUPS)
    cache.set_forward_hook({"lin": lin})
    with torch.enable_grad():
        torch.manual_seed(123)
        h = lin(x)
        args = SimpleNamespace(kl_topk=topk)
        if loss_kind == "kl":
            loss = g2.compute_kl_grad_hessian(head(h), head(x), cache, NullGradCache(), {}, args)
        else:
            loss = g2.compute_mse_grad_hessian(h, x, cache, NullGradCache(), {}, args)
    cache.clear_hook()
    tw.reset()
    return lin.weight.grad.clone(), loss


def _manual_gradient(loss_kind, weights, seed=0):
    lin, head, x = _toy(seed)
    with torch.enable_grad():
        h = lin(x)
        if loss_kind == "kl":
            per_tok = torch.nn.functional.kl_div(
                torch.log_softmax(head(h), -1), torch.softmax(head(x), -1), reduction="none").sum(-1)
            loss = (per_tok * weights).sum()
        else:
            per_tok = (h - x).pow(2).mean(-1)
            loss = (per_tok * weights).sum() / 2
        loss.backward()
    return lin.weight.grad.clone()


def test_ones_equal_plain():
    ones = torch.ones(B, T)
    for kind in ("mse", "kl"):
        assert torch.equal(_saliency(kind, None), _saliency(kind, ones)), kind
        g0, l0 = _gradient(kind, None)
        g1, l1 = _gradient(kind, ones)
        assert torch.allclose(g0, g1, rtol=1e-6, atol=1e-7), kind
        assert math.isclose(float(l0), float(l1), rel_tol=1e-6), kind


def test_saliency_scaled_per_token():
    w = torch.rand(B, T) * 3
    for kind in ("mse", "kl"):
        base = _saliency(kind, None)
        weighted = _saliency(kind, w)
        assert torch.allclose(weighted, base * w.reshape(-1, 1), rtol=1e-6, atol=1e-8), kind


def test_gradient_is_weighted_loss_gradient():
    w = torch.rand(B, T) * 3
    for kind in ("mse", "kl"):
        got, _ = _gradient(kind, w)
        want = _manual_gradient(kind, w)
        assert torch.allclose(got, want, rtol=1e-5, atol=1e-7), kind
        plain, _ = _gradient(kind, None)
        assert not torch.allclose(got, plain), kind


def test_kl_topk_path_runs():
    w = torch.rand(B, T)
    g, loss = _gradient("kl", w, topk=4)
    assert torch.isfinite(g).all() and torch.isfinite(loss)


def test_load_checks_hash_shape_and_slices():
    tokens = [torch.randint(0, 100, (T,)) for _ in range(4)]
    w = torch.rand(4, T) + 0.5
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.pt")
        torch.save({"weights": w, "tokens_sha256": tw.tokens_sha256(tokens), "meta": {}}, path)
        tw.load(path, tokens, start=2, end=4)
        assert torch.equal(tw._STATE["weights"], w[2:4])
        tw.set_batch(1, 1, "cpu")
        assert torch.equal(tw.current(), w[3:4])
        tw.reset()

        other = [t.clone() for t in tokens]
        other[0][0] += 1
        try:
            tw.load(path, other)
        except ValueError as e:
            assert "different calibration tokens" in str(e)
        else:
            raise AssertionError("hash mismatch not detected")

        torch.save({"weights": w[:3], "tokens_sha256": tw.tokens_sha256(tokens)}, path)
        try:
            tw.load(path, tokens)
        except ValueError as e:
            assert "do not match" in str(e)
        else:
            raise AssertionError("shape mismatch not detected")
    tw.reset()


def test_mismatched_token_count_raises():
    tw.reset()
    tw._STATE["weights"] = torch.ones(1, 3)
    tw.set_batch(0, 1, "cpu")
    try:
        tw.apply_flat(torch.ones(7, 2))
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatch not detected")
    tw.reset()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
