"""Check the token-importance signal math against naive loops (run: python tests/test_token_weight_signals.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ours", "token_weights"))
from compute_token_weights import retention, normalize_weights, combine  # noqa: E402


def test_retention_matches_loop():
    torch.manual_seed(0)
    B, T, H, k = 2, 9, 3, 4
    beta = torch.rand(B, T, H)
    g = -torch.rand(B, T, H)
    got = retention(beta, g, k)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                fut = list(range(t + 1, min(t + 1 + k, T)))
                if fut:
                    s = sum(g[b, s, h] for s in fut) * k / len(fut)
                else:
                    s = g[b, :, h].mean() * k
                want = beta[b, t, h] * torch.exp(s)
                assert torch.allclose(got[b, t, h], want, atol=1e-6), (b, t, h)


def test_weights_mean_one_and_clipped():
    torch.manual_seed(0)
    w = normalize_weights(torch.randn(4, 100) * 3, tau=1.0, w_min=0.2, w_max=5.0)
    assert abs(float(w.mean()) - 1) < 1e-5
    assert float(w.min()) >= 0.2 - 1e-3 and float(w.max()) <= 5.0 * 1.05


def test_prediction_mode_shifts_by_one():
    torch.manual_seed(0)
    attn = torch.rand(2, 10)
    src = combine(attn, None, 0.0, 1.0, 1e-6, 1e6, "source")
    pred = combine(attn, None, 0.0, 1.0, 1e-6, 1e6, "prediction")
    # same ordering of the underlying scores, shifted by one position
    assert torch.equal(src[:, 1:].argsort(), pred[:, :-1].argsort())


def test_sink_replaced():
    attn = torch.ones(1, 6)
    attn[0, 0] = 1000.0
    w = combine(attn, None, 0.0, 1.0, 1e-6, 1e6, "source")
    assert torch.allclose(w, torch.ones_like(w)), "position 0 must not dominate"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
