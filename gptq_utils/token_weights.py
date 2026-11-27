"""Per-token importance weights for the G²PTQ calibration objective (Idea 2).

A weights file is produced once, offline, by ``ours/token_weights/compute_token_weights.py``
from the BF16 model and the *exact* calibration tokens used for quantization. It is a dict:

    {"weights": FloatTensor[nsamples, seq_len],   # rescaled to mean 1
     "tokens_sha256": str,                         # hash of the calibration tokens
     "meta": {...}}                                # how the weights were computed

During G²PTQ, the batch's weights are made "current" before each block forward/backward.
They then scale (a) each token's contribution to the saliency that builds the Hessian,
H' = sum_t w_t F_t (see ``SaliencyCache.grad_hook``), and (b) each token's term in the block
loss whose gradient gives g' (see ``compute_kl_grad_hessian`` / ``compute_mse_grad_hessian``).
Because the weights have mean 1, the existing 1/n_tokens normalisations equal 1/sum(w).
With no weights loaded, every code path is unchanged.
"""

import hashlib
import logging

import torch

_STATE = {"weights": None, "current": None}


def tokens_sha256(tokens):
    """Hash of a list of 1-D token tensors (or (input_ids, target) tuples)."""
    h = hashlib.sha256()
    for t in tokens:
        if isinstance(t, (tuple, list)):
            t = t[0]
        h.update(t.reshape(-1).to(torch.int64).cpu().numpy().tobytes())
    return h.hexdigest()


def load(path, tokens, start=0, end=None):
    """Load weights for the calibration ``tokens`` and keep rows [start:end] (data-parallel rank)."""
    blob = torch.load(path, map_location="cpu")
    weights = blob["weights"].float()
    n, seq_len = len(tokens), (tokens[0][0] if isinstance(tokens[0], (tuple, list)) else tokens[0]).shape[-1]
    if tuple(weights.shape) != (n, seq_len):
        raise ValueError(f"Token weights {tuple(weights.shape)} do not match calibration tokens ({n}, {seq_len})")
    expected = blob.get("tokens_sha256")
    actual = tokens_sha256(tokens)
    if expected != actual:
        raise ValueError(
            "Token weights were computed for different calibration tokens "
            f"(file {expected[:12] if expected else None}..., run {actual[:12]}...). "
            "Recompute them with the same --model/--dataset/--nsamples/--seq_len/--seed."
        )
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Token weights must be finite and non-negative")
    _STATE["weights"] = weights[start:end].contiguous()
    _STATE["current"] = None
    logging.info(
        f"Loaded token weights from {path}: shape {tuple(_STATE['weights'].shape)}, "
        f"mean {weights.mean():.3f}, min {weights.min():.3f}, max {weights.max():.3f}, "
        f"meta {blob.get('meta')}"
    )


def enabled():
    return _STATE["weights"] is not None


def set_batch(j, bsz, device):
    """Make rows [j:j+bsz] the current batch's weights (no-op when disabled)."""
    if _STATE["weights"] is None:
        _STATE["current"] = None
        return
    _STATE["current"] = _STATE["weights"][j: j + bsz].to(device, non_blocking=True)


def clear_batch():
    _STATE["current"] = None


def current():
    """Weights of the current batch, shape [bsz, seq_len], or None."""
    return _STATE["current"]


def reset():
    _STATE["weights"] = None
    _STATE["current"] = None


def apply_flat(per_token, name=""):
    """Multiply a per-token tensor whose first dim is (bsz*seq_len) by the current weights."""
    w = _STATE["current"]
    if w is None:
        return per_token
    w = w.reshape(-1)
    if per_token.shape[0] != w.numel():
        raise RuntimeError(
            f"Token weights ({w.numel()} tokens) do not match {name or 'tensor'} "
            f"with {per_token.shape[0]} tokens; token weighting supports dense (non-MoE) layers only."
        )
    shape = (-1,) + (1,) * (per_token.dim() - 1)
    return per_token * w.to(per_token.dtype).view(shape)
