"""Compute per-token importance weights for token-weighted G²PTQ (Idea 2).

Runs the BF16 model once over the *exact* calibration tokens that ptq.py will use (same
--model/--dataset/--nsamples/--seq_len/--seed/--cache_dir, so the cached token file is shared)
and records two signals per token:

  attention received (RSQ-style), from the full-attention layers:
      recv[t] = sum_q A[q, t] / (T - t)          # mean attention t gets from the queries that can see it
      averaged over heads and full-attention layers; position 0 (attention sink) is replaced by
      the sequence median.

  DeltaNet gate retention (new), from the Gated DeltaNet layers:
      gate[t] = beta_t * exp(sum_{s=t+1}^{t+k} g_s)  # how strongly t is written, times how much of it
                                                    # survives the next k decay steps
      with beta = sigmoid(in_proj_b(x)), g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias),
      averaged over value heads and DeltaNet layers.

Each signal is robustly standardised over all calibration tokens (median / IQR, clipped to +-5),
combined as  s = z_attn + lam * z_gate  ("source" weighting: a token's own importance), or
shifted by one position ("prediction" weighting: the importance of the token being predicted),
mapped to w = exp(tau * s), and rescaled to mean 1 with clipping to [w_min, w_max].

Usage (GPU):
    python ours/token_weights/compute_token_weights.py --model ./modelzoo/Qwen3.5/Qwen3.5-4B \
        --dataset calib_ifchat --nsamples 128 --seq_len 2048 --lam 1.0 --out ./cache/token_weights/ifchat.pt
"""

import argparse
import json
import logging
import os
import sys
import types

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def _import_repo():
    try:
        import fast_hadamard_transform  # noqa: F401
    except ImportError:  # CUDA extension not needed here
        sys.modules["fast_hadamard_transform"] = types.ModuleType("fast_hadamard_transform")
    from utils import dist_utils  # noqa: F401  (repo import order)
    from utils import data_utils
    from gptq_utils.token_weights import tokens_sha256
    return data_utils, tokens_sha256


def get_text_layers(model):
    for path in ("model.language_model.layers", "model.layers", "language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not find decoder layers")


class SignalRecorder:
    """Forward hooks that turn one batch's activations into per-token signals."""

    def __init__(self, layers, window):
        self.window = window
        self.handles = []
        self.attn, self.gate = [], []
        self._ab = {}
        self.n_attn_layers = self.n_gate_layers = 0
        for idx, layer in enumerate(layers):
            if getattr(layer, "layer_type", None) == "linear_attention" or hasattr(layer, "linear_attn"):
                la = layer.linear_attn
                self.handles.append(la.in_proj_a.register_forward_hook(self._make_ab_hook(idx, "a", la)))
                self.handles.append(la.in_proj_b.register_forward_hook(self._make_ab_hook(idx, "b", la)))
                self.n_gate_layers += 1
            elif hasattr(layer, "self_attn"):
                self.handles.append(layer.self_attn.register_forward_hook(self._attn_hook))
                self.n_attn_layers += 1

    def _attn_hook(self, module, inputs, output):
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError("Attention weights not returned; load the model with attn_implementation='eager'")
        A = weights.float()  # [B, H, Tq, Tk]
        T = A.shape[-1]
        received = A.sum(dim=2).mean(dim=1)  # [B, Tk], summed over queries, mean over heads
        n_queries = torch.arange(T, 0, -1, device=A.device, dtype=A.dtype)  # T - t queries see t
        self.attn.append((received / n_queries).cpu())

    def _make_ab_hook(self, idx, which, la):
        def hook(module, inputs, output):
            self._ab[(idx, which)] = output.detach().float()
            if (idx, "a") in self._ab and (idx, "b") in self._ab:  # both projections seen (any order)
                a = self._ab.pop((idx, "a"))
                b = self._ab.pop((idx, "b"))
                beta = b.sigmoid()  # [B, T, Hv]
                g = -la.A_log.float().exp() * F.softplus(a + la.dt_bias.float())  # log-decay, <= 0
                self.gate.append(retention(beta, g, self.window).mean(-1).cpu())  # [B, T]
        return hook

    def pop_batch(self):
        attn = torch.stack(self.attn).mean(0) if self.attn else None
        gate = torch.stack(self.gate).mean(0) if self.gate else None
        self.attn, self.gate = [], []
        return attn, gate

    def remove(self):
        for h in self.handles:
            h.remove()


def retention(beta, g, window):
    """beta_t * exp(sum of the next `window` log-decays); tail tokens extrapolate from what remains."""
    B, T, H = g.shape
    G = torch.cat([torch.zeros_like(g[:, :1]), g.cumsum(1)], dim=1)  # G[i] = sum g[0:i]
    t = torch.arange(T, device=g.device)
    end = torch.clamp(t + 1 + window, max=T)  # exclusive
    n = (end - (t + 1)).clamp(min=0)
    future = G[:, end] - G[:, t + 1]  # sum_{s=t+1}^{end-1} g_s   [B, T, H]
    mean_g = g.mean(1, keepdim=True)  # for the last token, which has no future steps
    scale = torch.where(n > 0, window / n.clamp(min=1).float(), torch.zeros_like(n, dtype=torch.float))
    future = torch.where((n > 0).view(1, T, 1), future * scale.view(1, T, 1), mean_g * window)
    return beta * future.exp()


def robust_z(x, clip=5.0):
    flat = x.flatten().float()
    med = flat.median()
    q1, q3 = torch.quantile(flat[torch.randperm(flat.numel())[:1_000_000]] if flat.numel() > 1_000_000 else flat,
                            torch.tensor([0.25, 0.75]))
    scale = (q3 - q1) / 1.349
    if scale <= 0:
        scale = flat.std().clamp(min=1e-8)
    return ((x - med) / scale).clamp(-clip, clip)


def normalize_weights(s, tau, w_min, w_max, iters=5):
    w = torch.exp(tau * s)
    for _ in range(iters):
        w = w / w.mean()
        w = w.clamp(w_min, w_max)
    return w / w.mean()


def combine(attn, gate, lam, tau, w_min, w_max, mode):
    parts = []
    if attn is not None:
        attn = attn.clone()
        attn[:, 0] = attn[:, 1:].median(dim=1).values  # attention sink
        parts.append(robust_z(attn))
    if gate is not None and lam != 0:
        parts.append(lam * robust_z(gate))
    if not parts:
        raise ValueError("No signals selected")
    s = sum(parts)
    if mode == "prediction":  # weight position t by the importance of the token it predicts (t+1)
        s = torch.cat([s[:, 1:], s.new_zeros(s.shape[0], 1)], dim=1)
    return normalize_weights(s, tau, w_min, w_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42, help="Must match ptq.py --seed")
    ap.add_argument("--cache_dir", default="./cache", help="Must match ptq.py --cache_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--signals", default="attn,gate", help="Comma list from {attn, gate}")
    ap.add_argument("--lam", type=float, default=1.0, help="Weight of the DeltaNet gate signal")
    ap.add_argument("--window", type=int, default=64, help="Decay steps k for gate retention")
    ap.add_argument("--mode", choices=["source", "prediction"], default="source")
    ap.add_argument("--tau", type=float, default=0.5, help="Sharpness of w = exp(tau * s)")
    ap.add_argument("--w_min", type=float, default=0.2)
    ap.add_argument("--w_max", type=float, default=5.0)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_signals", action="store_true", help="Also save the raw signals next to --out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    signals = {s.strip() for s in args.signals.split(",") if s.strip()}
    assert signals <= {"attn", "gate"} and signals, f"bad --signals {args.signals}"

    data_utils, tokens_sha256 = _import_repo()
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_name = args.model.rstrip("/").split("/")[-1]
    tokens_cache_path = f"{args.cache_dir}/tokens/{model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}.pt"
    tokens = data_utils.get_tokens(args.dataset, "train", tokenizer, args.seq_len, args.nsamples,
                                   tokens_cache_path, args.seed)
    tokens = [t[0].reshape(-1) if isinstance(t, (tuple, list)) else t for t in tokens]

    config = AutoConfig.from_pretrained(args.model)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == "qwen3_5" else AutoModelForCausalLM
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model = cls.from_pretrained(args.model, dtype=dtype, attn_implementation="eager").to(args.device).eval()
    recorder = SignalRecorder(get_text_layers(model), args.window)
    logging.info(f"Hooked {recorder.n_attn_layers} full-attention and {recorder.n_gate_layers} DeltaNet layers")

    attn_all, gate_all = [], []
    with torch.no_grad():
        for i in range(0, len(tokens), args.bsz):
            ids = torch.stack(tokens[i: i + args.bsz]).to(args.device)
            model(input_ids=ids, use_cache=False)
            a, g = recorder.pop_batch()
            attn_all.append(a)
            gate_all.append(g)
            if (i // args.bsz) % 16 == 0:
                logging.info(f"{i + ids.shape[0]}/{len(tokens)} sequences")
    recorder.remove()
    attn = torch.cat(attn_all) if "attn" in signals and attn_all[0] is not None else None
    gate = torch.cat(gate_all) if "gate" in signals and gate_all[0] is not None else None

    weights = combine(attn, gate, args.lam, args.tau, args.w_min, args.w_max, args.mode)
    meta = {k: v for k, v in vars(args).items() if k not in ("out",)}
    meta.update({"n_attn_layers": recorder.n_attn_layers, "n_gate_layers": recorder.n_gate_layers,
                 "weight_stats": {"mean": float(weights.mean()), "std": float(weights.std()),
                                  "min": float(weights.min()), "max": float(weights.max()),
                                  "frac_at_min": float((weights <= weights.min() + 1e-6).float().mean()),
                                  "frac_at_max": float((weights >= weights.max() - 1e-6).float().mean())}})
    if attn is not None and gate is not None:
        meta["attn_gate_spearman"] = float(torch.corrcoef(torch.stack([
            attn.flatten().argsort().argsort().float(), gate.flatten().argsort().argsort().float()]))[0, 1])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"weights": weights, "tokens_sha256": tokens_sha256(tokens), "meta": meta}, args.out)
    if args.save_signals:
        torch.save({"attn": attn, "gate": gate, "tokens": torch.stack(tokens)},
                   os.path.splitext(args.out)[0] + "_signals.pt")
    logging.info(f"Saved {tuple(weights.shape)} weights to {args.out}\n{json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
