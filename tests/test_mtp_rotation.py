"""Mathematical equivalence tests for MTP rotation compensation with an independent lm_head.

Goal: after G2PTQ applies a global rotation R1 to the main model, the MTP
inputs (token embeddings, main model final hidden states) are all in the
rotated coordinate frame. The MTP is rotated layer-by-layer consistently with
the main model (norm column-wise fusion, fc similarity transform
R1^T*W*block_diag(R1,R1), input-side layers W*R1, output-side layers R1^T*W,
independent head fused with mtp.norm then W*R1), so the whole MTP runs in the
rotated coordinate frame and the recursion chain stays coordinate-consistent.

The multi-step recursion test is the key regression: the first MTP step takes
its hidden input from the main model (rotated frame), later steps take it from
the previous MTP step output (also rotated frame). Each step's logits must
match the unrotated model exactly.

Run from the repository root with:

    python -m unittest tests.test_mtp_rotation -v
"""

import unittest

import torch
import torch.nn as nn

from utils.mtp_utils import (
    rotate_mtp_state_dict,
    rotate_mtp_file,
    fuse_mtp_norms,
)


def random_hadamard(n):
    h = torch.tensor([[1.0]])
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    D = torch.diag(2 * torch.randint(0, 2, (n,)).float() - 1)
    return D @ h / (n ** 0.5)


class RMSNorm(nn.Module):
    """Qwen3.5-style zero-centered RMSNorm: x * (1 + w)."""

    def __init__(self, d, rand=True):
        super().__init__()
        if rand:
            self.weight = nn.Parameter(torch.randn(d) * 0.3)
        else:
            self.weight = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        return x * (1.0 + self.weight)


class Attn(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, k, v):
        q = self.q_proj(x).view(*x.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -3).reshape(*x.shape[:-1], -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter)
        self.up_proj = nn.Linear(d, inter)
        self.down_proj = nn.Linear(inter, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MtpAttn(Attn):
    def __init__(self, d, n_heads=4):
        super().__init__(d, n_heads)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, k, v):
        q = self.q_proj(x).view(*x.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -3).reshape(*x.shape[:-1], -1)
        return self.o_proj(out)


class MtpLayer35(nn.Module):
    """Qwen3.5-style MTP decoder layer (with q_norm/k_norm)."""

    def __init__(self, d, inter):
        super().__init__()
        self.input_layernorm = RMSNorm(d)
        self.self_attn = MtpAttn(d)
        self.post_attention_layernorm = RMSNorm(d)
        self.mlp = MLP(d, inter)

    def forward(self, x, k, v):
        h = x + self.self_attn(self.input_layernorm(x), k, v)
        return h + self.mlp(self.post_attention_layernorm(h))


class MainModel(nn.Module):
    def __init__(self, d, vocab, nlayer, inter):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([MtpLayer35(d, inter) for _ in range(nlayer)])
        self.final_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        k, v = None, None
        for l in self.layers:
            if k is None:
                h = l.input_layernorm(x)
                k = l.self_attn.k_proj(h).view(*h.shape[:-1], 4, -1).transpose(-2, -3)
                v = l.self_attn.v_proj(h).view(*h.shape[:-1], 4, -1).transpose(-2, -3)
            x = l(x, k, v)
        last_hidden = x
        logits = self.lm_head(self.final_norm(x))
        return logits, last_hidden


class ModelWithMtp35(nn.Module):
    """Qwen3.5-style MTP: pre_fc_norm_embedding/hidden -> fc(cat) -> layers.0 -> norm,
    output mapped through an independent mtp.head (decoupled from the main
    lm_head so that norm fusion can be verified)."""

    def __init__(self, d, vocab, nlayer, inter):
        super().__init__()
        self.model = MainModel(d, vocab, nlayer, inter)
        self.mtp = nn.ModuleDict({
            "pre_fc_norm_embedding": RMSNorm(d),
            "pre_fc_norm_hidden": RMSNorm(d),
            "fc": nn.Linear(2 * d, d, bias=False),
            "layers": nn.ModuleDict({"0": MtpLayer35(d, inter)}),
            "norm": RMSNorm(d),
            "lm_head": nn.Linear(d, vocab, bias=False),
        })

    def mtp_forward(self, next_ids, last_hidden, kv_from_model):
        k, v = kv_from_model
        emb = self.mtp.pre_fc_norm_embedding(self.model.embed_tokens(next_ids))
        hid = self.mtp.pre_fc_norm_hidden(last_hidden)
        h = self.mtp.fc(torch.cat([emb, hid], dim=-1))
        h = self.mtp.layers["0"](h, k, v)
        h = self.mtp.norm(h)
        return self.mtp.lm_head(h)

    def mtp_forward_multi(self, token_embeds, first_hidden, kv_from_model):
        """Multi-step recursion: each step takes one token embedding as input;
        the hidden input is first_hidden on the first step and the previous
        step's layer output (pre-norm, matching the recursion semantics of the
        rotated MTP) afterwards; logits use the final norm output."""
        k, v = kv_from_model
        h = first_hidden
        logits = []
        for i in range(len(token_embeds)):
            emb = self.mtp.pre_fc_norm_embedding(token_embeds[i])
            hid = self.mtp.pre_fc_norm_hidden(h)
            h = self.mtp.fc(torch.cat([emb, hid], dim=-1))
            h = self.mtp.layers["0"](h, k, v)
            logits.append(self.mtp.lm_head(self.mtp.norm(h)))
        return logits


def fuse_ln(linear, ln):
    """Zero-centered RMSNorm fusion: absorb (1+w) column-wise into the linear
    layer and reset the norm weight to 0."""
    with torch.no_grad():
        linear.weight.mul_(1.0 + ln.weight)
        ln.weight.fill_(0.0)


def fuse_ln_multi(linears, ln):
    """Zero-centered RMSNorm fusion into multiple linear layers.

    Same semantics as utils/rotation_utils.fuse_ln_linears: absorb the scale
    (1+w) into all downstream linear layers first and only then reset the norm
    weight to 0 (otherwise a later linear layer would read the already-zeroed
    norm and lose the fused scale).
    """
    with torch.no_grad():
        scale = (1.0 + ln.weight).double()
        for linear in linears:
            linear.weight.mul_(scale)
        ln.weight.fill_(0.0)


def apply_main_rotation(model, R1):
    """Replicate the R1 global rotation of utils/rotation_utils.py (LN fusion +
    consumer/writer projection transforms)."""
    for l in model.layers:
        fuse_ln_multi([l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj],
                      l.input_layernorm)
        fuse_ln_multi([l.mlp.gate_proj, l.mlp.up_proj], l.post_attention_layernorm)
    fuse_ln(model.lm_head, model.final_norm)
    R1T = R1.T
    with torch.no_grad():
        model.embed_tokens.weight.data = model.embed_tokens.weight.data @ R1
        model.lm_head.weight.data = model.lm_head.weight.data @ R1
        for l in model.layers:
            for lin in (l.self_attn.q_proj, l.self_attn.k_proj, l.self_attn.v_proj,
                        l.mlp.gate_proj, l.mlp.up_proj):
                lin.weight.data = lin.weight.data @ R1
            for lin in (l.self_attn.o_proj, l.mlp.down_proj):
                lin.weight.data = R1T @ lin.weight.data


class TestMtpRotation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.hidden, self.vocab, self.nlayer, self.inter = 32, 64, 2, 48
        self.R1 = random_hadamard(self.hidden).double()

    def _make_model(self, seed):
        torch.manual_seed(seed)
        model = ModelWithMtp35(self.hidden, self.vocab, self.nlayer, self.inter)
        model.double()
        return model

    def _kv_from_model(self, model, ids):
        x = model.embed_tokens(ids)
        for l in model.layers:
            h = l.input_layernorm(x)
            k = l.self_attn.k_proj(h).view(*h.shape[:-1], 4, -1).transpose(-2, -3)
            v = l.self_attn.v_proj(h).view(*h.shape[:-1], 4, -1).transpose(-2, -3)
            x = l(x, k, v)
        return k, v

    def test_mtp_rotation_equivalence_qwen35_multi_step(self):
        """Multi-step recursion equivalence: every step (including step 2 and
        later) must produce the same logits as the reference.

        This is the regression test for the coordinate mismatch on the second
        recursion step.
        """
        model = self._make_model(seed=1)
        ids = torch.randint(0, self.vocab, (2, 8))
        n_steps = 4
        next_ids = torch.randint(0, self.vocab, (2, n_steps))

        with torch.no_grad():
            _, last_hidden_ref = model.model(ids)
            kv_ref = self._kv_from_model(model.model, ids)
            first_hidden_ref = last_hidden_ref[:, -1:, :]
            emb_ref = [model.model.embed_tokens(next_ids[:, i]).unsqueeze(1)
                       for i in range(n_steps)]
            ref_logits = model.mtp_forward_multi(emb_ref, first_hidden_ref, kv_ref)

        # Apply R1 to the main model; the MTP weights go through the same
        # transforms (norm fusion + per-layer rotation + independent head)
        apply_main_rotation(model.model, self.R1)
        mtp_sd = {k: v.clone() for k, v in model.mtp.state_dict().items()}
        rotated_mtp_sd = rotate_mtp_state_dict(mtp_sd, self.R1)
        model.mtp.load_state_dict(rotated_mtp_sd)

        with torch.no_grad():
            _, last_hidden_rot = model.model(ids)
            kv_rot = self._kv_from_model(model.model, ids)
            # After rotation the main model embed output is in the rotated
            # frame, which is the true embedding input of the i-th MTP step
            first_hidden_rot = last_hidden_rot[:, -1:, :]
            emb_rot = [model.model.embed_tokens(next_ids[:, i]).unsqueeze(1)
                       for i in range(n_steps)]
            rot_logits = model.mtp_forward_multi(emb_rot, first_hidden_rot, kv_rot)

        self.assertEqual(len(ref_logits), n_steps)
        for step, (r, q) in enumerate(zip(ref_logits, rot_logits)):
            max_err = (q - r).abs().max().item()
            self.assertAlmostEqual(
                max_err, 0.0, places=6,
                msg=f"step {step} MTP logits max diff: {max_err}",
            )
            self.assertTrue(
                torch.equal(r.argmax(-1), q.argmax(-1)),
                f"step {step} MTP top-1 agreement must be 100%",
            )

    def test_mtp_rotation_without_norm_fusion_fails(self):
        """Regression guard: without norm fusion the rotation equivalence must break."""
        model = self._make_model(seed=2)
        ids = torch.randint(0, self.vocab, (2, 8))
        next_ids = torch.randint(0, self.vocab, (2, 8))

        with torch.no_grad():
            _, last_hidden_ref = model.model(ids)
            kv_ref = self._kv_from_model(model.model, ids)
            ref_mtp_logits = model.mtp_forward(next_ids, last_hidden_ref, kv_ref)

        apply_main_rotation(model.model, self.R1)
        mtp_sd = {k: v.clone() for k, v in model.mtp.state_dict().items()}
        rotated_mtp_sd = rotate_mtp_state_dict(mtp_sd, self.R1, fuse_norms=False)
        model.mtp.load_state_dict(rotated_mtp_sd)

        with torch.no_grad():
            _, last_hidden_rot = model.model(ids)
            kv_rot = self._kv_from_model(model.model, ids)
            rot_mtp_logits = model.mtp_forward(next_ids, last_hidden_rot, kv_rot)

        max_err = (rot_mtp_logits - ref_mtp_logits).abs().max().item()
        self.assertGreater(max_err, 1e-2,
                           "Without norm fusion the rotation must break equivalence")

    def test_fuse_mtp_norms_scheme_a(self):
        """All four norms are fused column-wise into their downstream layers and
        zeroed (same-source norms are absorbed first, then zeroed)."""
        d, vocab, inter = self.hidden, self.vocab, self.inter
        w_emb = torch.randn(d).double()
        w_hid = torch.randn(d).double()
        w_il = torch.randn(d).double()
        w_pal = torch.randn(d).double()
        w_n = torch.randn(d).double()
        fc = torch.randn(d, 2 * d).double()
        q = torch.randn(d, d).double()
        k = torch.randn(d, d).double()
        v = torch.randn(d, d).double()
        gate = torch.randn(inter, d).double()
        up = torch.randn(inter, d).double()
        head = torch.randn(vocab, d).double()
        sd = {
            "mtp.pre_fc_norm_embedding.weight": w_emb.clone(),
            "mtp.pre_fc_norm_hidden.weight": w_hid.clone(),
            "mtp.fc.weight": fc.clone(),
            "mtp.layers.0.input_layernorm.weight": w_il.clone(),
            "mtp.layers.0.self_attn.q_proj.weight": q.clone(),
            "mtp.layers.0.self_attn.k_proj.weight": k.clone(),
            "mtp.layers.0.self_attn.v_proj.weight": v.clone(),
            "mtp.layers.0.post_attention_layernorm.weight": w_pal.clone(),
            "mtp.layers.0.mlp.gate_proj.weight": gate.clone(),
            "mtp.layers.0.mlp.up_proj.weight": up.clone(),
            "mtp.norm.weight": w_n.clone(),
            "mtp.lm_head.weight": head.clone(),
        }
        out = fuse_mtp_norms(sd)
        # pre_fc_norm zeroed, fc absorbs the corresponding columns
        self.assertTrue(torch.equal(out["mtp.pre_fc_norm_embedding.weight"],
                                    torch.zeros_like(w_emb)))
        self.assertTrue(torch.equal(out["mtp.pre_fc_norm_hidden.weight"],
                                    torch.zeros_like(w_hid)))
        self.assertTrue(torch.allclose(out["mtp.fc.weight"][:, :d],
                                       fc[:, :d] * (1 + w_emb).unsqueeze(0)))
        self.assertTrue(torch.allclose(out["mtp.fc.weight"][:, d:],
                                       fc[:, d:] * (1 + w_hid).unsqueeze(0)))
        # input_layernorm zeroed, q/k/v all absorb (1+w_il)
        self.assertTrue(torch.equal(out["mtp.layers.0.input_layernorm.weight"],
                                    torch.zeros_like(w_il)))
        for name, W in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
            self.assertTrue(torch.allclose(
                out[f"mtp.layers.0.self_attn.{name}.weight"],
                W * (1 + w_il).unsqueeze(0)), name)
        # post_attention_layernorm zeroed, gate/up absorb (1+w_pal)
        self.assertTrue(torch.equal(out["mtp.layers.0.post_attention_layernorm.weight"],
                                    torch.zeros_like(w_pal)))
        for name, W in (("gate_proj", gate), ("up_proj", up)):
            self.assertTrue(torch.allclose(
                out[f"mtp.layers.0.mlp.{name}.weight"],
                W * (1 + w_pal).unsqueeze(0)), name)
        # mtp.norm zeroed, head absorbs (1+w_n)
        self.assertTrue(torch.equal(out["mtp.norm.weight"], torch.zeros_like(w_n)))
        self.assertTrue(torch.allclose(out["mtp.lm_head.weight"],
                                       head * (1 + w_n).unsqueeze(0)))
        # the input dict is not modified
        self.assertTrue(torch.equal(sd["mtp.fc.weight"], fc))
        self.assertTrue(torch.equal(sd["mtp.layers.0.self_attn.q_proj.weight"], q))

    def test_fuse_mtp_norms_without_head_keeps_final_norm(self):
        """Shared-head scenario: without an independent mtp.lm_head, mtp.norm is
        kept raw and a warning is logged."""
        d = self.hidden
        w_n = torch.randn(d).double()
        sd = {
            "mtp.pre_fc_norm_embedding.weight": torch.randn(d).double(),
            "mtp.pre_fc_norm_hidden.weight": torch.randn(d).double(),
            "mtp.fc.weight": torch.randn(d, 2 * d).double(),
            "mtp.layers.0.input_layernorm.weight": torch.randn(d).double(),
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(d, d).double(),
            "mtp.layers.0.post_attention_layernorm.weight": torch.randn(d).double(),
            "mtp.layers.0.mlp.gate_proj.weight": torch.randn(self.inter, d).double(),
            "mtp.layers.0.mlp.up_proj.weight": torch.randn(self.inter, d).double(),
            "mtp.norm.weight": w_n.clone(),
        }
        out = fuse_mtp_norms(sd)
        self.assertTrue(torch.equal(out["mtp.norm.weight"], w_n),
                        "without independent head the final mtp.norm must stay raw")
        self.assertTrue(torch.equal(sd["mtp.norm.weight"], w_n),
                        "input dict must not be modified")

    def test_rotate_mtp_state_dict_scheme_a(self):
        """fc similarity transform, input-side W*R1, output-side R1^T*W,
        head W*R1, q_norm/k_norm kept as-is."""
        d, vocab, inter = self.hidden, self.vocab, self.inter
        R1 = self.R1
        sd = {
            "mtp.pre_fc_norm_embedding.weight": torch.randn(d).double(),
            "mtp.pre_fc_norm_hidden.weight": torch.randn(d).double(),
            "mtp.fc.weight": torch.randn(d, 2 * d).double(),
            "mtp.layers.0.input_layernorm.weight": torch.randn(d).double(),
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(d, d).double(),
            "mtp.layers.0.self_attn.k_proj.weight": torch.randn(d, d).double(),
            "mtp.layers.0.self_attn.v_proj.weight": torch.randn(d, d).double(),
            "mtp.layers.0.self_attn.o_proj.weight": torch.randn(d, d).float(),
            "mtp.layers.0.self_attn.q_norm.weight": torch.randn(8).double(),
            "mtp.layers.0.self_attn.k_norm.weight": torch.randn(8).float(),
            "mtp.layers.0.post_attention_layernorm.weight": torch.randn(d).double(),
            "mtp.layers.0.mlp.gate_proj.weight": torch.randn(inter, d).double(),
            "mtp.layers.0.mlp.up_proj.weight": torch.randn(inter, d).double(),
            "mtp.layers.0.mlp.down_proj.weight": torch.randn(d, inter).float(),
            "mtp.norm.weight": torch.randn(d).double(),
            "mtp.lm_head.weight": torch.randn(vocab, d).float(),
        }
        src = {k: v.clone() for k, v in sd.items()}
        out = rotate_mtp_state_dict(src, R1)
        # keys, dtypes and shapes unchanged
        self.assertEqual(set(out.keys()), set(sd.keys()))
        for name, tensor in out.items():
            self.assertEqual(tensor.dtype, sd[name].dtype, name)
            self.assertEqual(tensor.shape, sd[name].shape, name)
        # the input dict is not modified
        for name, v in src.items():
            self.assertTrue(torch.equal(v, sd[name]), name)
        # fc = R1^T * (fused pre_fc_norm) * block_diag(R1, R1)
        w_emb = sd["mtp.pre_fc_norm_embedding.weight"]
        w_hid = sd["mtp.pre_fc_norm_hidden.weight"]
        fc_fused = sd["mtp.fc.weight"].clone()
        fc_fused[:, :d] = fc_fused[:, :d] * (1 + w_emb).unsqueeze(0)
        fc_fused[:, d:] = fc_fused[:, d:] * (1 + w_hid).unsqueeze(0)
        expected_fc = R1.T @ fc_fused.double() @ torch.block_diag(R1, R1)
        self.assertTrue(torch.allclose(out["mtp.fc.weight"], expected_fc))
        # Input-side q/k/v/gate/up = (fused) * R1
        w_il = sd["mtp.layers.0.input_layernorm.weight"]
        for name in ("q_proj", "k_proj", "v_proj"):
            W = sd[f"mtp.layers.0.self_attn.{name}.weight"] * (1 + w_il).unsqueeze(0)
            self.assertTrue(torch.allclose(
                out[f"mtp.layers.0.self_attn.{name}.weight"], W.double() @ R1), name)
        w_pal = sd["mtp.layers.0.post_attention_layernorm.weight"]
        for name in ("gate_proj", "up_proj"):
            W = sd[f"mtp.layers.0.mlp.{name}.weight"] * (1 + w_pal).unsqueeze(0)
            self.assertTrue(torch.allclose(
                out[f"mtp.layers.0.mlp.{name}.weight"], W.double() @ R1), name)
        # Output-side o/down = R1^T*W (no fusion); loose tolerance for float32
        self.assertTrue(torch.allclose(
            out["mtp.layers.0.self_attn.o_proj.weight"],
            (R1.T @ sd["mtp.layers.0.self_attn.o_proj.weight"].double()).float(),
            atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(
            out["mtp.layers.0.mlp.down_proj.weight"],
            (R1.T @ sd["mtp.layers.0.mlp.down_proj.weight"].double()).float(),
            atol=1e-5, rtol=1e-4))
        # head = (W_head * (1+w_n)) * R1; loose tolerance for float32
        w_n = sd["mtp.norm.weight"]
        head_fused = sd["mtp.lm_head.weight"] * (1 + w_n).unsqueeze(0)
        self.assertTrue(torch.allclose(out["mtp.lm_head.weight"],
                                       (head_fused.double() @ R1).float(),
                                       atol=1e-5, rtol=1e-4))
        # q_norm/k_norm kept raw (head_dim axis, not affected by R1)
        self.assertTrue(torch.equal(out["mtp.layers.0.self_attn.q_norm.weight"],
                                    sd["mtp.layers.0.self_attn.q_norm.weight"]))
        self.assertTrue(torch.equal(out["mtp.layers.0.self_attn.k_norm.weight"],
                                    sd["mtp.layers.0.self_attn.k_norm.weight"]))

    def test_rotate_mtp_file_roundtrip(self):
        import os
        import tempfile
        import safetensors.torch

        with tempfile.TemporaryDirectory() as tmp:
            shard = os.path.join(tmp, "model_mtp.safetensors")
            sd = {
                "mtp.pre_fc_norm_embedding.weight": torch.randn(self.hidden).float(),
                "mtp.pre_fc_norm_hidden.weight": torch.randn(self.hidden).float(),
                "mtp.fc.weight": torch.randn(self.hidden, 2 * self.hidden).float(),
                "mtp.layers.0.input_layernorm.weight": torch.randn(self.hidden).float(),
                "mtp.layers.0.self_attn.q_proj.weight": torch.randn(self.hidden, self.hidden).float(),
                "mtp.layers.0.self_attn.o_proj.weight": torch.randn(self.hidden, self.hidden).float(),
                "mtp.layers.0.self_attn.q_norm.weight": torch.randn(8).float(),
                "mtp.layers.0.post_attention_layernorm.weight": torch.randn(self.hidden).float(),
                "mtp.layers.0.mlp.gate_proj.weight": torch.randn(self.inter, self.hidden).float(),
                "mtp.layers.0.mlp.down_proj.weight": torch.randn(self.hidden, self.inter).float(),
                "mtp.norm.weight": torch.randn(self.hidden).float(),
                "mtp.lm_head.weight": torch.randn(self.vocab, self.hidden).float(),
            }
            safetensors.torch.save_file(sd, shard)
            self.assertTrue(rotate_mtp_file(tmp, self.R1))
            reloaded = safetensors.torch.load_file(shard)
            expected = rotate_mtp_state_dict(sd, self.R1)
            for k, v in expected.items():
                self.assertTrue(torch.equal(reloaded[k], v), k)
            # missing shard returns False without raising
            self.assertFalse(rotate_mtp_file(tmp, self.R1, shard_name="nope.safetensors"))

    def test_add_mtp_lm_head_to_shard(self):
        """Independent lm_head: copy the main W_head from the raw model into the
        MTP shard and update the index."""
        import json
        import os
        import tempfile
        import safetensors.torch
        from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
        from utils.mtp_utils import add_mtp_lm_head_to_shard

        hidden, vocab = self.hidden, self.vocab
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            raw_head = torch.randn(vocab, hidden).float()
            src_state = {
                "lm_head.weight": raw_head.clone(),
                "mtp.fc.weight": torch.randn(hidden, 2 * hidden).float(),
                "mtp.pre_fc_norm_embedding.weight": torch.randn(hidden).float(),
                "mtp.pre_fc_norm_hidden.weight": torch.randn(hidden).float(),
                "mtp.norm.weight": torch.randn(hidden).float(),
            }
            safetensors.torch.save_file(src_state, os.path.join(src, "model.safetensors"))

            # dst preloads a quantized main-model shard with a fake rotated
            # head whose value must differ from the raw head
            fake_head = torch.randn(vocab, hidden).float()
            safetensors.torch.save_file(
                {"lm_head.weight": fake_head.clone()},
                os.path.join(dst, "model.safetensors"),
            )

            save_mtp_tensors_to_checkpoint(source_model=src, dest_dir=dst)
            self.assertTrue(add_mtp_lm_head_to_shard(dst, src))

            shard = os.path.join(dst, "model_mtp.safetensors")
            loaded = safetensors.torch.load_file(shard)
            self.assertIn("mtp.lm_head.weight", loaded)
            # The head copied is the raw W_head, not the fake rotated head in dst
            self.assertTrue(torch.equal(loaded["mtp.lm_head.weight"], raw_head))
            self.assertFalse(torch.equal(loaded["mtp.lm_head.weight"], fake_head))
            # index updated: mtp.lm_head.weight points to the MTP shard
            with open(os.path.join(dst, "model.safetensors.index.json")) as f:
                index = json.load(f)
            self.assertEqual(index["weight_map"]["mtp.lm_head.weight"], "model_mtp.safetensors")
            self.assertEqual(index["weight_map"]["mtp.fc.weight"], "model_mtp.safetensors")
            self.assertEqual(index["weight_map"]["lm_head.weight"], "model.safetensors")

        # source_model without lm_head returns False without raising
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            safetensors.torch.save_file(
                {"mtp.fc.weight": torch.randn(hidden, 2 * hidden).float()},
                os.path.join(src, "model.safetensors"),
            )
            safetensors.torch.save_file(
                {"lm_head.weight": fake_head.clone()},
                os.path.join(dst, "model.safetensors"),
            )
            save_mtp_tensors_to_checkpoint(source_model=src, dest_dir=dst)
            self.assertFalse(add_mtp_lm_head_to_shard(dst, src))


if __name__ == "__main__":
    unittest.main()
