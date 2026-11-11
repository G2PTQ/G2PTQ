"""MTP rotation compensation with an independent lm_head.

Background: after G2PTQ applies a global rotation R1 to the main model (see
utils/rotation_utils.py), the main model's token embeddings and the final
residual stream are in the rotated coordinate frame, while the MTP
(Multi-Token Prediction) weights are not rotated. The MTP inputs (token
embeddings, main model hidden states) therefore mismatch the MTP weights and
the MTP output becomes wrong.

Approach: rotate every MTP layer consistently with the main model so the
whole MTP runs in the rotated coordinate frame and the recursion chain stays
coordinate-consistent. (Absorbing the rotation only on the fc input side
leaves the fc output in the original frame, so the second recursion step sees
a coordinate jump in the hidden segment.)

  - pre_fc_norm_embedding / pre_fc_norm_hidden (inputs in the rotated frame):
       fused column-wise into the corresponding fc columns and zeroed. A
       non-identity diagonal norm weight (1+w) does not commute with the
       dense R1 and cannot be evaluated in the rotated frame, so it must be
       absorbed first.
  - fc (fusion layer): W <- R1^T * fuse(fc) * block_diag(R1, R1), a similarity
       transform. The input cat([emb*R1, hid*R1]) is brought back to the
       original frame via block_diag(R1, R1), computed there, then written
       back to the rotated frame.
  - input_layernorm / post_attention_layernorm: fused column-wise into q/k/v
       and gate/up respectively, then zeroed.
  - q/k/v/gate/up (input-side projections): W <- W*R1 (absorb the input
       rotation; q/k outputs stay in the original frame to match the main
       model KV cache).
  - o/down (output-side projections): W <- R1^T*W (write back into the
       rotated residual stream).
  - q_norm / k_norm: operate on the head_dim axis, unaffected by R1, kept as-is.
  - mtp.norm (final norm): fused column-wise into the independent mtp.lm_head
       and zeroed.
  - mtp.lm_head (independent head): W <- W*R1 after fusing mtp.norm, i.e.
       W_head * diag(1+w_n) * R1: rotated hidden -> original-frame logits,
       consistent with the main model lm_head.

PyTorch Linear forward is y = x @ W.T, which the transforms below follow.
"""

import logging
import os
import tempfile

import torch
import safetensors.torch


_INPUT_PROJ_SUFFIXES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_OUTPUT_PROJ_SUFFIXES = ("o_proj", "down_proj")


def fuse_mtp_norms(mtp_state_dict: dict, zero_centered: bool = True) -> dict:
    """Fuse every norm on the rotated-coordinate input side into its downstream layers.

    With the MTP running in the rotated coordinate frame, each norm's input
    (fc output, layer residual stream, head input) is in the rotated frame;
    its diagonal scale (1+w) does not commute with the dense R1 and cannot be
    evaluated there, so it must be absorbed column-wise into the downstream
    linear layers and the norm zeroed:

      - pre_fc_norm_embedding / pre_fc_norm_hidden -> fc left/right halves;
      - input_layernorm -> q/k/v (absorb into all downstream first, then zero);
      - post_attention_layernorm -> gate/up (absorb into all downstream first,
        then zero);
      - mtp.norm -> independent mtp.lm_head (column-wise absorption).

    When one norm feeds several linear layers (e.g. input_layernorm into
    q/k/v), all downstream layers must be scaled with the raw norm weight
    before the norm is zeroed, otherwise the fused scale is lost.

    The absorption rule matches fuse_ln_linears in utils/rotation_utils.py.

    Args:
        mtp_state_dict: MTP weight dict.
        zero_centered: whether the norms are zero-centered (x * (1+w)); if
            True the fuse scale is 1+w and the norm is reset to 0, else it is
            w and the norm is reset to 1.

    Returns:
        A new dict with the fused weights (the input dict is not modified).
    """
    sd = {k: v.clone() for k, v in mtp_state_dict.items()}

    # Accept keys with or without the "mtp." prefix (real checkpoints use
    # mtp.fc.weight, an nn.ModuleDict's state_dict() uses fc.weight).
    prefix = "mtp." if any(k.startswith("mtp.") for k in sd) else ""

    def scale(w):
        return (1.0 + w) if zero_centered else w

    def reset(w):
        return torch.zeros_like(w) if zero_centered else torch.ones_like(w)

    def absorb(linear_key, norm_key, col_slice=None):
        if linear_key not in sd or norm_key not in sd:
            return
        t, n = sd[linear_key], sd[norm_key]
        cols = t.shape[1] if col_slice is None else (col_slice.stop - col_slice.start)
        if t.ndim != 2 or n.numel() != cols:
            return
        s = scale(n).to(t.dtype).unsqueeze(0)
        if col_slice is None:
            sd[linear_key] = t.clone() * s
        else:
            new_t = t.clone()
            new_t[:, col_slice] = new_t[:, col_slice] * s
            sd[linear_key] = new_t

    # 1) pre_fc_norm -> fc columns (inputs are in the rotated frame)
    fc_key = f"{prefix}fc.weight"
    if fc_key in sd and sd[fc_key].ndim == 2:
        hidden = sd[fc_key].shape[1] // 2
        absorb(fc_key, f"{prefix}pre_fc_norm_embedding.weight", slice(0, hidden))
        absorb(fc_key, f"{prefix}pre_fc_norm_hidden.weight", slice(hidden, 2 * hidden))
        for k in (f"{prefix}pre_fc_norm_embedding.weight",
                  f"{prefix}pre_fc_norm_hidden.weight"):
            if k in sd:
                sd[k] = reset(sd[k])

    # 2) input_layernorm -> q/k/v (absorb into all first, then zero)
    ln_key = f"{prefix}layers.0.input_layernorm.weight"
    for proj in ("q_proj", "k_proj", "v_proj"):
        absorb(f"{prefix}layers.0.self_attn.{proj}.weight", ln_key)
    if ln_key in sd:
        sd[ln_key] = reset(sd[ln_key])

    # 3) post_attention_layernorm -> gate/up (absorb into all first, then zero)
    ln_key = f"{prefix}layers.0.post_attention_layernorm.weight"
    for proj in ("gate_proj", "up_proj"):
        absorb(f"{prefix}layers.0.mlp.{proj}.weight", ln_key)
    if ln_key in sd:
        sd[ln_key] = reset(sd[ln_key])

    # 4) mtp.norm -> independent head (needs mtp.lm_head to be present)
    head_key = f"{prefix}lm_head.weight"
    norm_key = f"{prefix}norm.weight"
    if head_key in sd and norm_key in sd:
        absorb(head_key, norm_key)
        sd[norm_key] = reset(sd[norm_key])
    elif norm_key in sd:
        logging.warning(
            "No independent mtp.lm_head.weight found, final "
            "mtp.norm cannot be fused; MTP logits will be inexact."
        )

    return sd


def rotate_mtp_state_dict(
    mtp_state_dict: dict,
    R1: torch.Tensor,
    fuse_norms: bool = True,
    zero_centered_norms: bool = True,
) -> dict:
    """Apply the MTP rotation transforms (per-layer rotation + independent lm_head).

    Transforms:
      - norms (pre_fc_norm, input/post_attention_layernorm, mtp.norm): fused
        column-wise into their downstream layers and zeroed (see fuse_mtp_norms);
      - fc: W <- R1^T * W * block_diag(R1, R1) (similarity transform, input and
        output both in the rotated frame);
      - q/k/v/gate/up (input side): W <- W*R1 (absorb the input rotation);
      - o/down (output side): W <- R1^T*W (write back to the rotated frame);
      - mtp.lm_head (independent head): W <- W*R1, after fusing mtp.norm
        (combined W_head * diag(1+w_n) * R1);
      - q_norm / k_norm (head_dim axis) and remaining weights: kept as-is.

    The MTP recursion input (previous step output) is always in the rotated
    frame, matching the fc input-side block_diag(R1, R1), so the recursion
    chain stays coordinate-consistent.

    Args:
        mtp_state_dict: MTP weight dict (name -> tensor).
        R1: main model rotation matrix of shape [hidden, hidden].
        fuse_norms: whether to fuse the rotated-frame norms first (required
            for correctness unless the norms are identity).
        zero_centered_norms: whether the norms are zero-centered (x*(1+w)).

    Returns:
        The rotated state dict (the input dict is not modified).
    """
    R1 = R1.detach().cpu().to(torch.float64)
    if fuse_norms:
        mtp_state_dict = fuse_mtp_norms(mtp_state_dict, zero_centered=zero_centered_norms)

    sd = {k: v.clone() for k, v in mtp_state_dict.items()}
    prefix = "mtp." if any(k.startswith("mtp.") for k in sd) else ""
    hidden = R1.shape[0]

    # fc: similarity transform R1^T * W * block_diag(R1, R1)
    fc_key = f"{prefix}fc.weight"
    if fc_key in sd:
        t = sd[fc_key]
        if t.ndim == 2 and t.shape[1] == 2 * hidden:
            sd[fc_key] = torch.matmul(
                torch.matmul(R1.T, t.double()), torch.block_diag(R1, R1)
            ).to(t.dtype)
        elif t.ndim == 2 and t.shape[1] == hidden:
            sd[fc_key] = torch.matmul(
                torch.matmul(R1.T, t.double()), R1
            ).to(t.dtype)
        else:
            in_dim = t.shape[1] if t.ndim == 2 else None
            logging.warning(
                f"MTP fc in_dim={in_dim} not matching hidden={hidden}, "
                f"skip similarity rotation for shape {tuple(t.shape)}."
            )

    # Independent head: input absorption W*R1 (rotated hidden -> original logits)
    head_key = f"{prefix}lm_head.weight"
    if head_key in sd:
        t = sd[head_key]
        if t.ndim == 2 and t.shape[1] == hidden:
            sd[head_key] = torch.matmul(t.double(), R1).to(t.dtype)

    # Input-side projections (q/k/v/gate/up): W*R1; output-side (o/down): R1^T*W
    for name, t in list(sd.items()):
        if not name.endswith(".weight") or t.ndim != 2:
            continue
        stem = name[len(prefix):] if name.startswith(prefix) else name
        proj = stem[:-len(".weight")].rsplit(".", 1)[-1]
        if proj in _INPUT_PROJ_SUFFIXES and t.shape[1] == hidden:
            sd[name] = torch.matmul(t.double(), R1).to(t.dtype)
        elif proj in _OUTPUT_PROJ_SUFFIXES and t.shape[0] == hidden:
            sd[name] = torch.matmul(R1.T, t.double()).to(t.dtype)

    return sd


def rotate_mtp_file(
    save_path: str,
    R1: torch.Tensor,
    shard_name: str = "model_mtp.safetensors",
    **kwargs,
) -> bool:
    """Apply the MTP rotation transforms in-place to the exported MTP shard.

    The weight names, shapes and structure seen by the inference engine stay
    unchanged; only the values are transformed (norm column-wise fusion, fc
    similarity transform, input/output projection rotation, independent head
    input absorption), so the engine is not affected.

    Args:
        save_path: compressed model export directory.
        R1: main model rotation matrix.
        shard_name: MTP shard file name, default model_mtp.safetensors.
        **kwargs: forwarded to rotate_mtp_state_dict.

    Returns:
        Whether the transform was applied (False when the shard is missing).
    """
    shard_path = os.path.join(save_path, shard_name)
    if not os.path.exists(shard_path):
        logging.warning(f"MTP shard {shard_path} not found, skip rotation.")
        return False

    state_dict = safetensors.torch.load_file(shard_path)
    rotated = rotate_mtp_state_dict(state_dict, R1, **kwargs)

    fd, tmp_path = tempfile.mkstemp(
        dir=save_path, prefix=".model_mtp_tmp_", suffix=".safetensors"
    )
    os.close(fd)
    try:
        safetensors.torch.save_file(rotated, tmp_path)
        os.replace(tmp_path, shard_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    logging.info(
        f"Rotated {len(rotated)} MTP tensors in {shard_path} "
        f"(similarity rotation + norm fusion + independent lm_head)."
    )
    return True


_LM_HEAD_KEY_CANDIDATES = (
    "lm_head.weight",
    "model.language_model.lm_head.weight",
)


def add_mtp_lm_head_to_shard(
    save_path: str,
    source_model: str,
    shard_name: str = "model_mtp.safetensors",
) -> bool:
    """Copy the main model's W_head into the MTP shard as an independent mtp.lm_head.

    In the raw checkpoint the MTP lm_head is tied to the main model's lm_head,
    so the state dict has no separate mtp.lm_head.weight and it must be
    constructed at export time: read the main model's lm_head.weight from
    source_model (the unquantized raw model), append it to the MTP shard under
    the mtp.lm_head.weight key, and update the safetensors index and total_size.

    The raw W_head is written unchanged; its transform (fuse mtp.norm and
    absorb R1, i.e. W_head * diag(1+w_n) * R1) is applied later by
    rotate_mtp_file, so this function must run before rotate_mtp_file.

    Args:
        save_path: compressed model export directory.
        source_model: raw (unquantized) model directory or single safetensors file.
        shard_name: MTP shard file name, default model_mtp.safetensors.

    Returns:
        Whether the head was written (False when the shard is missing or no
        lm_head is found).
    """
    from safetensors import safe_open
    from compressed_tensors.utils import (
        get_weight_mappings,
        update_safetensors_index,
    )
    import json

    shard_path = os.path.join(save_path, shard_name)
    if not os.path.exists(shard_path):
        logging.warning(f"MTP shard {shard_path} not found, skip lm_head copy.")
        return False

    src_mapping = get_weight_mappings(source_model)
    lm_head_key = next(
        (k for k in _LM_HEAD_KEY_CANDIDATES if k in src_mapping), None
    )
    if lm_head_key is None:
        logging.warning(
            f"No lm_head.weight found in {source_model}, skip MTP lm_head copy."
        )
        return False
    with safe_open(src_mapping[lm_head_key], framework="pt", device="cpu") as f:
        lm_head = f.get_tensor(lm_head_key)

    state_dict = safetensors.torch.load_file(shard_path)
    state_dict["mtp.lm_head.weight"] = lm_head

    fd, tmp_path = tempfile.mkstemp(
        dir=save_path, prefix=".model_mtp_tmp_", suffix=".safetensors"
    )
    os.close(fd)
    try:
        safetensors.torch.save_file(state_dict, tmp_path)
        os.replace(tmp_path, shard_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # Update the safetensors index by reading the existing index directly
    # (save_mtp_tensors_to_checkpoint has just updated it and holds the most
    # complete mapping), avoiding get_weight_mappings which prefers the single
    # model.safetensors branch for a directory and drops the MTP keys.
    index_path = os.path.join(save_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {
            k: os.path.basename(v)
            for k, v in get_weight_mappings(save_path).items()
        }
    weight_map["mtp.lm_head.weight"] = shard_name
    total_size = sum(
        os.path.getsize(os.path.join(save_path, s))
        for s in set(weight_map.values())
    )
    update_safetensors_index(save_path, total_size, weight_map)

    logging.info(
        f"Added independent mtp.lm_head.weight (raw W_head) to {shard_path}."
    )
    return True
