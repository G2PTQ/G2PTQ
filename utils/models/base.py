"""``ModelSpec`` — the per-architecture description ``ModelAnalyzer`` delegates to.

A spec is *stateless with respect to the model*: it is constructed from the (text) config only and
holds no module references, so it can be resolved before the model exists (``load_model`` needs
:meth:`ModelSpec.load_model_class`). Everything model-dependent is passed in as an argument.

Module names in the declarative attributes are **layer-relative** (``"self_attn.q_proj"``,
``"mlp.down_proj"``) and are resolved with :func:`resolve`, which returns ``None`` for a missing
attribute — same lenient walk the analyzer has always used.

Subclasses customize in two ways:

- **class attributes** for the uniform parts (module-name lists, flags, the decoder-layer class);
- **method overrides** where the answer is genuinely per-layer conditional (``qwen3_5``'s
  ``layer_type``, ``glm_moe_dsa``'s dense-vs-sparse MLP, ``gemma4``'s KV-shared layers) or where an
  arch needs a real hook body (logit softcapping, block-internals capture, DSA index buffers).
"""

import contextlib

from transformers import AutoModelForCausalLM
from transformers.conversion_mapping import _MODEL_TO_CONVERSION_PATTERN


def resolve(attr_str, module):
    """Walk a dotted attribute path from ``module``, returning ``None`` if any hop is missing."""
    try:
        for attrib_name in attr_str.split('.'):
            module = getattr(module, attrib_name)
    except AttributeError:
        module = None
    return module


class ModelSpec:
    """Base description of a model architecture's layout and arch-specific behaviour."""

    # ---- Top-level layout -------------------------------------------------------------------
    #: Dotted path from the top-level model to the decoder stack's parent (the module holding
    #: ``embed_tokens`` / ``layers`` / ``norm``). Multimodal wrappers override this.
    MODEL_PREFIX = "model"
    #: Attribute name of the LM head on the top-level model.
    LM_HEAD = "lm_head"
    #: Names under ``MODEL_PREFIX`` that run before the decoder blocks.
    PRE_BLOCK_MODULES = ("embed_tokens", "rotary_emb")

    # ---- Per-layer layout (all names are layer-relative) ------------------------------------
    #: Quantizable attention modules, grouped in sequential order.
    ATTN_GROUPS = (
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ("self_attn.o_proj",),
    )
    #: Quantizable MLP modules, grouped in sequential order.
    MLP_GROUPS = (
        ("mlp.up_proj", "mlp.gate_proj"),
        ("mlp.down_proj",),
    )
    #: Modules consuming the layer's two residual-stream inputs (post-LN), for LN fusion/rotation.
    ATTN_INPUTS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
    MLP_INPUTS = ("mlp.up_proj", "mlp.gate_proj")
    #: Modules writing back into the residual stream, for output-side rotation.
    ATTN_OUTPUTS = ("self_attn.o_proj",)
    MLP_OUTPUTS = ("mlp.down_proj",)
    #: Quantizable modules that receive the *identical* input tensor at forward-hook time, so a
    #: consumer capturing those inputs can store one copy per group instead of one per module.
    ATTN_SHARED_INPUTS = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),)
    MLP_SHARED_INPUTS = (("mlp.up_proj", "mlp.gate_proj"),)
    #: The MLP down-projection(s), for the online Hadamard on the MLP's inner dimension.
    DOWN_PROJ = ("mlp.down_proj",)
    O_PROJ = "self_attn.o_proj"
    V_PROJ = "self_attn.v_proj"
    #: LayerNorms fused into ``ATTN_INPUTS`` / ``MLP_INPUTS`` respectively (order must match).
    LAYERNORMS = ("input_layernorm", "post_attention_layernorm")
    #: True when the norm weight is stored as an offset from 1 (``(1 + w) * x``).
    LN_ZERO_CENTERED = False
    #: Last module before the layer's final residual add.
    MODULE_BEFORE_FINAL_RESIDUAL = "mlp"

    # ---- Capabilities ------------------------------------------------------------------------
    IS_MOE = False
    IS_MULTIMODAL = False
    USES_DSA_INDEXSHARE = False
    #: Whether the o/v projections can take the head-dim (R2) rotation.
    CAN_ROTATE_OV = False
    #: Whether LN fusion + Hadamard rotation (``--rotate``) is validated for this arch. Set False
    #: for archs with extra un-fused norms in the residual path, where rotating would silently
    #: change the computation.
    CAN_ROTATE_GLOBAL = True
    #: Whether flash attention is validated for this arch. Set False for archs whose transformers
    #: model class sets ``_supports_flash_attn = False``.
    SUPPORTS_FLASH_ATTN = True
    #: Whether KV-cache quantization (``--k_bits`` *or* ``--v_bits`` < 16) is available. Set False
    #: for archs.
    SUPPORTS_KV_QUANT = True
    #: Decoder layer class name, used as accelerate's ``no_split_module_classes``.
    DECODER_LAYER_CLASS = None
    #: ``model_type`` whose checkpoint conversion pattern this arch borrows, enabling direct 2D
    #: expert weight loading for MoE archs transformers has no 2D mapping of its own for. ``None``
    #: leaves the arch's own (or absent) mapping alone. See :meth:`register_conversion_mappings`.
    CONVERSION_PATTERN = None
    #: Parameter/module paths that must remain frozen when gradient computation is enabled.
    #: Entries are matched as dotted path prefixes (and suffixes for full-model names), so
    #: architecture-specific declarations can be reused for both a complete model and an
    #: individual decoder layer.
    NO_GRAD_PARAM_PREFIXES = ()

    def __init__(self, config):
        #: The text config (``config.text_config`` when the model is a multimodal wrapper).
        self.config = config

    @property
    def residual_width(self):
        """Channel width of the tensor passed *between* decoder blocks.

        Equal to ``hidden_size`` for every conventional arch. Archs carrying several parallel
        residual streams (``qwen4_exp``'s gated residual) widen it, and the block-wise drivers
        size their inter-layer buffers from this rather than from ``hidden_size``.
        """
        return self.config.hidden_size

    # ---- Loading hooks ---------------------------------------------------------------------
    @classmethod
    def load_model_class(cls, config):
        """Return the class to load the checkpoint with."""
        return AutoModelForCausalLM

    @classmethod
    def register_conversion_mappings(cls, config):
        """Point transformers' checkpoint conversion at :attr:`CONVERSION_PATTERN`.

        Called by ``load_model`` before the checkpoint is read. ``_MODEL_TO_CONVERSION_PATTERN``
        is what ``llmcompressor``'s ``has_linearize_load_mappings`` /
        ``get_linearize_load_mappings`` consult to find an arch's 2D expert mapping, so borrowing
        another arch's pattern is what lets a MoE checkpoint load its experts directly as 2D
        per-expert tensors instead of fused 3D ones.
        """
        if cls.CONVERSION_PATTERN is not None:
            _MODEL_TO_CONVERSION_PATTERN[config.model_type] = cls.CONVERSION_PATTERN

    @classmethod
    def post_load(cls, model, config):
        """Post-process a freshly loaded model (e.g. unwrap a text model). Returns the model."""
        return model

    @contextlib.contextmanager
    def _enter_quantization_context(self, args, model):
        """Temporarily apply architecture-specific state for the weight-quantization pass."""
        yield

    def embed_tokens_path(self):
        """Dotted path (from the top-level model) to the input embedding."""
        return f"{self.MODEL_PREFIX}.embed_tokens"

    # ---- Top-level accessors ---------------------------------------------------------------
    def get_lm_head(self, model):
        return resolve(self.LM_HEAD, model)

    def get_embed_layer(self, model):
        return resolve(self.embed_tokens_path(), model)

    def get_layernorm_before_head(self, model):
        return resolve(f"{self.MODEL_PREFIX}.norm", model)

    def get_layers(self, model):
        return resolve(f"{self.MODEL_PREFIX}.layers", model)

    def get_pre_block_modules(self, model):
        return [resolve(f"{self.MODEL_PREFIX}.{name}", model) for name in self.PRE_BLOCK_MODULES]

    # ---- Per-layer accessors ---------------------------------------------------------------
    def get_attn_groups(self, layer):
        """Quantizable attention module names, grouped in sequential order."""
        return [list(group) for group in self.ATTN_GROUPS]

    def get_mlp_groups(self, layer):
        """Quantizable MLP module names, grouped in sequential order."""
        return [list(group) for group in self.MLP_GROUPS]

    def get_attn_inputs(self, layer):
        return [resolve(name, layer) for name in self.ATTN_INPUTS]

    def get_mlp_inputs(self, layer):
        return [resolve(name, layer) for name in self.MLP_INPUTS]

    def get_attn_outputs(self, layer):
        return [resolve(name, layer) for name in self.ATTN_OUTPUTS]

    def get_mlp_outputs(self, layer):
        return [resolve(name, layer) for name in self.MLP_OUTPUTS]

    def get_down_proj(self, layer):
        return [resolve(name, layer) for name in self.DOWN_PROJ]

    def get_kv_attn_module(self, layer):
        """Attention module to install the K-cache quantizer on, or ``None`` to skip this layer.
        """
        return resolve("self_attn", layer)

    def get_o_proj(self, layer):
        return resolve(self.O_PROJ, layer)

    def get_v_proj(self, layer):
        return resolve(self.V_PROJ, layer)

    def get_layernorms(self, layer):
        """Return ``[(LN module, is_zero_centered), ...]``, aligned with the input-module groups."""
        return [(resolve(name, layer), self.LN_ZERO_CENTERED) for name in self.LAYERNORMS]

    def get_shared_input_groups(self, layer):
        """Return a full partition of the layer's quantizable names into shared-input groups.

        Each tuple identifies modules that receive the *identical* input tensor at the forward-hook
        capture point (so a consumer can allocate one buffer per group instead of one per module).
        """
        attn_groups = self._shared_input_groups_attn(layer)
        mlp_groups = self._shared_input_groups_mlp(layer)
        declared_names = {name for group in attn_groups + mlp_groups for name in group}

        all_names = {
            name
            for group in self.get_attn_groups(layer) + self.get_mlp_groups(layer)
            for name in group
        }
        singletons = [(name,) for name in all_names if name not in declared_names]
        return attn_groups + mlp_groups + singletons

    def _shared_input_groups_attn(self, layer):
        """Attention-side shared-input groups; override for per-layer conditionals."""
        return [tuple(group) for group in self.ATTN_SHARED_INPUTS]

    def _shared_input_groups_mlp(self, layer):
        """MLP-side shared-input groups; override for MoE per-expert pairs."""
        return [tuple(group) for group in self.MLP_SHARED_INPUTS]

    def get_module_before_final_residual(self, layer):
        return resolve(self.MODULE_BEFORE_FINAL_RESIDUAL, layer)

    def get_layer_scalar(self, layer):
        """Scalar the layer multiplies the hidden states by after the final residual."""
        return 1.0

    # ---- MoE hooks (see MoEMixin) ----------------------------------------------------------
    def get_experts(self, layer):
        """Return the layer's experts container, or ``None`` when the layer's MLP is dense."""
        return None

    def get_gate(self, layer):
        """Return the layer's MoE router, or ``None`` when the layer's MLP is dense."""
        return None

    def patch_experts_forward(self, experts):
        """Install the forward implementation for an experts container with 2D weights."""
        raise NotImplementedError(f"{type(self).__name__} has no experts forward patch")

    def stack_experts(self, experts, weight_packed=False):
        raise NotImplementedError(f"{type(self).__name__} has no expert stacking")

    def wrap_gate_forward(self, gate_module, func):
        """Patch ``gate_module.forward`` so ``func`` can observe/replace the router logits."""
        raise NotImplementedError(f"{type(self).__name__} has no router patch")

    # ---- Forward hooks ---------------------------------------------------------------------
    def post_process_logits(self, logits):
        """Apply any arch-specific transform to the LM head's output."""
        return logits

    @contextlib.contextmanager
    def capture_block_internals(self, model):
        """Capture per-layer forward inputs that block-wise execution must reproduce by hand.

        Yields an opaque object handed back to :meth:`build_block_kwargs` /
        :meth:`finalize_block`, or ``None`` when the arch needs nothing.
        """
        yield None

    def alloc_index_buffer(self, nsamples, seqlen, device):
        """Allocate the cross-layer index buffer written by :meth:`finalize_block`.

        Returns ``None`` when the arch shares nothing across layers. The caller owns the buffer
        (rather than it living inside ``internals``) because it is prefetched batch-wise next to
        the hidden states — see ``offload_utils.prefetch_generator``.
        """
        return None

    def build_block_kwargs(self, internals, kwargs, *, prev_topk_indices,
                           layer_idx, sample_idx, bsz, dev):
        """Add arch-specific entries to a decoder layer's forward kwargs, in place.

        Runs immediately before the layer call. ``internals`` is whatever
        :meth:`capture_block_internals` yielded (``None`` if nothing), and ``prev_topk_indices``
        is this batch's slice of the index buffer (``None`` on the first layer or when unused).
        """
        pass

    def finalize_block(self, internals, index_buffer, extras, *, layer_idx, sample_idx, bsz):
        """Consume the layer's outputs and release its per-batch state.

        Runs immediately after the layer call. ``extras`` holds the layer's return values past
        the hidden states (GLM-5.2's freshly computed top-k indices); write anything later layers
        need into ``index_buffer``, which is ``None`` on read-only passes.
        """
        pass
