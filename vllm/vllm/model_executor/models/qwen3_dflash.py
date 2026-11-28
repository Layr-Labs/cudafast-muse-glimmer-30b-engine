# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.transformers_utils.repo_utils import get_hf_file_bytes
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


_SLIDING_ATTENTION = "sliding_attention"


def _dflash_layer_causal(config: Qwen3Config, layer_idx: int) -> bool:
    """``dflash_config.causal`` overrides all layers; else only SWA layers causal."""
    override = (getattr(config, "dflash_config", None) or {}).get("causal")
    if override is not None:
        return override
    if os.environ.get("MUSE_DFLASH_BIDIR", "1") != "0":
        archs = getattr(config, "architectures", None) or ()
        if any("MuseGlimmerAssistant" in a for a in archs):
            return False
    layer_types = getattr(config, "layer_types", None)
    return bool(layer_types) and layer_types[layer_idx] == _SLIDING_ATTENTION


def dflash_has_any_non_causal(config: Qwen3Config) -> bool:
    """Whether the draft needs a non-causal-capable backend, resolved from config
    (config mirror of the model's ``get_draft_attn_causal``, usable pre-build)."""
    return not all(
        _dflash_layer_causal(config, i) for i in range(config.num_hidden_layers)
    )


def dflash_query_len(config: Qwen3Config | None, num_spec_tokens: int) -> int:
    """Resolve bidirectional DFlash query block length (VLLM_DFLASH_QUERY_LEN).

    Default 16 runs one full m16n8k16 MMA tile (1 anchor + num_spec_tokens sampled
    masks + trailing unsampled context masks matching DFlash's block-16 training
    distribution). Set VLLM_DFLASH_QUERY_LEN=7 (or 0) to restore stock 1 + N.
    """
    base_q = 1 + max(0, int(num_spec_tokens))
    if config is None or num_spec_tokens <= 0:
        return base_q
    raw = os.environ.get("VLLM_DFLASH_QUERY_LEN", "16").strip()
    try:
        wanted = int(raw)
    except ValueError:
        return base_q
    if wanted <= base_q:
        return base_q
    dflash_config = getattr(config, "dflash_config", None) or {}
    if dflash_config.get("sample_from_anchor", False):
        return base_q
    if not dflash_has_any_non_causal(config):
        return base_q
    return min(wanted, 16)


def _get_dflash_fc_input_size(vllm_config: VllmConfig) -> int:
    spec_config = vllm_config.speculative_config
    config = spec_config.draft_model_config.hf_config
    aux_layers = get_eagle3_aux_layers_from_config(spec_config)
    num_features_to_use = len(aux_layers) if aux_layers else config.num_hidden_layers
    target_hidden_size = (
        getattr(config, "target_hidden_size", None) or config.hidden_size
    )
    return target_hidden_size * num_features_to_use


def _resolve_layer_attention(
    config: Qwen3Config, layer_idx: int
) -> tuple[int | None, bool]:
    """Resolve ``(sliding_window, causal)`` for one DFlash draft layer.

    +----------------------+-------------------------+--------------------------------+
    | Config               | ``layer_type``          | *``causal``                    |
    +======================+=========================+================================+
    | ``layer_types``      | SWA if ``use_swa``      | True if ``layer_types[i]=SWA`` |
    |                      | else ``layer_types[i]`` | else False                     |
    +----------------------+-------------------------+--------------------------------+
    | ``layer_types=None`` | SWA                     | False                          |
    | + ``use_swa=True``   |                         |                                |
    +----------------------+-------------------------+--------------------------------+
    | ``layer_types=None`` | Full                    | False                          |
    | + ``use_swa=False``  |                         |                                |
    +----------------------+-------------------------+--------------------------------+
    * If ``dflash_config.causal`` is set, its value overrides ``causal`` for all layers.

    This is to support a varied ecosystem of checkpoints, including:
    - XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash (sets "use_swa", assumes non-causal)
    - z-lab/gemma-4-31B-it-DFlash (has mixed layer types, assumes causal only for SWA)
    - z-lab/Qwen3.5-9B-DFlash ("standard" DFlash, all full attn, assumes non-causal)
    """
    dflash_config = getattr(config, "dflash_config", None) or {}
    layer_types = getattr(config, "layer_types", None)
    use_swa = dflash_config.get("use_swa", False)

    any_sliding = False
    if layer_types is not None:
        num_sliding = sum(lt == _SLIDING_ATTENTION for lt in layer_types)
        any_sliding = num_sliding > 0
        # Mixed sliding/full attention needs multiple KV groups (V2 runner only).
        if (
            0 < num_sliding < len(layer_types)
            and not get_current_vllm_config().use_v2_model_runner
        ):
            raise NotImplementedError(
                "DFlash drafters with mixed sliding/full attention require "
                "the V2 model runner; relaunch with "
                "VLLM_USE_V2_MODEL_RUNNER=1."
            )

    # ``use_swa`` forces SWA on every layer, even an all-full ``layer_types``.
    if layer_types is None or (use_swa and not any_sliding):
        is_sliding = use_swa
    else:
        is_sliding = layer_types[layer_idx] == _SLIDING_ATTENTION

    sliding_window = None
    if is_sliding:
        sliding_window = dflash_config.get(
            "swa_window_size", getattr(config, "sliding_window", None)
        )
        if sliding_window is None:
            raise ValueError(
                "DFlash sliding attention requires a window size configured in "
                "dflash_config.swa_window_size or the top-level sliding_window."
            )

    return sliding_window, _dflash_layer_causal(config, layer_idx)


class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        sliding_window: int | None = None,
        causal: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # DFlash has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        self.sliding_window = sliding_window
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            sinks=self.attention_sink_bias,
        )
        self.causal = causal
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        tail_slice_m6: bool = False,
    ) -> torch.Tensor:
        """DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv"""
        qkv, _ = self.qkv_proj(hidden_states)
        if (
            getattr(self, "_dflash_fuse_qkv_ok", False)
            and qkv.dtype == torch.bfloat16
            and qkv.dim() == 2
            and positions.dtype == torch.int64
        ):
            q, k = torch.ops.muse.dflash_qkv_norm_rope(
                qkv,
                positions,
                self.q_norm.weight,
                self.k_norm.weight,
                self._dflash_rope_cos_sin_cache,
                self.num_heads,
                self.num_kv_heads,
                float(self.q_norm.variance_epsilon),
                bool(self.rotary_emb.is_neox_style),
                self.attn.layer_name if getattr(self, "_dflash_qkv_cache_ok", False) else "",
            )
            v = qkv[..., self.q_size + self.kv_size :]
            attn_output = self.attn(q, k, v)
            if tail_slice_m6:
                attn_output = attn_output[1:7]
            output, _ = self.o_proj(attn_output)
            return output
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        if tail_slice_m6:
            attn_output = attn_output[1:7]
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER

        # DFlash drafts store the sink-bias flag inside dflash_config; fall back
        # to the top-level attribute used by other (e.g. MiMo) configs.
        dflash_config = getattr(config, "dflash_config", None) or {}
        add_swa_attention_sink_bias = dflash_config.get(
            "attention_sink_bias",
            getattr(config, "add_swa_attention_sink_bias", False),
        )

        # Resolve this layer's attention mode (full vs sliding window, causal vs
        # non-causal) from the draft config.
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            add_swa_attention_sink_bias=add_swa_attention_sink_bias,
            sliding_window=sliding_window,
            causal=causal,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        tail_slice_m6: bool = False,
        skip_input_norm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_fuse_norm = (
            getattr(self, "_dflash_fuse_norm_ok", False)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[-1] <= 8192
            and hidden_states.shape[-1] % 8 == 0
        )
        if not skip_input_norm:
            if residual is not None:
                if use_fuse_norm:
                    hidden_states, residual = torch.ops.muse.dflash_fused_add_rms_norm(
                        hidden_states,
                        residual,
                        self.input_layernorm.weight,
                        float(self.input_layernorm.variance_epsilon),
                    )
                else:
                    hidden_states, residual = self.input_layernorm(hidden_states, residual)
            else:
                residual = hidden_states
                if use_fuse_norm:
                    hidden_states = torch.ops.muse.dflash_rms_norm(
                        hidden_states,
                        self.input_layernorm.weight,
                        float(self.input_layernorm.variance_epsilon),
                    )
                else:
                    hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            tail_slice_m6=tail_slice_m6,
        )
        if tail_slice_m6:
            residual = residual[1:7]

        if use_fuse_norm:
            hidden_states, residual = torch.ops.muse.dflash_fused_add_rms_norm(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight,
                float(self.post_attention_layernorm.variance_epsilon),
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        if getattr(self.mlp, "_mi3_swiglu", None) is not None:
            from vllm.model_executor.layers import muse_draft_fp8

            inter = muse_draft_fp8.mlp_swiglu_apply(self.mlp, hidden_states)
            hidden_states, _ = self.mlp.down_proj(inter)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlashQwen3Model(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={"midlayer.": "layers.0."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Masked query slots are fed to the draft as `mask_token_id`. Most DFlash
        # checkpoints will have the mask embedding in the vocabulary embedding table
        # at that slot id. Some checkpoints (XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash) ship
        # with a separate mask embedding tensor to use instead. When present, we load it
        # and substitute it for embed_tokens[mask_token_id] when computing embeddings.
        self.mask_token_id = drafter_config.get("mask_token_id")
        self.mask_embedding = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False

        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            self.fc = ReplicatedLinear(
                input_size=_get_dflash_fc_input_size(
                    vllm_config,
                ),
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            # Replace masked slots with the dedicated mask embedding.
            is_mask = (input_ids == self.mask_token_id).unsqueeze(-1)
            embeds = torch.where(is_mask, self.mask_embedding.to(embeds.dtype), embeds)
        return embeds

    def _build_context_kv_buffers(
        self,
        layers_attn: list[nn.Module],
        has_bias: bool,
    ) -> None:
        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights stacked into one contiguous [num_layers, head_dim]
        # tensor so the per-layer K-norm runs as a single grouped kernel.
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._build_context_kv_buffers(layers_attn, has_bias)

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def _project_context_kv_flat(
        self,
        context_states: torch.Tensor,
    ) -> torch.Tensor:
        # --- Fused KV projection (one GEMM for all layers) ---
        if (
            getattr(self, "_dflash_fuse_norm_ok", False)
            and os.environ.get("MUSE_DFLASH_FUSE_NORM", "1") != "0"
            and context_states.is_cuda
            and context_states.dtype == torch.bfloat16
            and context_states.dim() == 2
            and context_states.shape[-1] <= 8192
            and context_states.shape[-1] % 8 == 0
        ):
            normed_context_states = torch.ops.muse.dflash_rms_norm(
                context_states,
                self._hidden_norm_weight,
                float(self._rms_norm_eps),
            )
        else:
            normed_context_states = torch.empty_like(context_states)
            ops.rms_norm(
                normed_context_states,
                context_states,
                self._hidden_norm_weight,
                self._rms_norm_eps,
            )
        if self._fused_kv_bias is None:
            all_kv_flat = None
            if (
                getattr(self, "_fkv_mi4", None) is not None
                or getattr(self, "_fkv_mi3", None) is not None
                or getattr(self, "_fkv_mnv4", None) is not None
                or getattr(self, "_fkv_mfp8", None) is not None
            ):
                from vllm.model_executor.layers import muse_draft_fp8
                all_kv_flat = muse_draft_fp8.tensor_apply(self, "fkv", normed_context_states, self._fused_kv_weight)
            if all_kv_flat is None:
                from vllm.model_executor.layers import muse_bfx
                all_kv_flat = muse_bfx.tensor_apply(self, "fkv", normed_context_states, self._fused_kv_weight)
        else:
            all_kv_flat = F.linear(
                normed_context_states, self._fused_kv_weight, self._fused_kv_bias
            )
        return all_kv_flat

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_kv_flat = self._project_context_kv_flat(context_states)
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous
        return all_k, all_v

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        # --- Grouped RMSNorm K across all layers ([L, num_ctx, nkv, hd]) ---
        # The weight is selected per layer by the outermost (layer) index.
        all_k_normed = torch.empty_like(all_k)
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # Fused 1-kernel context-KV post-GEMM (replaces permute + k_norm + pos.repeat + RoPE + 5x cache scatter).
        if (
            context_slot_mapping is not None
            and getattr(self, "_dflash_fuse_kv_ok", False)
            and os.environ.get("MUSE_DFLASH_FUSE_KV_POST", "1") != "0"
            and hd == 128
            and self._rope_head_size == 128
            and 1 <= L <= 8
            and context_states.is_cuda
            and context_states.dtype == torch.bfloat16
            and context_positions.is_cuda
            and context_positions.dtype == torch.int64
        ):
            per_layer = isinstance(context_slot_mapping, (list, tuple))
            slots_list = (
                list(context_slot_mapping) if per_layer else [context_slot_mapping] * L
            )
            if all(s is not None for s in slots_list):
                try:
                    kv_ptrs = tuple(
                        (None if a.kv_cache is None else a.kv_cache.data_ptr())
                        for a in self._attn_layers
                    )
                    cache_entry = getattr(self, "_dflash_kv_post_cache", None)
                    if cache_entry is None or cache_entry[0] != kv_ptrs:
                        import muse_bfx_ext as _bfx

                        kv_post_fn = getattr(_bfx, "dflash_kv_post", None)
                        k_caches = []
                        v_caches = []
                        block_size = 0
                        ok_layout = kv_post_fn is not None
                        if ok_layout:
                            for i in range(L):
                                attn = self._attn_layers[i]
                                if getattr(attn.impl, "kv_cache_dtype", "auto") != "auto":
                                    ok_layout = False
                                    break
                                kvc = attn.kv_cache
                                if kvc is None or not kvc.is_cuda or kvc.dtype != torch.bfloat16:
                                    ok_layout = False
                                    break
                                if kvc.dim() == 4 and kvc.shape[1] == nkv and kvc.shape[3] == 2 * hd:
                                    kc, vc = kvc.transpose(1, 2).split(hd, dim=-1)
                                elif kvc.dim() == 5 and kvc.shape[0] == 2 and kvc.shape[3] == nkv and kvc.shape[4] == hd:
                                    kc, vc = kvc[0], kvc[1]
                                else:
                                    ok_layout = False
                                    break
                                if kc.stride(3) != 1 or vc.stride(3) != 1:
                                    ok_layout = False
                                    break
                                bs_i = int(kc.shape[1])
                                if block_size == 0:
                                    block_size = bs_i
                                elif block_size != bs_i:
                                    ok_layout = False
                                    break
                                k_caches.append(kc)
                                v_caches.append(vc)
                        if ok_layout and block_size > 0:
                            cache_entry = (kv_ptrs, kv_post_fn, k_caches, v_caches, block_size)
                        else:
                            cache_entry = (kv_ptrs, None, None, None, 0)
                        self._dflash_kv_post_cache = cache_entry

                    _, kv_post_fn, k_caches, v_caches, block_size = cache_entry
                    if kv_post_fn is not None:
                        cos_sin_cache = self._rope_cos_sin_cache
                        if cos_sin_cache.dtype != context_states.dtype:
                            cast_cache = getattr(self, "_rope_cos_sin_cache_cast", None)
                            if (
                                cast_cache is None
                                or cast_cache.dtype != context_states.dtype
                                or cast_cache.device != context_states.device
                            ):
                                cast_cache = cos_sin_cache.to(
                                    device=context_states.device, dtype=context_states.dtype
                                )
                                self._rope_cos_sin_cache_cast = cast_cache
                            cos_sin_cache = cast_cache
                        all_kv_flat = self._project_context_kv_flat(context_states)
                        if not all_kv_flat.is_contiguous():
                            all_kv_flat = all_kv_flat.contiguous()
                        pos_arg = (
                            context_positions
                            if context_positions.is_contiguous()
                            else context_positions.contiguous()
                        )
                        if not all(s.is_contiguous() for s in slots_list):
                            slots_list = [s if s.is_contiguous() else s.contiguous() for s in slots_list]
                        kv_post_fn(
                            all_kv_flat,
                            pos_arg,
                            self._k_norm_weights,
                            cos_sin_cache,
                            k_caches,
                            v_caches,
                            slots_list,
                            nkv,
                            block_size,
                            float(self._rms_norm_eps),
                            bool(self._rope_is_neox),
                        )
                        return
                except Exception as exc:  # noqa: BLE001
                    logger.warning_once("dflash_kv_post skipped (%s); using stock path", exc)

        all_k, all_v = self._project_context_kv(context_states, num_ctx, L, nkv, hd)
        all_k_normed = self._normalize_context_k(all_k)

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cast_cache = getattr(self, "_rope_cos_sin_cache_cast", None)
            if (
                cast_cache is None
                or cast_cache.dtype != all_k_flat.dtype
                or cast_cache.device != all_k_flat.device
            ):
                cast_cache = cos_sin_cache.to(
                    device=all_k_flat.device, dtype=all_k_flat.dtype
                )
                self._rope_cos_sin_cache_cast = cast_cache
            cos_sin_cache = cast_cache
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i in range(L):
            slot_mapping = (
                context_slot_mapping[i] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue  # dummy run: skip cache ops
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skip_l0_norm = False
        if (
            input_embeds is None
            and getattr(self, "_dflash_fuse_embed_norm_ok", False)
            and len(self.layers) > 0
            and input_ids.is_cuda
            and input_ids.dim() == 1
            and input_ids.dtype in (torch.int32, torch.int64)
        ):
            l0_norm = self.layers[0].input_layernorm
            hidden_states, residual = torch.ops.muse.dflash_embed_rms_norm(
                input_ids,
                self.embed_tokens.weight,
                l0_norm.weight,
                float(l0_norm.variance_epsilon),
            )
            skip_l0_norm = True
        else:
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            hidden_states = input_embeds
            residual = None

        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                skip_input_norm=(skip_l0_norm and idx == 0),
            )
        if (
            getattr(self, "_dflash_fuse_norm_ok", False)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[-1] <= 8192
            and hidden_states.shape[-1] % 8 == 0
            and residual is not None
        ):
            hidden_states, _ = torch.ops.muse.dflash_fused_add_rms_norm(
                hidden_states,
                residual,
                self.norm.weight,
                float(self.norm.variance_epsilon),
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_tail_m6(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Eager/CUDA-graph forward slicing Layer-4 post-attention [1:7] -> M=6 (0-ULP exact)."""
        skip_l0_norm = False
        if (
            input_embeds is None
            and getattr(self, "_dflash_fuse_embed_norm_ok", False)
            and len(self.layers) > 0
            and input_ids.is_cuda
            and input_ids.dim() == 1
            and input_ids.dtype in (torch.int32, torch.int64)
        ):
            l0_norm = self.layers[0].input_layernorm
            hidden_states, residual = torch.ops.muse.dflash_embed_rms_norm(
                input_ids,
                self.embed_tokens.weight,
                l0_norm.weight,
                float(l0_norm.variance_epsilon),
            )
            skip_l0_norm = True
        else:
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            hidden_states = input_embeds
            residual = None

        last_idx = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                tail_slice_m6=(idx == last_idx),
                skip_input_norm=(skip_l0_norm and idx == 0),
            )
        if (
            getattr(self, "_dflash_fuse_norm_ok", False)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[-1] <= 8192
            and hidden_states.shape[-1] % 8 == 0
            and residual is not None
        ):
            hidden_states, _ = torch.ops.muse.dflash_fused_add_rms_norm(
                hidden_states,
                residual,
                self.norm.weight,
                float(self.norm.variance_epsilon),
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _preprocess(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for name, loaded_weight in weights:
            if "attention_sink_bias" in name:
                # Sink bias is per-head; shard it across TP ranks like the
                # attention heads themselves.
                heads_per_rank = loaded_weight.shape[0] // tp_size
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * heads_per_rank, heads_per_rank
                )
            yield name, loaded_weight

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            self._preprocess(weights), mapper=self.hf_to_vllm_mapper
        )


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        tail_slice_m6: bool = False,
    ) -> torch.Tensor:
        if tail_slice_m6:
            return self.model.forward_tail_m6(input_ids, positions, inputs_embeds)
        return self.model(input_ids, positions, inputs_embeds)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        """Per-layer attention causality, aligned with
        get_draft_kv_cache_layer_names."""
        return [layer.self_attn.causal for layer in self.model.layers]

    def update_draft_prompt_tail(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        num_rejected: torch.Tensor | None = None,
        extra_ids: torch.Tensor | None = None,
    ) -> None:
        lm_head = self.lm_head
        if not getattr(lm_head, "prompt_tail_admitted", False):
            return
        if not input_ids.is_cuda or input_ids.numel() == 0:
            return
        try:
            import muse_bfx_ext as _bfx

            try:
                _bfx.draft_vocab_update_tail(
                    input_ids,
                    positions,
                    lm_head.in_static_vocab,
                    lm_head.tail_ids,
                    lm_head.tail_count,
                    lm_head.tail_seen,
                    num_rejected,
                    extra_ids,
                )
            except TypeError:
                _bfx.draft_vocab_update_tail(
                    input_ids,
                    positions,
                    lm_head.in_static_vocab,
                    lm_head.tail_ids,
                    lm_head.tail_count,
                    lm_head.tail_seen,
                )
        except Exception:  # noqa: BLE001
            pass

    def draft_argmax(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.draft_id_to_target_id is not None:
            return None
        vocab_map = getattr(self, "draft_vocab_token_ids", None)
        if vocab_map is None:
            return None
        from vllm.v1.worker.gpu.spec_decode.dflash import native_leaf as _nl

        alt_buf = _nl.get_alt_buf(hidden_states.device) if _nl.ENABLED else None
        margin_buf = _nl.get_margin_buf(hidden_states.device) if _nl.ENABLED else None
        tree_top_buf = _nl.get_tree_top_buf(hidden_states.device) if _nl.ENABLED else None
        tree_vals_buf = _nl.get_tree_vals_buf(hidden_states.device) if _nl.ENABLED else None
        lm_head = self.lm_head
        if _nl.ENABLED and (
            getattr(lm_head, "prompt_tail_admitted", False)
            or getattr(lm_head, "fast_dhead_admitted", False)
        ):
            _nl.register_draft_tail_bufs(
                lm_head.in_static_vocab,
                lm_head.tail_ids,
                lm_head.tail_count,
                lm_head.tail_seen,
            )
        if (
            getattr(lm_head, "fast_dhead_admitted", False)
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and 1 <= hidden_states.shape[0] <= 8
        ):
            import muse_bfx_ext as _bfx

            x_c = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
            logit_scale = float(getattr(self.logits_processor, "scale", 1.0))
            return _bfx.dhead_fast_argmax(
                x_c,
                lm_head.fast_dhead_blob,
                lm_head.fast_dhead_row_scales,
                lm_head.fast_dhead_col_scales,
                lm_head.fast_dhead_tok_ids,
                lm_head.tail_ids,
                lm_head.tail_count,
                lm_head.full_raw_w,
                lm_head.full_raw_s,
                lm_head.full_global_scale,
                logit_scale,
                int(lm_head.fast_dhead_mode),
                int(lm_head.fast_dhead_topk),
                alt_buf,
                margin_buf,
                tree_top_buf,
                tree_vals_buf,
            )
        static_logits = self.logits_processor(self.lm_head, hidden_states)
        if static_logits is None:
            return None
        if (
            getattr(lm_head, "prompt_tail_admitted", False)
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and 1 <= hidden_states.shape[0] <= 8
            and static_logits.dtype == torch.bfloat16
        ):
            import muse_bfx_ext as _bfx

            x_c = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
            sl_c = static_logits if static_logits.is_contiguous() else static_logits.contiguous()
            logit_scale = float(getattr(self.logits_processor, "scale", 1.0))
            vm_sorted = getattr(self, "_draft_vocab_map_sorted", None)
            if vm_sorted is None:
                if os.environ.get("MUSE_DVT_PACKED_TOP6", "1") == "0":
                    vm_sorted = False
                    self._draft_vocab_map_sorted = False
                elif torch.cuda.is_current_stream_capturing():
                    vm_sorted = False
                else:
                    vm_sorted = bool(
                        vocab_map.dim() == 1
                        and (
                            vocab_map.numel() <= 1
                            or bool((vocab_map[1:] > vocab_map[:-1]).all().item())
                        )
                    )
                    self._draft_vocab_map_sorted = vm_sorted
            try:
                return _bfx.draft_vocab_tail_argmax(
                    x_c,
                    sl_c,
                    vocab_map,
                    lm_head.tail_ids,
                    lm_head.tail_count,
                    lm_head.full_raw_w,
                    lm_head.full_raw_s,
                    lm_head.full_global_scale,
                    logit_scale,
                    alt_buf,
                    margin_buf,
                    tree_top_buf,
                    tree_vals_buf,
                    vm_sorted,
                )
            except TypeError:
                try:
                    return _bfx.draft_vocab_tail_argmax(
                        x_c,
                        sl_c,
                        vocab_map,
                        lm_head.tail_ids,
                        lm_head.tail_count,
                        lm_head.full_raw_w,
                        lm_head.full_raw_s,
                        lm_head.full_global_scale,
                        logit_scale,
                        alt_buf,
                        margin_buf,
                        tree_top_buf,
                        tree_vals_buf,
                    )
                except TypeError:
                    return _bfx.draft_vocab_tail_argmax(
                        x_c,
                        sl_c,
                        vocab_map,
                        lm_head.tail_ids,
                        lm_head.tail_count,
                        lm_head.full_raw_w,
                        lm_head.full_raw_s,
                        lm_head.full_global_scale,
                        logit_scale,
                    )
        if static_logits.dim() == 2 and static_logits.shape[0] >= 1:
            if tree_top_buf is not None and tree_top_buf.numel() >= 48:
                n_tt = min(6, static_logits.shape[0])
                tk8 = static_logits[:n_tt].topk(8, dim=-1)
                tree_top_buf.fill_(-1)
                tree_top_buf[: 8 * n_tt].copy_(vocab_map[tk8.indices].reshape(-1))
                if tree_vals_buf is not None and tree_vals_buf.numel() >= 48:
                    tree_vals_buf.fill_(-1e30)
                    tree_vals_buf[: 8 * n_tt].copy_(tk8.values.to(torch.float32).reshape(-1))
                if alt_buf is not None:
                    n_t2 = min(2, n_tt, alt_buf.numel() // 2)
                    alt_buf[: 2 * n_t2].copy_(vocab_map[tk8.indices[:n_t2, :2]].reshape(-1))
                if margin_buf is not None:
                    n_t2 = min(2, n_tt, margin_buf.numel())
                    margin_buf[:n_t2].copy_((tk8.values[:n_t2, 0] - tk8.values[:n_t2, 1]).to(torch.float32))
            elif tree_top_buf is not None and tree_top_buf.numel() >= 36:
                n_tt = min(6, static_logits.shape[0])
                tk6 = static_logits[:n_tt].topk(6, dim=-1)
                tree_top_buf.fill_(-1)
                tree_top_buf[: 6 * n_tt].copy_(vocab_map[tk6.indices].reshape(-1))
                if tree_vals_buf is not None and tree_vals_buf.numel() >= 36:
                    tree_vals_buf.fill_(-1e30)
                    tree_vals_buf[: 6 * n_tt].copy_(tk6.values.to(torch.float32).reshape(-1))
                if alt_buf is not None:
                    n_t2 = min(2, n_tt, alt_buf.numel() // 2)
                    alt_buf[: 2 * n_t2].copy_(vocab_map[tk6.indices[:n_t2, :2]].reshape(-1))
                if margin_buf is not None:
                    n_t2 = min(2, n_tt, margin_buf.numel())
                    margin_buf[:n_t2].copy_((tk6.values[:n_t2, 0] - tk6.values[:n_t2, 1]).to(torch.float32))
            elif tree_top_buf is not None and tree_top_buf.numel() >= 20:
                n_tt = min(5, static_logits.shape[0])
                tk4 = static_logits[:n_tt].topk(4, dim=-1)
                tree_top_buf.fill_(-1)
                tree_top_buf[: 4 * n_tt].copy_(vocab_map[tk4.indices].reshape(-1))
                if alt_buf is not None:
                    n_t2 = min(2, n_tt, alt_buf.numel() // 2)
                    alt_buf[: 2 * n_t2].copy_(vocab_map[tk4.indices[:n_t2, :2]].reshape(-1))
                if margin_buf is not None:
                    n_t2 = min(2, n_tt, margin_buf.numel())
                    margin_buf[:n_t2].copy_((tk4.values[:n_t2, 0] - tk4.values[:n_t2, 1]).to(torch.float32))
            elif alt_buf is not None:
                n_t2 = min(2, static_logits.shape[0], alt_buf.numel() // 2)
                tk = static_logits[:n_t2].topk(2, dim=-1)
                alt_buf[: 2 * n_t2].copy_(vocab_map[tk.indices].reshape(-1))
                if margin_buf is not None and margin_buf.numel() >= n_t2:
                    margin_buf[:n_t2].copy_((tk.values[:, 0] - tk.values[:, 1]).to(torch.float32))
        return vocab_map[static_logits.argmax(dim=-1)]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states_aux5(
        self,
        aux_hidden_states: list[torch.Tensor],
        num_tokens: int,
    ) -> torch.Tensor | None:
        """Zero-cat 5-aux-tensor INT4 fc projection for small M <= 16 (CUDA-graph safe)."""
        if (
            not self.model.use_aux_hidden_state
            or len(aux_hidden_states) != 5
            or not (1 <= num_tokens <= 16)
        ):
            return None
        fc = self.model.fc
        if not getattr(fc, "_mi4_aux5_ok", False):
            return None
        mi4 = getattr(fc, "_mi4", None)
        if mi4 is None:
            return None
        n, k = mi4
        k_sub = k // 5
        hs: list[torch.Tensor] = []
        for h in aux_hidden_states:
            if (
                not h.is_cuda
                or h.dtype != torch.bfloat16
                or h.dim() != 2
                or h.shape[0] < num_tokens
                or h.shape[1] != k_sub
                or h.stride(1) != 1
                or (h.stride(0) % 8 != 0)
                or (h.data_ptr() % 16 != 0)
            ):
                return None
            hs.append(h[:num_tokens] if h.shape[0] > num_tokens else h)
        return torch.ops.muse.draft_int4_aux5_linear(
            hs[0], hs[1], hs[2], hs[3], hs[4], fc._mi4_blob, n, k
        )

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        expected = self.model.fc.input_size
        if hidden_states.shape[-1] != expected:
            raise ValueError(
                f"DFlash drafter expects {expected} concatenated aux hidden "
                f"features but received {hidden_states.shape[-1]}. This usually "
                "means the draft model's target_layer_ids reference layers that "
                "do not exist in the target model (incompatible draft/target pair)."
            )
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash embeds masked slots via mask_token_id (optionally "
                "overridden by a mask_embedding.pt file); it should not ship a "
                "mask_hidden weight."
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # Route the separately-trained mask embedding (if shipped) through the
        # standard weight loader alongside the rest of the draft weights.
        mask_embedding = self._read_mask_embedding()
        if mask_embedding is not None:
            model_weights["model.mask_embedding"] = mask_embedding
            self.model.has_separate_mask_embedding = True

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if not self.model.has_separate_mask_embedding:
            skip_substrs.append("mask_embedding")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
        import os
        try:
            from vllm.model_executor.layers import muse_draft_fp8
            muse_draft_fp8.attach_drafter(self.model)
        except Exception as e:  # noqa: BLE001
            logger.warning("muse_draft_fp8: skipped (%s)", e)
        if os.environ.get("MUSE_BFX", "1") != "0":
            from vllm.model_executor.layers import muse_bfx
            n = muse_bfx.attach_drafter(self.model)
            has_q_fkv = (
                getattr(self.model, "_fkv_mi4", None) is not None
                or getattr(self.model, "_fkv_mi3", None) is not None
                or getattr(self.model, "_fkv_mnv4", None) is not None
                or getattr(self.model, "_fkv_mfp8", None) is not None
            )
            kv = (
                muse_bfx.attach_tensor(self.model, "fkv", self.model._fused_kv_weight)
                if (self.model._fused_kv_bias is None and not has_q_fkv)
                else False
            )
            if n > 0 or kv:
                muse_bfx.place_drafter_rans(self.model)
            logger.info("muse_bfx: coded %d drafter linears, fused kv %s", n, kv)

    def _read_mask_embedding(self) -> torch.Tensor | None:
        """Checks for an override mask embedding in `mask_embedding.pt` and returns it.

        Some checkpoints ship a separately-trained mask embedding for the mask token,
        which we use to overwrite the embedding for `mask_token_id`. This helper
        checks for the file, loads the pytorch tensor, and returns the embedding to use.

        Returns None if the override file is not present.
        """
        mask_token_id = self.model.mask_token_id
        if mask_token_id is None:
            return None

        MASK_EMBEDDING_FILENAME = "mask_embedding.pt"
        data = get_hf_file_bytes(
            MASK_EMBEDDING_FILENAME,
            self.draft_model_config.model,
            self.draft_model_config.revision,
        )
        if data is None:
            return None

        state = torch.load(io.BytesIO(data), weights_only=True)
        if isinstance(state, dict):
            if state.get("mask_token_id", mask_token_id) != mask_token_id:
                raise ValueError(
                    f"{MASK_EMBEDDING_FILENAME} mask_token_id does not match "
                    f"dflash_config.mask_token_id ({mask_token_id}). "
                    f"Got {state.get('mask_token_id')}."
                )
            state = state["embedding"]

        logger.info(
            "Loaded DFlash mask embedding for mask_token_id %s from %s",
            mask_token_id,
            MASK_EMBEDDING_FILENAME,
        )
        return state.reshape(-1)


class DFlashMuseGlimmerAssistantForCausalLM(DFlashQwen3ForCausalLM):
    """Muse Glimmer's DFlash draft head (meta-models/Muse-Glimmer-30B-assistant).

    The head is the generic DFlash Qwen3 draft, but its checkpoint keeps the
    context projection and the norm after it in an ``encoder.`` block:
    ``encoder.fc`` (the concatenated target-layer hidden states -> hidden size)
    and ``encoder.output_norm_enc``. The generic module names those two ``fc``
    and ``hidden_norm``; every other tensor already has the generic name. Only
    the two names are rewritten, so a checkpoint that already uses the generic
    names loads unchanged.
    """

    _CHECKPOINT_RENAMES = (
        ("encoder.fc.", "fc."),
        ("encoder.output_norm_enc.", "hidden_norm."),
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def renamed() -> Iterable[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                for old, new in self._CHECKPOINT_RENAMES:
                    if name.startswith(old):
                        name = new + name[len(old) :]
                        break
                yield name, loaded_weight

        return super().load_weights(renamed())
