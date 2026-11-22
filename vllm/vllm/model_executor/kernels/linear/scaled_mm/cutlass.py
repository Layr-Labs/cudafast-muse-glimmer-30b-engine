# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Sequence

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    CUTLASS_BLOCK_FP8_SUPPORTED,
    convert_to_channelwise,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_MUSE_FP8_KERNELS_READY = False
try:
    from muse_kernels import load_muse_kernels as _load_muse_kernels

    _MUSE_FP8_KERNELS_READY = _load_muse_kernels()
except Exception:
    _MUSE_FP8_KERNELS_READY = False

_FAST_FP8_PACKS: list[tuple[int, object, int, bool, object]] = []
_FAST_FP8_GU_LAYERS: list[tuple[torch.nn.Module, torch.Tensor]] = []


def _muse_fp8_scaled_mm_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    pack_id: int,
) -> torch.Tensor:
    orig_ptr, payload, n, is_rans, m512_pack = _FAST_FP8_PACKS[pack_id]
    m = a.shape[0]
    if (
        _MUSE_FP8_KERNELS_READY
        and scale_a.numel() == 1
        and b.data_ptr() == orig_ptr
    ):
        if m == 1:
            a_c = a if a.is_contiguous() else a.contiguous()
            if is_rans:
                sm_pack, exp_stream, tile_chunk_off, init_states, lut = payload
                return torch.ops.muse_fast.fp8_rans_m1(
                    a_c, sm_pack, exp_stream, tile_chunk_off, init_states, lut, scale_a, scale_b, n
                )
            return torch.ops.muse_fast.fp8_mma_repacked_m1(
                a_c, payload, scale_a, scale_b, n
            )
        if m512_pack is not None and m >= 128 and m % 128 == 0:
            a_c = a if a.is_contiguous() else a.contiguous()
            return torch.ops.muse_fast.fp8_linear_contig_m512(
                a_c, m512_pack, scale_a, scale_b, n
            )
    return ops.cutlass_scaled_mm(
        a, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, bias=None
    )


def _muse_fp8_scaled_mm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    pack_id: int,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=a.device)


def _muse_fp8_bf16_linear_impl(
    x: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    pack_id: int,
) -> torch.Tensor:
    orig_ptr, payload, n, is_rans, m512_pack = _FAST_FP8_PACKS[pack_id]
    x_2d = x.reshape(-1, x.shape[-1])
    m = x_2d.shape[0]
    out_shape = (*x.shape[:-1], n)
    if (
        _MUSE_FP8_KERNELS_READY
        and scale_a.numel() == 1
        and b.data_ptr() == orig_ptr
    ):
        if m == 1:
            if x_2d.dtype == torch.bfloat16 and x_2d.shape[1] <= 8192:
                x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
                if is_rans:
                    sm_pack, exp_stream, tile_chunk_off, init_states, lut = payload
                    return torch.ops.muse_fast.fp8_rans_m1(
                        x_c, sm_pack, exp_stream, tile_chunk_off, init_states, lut, scale_a, scale_b, n
                    ).reshape(out_shape)
                return torch.ops.muse_fast.fp8_mma_repacked_m1(
                    x_c, payload, scale_a, scale_b, n
                ).reshape(out_shape)
            x_q = torch.ops.muse_fast.quant_fp8_triton_exact(
                x_2d if x_2d.is_contiguous() else x_2d.contiguous(), scale_a
            )
            if is_rans:
                sm_pack, exp_stream, tile_chunk_off, init_states, lut = payload
                return torch.ops.muse_fast.fp8_rans_m1(
                    x_q, sm_pack, exp_stream, tile_chunk_off, init_states, lut, scale_a, scale_b, n
                ).reshape(out_shape)
            return torch.ops.muse_fast.fp8_mma_repacked_m1(
                x_q, payload, scale_a, scale_b, n
            ).reshape(out_shape)
        if m512_pack is not None and m >= 128 and m % 128 == 0 and x_2d.dtype == torch.bfloat16:
            x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
            x_q = torch.ops.muse_fast.quant_fp8_triton_exact(x_c, scale_a)
            return torch.ops.muse_fast.fp8_linear_contig_m512(
                x_q, m512_pack, scale_a, scale_b, n
            ).reshape(out_shape)
    if _MUSE_FP8_KERNELS_READY and x_2d.dtype == torch.bfloat16 and scale_a.numel() == 1:
        x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
        x_q = torch.ops.muse_fast.quant_fp8_triton_exact(x_c, scale_a)
    else:
        x_q, _ = ops.scaled_fp8_quant(x_2d, scale_a)
    return ops.cutlass_scaled_mm(
        x_q, b, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b, bias=None
    ).reshape(out_shape)


def _muse_fp8_bf16_linear_fake(
    x: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    pack_id: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], b.shape[1]), dtype=torch.bfloat16, device=x.device)


def _muse_fp8_gate_up_silu_impl(x: torch.Tensor, gu_id: int) -> torch.Tensor:
    layer, gu512_pack = _FAST_FP8_GU_LAYERS[gu_id]
    x_2d = x.reshape(-1, x.shape[-1])
    m = x_2d.shape[0]
    out_shape = (*x.shape[:-1], 19968)
    if (
        _MUSE_FP8_KERNELS_READY
        and x_2d.dtype == torch.bfloat16
        and layer.input_scale.numel() == 1
    ):
        x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
        if m >= 128 and m % 128 == 0:
            x_q = torch.ops.muse_fast.quant_fp8_triton_exact(x_c, layer.input_scale)
            return torch.ops.muse_fast.fp8_gate_up_silu_contig_m512(
                x_q, gu512_pack, layer.input_scale, layer.weight_scale, 19968
            ).reshape(out_shape)
        gu = _muse_fp8_bf16_linear_impl(
            x_c, layer.weight, layer.input_scale, layer.weight_scale, layer._muse_fp8_pack_id
        )
        return torch.ops.muse_fast.silu_mul_mode0(gu).reshape(out_shape)
    gu = _muse_fp8_bf16_linear_impl(
        x_2d, layer.weight, layer.input_scale, layer.weight_scale, layer._muse_fp8_pack_id
    )
    g = gu[:, :19968].float()
    u = gu[:, 19968:].float()
    return (g / (1.0 + torch.exp(-g)) * u).to(gu.dtype).reshape(out_shape)


def _muse_fp8_gate_up_silu_fake(x: torch.Tensor, gu_id: int) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], 19968), dtype=torch.bfloat16, device=x.device)


def _muse_fp8_gate_up_silu_fp8quant_impl(
    x: torch.Tensor, gu_id: int, out_scale_a: torch.Tensor
) -> torch.Tensor:
    layer, gu512_pack = _FAST_FP8_GU_LAYERS[gu_id]
    x_2d = x.reshape(-1, x.shape[-1])
    m = x_2d.shape[0]
    out_shape = (*x.shape[:-1], 19968)
    if (
        _MUSE_FP8_KERNELS_READY
        and x_2d.dtype == torch.bfloat16
        and layer.input_scale.numel() == 1
        and out_scale_a.numel() == 1
    ):
        x_c = x_2d if x_2d.is_contiguous() else x_2d.contiguous()
        if m >= 128 and m % 128 == 0:
            x_q = torch.ops.muse_fast.quant_fp8_triton_exact(x_c, layer.input_scale)
            return torch.ops.muse_fast.fp8_gate_up_silu_fp8quant_contig_m512(
                x_q, gu512_pack, layer.input_scale, layer.weight_scale, out_scale_a, 19968
            ).reshape(out_shape)
        gu = _muse_fp8_bf16_linear_impl(
            x_c, layer.weight, layer.input_scale, layer.weight_scale, layer._muse_fp8_pack_id
        )
        return torch.ops.muse_fast.silu_mul_fp8quant_mode0(gu, out_scale_a).reshape(out_shape)
    gu = _muse_fp8_bf16_linear_impl(
        x_2d, layer.weight, layer.input_scale, layer.weight_scale, layer._muse_fp8_pack_id
    )
    g = gu[:, :19968].float()
    u = gu[:, 19968:].float()
    prod = (g / (1.0 + torch.exp(-g))) * u
    return (prod * (1.0 / out_scale_a.float())).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(out_shape)


def _muse_fp8_gate_up_silu_fp8quant_fake(
    x: torch.Tensor, gu_id: int, out_scale_a: torch.Tensor
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], 19968), dtype=torch.float8_e4m3fn, device=x.device)


direct_register_custom_op(
    op_name="muse_fp8_scaled_mm",
    op_func=_muse_fp8_scaled_mm_impl,
    fake_impl=_muse_fp8_scaled_mm_fake,
)
direct_register_custom_op(
    op_name="muse_fp8_bf16_linear",
    op_func=_muse_fp8_bf16_linear_impl,
    fake_impl=_muse_fp8_bf16_linear_fake,
)
direct_register_custom_op(
    op_name="muse_fp8_gate_up_silu",
    op_func=_muse_fp8_gate_up_silu_impl,
    fake_impl=_muse_fp8_gate_up_silu_fake,
)
direct_register_custom_op(
    op_name="muse_fp8_gate_up_silu_fp8quant",
    op_func=_muse_fp8_gate_up_silu_fp8quant_impl,
    fake_impl=_muse_fp8_gate_up_silu_fp8quant_fake,
)

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class CutlassInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        config = self.config
        # WEIGHT
        # Cutlass kernels need transposed weight.
        weight = getattr(layer, w_q_name)
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(weight.t().data, requires_grad=False),
        )

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if config.is_static_input_scheme:
            input_scale = getattr(layer, i_s_name)

            if config.input_symmetric:
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
                setattr(layer, i_zp_name, None)
            else:
                input_zero_point = getattr(layer, i_zp_name)

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
                )

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(
                    layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
                )

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/w8a8/cutlass/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/Epilogues.md
        if not config.input_symmetric:
            weight = getattr(layer, w_q_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if config.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = getattr(layer, i_zp_name) * azp_adj
            setattr(
                layer,
                azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        x_q, x_s, x_zp = ops.scaled_int8_quant(
            x.contiguous(), i_s, i_zp, symmetric=symmetric
        )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            return ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        return ops.cutlass_scaled_mm(
            x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
        )


class CutlassFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        self.logical_output_size: int | None = None
        self._muse_fp8_pack_id: int = -1
        self._muse_fp8_gu_pack: torch.Tensor | None = None
        super().__init__(c, layer_param_names)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def input_quant_key(self) -> QuantKey | None:
        """Only static per-tensor activation quantization is supported for external
        quantization."""
        if self.config.activation_quant_key == kFp8StaticTensorSym:
            return kFp8StaticTensorSym
        return None

    @staticmethod
    def _pad_to_alignment(
        x: torch.Tensor, dim: int, alignment: int, value: float = 0.0
    ) -> torch.Tensor:
        """Pad tensor ``x`` along ``dim`` to the next multiple of
        ``alignment``."""
        remainder = x.shape[dim] % alignment
        if remainder == 0:
            return x
        pad_size = alignment - remainder
        pad_spec = [0] * (2 * x.dim())
        pad_spec[-(2 * dim + 1)] = pad_size
        return torch.nn.functional.pad(x, pad_spec, value=value)

    @staticmethod
    def padded_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.shape != param.shape:
            slices = tuple(slice(0, s) for s in loaded_weight.shape)
            param.data[slices].copy_(loaded_weight)
        else:
            param.data.copy_(loaded_weight.view(param.shape))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_name, weight_scale_name, _, _ = self.layer_param_names
        weight = getattr(layer, weight_name)

        # keep the logical output width so runtime can slice away static padding.
        self.logical_output_size = weight.shape[1]

        pad_k = (16 - weight.shape[0] % 16) % 16
        pad_n = (16 - weight.shape[1] % 16) % 16
        if pad_k == 0 and pad_n == 0:
            self._maybe_repack_fp8_for_m1(weight)
            layer._muse_fp8_pack_id = self._muse_fp8_pack_id
            if self._muse_fp8_gu_pack is not None:
                gu_idx = len(_FAST_FP8_GU_LAYERS)
                _FAST_FP8_GU_LAYERS.append((layer, self._muse_fp8_gu_pack))
                layer._muse_fp8_gu_id = gu_idx
            return

        # B is column-major [K, N]
        padded_weight = torch.nn.functional.pad(
            weight.t().contiguous(),
            (0, pad_k, 0, pad_n),
        ).t()
        replace_parameter(layer, weight_name, padded_weight.data)
        set_weight_attrs(
            getattr(layer, weight_name),
            {
                "weight_loader": self.padded_weight_loader,
            },
        )
        self._maybe_repack_fp8_for_m1(getattr(layer, weight_name))
        layer._muse_fp8_pack_id = self._muse_fp8_pack_id
        if self._muse_fp8_gu_pack is not None:
            gu_idx = len(_FAST_FP8_GU_LAYERS)
            _FAST_FP8_GU_LAYERS.append((layer, self._muse_fp8_gu_pack))
            layer._muse_fp8_gu_id = gu_idx

        weight_scale = getattr(layer, weight_scale_name, None)
        if weight_scale is not None and pad_n > 0 and weight_scale.numel() > 1:
            flat_scale = weight_scale.reshape(-1)
            padded_scale = self._pad_to_alignment(
                flat_scale, dim=0, alignment=16, value=1.0
            ).view(-1, *weight_scale.shape[1:])
            replace_parameter(layer, weight_scale_name, padded_scale.data)
            set_weight_attrs(
                getattr(layer, weight_name),
                {
                    "weight_loader": self.padded_weight_loader,
                },
            )

    def _maybe_repack_fp8_for_m1(self, weight: torch.Tensor) -> None:
        if (
            not _MUSE_FP8_KERNELS_READY
            or not weight.is_cuda
            or weight.dtype != torch.float8_e4m3fn
            or weight.dim() != 2
            or weight.stride(0) != 1
        ):
            return
        k, n = weight.shape
        if n % 8 != 0 or k % 64 != 0 or k > 24576:
            return
        rans_pack = torch.ops.muse_fast.repack_fp8_rans(weight)
        # Load-time bit-exact self-check vs stock CUTLASS on a deterministic M=1 probe.
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(12345)
        probe_a = (
            torch.randn((1, k), dtype=torch.bfloat16, device=weight.device, generator=gen)
            .to(torch.float8_e4m3fn)
        )
        one = torch.ones((1,), dtype=torch.float32, device=weight.device)
        ref = ops.cutlass_scaled_mm(
            probe_a, weight, out_dtype=torch.bfloat16, scale_a=one, scale_b=one, bias=None
        )
        sm_pack, exp_stream, tile_chunk_off, init_states, lut = rans_pack
        out = torch.ops.muse_fast.fp8_rans_m1(
            probe_a, sm_pack, exp_stream, tile_chunk_off, init_states, lut, one, one, n
        )
        m512_pack = None
        if (k, n) in ((19968, 6656), (6656, 4608)):
            cand_512 = torch.ops.muse_fast.repack_fp8_m512_contig(weight, False)
            probe_128 = (
                torch.randn((128, k), dtype=torch.bfloat16, device=weight.device, generator=gen)
                .to(torch.float8_e4m3fn)
            )
            ref_128 = ops.cutlass_scaled_mm(
                probe_128, weight, out_dtype=torch.bfloat16, scale_a=one, scale_b=one, bias=None
            )
            out_128 = torch.ops.muse_fast.fp8_linear_contig_m512(
                probe_128, cand_512, one, one, n
            )
            if torch.equal(out_128, ref_128):
                m512_pack = cand_512
        elif (k, n) == (6656, 39936):
            cand_gu = torch.ops.muse_fast.repack_fp8_m512_contig(weight, True)
            probe_128 = (
                torch.randn((128, k), dtype=torch.bfloat16, device=weight.device, generator=gen)
                .to(torch.float8_e4m3fn)
            )
            ref_gu = ops.cutlass_scaled_mm(
                probe_128, weight, out_dtype=torch.bfloat16, scale_a=one, scale_b=one, bias=None
            )
            ref_silu = torch.ops.muse_fast.silu_mul_mode0(ref_gu)
            out_silu = torch.ops.muse_fast.fp8_gate_up_silu_contig_m512(
                probe_128, cand_gu, one, one, 19968
            )
            if torch.equal(out_silu, ref_silu):
                self._muse_fp8_gu_pack = cand_gu
        idx = len(_FAST_FP8_PACKS)
        if torch.equal(out, ref):
            _FAST_FP8_PACKS.append((weight.data_ptr(), rans_pack, n, True, m512_pack))
        else:
            b_pack = torch.ops.muse_fast.repack_fp8_mma(weight)
            _FAST_FP8_PACKS.append((weight.data_ptr(), b_pack, n, False, m512_pack))
        self._muse_fp8_pack_id = idx

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        padded_k, padded_n = B.shape
        output_size = self.logical_output_size
        assert output_size is not None
        pad_k = padded_k - A.shape[1]
        pad_n = padded_n - output_size

        if pad_k > 0:
            A = self._pad_to_alignment(A, dim=1, alignment=16)
        if pad_n > 0 and bias is not None:
            bias = self._pad_to_alignment(bias, dim=0, alignment=16)

        if (
            self._muse_fp8_pack_id >= 0
            and bias is None
            and out_dtype == torch.bfloat16
        ):
            output = torch.ops.vllm.muse_fp8_scaled_mm(
                A, B, As, Bs, self._muse_fp8_pack_id
            )
        else:
            output = ops.cutlass_scaled_mm(
                A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
            )

        if pad_n > 0:
            output = output[..., :output_size].contiguous()

        return output.view(*output_shape[:-1], output_size)


class CutlassFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
            column_major_scales=True,
        )

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not CUTLASS_BLOCK_FP8_SUPPORTED:
            return (
                False,
                "The device compute capability of"
                f"{compute_capability} is not supported.",
            )
        return True, None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        return ops.cutlass_scaled_mm(
            A,
            B.T,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.T,
        )


def cutlass_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return ops.cutlass_scaled_mm(
        A,
        B.T,
        out_dtype=output_dtype,
        scale_a=As,
        scale_b=Bs.T,
    )
