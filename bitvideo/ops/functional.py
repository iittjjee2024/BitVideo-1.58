"""Functional W1.58A8 operators and three-tier backend dispatch."""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn.functional as F

from .backends import (
    get_extension,
    get_triton_forward,
    is_compiling,
    requested_backend,
)
from .packing import PackedTernaryWeight, _validate_eps, unpack_ternary_weight
from .types import (
    ActivationGranularity,
    Backend,
    KernelVariant,
    OutputDType,
    ScaleMode,
    WeightLayout,
    normalize_granularity,
    normalize_output_dtype,
    normalize_variant,
)

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_EXTENSION_GEMV_VARIANTS = {
    KernelVariant.GEMV_1WARP,
    KernelVariant.GEMV_SPLIT_K,
    KernelVariant.GEMV_WIDE,
}
_EXTENSION_DP4A_VARIANTS = {
    KernelVariant.GEMM_DP4A_64X64,
    KernelVariant.GEMM_DP4A_128X64,
    KernelVariant.GEMM_DP4A_128X128,
    KernelVariant.GEMM_DP4A_64X128,
}
_EXTENSION_MMA_VARIANTS = {
    KernelVariant.GEMM_MMA_64X64,
    KernelVariant.GEMM_MMA_128X64,
    KernelVariant.GEMM_MMA_128X128,
    KernelVariant.GEMM_MMA_64X128,
}
BitLinearResult: TypeAlias = torch.Tensor


def _validate_activation(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim < 1:
        raise ValueError("x must have shape [..., K]")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError("x must use float16, bfloat16, or float32")
    if x.shape[-1] <= 0:
        raise ValueError("x.shape[-1] must be positive")
    return x.contiguous()


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if value != value or value < -_FLOAT32_MAX or value > _FLOAT32_MAX:
        raise ValueError(f"alpha must be finite and representable as float32; got {alpha!r}")
    return value


def _sanitize_for_quantization(x_2d: torch.Tensor) -> torch.Tensor:
    """Convert a floating ``[M, K]`` tensor to finite float32 quantizer input."""

    # values: [M, K] float32. NaN maps to zero; infinities saturate to the
    # largest finite float so scale computation and rounding remain defined.
    values = x_2d.to(torch.float32)
    limit = torch.finfo(torch.float32).max
    return torch.nan_to_num(values, nan=0.0, posinf=limit, neginf=-limit)


def _quantize_activations_torch(
    x: torch.Tensor,
    granularity: ActivationGranularity,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch symmetric INT8 quantization matching the native contract."""

    original_shape = tuple(x.shape)
    k = x.shape[-1]
    m = x.numel() // k
    # values: [M, K] float32.
    values = _sanitize_for_quantization(x.reshape(m, k))

    if granularity is ActivationGranularity.PER_TENSOR:
        if m == 0:
            # scales: [1]; q_2d: [0, K].
            scales = torch.full((1,), eps / 127.0, dtype=torch.float32, device=x.device)
            q_2d = torch.empty((0, k), dtype=torch.int8, device=x.device)
            return q_2d.reshape(original_shape), scales
        # maximum/scales/inverse_scales: [1].
        maximum = values.abs().amax().reshape(1)
        scales = torch.clamp_min(maximum, eps) / 127.0
        inverse_scales = scales.reciprocal()
        q_2d = torch.round(values * inverse_scales.view(1, 1)).clamp_(-127, 127).to(torch.int8)
        return q_2d.reshape(original_shape), scales.contiguous()

    if granularity is ActivationGranularity.PER_TOKEN:
        # maximum/scales/inverse_scales: [M]; q_2d: [M, K]. This also handles M=0.
        maximum = values.abs().amax(dim=1)
        scales = torch.clamp_min(maximum, eps) / 127.0
        inverse_scales = scales.reciprocal()
        q_2d = torch.round(values * inverse_scales.view(m, 1)).clamp_(-127, 127).to(torch.int8)
        return q_2d.reshape(original_shape), scales.contiguous()

    groups = (k + 127) // 128
    padded_k = groups * 128
    # grouped: [M, groups, 128], with zero-valued K padding.
    grouped = F.pad(values, (0, padded_k - k)).view(m, groups, 128)
    # scales/inverse_scales: [M, groups]; expanded_inverse_scales: [M, K].
    scales = torch.clamp_min(grouped.abs().amax(dim=2), eps) / 127.0
    inverse_scales = scales.reciprocal()
    expanded_inverse_scales = inverse_scales.repeat_interleave(128, dim=1)[:, :k]
    q_2d = torch.round(values * expanded_inverse_scales).clamp_(-127, 127).to(torch.int8)
    return q_2d.reshape(original_shape), scales.contiguous()


def quantize_activations(
    x: torch.Tensor,
    *,
    granularity: int | ActivationGranularity = ActivationGranularity.PER_TOKEN,
    eps: float = 1.0e-5,
    backend: str | Backend | None = Backend.AUTO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize ``x [..., K]`` to INT8 and return ``(q, scale)``.

    Scale shapes are ``[1]`` (tensor), ``[M]`` (token), or
    ``[M, ceil(K/128)]`` (group), where ``M = x.numel() / K``.
    """

    x = _validate_activation(x)
    normalized_granularity = normalize_granularity(granularity)
    epsilon = _validate_eps(eps)
    selected = requested_backend(backend)
    if selected is Backend.TRITON:
        raise ValueError("the standalone activation quantizer has extension and torch tiers")

    extension = None
    if selected is Backend.CUDA_EXTENSION:
        if not x.is_cuda:
            raise RuntimeError("the CUDA extension quantizer requires a CUDA tensor")
        extension = get_extension(required=True)
    elif selected is Backend.AUTO and x.is_cuda and not is_compiling():
        extension = get_extension()
    if extension is not None:
        return extension.quantize_activations(x, int(normalized_granularity), epsilon)
    return _quantize_activations_torch(x, normalized_granularity, epsilon)


def _padded_cuda_int_mm(q: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Run CUDA ``torch._int_mm`` after satisfying its conservative tile contract."""

    m, k = q.shape
    n = rhs.shape[1]
    m_padded = max(32, ((m + 31) // 32) * 32)
    k_padded = ((k + 7) // 8) * 8
    n_padded = ((n + 7) // 8) * 8
    if (m == m_padded and k == k_padded and n == n_padded):
        return torch._int_mm(q, rhs)

    # q_padded: [Mp, Kp]; rhs_padded: [Kp, Np], zero outside logical extents.
    q_padded = torch.zeros((m_padded, k_padded), dtype=torch.int8, device=q.device)
    rhs_padded = torch.zeros((k_padded, n_padded), dtype=torch.int8, device=q.device)
    q_padded[:m, :k].copy_(q)
    rhs_padded[:k, :n].copy_(rhs)
    # accumulator: [M, N] int32 after removing padded rows and columns.
    accumulator = torch._int_mm(q_padded, rhs_padded)
    return accumulator[:m, :n].contiguous()


def _exact_float_int_mm(q: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Exact INT8 matmul fallback using bounded float32 partial sums.

    Each 512-wide partial has magnitude at most ``512 * 128 * 128 = 2**23``;
    every intermediate integer is therefore exactly representable in float32.
    Partials are rounded to int32 before integer accumulation.
    """

    m, k = q.shape
    n = rhs.shape[1]
    # accumulator: [M, N] int32.
    accumulator = torch.zeros((m, n), dtype=torch.int32, device=q.device)
    chunk = 512
    for begin in range(0, k, chunk):
        end = min(k, begin + chunk)
        # partial: [M, N] float32 with exact integer values within 2**23.
        partial = q[:, begin:end].to(torch.float32) @ rhs[begin:end, :].to(torch.float32)
        accumulator.add_(torch.round(partial).to(torch.int32))
    return accumulator


def int8_mm(q: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Compute exact ``int8[M,K] @ int8[K,N] -> int32[M,N]``.

    ``torch._int_mm`` is preferred. CUDA inputs are transparently zero-padded
    to tile-compatible extents; a bounded-partial PyTorch implementation keeps
    the operation available on devices without an INT8 matmul kernel.
    """

    if not isinstance(q, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("q and rhs must be torch.Tensor instances")
    if q.dtype is not torch.int8 or rhs.dtype is not torch.int8:
        raise TypeError("q and rhs must both use torch.int8")
    if q.ndim != 2 or rhs.ndim != 2 or q.shape[1] != rhs.shape[0]:
        raise ValueError(f"expected [M,K] @ [K,N], got {tuple(q.shape)} @ {tuple(rhs.shape)}")
    if q.device != rhs.device:
        raise ValueError("q and rhs must be on the same device")
    if q.shape[1] <= 0:
        raise ValueError("K must be positive")
    m, _ = q.shape
    n = rhs.shape[1]
    if m == 0 or n == 0:
        return torch.empty((m, n), dtype=torch.int32, device=q.device)

    q = q.contiguous()
    rhs = rhs.contiguous()
    int_mm = getattr(torch, "_int_mm", None)
    if callable(int_mm):
        try:
            if q.is_cuda:
                return _padded_cuda_int_mm(q, rhs)
            return int_mm(q, rhs)
        except (NotImplementedError, RuntimeError):
            pass
    if q.device.type == "cpu":
        return q.to(torch.int32) @ rhs.to(torch.int32)
    return _exact_float_int_mm(q, rhs)


def _extension_ineligibility(
    x: torch.Tensor,
    packed: PackedTernaryWeight,
    granularity: ActivationGranularity,
    variant: KernelVariant,
) -> str | None:
    if is_compiling():
        return "Python extension calls are bypassed while torch.compile is tracing"
    if get_extension() is None:
        return "bitvideo._C is unavailable"
    if not x.is_cuda:
        return "the native kernel requires CUDA input"
    if x.device != packed.device:
        return "input and packed weight are on different devices"
    if granularity is ActivationGranularity.GROUP_128:
        return "group-128 scales require a groupwise GEMM epilogue"
    if packed.layout is WeightLayout.COLUMN_MAJOR:
        return "column-major weights are an ablation/conversion layout"
    m = x.numel() // x.shape[-1]
    if packed.layout is WeightLayout.ROW_MAJOR and m > 8:
        return "row-major native kernels are GEMV-only (M <= 8)"
    if variant in _EXTENSION_GEMV_VARIANTS and packed.layout is not WeightLayout.ROW_MAJOR:
        return "the selected GEMV variant requires row-major weights"
    if variant in _EXTENSION_GEMV_VARIANTS and m > 8:
        return "the selected GEMV variant requires M <= 8"
    if variant in _EXTENSION_DP4A_VARIANTS and packed.layout is not WeightLayout.BLOCKED_N64:
        return "the selected DP4A variant requires blocked-N64 weights"
    if variant in _EXTENSION_MMA_VARIANTS and packed.layout is not WeightLayout.MMA_INTERLEAVED:
        return "the selected MMA variant requires MMA-interleaved weights"
    try:
        major, _ = torch.cuda.get_device_capability(x.device)
    except (AssertionError, RuntimeError):
        return "CUDA device capability could not be queried"
    if major < 8:
        return "the native kernels require compute capability 8.0 or newer"
    return None


def select_backend(
    x: torch.Tensor,
    packed: PackedTernaryWeight,
    *,
    backend: str | Backend | None = Backend.AUTO,
    activation_granularity: int | ActivationGranularity = ActivationGranularity.PER_TOKEN,
    variant: int | KernelVariant = KernelVariant.AUTO,
) -> Backend:
    """Resolve the execution backend for a concrete BitLinear problem."""

    x = _validate_activation(x)
    if not isinstance(packed, PackedTernaryWeight):
        raise TypeError("packed must be a PackedTernaryWeight")
    if x.shape[-1] != packed.in_features:
        raise ValueError(
            f"x.shape[-1] must equal packed.in_features={packed.in_features}; "
            f"got {x.shape[-1]}"
        )
    if x.device != packed.device:
        raise ValueError(f"x must be on {packed.device}; got {x.device}")
    granularity = normalize_granularity(activation_granularity)
    kernel_variant = normalize_variant(variant)
    selected = requested_backend(backend)
    extension_reason = _extension_ineligibility(x, packed, granularity, kernel_variant)

    if selected is Backend.CUDA_EXTENSION:
        if extension_reason is not None:
            raise RuntimeError("CUDA extension backend is ineligible: " + extension_reason)
        return selected
    if selected is Backend.TRITON:
        if is_compiling():
            raise RuntimeError("the Triton Python adapter is bypassed while tracing")
        if not x.is_cuda:
            raise RuntimeError("the Triton backend requires CUDA input")
        get_triton_forward(required=True)
        return selected
    if selected is Backend.TORCH:
        return selected

    if extension_reason is None:
        return Backend.CUDA_EXTENSION
    if x.is_cuda and not is_compiling() and get_triton_forward() is not None:
        return Backend.TRITON
    return Backend.TORCH


def _normalize_bias(
    bias: torch.Tensor | None,
    *,
    n: int,
    device: torch.device,
) -> torch.Tensor | None:
    if bias is None:
        return None
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias must be a torch.Tensor or None")
    if bias.ndim != 1 or bias.numel() != n:
        raise ValueError(f"bias must have shape [{n}]; got {tuple(bias.shape)}")
    if bias.dtype not in _FLOAT_DTYPES:
        raise TypeError("bias must use float16, bfloat16, or float32")
    if bias.device != device:
        raise ValueError(f"bias must be on {device}; got {bias.device}")
    return bias.to(torch.float32).contiguous()


def _bit_linear_torch(
    x: torch.Tensor,
    packed: PackedTernaryWeight,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
    granularity: ActivationGranularity,
    alpha: float,
) -> torch.Tensor:
    k = packed.in_features
    n = packed.out_features
    m = x.numel() // k
    output_shape = (*x.shape[:-1], n)
    if m == 0:
        return torch.empty(output_shape, dtype=output_dtype, device=x.device)

    # q: [..., K] int8; activation_scale: [1] or [M].
    q, activation_scale = _quantize_activations_torch(x, granularity, 1.0e-5)
    # ternary: [N, K]; rhs: [K, N]; accumulator: [M, N] int32.
    ternary = unpack_ternary_weight(packed, backend=Backend.TORCH)
    rhs = ternary.transpose(0, 1).contiguous()
    accumulator = int8_mm(q.reshape(m, k), rhs)

    # activation_factor: [M, 1]; weight_factor: [1, N].
    if granularity is ActivationGranularity.PER_TOKEN:
        activation_factor = activation_scale.view(m, 1)
    else:
        activation_factor = activation_scale.view(1, 1)
    weight_factor = (
        packed.scale.view(1, n)
        if packed.scale_mode is ScaleMode.PER_CHANNEL
        else packed.scale.view(1, 1)
    )
    # output_2d: [M, N] float32 before the requested storage cast.
    output_2d = accumulator.to(torch.float32) * (alpha * activation_factor * weight_factor)
    if bias is not None:
        output_2d = output_2d + bias.view(1, n)
    return output_2d.to(output_dtype).view(output_shape)


def bit_linear(
    x: torch.Tensor,
    packed: PackedTernaryWeight,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | int | OutputDType | None = None,
    activation_granularity: int | ActivationGranularity = ActivationGranularity.PER_TOKEN,
    alpha: float = 1.0,
    backend: str | Backend | None = Backend.AUTO,
    variant: int | KernelVariant = KernelVariant.AUTO,
    split_k: int = 0,
    autotune: bool = False,
) -> BitLinearResult:
    """Run packed W1.58A8 linear inference with deterministic backend dispatch.

    Dispatch order for ``backend='auto'`` is native CUDA extension, registered
    Triton implementation, then the portable ``torch._int_mm`` path.
    """

    x = _validate_activation(x)
    if not isinstance(packed, PackedTernaryWeight):
        raise TypeError("packed must be a PackedTernaryWeight")
    if x.shape[-1] != packed.in_features:
        raise ValueError(
            f"x.shape[-1] must equal packed.in_features={packed.in_features}; "
            f"got {x.shape[-1]}"
        )
    if x.device != packed.device:
        raise ValueError(f"x must be on {packed.device}; got {x.device}")
    output_dtype, native_output_dtype = normalize_output_dtype(out_dtype, default=x.dtype)
    granularity = normalize_granularity(activation_granularity)
    if granularity is ActivationGranularity.GROUP_128:
        raise ValueError("bit_linear does not support group-128 scales without a groupwise epilogue")
    kernel_variant = normalize_variant(variant)
    alpha_f32 = _validate_alpha(alpha)
    if isinstance(split_k, bool) or not isinstance(split_k, int) or not 0 <= split_k <= 64:
        raise ValueError(f"split_k must be an integer in [0, 64]; got {split_k!r}")
    normalized_bias = _normalize_bias(bias, n=packed.out_features, device=x.device)
    selected = select_backend(
        x,
        packed,
        backend=backend,
        activation_granularity=granularity,
        variant=kernel_variant,
    )

    if selected is Backend.CUDA_EXTENSION:
        extension = get_extension(required=True)
        return extension.bit_linear_forward(
            x,
            packed.data,
            packed.scale,
            packed.out_features,
            packed.in_features,
            packed.n_padded,
            packed.k_padded,
            int(packed.layout),
            normalized_bias,
            int(native_output_dtype),
            int(packed.scale_mode),
            int(granularity),
            alpha_f32,
            int(kernel_variant),
            split_k,
            bool(autotune),
        )

    if selected is Backend.TRITON:
        forward = get_triton_forward(required=True)
        return forward(
            x=x,
            packed_weight=packed.data,
            weight_scale=packed.scale,
            out_features=packed.out_features,
            in_features=packed.in_features,
            n_padded=packed.n_padded,
            k_padded=packed.k_padded,
            layout=int(packed.layout),
            bias=normalized_bias,
            out_dtype=int(native_output_dtype),
            weight_scale_mode=int(packed.scale_mode),
            activation_granularity=int(granularity),
            alpha=alpha_f32,
            variant=int(kernel_variant),
            split_k=split_k,
            autotune=bool(autotune),
        )

    if is_compiling():
        # Keep the optional Python containers and backend probing outside the
        # captured graph; the registered raw-tensor op supplies FakeTensor metadata.
        from .library import bit_linear_fallback_op

        return bit_linear_fallback_op(
            x,
            packed,
            normalized_bias,
            out_dtype=output_dtype,
            activation_granularity=granularity,
            alpha=alpha_f32,
        )

    return _bit_linear_torch(
        x,
        packed,
        normalized_bias,
        output_dtype,
        granularity,
        alpha_f32,
    )
