"""Shared public types and shape contracts for BitVideo operators."""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import TypeVar

import torch

_INT32_MAX = 2**31 - 1


class WeightLayout(IntEnum):
    """Physical layout of a two-bit packed ``[N, K]`` ternary matrix."""

    ROW_MAJOR = 0
    COLUMN_MAJOR = 1
    BLOCKED_N64 = 2
    MMA_INTERLEAVED = 3


class ScaleMode(IntEnum):
    """Granularity of weight dequantization scales."""

    PER_TENSOR = 0
    PER_CHANNEL = 1


class ActivationGranularity(IntEnum):
    """Granularity of dynamic symmetric activation quantization."""

    PER_TENSOR = 0
    PER_TOKEN = 1
    GROUP_128 = 2


class OutputDType(IntEnum):
    """Native extension output dtype identifiers."""

    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2


class KernelVariant(IntEnum):
    """Legal sparse native-kernel variant identifiers."""

    AUTO = 0
    GEMV_1WARP = 10
    GEMV_SPLIT_K = 11
    GEMV_WIDE = 12
    GEMM_DP4A_64X64 = 20
    GEMM_DP4A_128X64 = 21
    GEMM_DP4A_128X128 = 22
    GEMM_DP4A_64X128 = 23
    GEMM_MMA_64X64 = 30
    GEMM_MMA_128X64 = 31
    GEMM_MMA_128X128 = 32
    GEMM_MMA_64X128 = 33


class Backend(str, Enum):
    """Execution backend requested for a fused packed BitLinear operation."""

    AUTO = "auto"
    CUDA_EXTENSION = "cuda_extension"
    TRITON = "triton"
    TORCH = "torch"


_EnumT = TypeVar("_EnumT", bound=IntEnum)


def _coerce_int_enum(value: int | _EnumT, enum_type: type[_EnumT], name: str) -> _EnumT:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer enum value, not bool")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        legal = ", ".join(f"{item.name}={item.value}" for item in enum_type)
        raise ValueError(f"invalid {name} {value!r}; expected one of {legal}") from exc


def normalize_layout(value: int | WeightLayout) -> WeightLayout:
    return _coerce_int_enum(value, WeightLayout, "layout")


def normalize_scale_mode(value: int | ScaleMode) -> ScaleMode:
    return _coerce_int_enum(value, ScaleMode, "scale_mode")


def normalize_granularity(value: int | ActivationGranularity) -> ActivationGranularity:
    return _coerce_int_enum(value, ActivationGranularity, "activation granularity")


def normalize_variant(value: int | KernelVariant) -> KernelVariant:
    return _coerce_int_enum(value, KernelVariant, "kernel variant")


def normalize_backend(value: str | Backend | None) -> Backend:
    if value is None:
        return Backend.AUTO
    if isinstance(value, Backend):
        return value
    aliases = {
        "cuda": Backend.CUDA_EXTENSION,
        "extension": Backend.CUDA_EXTENSION,
        "cuda_ext": Backend.CUDA_EXTENSION,
        "pytorch": Backend.TORCH,
        "reference": Backend.TORCH,
    }
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in aliases:
        return aliases[normalized]
    try:
        return Backend(normalized)
    except ValueError as exc:
        legal = ", ".join(item.value for item in Backend)
        raise ValueError(f"invalid backend {value!r}; expected one of {legal}") from exc


def normalize_output_dtype(
    value: torch.dtype | int | OutputDType | None,
    *,
    default: torch.dtype,
) -> tuple[torch.dtype, OutputDType]:
    """Return the torch dtype and corresponding native-extension identifier."""

    if value is None:
        dtype = default
    elif isinstance(value, torch.dtype):
        dtype = value
    else:
        enum_value = _coerce_int_enum(value, OutputDType, "output dtype")
        dtype = {
            OutputDType.FLOAT32: torch.float32,
            OutputDType.FLOAT16: torch.float16,
            OutputDType.BFLOAT16: torch.bfloat16,
        }[enum_value]
    mapping = {
        torch.float32: OutputDType.FLOAT32,
        torch.float16: OutputDType.FLOAT16,
        torch.bfloat16: OutputDType.BFLOAT16,
    }
    if dtype not in mapping:
        raise TypeError(
            "output dtype must be torch.float32, torch.float16, or torch.bfloat16; "
            f"got {dtype}"
        )
    return dtype, mapping[dtype]


def layout_alignment(layout: int | WeightLayout) -> tuple[int, int]:
    """Return ``(N multiple, K multiple)`` required by a packed layout."""

    normalized = normalize_layout(layout)
    if normalized is WeightLayout.BLOCKED_N64:
        return 64, 16
    if normalized is WeightLayout.MMA_INTERLEAVED:
        return 8, 256
    return 1, 16


def padded_extents(n: int, k: int, layout: int | WeightLayout) -> tuple[int, int]:
    """Compute minimum padded extents using the same contract as the C++ packer."""

    if isinstance(n, bool) or isinstance(k, bool) or not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("n and k must be integers")
    if n <= 0 or k <= 0:
        raise ValueError(f"n and k must be positive; got n={n}, k={k}")
    n_multiple, k_multiple = layout_alignment(layout)
    np = ((n + n_multiple - 1) // n_multiple) * n_multiple
    kp = ((k + k_multiple - 1) // k_multiple) * k_multiple
    if np > _INT32_MAX or kp > _INT32_MAX:
        raise OverflowError(f"padded extents exceed INT32_MAX: ({np}, {kp})")
    return np, kp


def packed_word_count(
    layout: int | WeightLayout,
    n_padded: int,
    k_padded: int,
) -> int:
    """Return the number of int32 words in a validated packed allocation."""

    normalized = normalize_layout(layout)
    if n_padded <= 0 or k_padded <= 0:
        raise ValueError("padded extents must be positive")
    n_multiple, k_multiple = layout_alignment(normalized)
    if n_padded % n_multiple or k_padded % k_multiple:
        raise ValueError(
            f"layout {normalized.name} requires padded multiples "
            f"({n_multiple}, {k_multiple}); got ({n_padded}, {k_padded})"
        )
    if normalized is WeightLayout.MMA_INTERLEAVED:
        return (n_padded // 8) * (k_padded // 256) * 128
    return n_padded * (k_padded // 16)
