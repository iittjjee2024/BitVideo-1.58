"""Dispatcher-registered BitVideo reference operators.

These raw-tensor entry points make the portable path visible to FakeTensor,
``torch.compile``, and export tooling without teaching those systems about the
:class:`PackedTernaryWeight` Python container.
"""

from __future__ import annotations

import torch

from .functional import (
    _bit_linear_torch,
    _normalize_bias,
    _quantize_activations_torch,
    _validate_activation,
    _validate_alpha,
)
from .packing import PackedTernaryWeight, _unpack_ternary_codes, _validate_eps
from .types import (
    ActivationGranularity,
    ScaleMode,
    WeightLayout,
    normalize_granularity,
    normalize_layout,
    normalize_output_dtype,
    normalize_scale_mode,
)

_DEF_LIBRARY = torch.library.Library("bitvideo", "DEF")
_DEF_LIBRARY.define(
    "unpack_ternary(Tensor packed, int n, int k, int n_padded, int k_padded, int layout) -> Tensor"
)
_DEF_LIBRARY.define(
    "quantize_activations(Tensor x, int granularity, float eps) -> (Tensor, Tensor)"
)
_DEF_LIBRARY.define(
    "bit_linear_fallback(Tensor x, Tensor packed, Tensor weight_scale, Tensor? bias, "
    "int n, int k, int n_padded, int k_padded, int layout, int weight_scale_mode, "
    "int activation_granularity, float alpha, int out_dtype) -> Tensor"
)

_COMPOSITE_LIBRARY = torch.library.Library("bitvideo", "IMPL", "CompositeExplicitAutograd")
_META_LIBRARY = torch.library.Library("bitvideo", "IMPL", "Meta")


def _unpack_impl(
    packed: torch.Tensor,
    n: int,
    k: int,
    n_padded: int,
    k_padded: int,
    layout: int,
) -> torch.Tensor:
    return _unpack_ternary_codes(
        packed,
        n,
        k,
        n_padded,
        k_padded,
        normalize_layout(layout),
    )


def _unpack_meta(
    packed: torch.Tensor,
    n: int,
    k: int,
    n_padded: int,
    k_padded: int,
    layout: int,
) -> torch.Tensor:
    del n_padded, k_padded, layout
    return packed.new_empty((n, k), dtype=torch.int8)


def _quantize_impl(
    x: torch.Tensor,
    granularity: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    validated = _validate_activation(x)
    return _quantize_activations_torch(
        validated,
        normalize_granularity(granularity),
        _validate_eps(eps),
    )


def _quantize_meta(
    x: torch.Tensor,
    granularity: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del eps
    normalized = normalize_granularity(granularity)
    k = x.shape[-1]
    m = x.numel() // k
    q = x.new_empty(x.shape, dtype=torch.int8)
    if normalized is ActivationGranularity.PER_TENSOR:
        scale_shape = (1,)
    elif normalized is ActivationGranularity.PER_TOKEN:
        scale_shape = (m,)
    else:
        scale_shape = (m, (k + 127) // 128)
    scales = x.new_empty(scale_shape, dtype=torch.float32)
    return q, scales


def _bit_linear_impl(
    x: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
    n_padded: int,
    k_padded: int,
    layout: int,
    weight_scale_mode: int,
    activation_granularity: int,
    alpha: float,
    out_dtype: int,
) -> torch.Tensor:
    validated_x = _validate_activation(x)
    scale_mode = normalize_scale_mode(weight_scale_mode)
    granularity = normalize_granularity(activation_granularity)
    if granularity is ActivationGranularity.GROUP_128:
        raise ValueError("bit_linear_fallback requires tensor or token activation scales")
    packed_object = PackedTernaryWeight(
        packed,
        weight_scale,
        n,
        k,
        n_padded,
        k_padded,
        WeightLayout(layout),
        ScaleMode(scale_mode),
    )
    output_dtype, _ = normalize_output_dtype(out_dtype, default=validated_x.dtype)
    normalized_bias = _normalize_bias(bias, n=n, device=validated_x.device)
    return _bit_linear_torch(
        validated_x,
        packed_object,
        normalized_bias,
        output_dtype,
        granularity,
        _validate_alpha(alpha),
    )


def _bit_linear_meta(
    x: torch.Tensor,
    packed: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    n: int,
    k: int,
    n_padded: int,
    k_padded: int,
    layout: int,
    weight_scale_mode: int,
    activation_granularity: int,
    alpha: float,
    out_dtype: int,
) -> torch.Tensor:
    del packed, weight_scale, bias, k, n_padded, k_padded
    del layout, weight_scale_mode, activation_granularity, alpha
    dtype, _ = normalize_output_dtype(out_dtype, default=x.dtype)
    output_shape = (*x.shape[:-1], n)
    return x.new_empty(output_shape, dtype=dtype)


_COMPOSITE_LIBRARY.impl("unpack_ternary", _unpack_impl)
_COMPOSITE_LIBRARY.impl("quantize_activations", _quantize_impl)
_COMPOSITE_LIBRARY.impl("bit_linear_fallback", _bit_linear_impl)
_META_LIBRARY.impl("unpack_ternary", _unpack_meta)
_META_LIBRARY.impl("quantize_activations", _quantize_meta)
_META_LIBRARY.impl("bit_linear_fallback", _bit_linear_meta)


def unpack_ternary_op(
    packed: torch.Tensor,
    n: int,
    k: int,
    n_padded: int,
    k_padded: int,
    layout: int | WeightLayout,
) -> torch.Tensor:
    return torch.ops.bitvideo.unpack_ternary(
        packed, n, k, n_padded, k_padded, int(layout)
    )


def quantize_activations_op(
    x: torch.Tensor,
    granularity: int | ActivationGranularity = ActivationGranularity.PER_TOKEN,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.bitvideo.quantize_activations(x, int(granularity), float(eps))


def bit_linear_fallback_op(
    x: torch.Tensor,
    packed: PackedTernaryWeight,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype | int | None = None,
    activation_granularity: int | ActivationGranularity = ActivationGranularity.PER_TOKEN,
    alpha: float = 1.0,
) -> torch.Tensor:
    _, dtype_enum = normalize_output_dtype(out_dtype, default=x.dtype)
    return torch.ops.bitvideo.bit_linear_fallback(
        x,
        packed.data,
        packed.scale,
        bias,
        packed.out_features,
        packed.in_features,
        packed.n_padded,
        packed.k_padded,
        int(packed.layout),
        int(packed.scale_mode),
        int(activation_granularity),
        float(alpha),
        int(dtype_enum),
    )
