"""Training-time ternary weight and symmetric activation quantization."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitvideo.ops.types import (
    ActivationGranularity,
    ScaleMode,
    normalize_granularity,
    normalize_scale_mode,
)

from .ste import ste_replace

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_FLOAT32_HALF_MIN_SUBNORMAL = float.fromhex("0x1p-150")


def _positive_float32_value(value: float, name: str) -> float:
    """Validate a positive finite scalar that remains positive in float32."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if result != result or result <= 0.0 or result > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite, positive, and representable as float32")
    # Round-to-nearest-even produces zero at exactly half the minimum subnormal.
    if result <= _FLOAT32_HALF_MIN_SUBNORMAL:
        raise ValueError(f"{name} must remain positive when rounded to float32")
    return result


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if result != result or result < -_FLOAT32_MAX or result > _FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and representable as float32")
    return result


@dataclass(frozen=True)
class WeightQuantizationConfig:
    """Configuration for BitNet-style absmean ternary weight quantization."""

    scale_mode: ScaleMode = ScaleMode.PER_CHANNEL
    eps: float = 1.0e-5
    threshold_factor: float = 0.5
    gradient_scale: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_mode", normalize_scale_mode(self.scale_mode))
        object.__setattr__(
            self,
            "eps",
            _positive_float32_value(self.eps, "weight quantization eps"),
        )
        threshold_factor = _finite_float(self.threshold_factor, "threshold_factor")
        if threshold_factor < 0.0:
            raise ValueError("threshold_factor must be non-negative")
        object.__setattr__(self, "threshold_factor", threshold_factor)
        object.__setattr__(
            self,
            "gradient_scale",
            _finite_float(self.gradient_scale, "gradient_scale"),
        )
        if not isinstance(self.enabled, bool):
            raise TypeError("weight quantization enabled must be bool")


@dataclass(frozen=True)
class ActivationQuantizationConfig:
    """Configuration for dynamic symmetric activation fake quantization."""

    granularity: ActivationGranularity = ActivationGranularity.PER_TOKEN
    bits: int = 8
    eps: float = 1.0e-5
    group_size: int = 128
    gradient_scale: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "granularity", normalize_granularity(self.granularity))
        if isinstance(self.bits, bool) or not isinstance(self.bits, int) or not 2 <= self.bits <= 8:
            raise ValueError("activation bits must be an integer in [2, 8]")
        object.__setattr__(
            self,
            "eps",
            _positive_float32_value(self.eps, "activation quantization eps"),
        )
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size <= 0
        ):
            raise ValueError("group_size must be a positive integer")
        object.__setattr__(
            self,
            "gradient_scale",
            _finite_float(self.gradient_scale, "gradient_scale"),
        )
        if not isinstance(self.enabled, bool):
            raise TypeError("activation quantization enabled must be bool")

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    @property
    def qmin(self) -> int:
        return -self.qmax


@dataclass(frozen=True)
class QuantizationConfig:
    """Combined QAT configuration used by :class:`BitLinear`."""

    weight: WeightQuantizationConfig = field(default_factory=WeightQuantizationConfig)
    activation: ActivationQuantizationConfig = field(default_factory=ActivationQuantizationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.weight, WeightQuantizationConfig):
            raise TypeError("weight must be a WeightQuantizationConfig")
        if not isinstance(self.activation, ActivationQuantizationConfig):
            raise TypeError("activation must be an ActivationQuantizationConfig")


@dataclass(frozen=True)
class TernaryQuantizedTensor:
    """Integer ternary values and their float32 dequantization scale."""

    values: torch.Tensor
    scale: torch.Tensor
    scale_mode: ScaleMode

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor) or not isinstance(self.scale, torch.Tensor):
            raise TypeError("ternary values and scale must be torch.Tensor instances")
        mode = normalize_scale_mode(self.scale_mode)
        object.__setattr__(self, "scale_mode", mode)
        if self.values.dtype is not torch.int8:
            raise TypeError("ternary values must use torch.int8")
        if self.values.ndim != 2 or self.values.shape[0] <= 0 or self.values.shape[1] <= 0:
            raise ValueError(
                f"ternary values must have positive shape [N, K]; got {tuple(self.values.shape)}"
            )
        if self.scale.dtype is not torch.float32 or self.scale.ndim != 1:
            raise TypeError("ternary scale must be a one-dimensional torch.float32 tensor")
        expected_scales = self.values.shape[0] if mode is ScaleMode.PER_CHANNEL else 1
        if self.scale.numel() != expected_scales:
            raise ValueError(
                f"ternary scale has {self.scale.numel()} values, expected {expected_scales}"
            )
        if not self.scale.is_contiguous():
            raise ValueError("ternary scale must be contiguous")
        if self.values.device != self.scale.device:
            raise ValueError("ternary values and scale must be on the same device")


@dataclass(frozen=True)
class ActivationQuantizedTensor:
    """Integer activation values and dynamic float32 scales."""

    values: torch.Tensor
    scale: torch.Tensor
    granularity: ActivationGranularity
    group_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor) or not isinstance(self.scale, torch.Tensor):
            raise TypeError("activation values and scale must be torch.Tensor instances")
        granularity = normalize_granularity(self.granularity)
        object.__setattr__(self, "granularity", granularity)
        if (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size <= 0
        ):
            raise ValueError("group_size must be a positive integer")
        if self.values.dtype is not torch.int8:
            raise TypeError("quantized activations must use torch.int8")
        if self.values.ndim < 1 or self.values.shape[-1] <= 0:
            raise ValueError(
                "quantized activations must have shape [..., K] with K > 0; "
                f"got {tuple(self.values.shape)}"
            )
        if self.scale.dtype is not torch.float32:
            raise TypeError("activation scales must use torch.float32")
        if not self.scale.is_contiguous():
            raise ValueError("activation scale must be contiguous")

        k = self.values.shape[-1]
        m = self.values.numel() // k
        if granularity is ActivationGranularity.PER_TENSOR:
            expected_shape = (1,)
        elif granularity is ActivationGranularity.PER_TOKEN:
            expected_shape = (m,)
        else:
            groups = (k + self.group_size - 1) // self.group_size
            expected_shape = (m, groups)
        if tuple(self.scale.shape) != expected_shape:
            raise ValueError(
                f"activation scale has shape {tuple(self.scale.shape)}, expected {expected_shape}"
            )
        if self.values.device != self.scale.device:
            raise ValueError("quantized activations and scales must be on the same device")


def _resolve_weight_config(
    config: WeightQuantizationConfig | None,
) -> WeightQuantizationConfig:
    if config is None:
        return WeightQuantizationConfig()
    if not isinstance(config, WeightQuantizationConfig):
        raise TypeError("config must be a WeightQuantizationConfig or None")
    return config


def _resolve_activation_config(
    config: ActivationQuantizationConfig | None,
) -> ActivationQuantizationConfig:
    if config is None:
        return ActivationQuantizationConfig()
    if not isinstance(config, ActivationQuantizationConfig):
        raise TypeError("config must be an ActivationQuantizationConfig or None")
    return config


def _validate_weight(weight: torch.Tensor) -> None:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N, K]; got {tuple(weight.shape)}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError("weight extents must be positive")
    if weight.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"weight must be floating point; got {weight.dtype}")
    if weight.device.type != "meta":
        # float32_weight: [N, K]. Finite float64 values outside the float32
        # range become infinities here and are rejected with literal infinities.
        float32_weight = weight.to(torch.float32)
        # representable: [N, K] bool; NaNs remain allowed and map to zero.
        representable = ~torch.isinf(float32_weight)
        torch._check_tensor_all(
            representable,
            lambda: "weight values must be finite or NaN and representable as float32",
        )


def weight_absmean_scale(
    weight: torch.Tensor,
    *,
    scale_mode: int | ScaleMode = ScaleMode.PER_CHANNEL,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Return FP64-accumulated absmean scales as float32 ``[1]`` or ``[N]``."""

    _validate_weight(weight)
    mode = normalize_scale_mode(scale_mode)
    epsilon = _positive_float32_value(eps, "eps")
    # wide_magnitude: [N, K] float64 for threshold-stable accumulation.
    wide_magnitude = weight.to(torch.float64).abs()
    if mode is ScaleMode.PER_CHANNEL:
        # wide_scale: [N] float64; scale: [N] float32.
        wide_scale = wide_magnitude.mean(dim=1)
    else:
        # wide_scale: [] float64; scale: [1] float32.
        wide_scale = wide_magnitude.mean()
    scale = wide_scale.to(torch.float32).reshape(-1)
    # epsilon_tensor/scale: [N] or [1] float32. fmax maps a NaN mean to eps.
    epsilon_tensor = torch.full_like(scale, epsilon)
    result = torch.fmax(scale, epsilon_tensor).contiguous()
    if result.device.type != "meta":
        # finite_scale: [N] or [1] bool.
        finite_scale = torch.isfinite(result)
        torch._check_tensor_all(
            finite_scale,
            lambda: "weight absmean scale must be finite",
        )
    return result


def quantize_ternary_weight(
    weight: torch.Tensor,
    config: WeightQuantizationConfig | None = None,
) -> TernaryQuantizedTensor:
    """Quantize a floating ``[N, K]`` matrix to integer ``{-1, 0, +1}``."""

    cfg = _resolve_weight_config(config)
    scale = weight_absmean_scale(weight, scale_mode=cfg.scale_mode, eps=cfg.eps)
    # threshold: [N, 1] or [1, 1]; values: [N, K] float32.
    threshold = (
        (cfg.threshold_factor * scale).view(-1, 1)
        if cfg.scale_mode is ScaleMode.PER_CHANNEL
        else (cfg.threshold_factor * scale).view(1, 1)
    )
    values = weight.to(torch.float32)
    # quantized: [N, K] int8. NaN comparisons are false and therefore map to zero.
    quantized = torch.where(
        values > threshold,
        torch.ones((), dtype=torch.int8, device=weight.device),
        torch.where(
            values < -threshold,
            -torch.ones((), dtype=torch.int8, device=weight.device),
            torch.zeros((), dtype=torch.int8, device=weight.device),
        ),
    ).contiguous()
    return TernaryQuantizedTensor(quantized, scale, cfg.scale_mode)


def dequantize_ternary_weight(quantized: TernaryQuantizedTensor) -> torch.Tensor:
    """Dequantize to float32 ``[N, K]``."""

    if not isinstance(quantized, TernaryQuantizedTensor):
        raise TypeError("quantized must be a TernaryQuantizedTensor")
    n = quantized.values.shape[0]
    # scale: [N, 1] or [1, 1]; output: [N, K] float32.
    scale = (
        quantized.scale.view(n, 1)
        if quantized.scale_mode is ScaleMode.PER_CHANNEL
        else quantized.scale.view(1, 1)
    )
    return quantized.values.to(torch.float32) * scale


def fake_quantize_ternary_weight(
    weight: torch.Tensor,
    config: WeightQuantizationConfig | None = None,
) -> torch.Tensor:
    """Return dequantized ternary weights with a straight-through gradient."""

    cfg = _resolve_weight_config(config)
    if not cfg.enabled:
        return weight
    quantized = quantize_ternary_weight(weight, cfg)
    # dequantized: [N, K] in the master-weight storage dtype.
    dequantized = dequantize_ternary_weight(quantized).to(weight.dtype)
    return ste_replace(weight, dequantized, gradient_scale=cfg.gradient_scale)


def _validate_activation(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim < 1 or x.shape[-1] <= 0:
        raise ValueError(f"x must have shape [..., K] with K > 0; got {tuple(x.shape)}")
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"x must be floating point; got {x.dtype}")


def quantize_activation_tensor(
    x: torch.Tensor,
    config: ActivationQuantizationConfig | None = None,
) -> ActivationQuantizedTensor:
    """Dynamically quantize ``x [..., K]`` to symmetric signed INT8 values."""

    _validate_activation(x)
    cfg = _resolve_activation_config(config)
    original_shape = tuple(x.shape)
    k = x.shape[-1]
    m = x.numel() // k
    qmax = cfg.qmax
    # values: [M, K] float32. The finite policy matches the inference operator.
    limit = torch.finfo(torch.float32).max
    values = torch.nan_to_num(
        x.reshape(m, k).to(torch.float32),
        nan=0.0,
        posinf=limit,
        neginf=-limit,
    )

    if cfg.granularity is ActivationGranularity.PER_TENSOR:
        if m == 0:
            # q: [..., K] int8; scale: [1] float32.
            q = torch.empty(original_shape, dtype=torch.int8, device=x.device)
            scale = torch.full((1,), cfg.eps / qmax, dtype=torch.float32, device=x.device)
            return ActivationQuantizedTensor(q, scale, cfg.granularity, cfg.group_size)
        # maximum/scale/inverse_scale: [1]; normalized: [M, K].
        maximum = values.abs().amax().reshape(1)
        scale = torch.clamp_min(maximum, cfg.eps) / qmax
        inverse_scale = scale.reciprocal()
        normalized = values * inverse_scale.view(1, 1)
    elif cfg.granularity is ActivationGranularity.PER_TOKEN:
        # maximum/scale/inverse_scale: [M]; normalized: [M, K].
        maximum = values.abs().amax(dim=1)
        scale = torch.clamp_min(maximum, cfg.eps) / qmax
        inverse_scale = scale.reciprocal()
        normalized = values * inverse_scale.view(m, 1)
    else:
        groups = (k + cfg.group_size - 1) // cfg.group_size
        padded_k = groups * cfg.group_size
        # grouped: [M, groups, group_size], zero padded along K.
        grouped = F.pad(values, (0, padded_k - k)).view(m, groups, cfg.group_size)
        # scale/inverse_scale: [M, groups]; element_inverse_scale: [M, K].
        scale = torch.clamp_min(grouped.abs().amax(dim=2), cfg.eps) / qmax
        inverse_scale = scale.reciprocal()
        element_inverse_scale = inverse_scale.repeat_interleave(cfg.group_size, dim=1)[:, :k]
        normalized = values * element_inverse_scale

    # q_2d: [M, K] int8; q: original x shape.
    q_2d = torch.round(normalized).clamp_(cfg.qmin, cfg.qmax).to(torch.int8)
    return ActivationQuantizedTensor(
        q_2d.reshape(original_shape),
        scale.contiguous(),
        cfg.granularity,
        cfg.group_size,
    )


def dequantize_activation(quantized: ActivationQuantizedTensor) -> torch.Tensor:
    """Dequantize a dynamic activation tensor to float32 with its original shape."""

    if not isinstance(quantized, ActivationQuantizedTensor):
        raise TypeError("quantized must be an ActivationQuantizedTensor")
    original_shape = tuple(quantized.values.shape)
    k = original_shape[-1]
    m = quantized.values.numel() // k
    # values: [M, K] float32.
    values = quantized.values.reshape(m, k).to(torch.float32)
    if quantized.granularity is ActivationGranularity.PER_TENSOR:
        # element_scale: [1, 1].
        element_scale = quantized.scale.view(1, 1)
    elif quantized.granularity is ActivationGranularity.PER_TOKEN:
        # element_scale: [M, 1].
        element_scale = quantized.scale.view(m, 1)
    else:
        # element_scale: [M, K].
        element_scale = quantized.scale.repeat_interleave(
            quantized.group_size,
            dim=1,
        )[:, :k]
    # output: original activation shape, float32.
    return (values * element_scale).reshape(original_shape)


def fake_quantize_activation(
    x: torch.Tensor,
    config: ActivationQuantizationConfig | None = None,
) -> torch.Tensor:
    """Return dequantized activations with a straight-through gradient."""

    cfg = _resolve_activation_config(config)
    if not cfg.enabled:
        return x
    quantized = quantize_activation_tensor(x, cfg)
    # dequantized: x.shape in the source activation storage dtype.
    dequantized = dequantize_activation(quantized).to(x.dtype)
    return ste_replace(x, dequantized, gradient_scale=cfg.gradient_scale)


class TernaryWeightQuantizer(nn.Module):
    """Reusable module wrapper around ternary weight fake quantization."""

    config: WeightQuantizationConfig

    def __init__(self, config: WeightQuantizationConfig | None = None) -> None:
        super().__init__()
        self.config = _resolve_weight_config(config)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return fake_quantize_ternary_weight(weight, self.config)

    def quantize(self, weight: torch.Tensor) -> TernaryQuantizedTensor:
        return quantize_ternary_weight(weight, self.config)

    def extra_repr(self) -> str:
        return (
            f"scale_mode={self.config.scale_mode.name}, eps={self.config.eps:g}, "
            f"threshold_factor={self.config.threshold_factor:g}, enabled={self.config.enabled}"
        )


class ActivationQuantizer(nn.Module):
    """Reusable module wrapper around dynamic activation fake quantization."""

    config: ActivationQuantizationConfig

    def __init__(self, config: ActivationQuantizationConfig | None = None) -> None:
        super().__init__()
        self.config = _resolve_activation_config(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quantize_activation(x, self.config)

    def quantize(self, x: torch.Tensor) -> ActivationQuantizedTensor:
        return quantize_activation_tensor(x, self.config)

    def extra_repr(self) -> str:
        return (
            f"granularity={self.config.granularity.name}, bits={self.config.bits}, "
            f"eps={self.config.eps:g}, group_size={self.config.group_size}, "
            f"enabled={self.config.enabled}"
        )
