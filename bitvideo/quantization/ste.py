"""Straight-through estimators used by BitVideo quantization-aware training."""

from __future__ import annotations

import torch

_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")


def _require_floating(x: torch.Tensor, name: str = "x") -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be floating point; got {x.dtype}")


def _finite_gradient_scale(value: float, name: str = "gradient_scale") -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if result != result or result < -_FLOAT32_MAX or result > _FLOAT32_MAX:
        raise ValueError(
            f"{name} must be finite and representable as float32; got {value!r}"
        )
    return result


def _assert_scalar_tensor(condition: torch.Tensor, message: str) -> None:
    """Issue a compile-visible assertion for a scalar boolean tensor."""

    if condition.device.type != "meta":
        # condition: [] bool. _check_tensor_all reports a recoverable host-side
        # runtime error on CUDA rather than launching a device assertion.
        torch._check_tensor_all(condition, lambda: message)


def ste_replace(
    source: torch.Tensor,
    quantized: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
) -> torch.Tensor:
    """Use ``quantized`` exactly in forward and a scaled source gradient.

    Both tensors must have the same shape. The returned tensor uses the source
    dtype/device. ``quantized`` is a non-differentiable numerical target.
    """

    _require_floating(source, "source")
    if not isinstance(quantized, torch.Tensor):
        raise TypeError("quantized must be a torch.Tensor")
    if source.shape != quantized.shape:
        raise ValueError(
            f"source and quantized shapes must match; got {tuple(source.shape)} and "
            f"{tuple(quantized.shape)}"
        )
    scale = _finite_gradient_scale(gradient_scale)
    # target: source.shape in source storage dtype/device.
    target = quantized.to(dtype=source.dtype, device=source.device).detach()
    # surrogate_zero: source.shape. Subtracting identical finite values first
    # produces exact zero without target cancellation or scale overflow. At
    # nonfinite source positions the surrogate and its gradient are both zero.
    finite_source = torch.isfinite(source)
    surrogate_zero = torch.where(
        finite_source,
        source - source.detach(),
        torch.zeros_like(source),
    )
    # output: source.shape with independent storage and exact finite target values.
    return target + surrogate_zero * scale


def round_ste(x: torch.Tensor, *, gradient_scale: float = 1.0) -> torch.Tensor:
    """Round in forward and pass a scaled identity gradient in backward."""

    _require_floating(x)
    # target/output: x.shape.
    return ste_replace(x, torch.round(x), gradient_scale=gradient_scale)


def floor_ste(x: torch.Tensor, *, gradient_scale: float = 1.0) -> torch.Tensor:
    """Floor in forward and pass a scaled identity gradient in backward."""

    _require_floating(x)
    # target/output: x.shape.
    return ste_replace(x, torch.floor(x), gradient_scale=gradient_scale)


def sign_ste(x: torch.Tensor, *, gradient_scale: float = 1.0) -> torch.Tensor:
    """Take the mathematical sign in forward with an identity-style gradient."""

    _require_floating(x)
    # target/output: x.shape.
    return ste_replace(x, torch.sign(x), gradient_scale=gradient_scale)


def clamp_ste(
    x: torch.Tensor,
    minimum: float | torch.Tensor,
    maximum: float | torch.Tensor,
    *,
    gradient_scale: float = 1.0,
) -> torch.Tensor:
    """Clamp in forward while passing gradient through saturated values."""

    _require_floating(x)
    # lower/upper: scalar or broadcastable bound shapes.
    lower = torch.as_tensor(minimum, dtype=x.dtype, device=x.device)
    upper = torch.as_tensor(maximum, dtype=x.dtype, device=x.device)
    # valid_bounds: [] bool after broadcasting the bound tensors.
    valid_bounds = torch.all(lower <= upper)
    _assert_scalar_tensor(valid_bounds, "minimum cannot exceed maximum or contain NaN")
    # quantized/output: x.shape after broadcasting bounds over x.
    quantized = torch.maximum(torch.minimum(x, upper), lower)
    return ste_replace(x, quantized, gradient_scale=gradient_scale)


def saturating_clamp_ste(
    x: torch.Tensor,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Clamp in forward and suppress gradients outside the clipping interval."""

    _require_floating(x)
    if isinstance(minimum, bool) or isinstance(maximum, bool):
        raise TypeError("minimum and maximum must be real numbers, not bool")
    lower, upper = float(minimum), float(maximum)
    if lower != lower or upper != upper or lower > upper:
        raise ValueError("minimum cannot exceed maximum or contain NaN")
    # lower_tensor/upper_tensor: []; valid_bounds: [] bool.
    lower_tensor = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    upper_tensor = torch.as_tensor(upper, dtype=x.dtype, device=x.device)
    _assert_scalar_tensor(
        torch.all(lower_tensor <= upper_tensor),
        "minimum cannot exceed maximum or contain NaN",
    )
    # mask/identity/clamped: x.shape. The detached mask produces dY/dX in {0,1}.
    mask = ((x >= lower_tensor) & (x <= upper_tensor)).to(x.dtype).detach()
    identity = x * mask
    clamped = torch.maximum(torch.minimum(x, upper_tensor), lower_tensor)
    return ste_replace(identity, clamped, gradient_scale=1.0)


def ternary_ste(
    x: torch.Tensor,
    threshold: float | torch.Tensor,
    *,
    scale: float | torch.Tensor = 1.0,
    gradient_scale: float = 1.0,
) -> torch.Tensor:
    """Quantize to ``{-scale, 0, +scale}`` with an identity-style gradient."""

    _require_floating(x)
    # threshold_tensor/scale_tensor: scalar or broadcastable parameter shapes.
    threshold_tensor = torch.as_tensor(threshold, dtype=x.dtype, device=x.device)
    scale_tensor = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    # Validity conditions are scalar booleans and apply equally to tensor inputs.
    valid_threshold = torch.all(torch.isfinite(threshold_tensor) & (threshold_tensor >= 0))
    valid_scale = torch.all(torch.isfinite(scale_tensor) & (scale_tensor >= 0))
    _assert_scalar_tensor(valid_threshold, "threshold must be finite and non-negative")
    _assert_scalar_tensor(valid_scale, "scale must be finite and non-negative")
    # quantized: x.shape via broadcasting threshold/scale over x.
    quantized = torch.where(
        x > threshold_tensor,
        scale_tensor,
        torch.where(x < -threshold_tensor, -scale_tensor, torch.zeros_like(x)),
    )
    return ste_replace(x, quantized, gradient_scale=gradient_scale)


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Keep the exact forward value while multiplying its gradient."""

    _require_floating(x)
    factor = _finite_gradient_scale(scale, "scale")
    # target/output: x.shape with independently allocated exact values.
    return ste_replace(x, x.detach(), gradient_scale=factor)
