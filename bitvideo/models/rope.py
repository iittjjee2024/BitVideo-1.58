"""Rotary position embeddings for sequence and factorized video coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _positive_even(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2:
        raise ValueError(f"{name} must be a positive even integer; got {value!r}")
    return value


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _triple(value: Sequence[int] | int, name: str) -> tuple[int, int, int]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integers, not bool")
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(value)
        if len(result) != 3:
            raise ValueError(f"{name} must contain exactly three values")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive integers; got {result}")
    return result


def _float_triple(
    value: Sequence[float] | float,
    name: str,
    *,
    positive: bool,
) -> tuple[float, float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = (float(value), float(value), float(value))
    else:
        result = tuple(float(item) for item in value)
        if len(result) != 3:
            raise ValueError(f"{name} must contain exactly three values")
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} values must be finite")
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _inverse_frequencies(
    dim: int,
    base: float,
    frequency_scale: float,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    # pair_index: [dim/2] float32; inverse: [dim/2] float32.
    pair_index = torch.arange(dim // 2, dtype=torch.float32, device=device)
    exponent = pair_index / float(dim // 2)
    return torch.pow(torch.tensor(base, dtype=torch.float32, device=device), -exponent) * float(
        frequency_scale
    )


@dataclass(frozen=True)
class RotaryFrequencies:
    """Cosine and sine tensors sharing shape ``[..., sequence, rotary_dim]``."""

    cos: torch.Tensor
    sin: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.cos, torch.Tensor) or not isinstance(self.sin, torch.Tensor):
            raise TypeError("cos and sin must be torch.Tensor instances")
        if self.cos.shape != self.sin.shape:
            raise ValueError(
                f"cos and sin shapes must match; got {tuple(self.cos.shape)} and "
                f"{tuple(self.sin.shape)}"
            )
        if self.cos.ndim < 2 or self.cos.shape[-1] <= 0 or self.cos.shape[-1] % 2:
            raise ValueError("rotary frequencies must have shape [..., sequence, even_dim]")
        if self.cos.dtype not in _FLOAT_DTYPES or self.sin.dtype != self.cos.dtype:
            raise TypeError("cos and sin must share a floating-point dtype")
        if self.cos.device != self.sin.device:
            raise ValueError("cos and sin must reside on the same device")

    @property
    def rotary_dim(self) -> int:
        return self.cos.shape[-1]

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "RotaryFrequencies":
        target_dtype = self.cos.dtype if dtype is None else dtype
        if target_dtype not in _FLOAT_DTYPES:
            raise TypeError(f"rotary frequency dtype must be floating point; got {target_dtype}")
        return RotaryFrequencies(
            # cos/sin: unchanged logical frequency shape.
            self.cos.to(device=device, dtype=target_dtype),
            self.sin.to(device=device, dtype=target_dtype),
        )


def rotate_half(x: torch.Tensor, *, interleaved: bool = False) -> torch.Tensor:
    """Rotate paired features by 90 degrees along the final dimension."""

    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch.Tensor")
    if x.shape[-1] <= 0 or x.shape[-1] % 2:
        raise ValueError(f"x.shape[-1] must be positive and even; got {x.shape[-1]}")
    if not isinstance(interleaved, bool):
        raise TypeError("interleaved must be bool")
    if interleaved:
        # paired: [..., rotary_dim/2, 2]; rotated: same shape.
        paired = x.unflatten(-1, (x.shape[-1] // 2, 2))
        rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
        return rotated.flatten(-2)
    # first/second: [..., rotary_dim/2]; output: [..., rotary_dim].
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _broadcast_frequencies(frequency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Align supplied batch/group prefixes and insert omitted head axes."""

    if frequency.ndim < 2 or frequency.ndim > x.ndim:
        raise ValueError("rotary frequency rank must be in [2, input rank]")
    frequency_prefix = tuple(frequency.shape[:-2])
    input_prefix = tuple(x.shape[:-2])
    if len(frequency_prefix) > len(input_prefix):
        raise ValueError("rotary frequency prefix rank exceeds the input prefix rank")
    for frequency_extent, input_extent in zip(frequency_prefix, input_prefix):
        if frequency_extent not in {1, input_extent}:
            raise ValueError(
                "rotary frequency batch/group prefixes must broadcast from the left; "
                f"got {frequency_prefix} for input prefix {input_prefix}"
            )
    # Missing singleton axes are inserted after the supplied batch/group prefix:
    # [B,L,R] -> [B,1,1,L,R] for x [B,G,H,L,D].
    missing_axes = (1,) * (len(input_prefix) - len(frequency_prefix))
    return frequency.reshape(*frequency_prefix, *missing_axes, *frequency.shape[-2:])


def apply_rotary_embedding(
    x: torch.Tensor,
    frequencies: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor],
    *,
    rotary_dim: int | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply RoPE to the leading rotary channels of ``x [..., L, D]``."""

    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("x must be a floating-point torch.Tensor")
    if x.ndim < 2 or x.shape[-2] <= 0 or x.shape[-1] <= 0:
        raise ValueError(f"x must have shape [..., L, D] with positive L,D; got {tuple(x.shape)}")
    if isinstance(frequencies, RotaryFrequencies):
        cos, sin = frequencies.cos, frequencies.sin
    else:
        if not isinstance(frequencies, tuple) or len(frequencies) != 2:
            raise TypeError("frequencies must be RotaryFrequencies or a (cos, sin) tuple")
        cos, sin = frequencies
        frequencies = RotaryFrequencies(cos, sin)
    selected_dim = frequencies.rotary_dim if rotary_dim is None else _positive_even(
        rotary_dim, "rotary_dim"
    )
    if selected_dim > x.shape[-1] or selected_dim > frequencies.rotary_dim:
        raise ValueError(
            f"rotary_dim={selected_dim} exceeds input/frequency dimensions "
            f"({x.shape[-1]}, {frequencies.rotary_dim})"
        )
    if cos.shape[-2] not in {1, x.shape[-2]}:
        raise ValueError(
            f"frequency sequence length must be 1 or {x.shape[-2]}; got {cos.shape[-2]}"
        )

    if not interleaved and selected_dim < frequencies.rotary_dim:
        # Non-interleaved partners occupy corresponding positions in the two
        # full-width halves; a plain prefix would select unmatched phases.
        selected_half = selected_dim // 2
        full_half = frequencies.rotary_dim // 2
        cos_selected = torch.cat(
            (cos[..., :selected_half], cos[..., full_half : full_half + selected_half]),
            dim=-1,
        )
        sin_selected = torch.cat(
            (sin[..., :selected_half], sin[..., full_half : full_half + selected_half]),
            dim=-1,
        )
    else:
        cos_selected = cos[..., :selected_dim]
        sin_selected = sin[..., :selected_dim]
    # cos/sin: broadcastable to x_rotary [..., L, selected_dim].
    cos = _broadcast_frequencies(cos_selected, x).to(device=x.device, dtype=x.dtype)
    sin = _broadcast_frequencies(sin_selected, x).to(device=x.device, dtype=x.dtype)
    # x_rotary: [..., L, selected_dim]; x_tail: [..., L, D-selected_dim].
    x_rotary = x[..., :selected_dim]
    x_tail = x[..., selected_dim:]
    rotated = x_rotary * cos + rotate_half(x_rotary, interleaved=interleaved) * sin
    # output: x.shape.
    return torch.cat((rotated, x_tail), dim=-1) if x_tail.shape[-1] else rotated


def apply_rotary_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    query_frequencies: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor],
    key_frequencies: RotaryFrequencies | tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    rotary_dim: int | None = None,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply potentially distinct RoPE frequencies to query and key tensors."""

    selected_key_frequencies = query_frequencies if key_frequencies is None else key_frequencies
    # rotated_query: query.shape; rotated_key: key.shape.
    rotated_query = apply_rotary_embedding(
        query,
        query_frequencies,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
    )
    rotated_key = apply_rotary_embedding(
        key,
        selected_key_frequencies,
        rotary_dim=rotary_dim,
        interleaved=interleaved,
    )
    return rotated_query, rotated_key


class RotaryEmbedding(nn.Module):
    """One-dimensional RoPE frequency generator with float32 phase math."""

    def __init__(
        self,
        dim: int,
        *,
        base: float = 10_000.0,
        frequency_scale: float = 1.0,
        interleaved: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dim = _positive_even(dim, "dim")
        self.base = _positive_float(base, "base")
        self.frequency_scale = _positive_float(frequency_scale, "frequency_scale")
        if not isinstance(interleaved, bool):
            raise TypeError("interleaved must be bool")
        self.interleaved = interleaved
        self.register_buffer(
            "inv_freq",
            _inverse_frequencies(
                self.dim,
                self.base,
                self.frequency_scale,
                device=device,
            ),
            persistent=True,
        )

    def _apply(self, fn, recurse: bool = True) -> "RotaryEmbedding":
        result = super()._apply(fn, recurse=recurse)
        # Module dtype conversion must not quantize the master phase frequencies.
        target_device = result.inv_freq.device
        result.inv_freq = _inverse_frequencies(
            result.dim,
            result.base,
            result.frequency_scale,
            device=target_device,
        )
        return result

    def forward(
        self,
        positions: int | torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> RotaryFrequencies:
        target_device = self.inv_freq.device if device is None else torch.device(device)
        target_dtype = torch.float32 if dtype is None else dtype
        if target_dtype not in _FLOAT_DTYPES:
            raise TypeError(f"dtype must be floating point; got {target_dtype}")
        if isinstance(positions, bool):
            raise TypeError("positions must be an integer length or a tensor")
        if isinstance(positions, int):
            if positions <= 0:
                raise ValueError("position count must be positive")
            # position_tensor: [L] float32.
            position_tensor = torch.arange(positions, device=target_device, dtype=torch.float32)
        elif isinstance(positions, torch.Tensor):
            if positions.ndim < 1 or positions.shape[-1] <= 0:
                raise ValueError("position tensor must have shape [..., L] with L > 0")
            if positions.dtype == torch.bool or positions.is_complex():
                raise TypeError("positions must use a real numeric dtype")
            # position_tensor: positions.shape in float32.
            position_tensor = positions.to(device=target_device, dtype=torch.float32)
        else:
            raise TypeError("positions must be an integer length or a torch.Tensor")

        # angles: positions.shape + [dim/2], accumulated in float32.
        angles = position_tensor.unsqueeze(-1) * self.inv_freq.to(target_device)
        if self.interleaved:
            # phase: positions.shape + [dim], adjacent features share a phase.
            phase = angles.repeat_interleave(2, dim=-1)
        else:
            # phase: positions.shape + [dim], half-split features share a phase.
            phase = torch.cat((angles, angles), dim=-1)
        return RotaryFrequencies(
            phase.cos().to(target_dtype),
            phase.sin().to(target_dtype),
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, base={self.base:g}, frequency_scale={self.frequency_scale:g}, "
            f"interleaved={self.interleaved}"
        )


class VideoRotaryEmbedding(nn.Module):
    """Factorized RoPE over temporal, height, and width token coordinates."""

    def __init__(
        self,
        dim: int,
        *,
        axis_dims: Sequence[int] | None = None,
        base: Sequence[float] | float = 10_000.0,
        frequency_scale: Sequence[float] | float = 1.0,
        interleaved: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dim = _positive_even(dim, "dim")
        if axis_dims is None:
            pairs = self.dim // 2
            if pairs < 3:
                raise ValueError("automatic 3D RoPE allocation requires dim >= 6")
            base_pairs, remainder = divmod(pairs, 3)
            pair_counts = tuple(base_pairs + int(axis < remainder) for axis in range(3))
            normalized_axis_dims = tuple(2 * count for count in pair_counts)
        else:
            normalized_axis_dims = tuple(axis_dims)
            if len(normalized_axis_dims) != 3:
                raise ValueError("axis_dims must contain temporal, height, and width dimensions")
            for axis, axis_dim in zip(("temporal", "height", "width"), normalized_axis_dims):
                _positive_even(axis_dim, f"{axis} axis dimension")
            if sum(normalized_axis_dims) != self.dim:
                raise ValueError(
                    f"axis_dims must sum to dim={self.dim}; got {normalized_axis_dims}"
                )
        self.axis_dims = normalized_axis_dims
        self.bases = _float_triple(base, "base", positive=True)
        self.frequency_scales = _float_triple(
            frequency_scale,
            "frequency_scale",
            positive=True,
        )
        if not isinstance(interleaved, bool):
            raise TypeError("interleaved must be bool")
        self.interleaved = interleaved
        for axis, axis_dim, axis_base, axis_scale in zip(
            ("t", "h", "w"),
            self.axis_dims,
            self.bases,
            self.frequency_scales,
        ):
            self.register_buffer(
                f"inv_freq_{axis}",
                _inverse_frequencies(
                    axis_dim,
                    axis_base,
                    axis_scale,
                    device=device,
                ),
                persistent=True,
            )

    def _apply(self, fn, recurse: bool = True) -> "VideoRotaryEmbedding":
        result = super()._apply(fn, recurse=recurse)
        # Regenerate all three float32 masters on the transformed device.
        target_device = result.inv_freq_t.device
        for axis, axis_dim, axis_base, axis_scale in zip(
            ("t", "h", "w"),
            result.axis_dims,
            result.bases,
            result.frequency_scales,
        ):
            setattr(
                result,
                f"inv_freq_{axis}",
                _inverse_frequencies(
                    axis_dim,
                    axis_base,
                    axis_scale,
                    device=target_device,
                ),
            )
        return result

    def positions(
        self,
        grid_size: Sequence[int] | int,
        *,
        offsets: Sequence[float] | float = (0.0, 0.0, 0.0),
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        temporal, height, width = _triple(grid_size, "grid_size")
        offset_t, offset_h, offset_w = _float_triple(offsets, "offsets", positive=False)
        target_device = self.inv_freq_t.device if device is None else torch.device(device)
        # t/h/w: [T], [H], [W] float32.
        t = torch.arange(temporal, device=target_device, dtype=torch.float32) + offset_t
        h = torch.arange(height, device=target_device, dtype=torch.float32) + offset_h
        w = torch.arange(width, device=target_device, dtype=torch.float32) + offset_w
        # mesh axes: [T,H,W]; positions: [T*H*W,3] in patch-flattening order.
        mesh_t, mesh_h, mesh_w = torch.meshgrid(t, h, w, indexing="ij")
        return torch.stack((mesh_t, mesh_h, mesh_w), dim=-1).reshape(-1, 3)

    def frequencies_from_positions(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> RotaryFrequencies:
        if not isinstance(positions, torch.Tensor):
            raise TypeError("positions must be a torch.Tensor")
        if positions.ndim < 2 or positions.shape[-1] != 3 or positions.shape[-2] <= 0:
            raise ValueError("positions must have shape [..., L, 3] with L > 0")
        if positions.dtype == torch.bool or positions.is_complex():
            raise TypeError("positions must use a real numeric dtype")
        target_dtype = torch.float32 if dtype is None else dtype
        if target_dtype not in _FLOAT_DTYPES:
            raise TypeError(f"dtype must be floating point; got {target_dtype}")
        # coordinates: three tensors with shape positions.shape[:-1].
        coordinates = positions.to(dtype=torch.float32).unbind(dim=-1)
        axis_angles: list[torch.Tensor] = []
        for coordinate, inverse in zip(
            coordinates,
            (self.inv_freq_t, self.inv_freq_h, self.inv_freq_w),
        ):
            # angles: [..., L, axis_dim/2].
            axis_angles.append(coordinate.unsqueeze(-1) * inverse.to(positions.device))
        if self.interleaved:
            # Interleaved pairs remain adjacent within each axis allocation.
            phases = [angles.repeat_interleave(2, dim=-1) for angles in axis_angles]
            full_phase = torch.cat(phases, dim=-1)
        else:
            # Half-split rotation pairs the first global half with the second;
            # concatenate all axis half-phases before duplicating them.
            combined_angles = torch.cat(axis_angles, dim=-1)
            full_phase = torch.cat((combined_angles, combined_angles), dim=-1)
        # full_phase/cos/sin: [..., L, dim].
        return RotaryFrequencies(
            full_phase.cos().to(target_dtype),
            full_phase.sin().to(target_dtype),
        )

    def forward(
        self,
        grid_size: Sequence[int] | int,
        *,
        offsets: Sequence[float] | float = (0.0, 0.0, 0.0),
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> RotaryFrequencies:
        # positions: [T*H*W,3]; frequencies: [T*H*W,dim].
        positions = self.positions(grid_size, offsets=offsets, device=device)
        return self.frequencies_from_positions(positions, dtype=dtype)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, axis_dims={self.axis_dims}, bases={self.bases}, "
            f"frequency_scales={self.frequency_scales}, interleaved={self.interleaved}"
        )
