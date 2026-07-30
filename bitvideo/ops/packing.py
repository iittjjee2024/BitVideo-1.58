"""Exact two-bit ternary packing, unpacking, and layout conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .backends import get_extension, is_compiling, requested_backend
from .types import (
    Backend,
    ScaleMode,
    WeightLayout,
    layout_alignment,
    normalize_backend,
    normalize_layout,
    normalize_scale_mode,
    packed_word_count,
    padded_extents,
)

_CODES_PER_WORD = 16
_MMA_TILE_BYTES = 512
_MMA_BYTES_PER_LANE = 16
_FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
_FLOAT32_HALF_MIN_SUBNORMAL = float.fromhex("0x1p-150")


@dataclass(frozen=True)
class PackedTernaryWeight:
    """A packed ternary matrix and all metadata required to consume it.

    ``data`` stores sixteen two-bit codes per signed ``torch.int32`` word.
    ``scale`` is always float32 and has shape ``[1]`` or ``[out_features]``.
    """

    data: torch.Tensor
    scale: torch.Tensor
    out_features: int
    in_features: int
    n_padded: int
    k_padded: int
    layout: WeightLayout = WeightLayout.ROW_MAJOR
    scale_mode: ScaleMode = ScaleMode.PER_TENSOR

    def __post_init__(self) -> None:
        layout = normalize_layout(self.layout)
        scale_mode = normalize_scale_mode(self.scale_mode)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "scale_mode", scale_mode)
        metadata = {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "n_padded": self.n_padded,
            "k_padded": self.k_padded,
        }
        for name, value in metadata.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer; got {type(value).__name__}")
            if value > 2**31 - 1:
                raise OverflowError(f"{name} exceeds INT32_MAX: {value}")
        if self.out_features <= 0 or self.in_features <= 0:
            raise ValueError("logical packed-weight extents must be positive")
        if self.n_padded < self.out_features or self.k_padded < self.in_features:
            raise ValueError("padded extents cannot be smaller than logical extents")
        n_multiple, k_multiple = layout_alignment(layout)
        if self.n_padded % n_multiple or self.k_padded % k_multiple:
            raise ValueError(
                f"layout {layout.name} requires padded multiples "
                f"({n_multiple}, {k_multiple})"
            )
        expected_words = packed_word_count(layout, self.n_padded, self.k_padded)
        if self.data.dtype is not torch.int32 or self.data.ndim != 1:
            raise TypeError("packed data must be a one-dimensional torch.int32 tensor")
        if not self.data.is_contiguous():
            raise ValueError("packed data must be contiguous")
        required_alignment = (
            16
            if layout in {WeightLayout.BLOCKED_N64, WeightLayout.MMA_INTERLEAVED}
            else 4
        )
        if self.data.device.type != "meta" and self.data.data_ptr() % required_alignment:
            raise ValueError(
                f"layout {layout.name} requires {required_alignment}-byte packed-data alignment"
            )
        if self.data.numel() != expected_words:
            raise ValueError(
                f"packed data has {self.data.numel()} words, expected {expected_words}"
            )
        if self.scale.dtype is not torch.float32 or self.scale.ndim != 1:
            raise TypeError("weight scale must be a one-dimensional torch.float32 tensor")
        if not self.scale.is_contiguous():
            raise ValueError("weight scale must be contiguous")
        expected_scales = self.out_features if scale_mode is ScaleMode.PER_CHANNEL else 1
        if self.scale.numel() != expected_scales:
            raise ValueError(
                f"weight scale has {self.scale.numel()} elements, expected {expected_scales}"
            )
        if self.scale.device.type != "meta":
            finite_positive = torch.isfinite(self.scale) & (self.scale > 0)
            if not bool(finite_positive.all().item()):
                raise ValueError("all weight scales must be finite and positive")
        if self.data.device != self.scale.device:
            raise ValueError("packed data and weight scale must reside on the same device")

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features

    @property
    def packed_bytes(self) -> int:
        return self.data.numel() * self.data.element_size()

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "PackedTernaryWeight":
        """Move packed storage and scales without changing their fixed dtypes."""

        return PackedTernaryWeight(
            data=self.data.to(device=device, non_blocking=non_blocking),
            scale=self.scale.to(device=device, non_blocking=non_blocking),
            out_features=self.out_features,
            in_features=self.in_features,
            n_padded=self.n_padded,
            k_padded=self.k_padded,
            layout=self.layout,
            scale_mode=self.scale_mode,
        )

    def detach(self) -> "PackedTernaryWeight":
        return PackedTernaryWeight(
            data=self.data.detach(),
            scale=self.scale.detach(),
            out_features=self.out_features,
            in_features=self.in_features,
            n_padded=self.n_padded,
            k_padded=self.k_padded,
            layout=self.layout,
            scale_mode=self.scale_mode,
        )

    def clone(self) -> "PackedTernaryWeight":
        return PackedTernaryWeight(
            data=self.data.clone(),
            scale=self.scale.clone(),
            out_features=self.out_features,
            in_features=self.in_features,
            n_padded=self.n_padded,
            k_padded=self.k_padded,
            layout=self.layout,
            scale_mode=self.scale_mode,
        )


def _flatten_packed(value: PackedTernaryWeight) -> tuple[list[torch.Tensor], tuple[Any, ...]]:
    context = (
        value.out_features,
        value.in_features,
        value.n_padded,
        value.k_padded,
        int(value.layout),
        int(value.scale_mode),
    )
    return [value.data, value.scale], context


def _unflatten_packed(
    tensors: list[torch.Tensor], context: tuple[Any, ...]
) -> PackedTernaryWeight:
    n, k, np, kp, layout, scale_mode = context
    return PackedTernaryWeight(
        tensors[0], tensors[1], n, k, np, kp, WeightLayout(layout), ScaleMode(scale_mode)
    )


try:
    from torch.utils import _pytree

    _pytree.register_pytree_node(PackedTernaryWeight, _flatten_packed, _unflatten_packed)
except (ImportError, ValueError):
    # ImportError covers minimal torch builds; ValueError means another import
    # path already registered this exact class.
    pass


def _validate_eps(eps: float) -> float:
    value = float(eps)
    if value != value or value <= 0.0 or value > _FLOAT32_MAX:
        raise ValueError(
            f"eps must be finite, positive, and representable as float32; got {eps!r}"
        )
    if value <= _FLOAT32_HALF_MIN_SUBNORMAL:
        raise ValueError(f"eps must remain positive when rounded to float32; got {eps!r}")
    return value


def _validate_weight(weight: torch.Tensor, *, floating: bool) -> torch.Tensor:
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N, K]; got {tuple(weight.shape)}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError(f"weight extents must be positive; got {tuple(weight.shape)}")
    if floating:
        if weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise TypeError("floating weight must use float16, bfloat16, or float32")
    elif weight.dtype is not torch.int8:
        raise TypeError("ternary weight must use torch.int8")
    return weight.contiguous()


def _mma_linear_slots(np: int, kp: int, device: torch.device) -> torch.Tensor:
    """Map every logical ``[Np, Kp]`` element to a packed slot index."""

    # n_index: [Np, 1]; k_index: [1, Kp].
    n_index = torch.arange(np, device=device, dtype=torch.int64).view(np, 1)
    k_index = torch.arange(kp, device=device, dtype=torch.int64).view(1, kp)
    k_blocks = kp // 256
    n_tile = n_index // 8
    g = n_index % 8
    k_block = k_index // 256
    kk = k_index % 256
    sub_tile = kk // 32
    k_inside = kk % 32
    half = k_inside // 16
    pos = k_inside % 16
    lane = 4 * g + pos // 4
    byte_in_lane = 2 * sub_tile + half
    tile = n_tile * k_blocks + k_block
    byte_offset = tile * _MMA_TILE_BYTES + lane * _MMA_BYTES_PER_LANE + byte_in_lane
    word = byte_offset // 4
    slot = (byte_offset % 4) * 4 + pos % 4
    # linear_slot: [Np, Kp], a permutation of range(Np*Kp).
    return word * _CODES_PER_WORD + slot


def _slots_to_words(slots: torch.Tensor) -> torch.Tensor:
    """Pack an integer ``[..., 16]`` code tensor into signed int32 words."""

    # shifts: [16], broadcast over every leading word dimension.
    shifts = 2 * torch.arange(_CODES_PER_WORD, device=slots.device, dtype=torch.int64)
    # words: slots.shape[:-1], low two bits from each slot.
    words = torch.sum((slots.to(torch.int64) & 3) << shifts, dim=-1)
    return words.to(torch.int32)


def _words_to_slots(words: torch.Tensor) -> torch.Tensor:
    """Expand signed int32 words into a ``[num_words, 16]`` code tensor."""

    # unsigned_words: [num_words, 1]; shifts: [1, 16].
    unsigned_words = (words.to(torch.int64) & 0xFFFFFFFF).unsqueeze(1)
    shifts = (2 * torch.arange(_CODES_PER_WORD, device=words.device, dtype=torch.int64)).view(1, -1)
    return (unsigned_words >> shifts) & 3


def _pack_ternary_codes(
    ternary: torch.Tensor,
    layout: WeightLayout,
    np: int,
    kp: int,
) -> torch.Tensor:
    """Pack an int8 ``[N, K]`` sign tensor into one of the four layouts."""

    n, k = ternary.shape
    # padded: [Np, Kp] int8, with zero codes in every tail position.
    padded = torch.zeros((np, kp), dtype=torch.int8, device=ternary.device)
    padded[:n, :k].copy_(ternary)
    # codes: [Np, Kp] int64 using 00=0, 01=+1, 10=-1.
    codes = (padded > 0).to(torch.int64) + 2 * (padded < 0).to(torch.int64)
    words = packed_word_count(layout, np, kp)

    if layout is WeightLayout.MMA_INTERLEAVED:
        # slot_values: [words*16], filled through a bijective logical->slot map.
        linear_slot = _mma_linear_slots(np, kp, ternary.device)
        slot_values = torch.empty(words * _CODES_PER_WORD, dtype=torch.int64, device=ternary.device)
        slot_values[linear_slot.reshape(-1)] = codes.reshape(-1)
        return _slots_to_words(slot_values.view(words, _CODES_PER_WORD)).contiguous()

    k_words = kp // _CODES_PER_WORD
    # row_slots: [Np, Kp/16, 16]; row_words: [Np, Kp/16].
    row_words = _slots_to_words(codes.view(np, k_words, _CODES_PER_WORD))
    if layout is WeightLayout.ROW_MAJOR:
        return row_words.reshape(-1).contiguous()
    if layout is WeightLayout.COLUMN_MAJOR:
        return row_words.transpose(0, 1).reshape(-1).contiguous()
    # blocked_words: [Np/64, Kp/16, 64].
    blocked_words = row_words.view(np // 64, 64, k_words).permute(0, 2, 1)
    return blocked_words.reshape(-1).contiguous()


def _unpack_ternary_codes(
    data: torch.Tensor,
    n: int,
    k: int,
    np: int,
    kp: int,
    layout: WeightLayout,
) -> torch.Tensor:
    """Decode packed words to a logical int8 ``[N, K]`` tensor."""

    words = packed_word_count(layout, np, kp)
    # slots: [words, 16] integer codes; decoded_slots: [words, 16] int8.
    slots = _words_to_slots(data)
    decoded_slots = (slots == 1).to(torch.int8) - (slots == 2).to(torch.int8)

    if layout is WeightLayout.MMA_INTERLEAVED:
        # linear_slot: [Np, Kp]; padded: [Np, Kp].
        linear_slot = _mma_linear_slots(np, kp, data.device)
        padded = decoded_slots.reshape(words * _CODES_PER_WORD)[linear_slot]
    else:
        k_words = kp // _CODES_PER_WORD
        if layout is WeightLayout.ROW_MAJOR:
            # row_slots: [Np, Kp/16, 16].
            row_slots = decoded_slots.view(np, k_words, _CODES_PER_WORD)
        elif layout is WeightLayout.COLUMN_MAJOR:
            # source: [Kp/16, Np, 16]; row_slots: [Np, Kp/16, 16].
            row_slots = decoded_slots.view(k_words, np, _CODES_PER_WORD).permute(1, 0, 2)
        else:
            # source: [Np/64, Kp/16, 64, 16].
            row_slots = decoded_slots.view(np // 64, k_words, 64, _CODES_PER_WORD).permute(
                0, 2, 1, 3
            )
        padded = row_slots.reshape(np, kp)
    return padded[:n, :k].contiguous()


def _torch_ternarize(
    weight: torch.Tensor,
    scale_mode: ScaleMode,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmean-ternarize a floating ``[N, K]`` tensor deterministically."""

    n, k = weight.shape
    # wide_abs: [N, K] float64; offline FP64 reduction avoids order-sensitive thresholds.
    wide_abs = weight.to(torch.float64).abs()
    if scale_mode is ScaleMode.PER_CHANNEL:
        # gamma: [N] float32.
        gamma = (wide_abs.sum(dim=1) / float(k)).to(torch.float32)
    else:
        # gamma: [1] float32.
        gamma = (wide_abs.sum().reshape(1) / float(n * k)).to(torch.float32)
    # fmax reproduces the native NaN rule: a poisoned mean falls back to eps.
    gamma = torch.fmax(gamma, torch.full_like(gamma, eps))
    # threshold: [N, 1] or [1, 1]; values: [N, K] float32.
    threshold = (0.5 * gamma).view(-1, 1) if scale_mode is ScaleMode.PER_CHANNEL else (0.5 * gamma).view(1, 1)
    values = weight.to(torch.float32)
    ternary = torch.where(
        values > threshold,
        torch.ones((), dtype=torch.int8, device=weight.device),
        torch.where(
            values < -threshold,
            -torch.ones((), dtype=torch.int8, device=weight.device),
            torch.zeros((), dtype=torch.int8, device=weight.device),
        ),
    )
    return ternary.contiguous(), gamma.contiguous()


def _extension_for_packing(backend: str | Backend | None, tensor: torch.Tensor):
    explicit = normalize_backend(backend)
    selected = requested_backend(backend)
    if selected is Backend.TRITON:
        if explicit is Backend.TRITON:
            raise ValueError("Triton is an execution backend, not an offline packing backend")
        # A process-wide Triton preference still requires the portable codec
        # because Triton only accelerates the fused execution path.
        return None
    if tensor.device.type not in {"cpu", "cuda"}:
        if selected is Backend.CUDA_EXTENSION:
            raise RuntimeError(
                "native offline packing only accepts CPU or CUDA tensors; "
                f"got {tensor.device.type}"
            )
        return None
    if selected is Backend.CUDA_EXTENSION:
        return get_extension(required=True)
    if selected is Backend.TORCH or is_compiling():
        return None
    return get_extension()


def pack_ternary_weight(
    weight: torch.Tensor,
    *,
    layout: int | WeightLayout = WeightLayout.ROW_MAJOR,
    scale_mode: int | ScaleMode = ScaleMode.PER_TENSOR,
    eps: float = 1.0e-5,
    backend: str | Backend | None = Backend.AUTO,
) -> PackedTernaryWeight:
    """Absmean-ternarize and pack a floating ``[N, K]`` weight matrix."""

    weight = _validate_weight(weight, floating=True)
    if bool(torch.isinf(weight).any().item()):
        raise ValueError("weight cannot contain positive or negative infinity")
    normalized_layout = normalize_layout(layout)
    normalized_scale_mode = normalize_scale_mode(scale_mode)
    epsilon = _validate_eps(eps)
    n, k = weight.shape
    extension = _extension_for_packing(backend, weight)
    if extension is not None:
        data, scale, np, kp = extension.pack_ternary_weights(
            weight, int(normalized_layout), int(normalized_scale_mode), epsilon
        )
        return PackedTernaryWeight(
            data, scale, n, k, int(np), int(kp), normalized_layout, normalized_scale_mode
        )

    np, kp = padded_extents(n, k, normalized_layout)
    with torch.no_grad():
        ternary, scale = _torch_ternarize(weight, normalized_scale_mode, epsilon)
        data = _pack_ternary_codes(ternary, normalized_layout, np, kp)
    return PackedTernaryWeight(
        data, scale, n, k, np, kp, normalized_layout, normalized_scale_mode
    )


def pack_ternary_int8(
    weight: torch.Tensor,
    *,
    layout: int | WeightLayout = WeightLayout.ROW_MAJOR,
    scale: torch.Tensor | float | None = None,
    scale_mode: int | ScaleMode = ScaleMode.PER_TENSOR,
    backend: str | Backend | None = Backend.AUTO,
) -> PackedTernaryWeight:
    """Pack an int8 ``[N, K]`` tensor by sign, with explicit dequant scales."""

    weight = _validate_weight(weight, floating=False)
    normalized_layout = normalize_layout(layout)
    normalized_scale_mode = normalize_scale_mode(scale_mode)
    n, k = weight.shape
    expected_scales = n if normalized_scale_mode is ScaleMode.PER_CHANNEL else 1
    if scale is None:
        scale_tensor = torch.ones(expected_scales, dtype=torch.float32, device=weight.device)
    elif isinstance(scale, torch.Tensor):
        scale_tensor = scale.to(device=weight.device, dtype=torch.float32).reshape(-1).contiguous()
    else:
        scale_tensor = torch.full((expected_scales,), float(scale), dtype=torch.float32, device=weight.device)
    if scale_tensor.numel() != expected_scales:
        raise ValueError(f"scale has {scale_tensor.numel()} values, expected {expected_scales}")
    if not bool(torch.isfinite(scale_tensor).all().item()) or not bool((scale_tensor > 0).all().item()):
        raise ValueError("all explicit weight scales must be finite and positive")

    extension = _extension_for_packing(backend, weight)
    if extension is not None:
        data, np, kp = extension.pack_ternary_int8(weight, int(normalized_layout))
    else:
        np, kp = padded_extents(n, k, normalized_layout)
        with torch.no_grad():
            data = _pack_ternary_codes(weight, normalized_layout, np, kp)
    return PackedTernaryWeight(
        data,
        scale_tensor,
        n,
        k,
        int(np),
        int(kp),
        normalized_layout,
        normalized_scale_mode,
    )


def unpack_ternary_weight(
    packed: PackedTernaryWeight,
    *,
    backend: str | Backend | None = Backend.AUTO,
) -> torch.Tensor:
    """Unpack to a logical int8 ``[N, K]`` matrix of ``{-1, 0, +1}``."""

    if not isinstance(packed, PackedTernaryWeight):
        raise TypeError("packed must be a PackedTernaryWeight")
    extension = _extension_for_packing(backend, packed.data)
    if extension is not None:
        return extension.unpack_ternary_weights(
            packed.data,
            packed.out_features,
            packed.in_features,
            packed.n_padded,
            packed.k_padded,
            int(packed.layout),
        )
    return _unpack_ternary_codes(
        packed.data,
        packed.out_features,
        packed.in_features,
        packed.n_padded,
        packed.k_padded,
        packed.layout,
    )


def convert_ternary_layout(
    packed: PackedTernaryWeight,
    layout: int | WeightLayout,
    *,
    backend: str | Backend | None = Backend.AUTO,
) -> PackedTernaryWeight:
    """Convert packed storage while preserving logical values and scales."""

    if not isinstance(packed, PackedTernaryWeight):
        raise TypeError("packed must be a PackedTernaryWeight")
    destination = normalize_layout(layout)
    if destination is packed.layout:
        return packed
    extension = _extension_for_packing(backend, packed.data)
    if extension is not None:
        data, np, kp = extension.convert_ternary_layout(
            packed.data,
            packed.out_features,
            packed.in_features,
            packed.n_padded,
            packed.k_padded,
            int(packed.layout),
            int(destination),
        )
    else:
        np, kp = padded_extents(packed.out_features, packed.in_features, destination)
        logical = _unpack_ternary_codes(
            packed.data,
            packed.out_features,
            packed.in_features,
            packed.n_padded,
            packed.k_padded,
            packed.layout,
        )
        data = _pack_ternary_codes(logical, destination, np, kp)
    return PackedTernaryWeight(
        data,
        packed.scale,
        packed.out_features,
        packed.in_features,
        int(np),
        int(kp),
        destination,
        packed.scale_mode,
    )
