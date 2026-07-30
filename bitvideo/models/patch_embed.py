"""Lossless video patchification and learned three-dimensional patch embedding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_ALLOWED_PADDING_MODES = {"constant", "reflect", "replicate", "circular"}


def _positive_triple(value: Sequence[int] | int, name: str) -> tuple[int, int, int]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must contain integers, not bool")
    if isinstance(value, int):
        result = (value, value, value)
    else:
        result = tuple(value)
        if len(result) != 3:
            raise ValueError(f"{name} must contain exactly three integers")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive integers; got {result}")
    return result


def _validate_padding_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _ALLOWED_PADDING_MODES:
        legal = ", ".join(sorted(_ALLOWED_PADDING_MODES))
        raise ValueError(f"padding_mode must be one of {legal}; got {mode!r}")
    return normalized


@dataclass(frozen=True)
class VideoPatchInfo:
    """Geometry required to map flattened patches back to a video tensor."""

    original_size: tuple[int, int, int]
    padded_size: tuple[int, int, int]
    grid_size: tuple[int, int, int]
    patch_size: tuple[int, int, int]

    def __post_init__(self) -> None:
        original = _positive_triple(self.original_size, "original_size")
        padded = _positive_triple(self.padded_size, "padded_size")
        grid = _positive_triple(self.grid_size, "grid_size")
        patch = _positive_triple(self.patch_size, "patch_size")
        object.__setattr__(self, "original_size", original)
        object.__setattr__(self, "padded_size", padded)
        object.__setattr__(self, "grid_size", grid)
        object.__setattr__(self, "patch_size", patch)
        if any(padded_axis < original_axis for padded_axis, original_axis in zip(padded, original)):
            raise ValueError("padded_size cannot be smaller than original_size")
        expected_padded = tuple(grid_axis * patch_axis for grid_axis, patch_axis in zip(grid, patch))
        if padded != expected_padded:
            raise ValueError(
                f"padded_size must equal grid_size * patch_size; got {padded} and "
                f"{expected_padded}"
            )

    @property
    def token_count(self) -> int:
        temporal, height, width = self.grid_size
        return temporal * height * width

    @property
    def patch_volume(self) -> int:
        temporal, height, width = self.patch_size
        return temporal * height * width

    @property
    def padding(self) -> tuple[int, int, int]:
        return tuple(
            padded_axis - original_axis
            for padded_axis, original_axis in zip(self.padded_size, self.original_size)
        )


def _patch_geometry(
    size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    *,
    pad: bool,
) -> VideoPatchInfo:
    if not isinstance(pad, bool):
        raise TypeError("pad must be bool")
    padded_size: list[int] = []
    grid_size: list[int] = []
    for axis_size, axis_patch in zip(size, patch_size):
        remainder = axis_size % axis_patch
        if remainder and not pad:
            raise ValueError(
                f"video size {size} is not divisible by patch_size={patch_size} and pad=False"
            )
        padded_axis = axis_size if remainder == 0 else axis_size + axis_patch - remainder
        padded_size.append(padded_axis)
        grid_size.append(padded_axis // axis_patch)
    return VideoPatchInfo(
        original_size=size,
        padded_size=tuple(padded_size),
        grid_size=tuple(grid_size),
        patch_size=patch_size,
    )


def _pad_video(
    video: torch.Tensor,
    info: VideoPatchInfo,
    *,
    mode: str,
    value: float,
) -> torch.Tensor:
    pad_t, pad_h, pad_w = info.padding
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return video
    # PyTorch pad order is W-left/right, H-left/right, T-left/right.
    padding = (0, pad_w, 0, pad_h, 0, pad_t)
    if mode == "constant":
        # padded: [B,C,Tpadded,Hpadded,Wpadded].
        return F.pad(video, padding, mode=mode, value=value)
    # padded: [B,C,Tpadded,Hpadded,Wpadded].
    return F.pad(video, padding, mode=mode)


def patchify_video(
    video: torch.Tensor,
    patch_size: Sequence[int] | int,
    *,
    pad: bool = True,
    padding_mode: str = "constant",
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, VideoPatchInfo]:
    """Convert ``video [B,C,T,H,W]`` into lossless flattened patch tokens."""

    if not isinstance(video, torch.Tensor):
        raise TypeError("video must be a torch.Tensor")
    if video.ndim != 5:
        raise ValueError(f"video must have shape [B,C,T,H,W]; got {tuple(video.shape)}")
    if not video.is_floating_point():
        raise TypeError(f"video must be floating point; got {video.dtype}")
    batch, channels, temporal, height, width = video.shape
    if channels <= 0 or temporal <= 0 or height <= 0 or width <= 0:
        raise ValueError("video channel, temporal, height, and width extents must be positive")
    normalized_patch = _positive_triple(patch_size, "patch_size")
    mode = _validate_padding_mode(padding_mode)
    value = float(padding_value)
    if not math.isfinite(value):
        raise ValueError("padding_value must be finite")
    info = _patch_geometry(
        (temporal, height, width),
        normalized_patch,
        pad=pad,
    )
    # padded: [B,C,Tg*Pt,Hg*Ph,Wg*Pw].
    padded = _pad_video(video, info, mode=mode, value=value)
    grid_t, grid_h, grid_w = info.grid_size
    patch_t, patch_h, patch_w = info.patch_size
    # blocked: [B,C,Tg,Pt,Hg,Ph,Wg,Pw].
    blocked = padded.reshape(
        batch,
        channels,
        grid_t,
        patch_t,
        grid_h,
        patch_h,
        grid_w,
        patch_w,
    )
    # patches: [B,Tg*Hg*Wg,C*Pt*Ph*Pw].
    patches = blocked.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        batch,
        info.token_count,
        channels * info.patch_volume,
    )
    return patches, info


def unpatchify_video(
    patches: torch.Tensor,
    info: VideoPatchInfo,
    *,
    channels: int | None = None,
) -> torch.Tensor:
    """Reconstruct ``[B,C,T,H,W]`` from lossless flattened patch tokens."""

    if not isinstance(patches, torch.Tensor):
        raise TypeError("patches must be a torch.Tensor")
    if not isinstance(info, VideoPatchInfo):
        raise TypeError("info must be a VideoPatchInfo")
    if patches.ndim != 3:
        raise ValueError(f"patches must have shape [B,L,P]; got {tuple(patches.shape)}")
    batch, token_count, patch_features = patches.shape
    if token_count != info.token_count:
        raise ValueError(f"patch token count must be {info.token_count}; got {token_count}")
    if channels is None:
        if patch_features % info.patch_volume:
            raise ValueError(
                f"patch feature size {patch_features} is not divisible by "
                f"patch volume {info.patch_volume}"
            )
        channels = patch_features // info.patch_volume
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise ValueError("channels must be a positive integer")
    expected_features = channels * info.patch_volume
    if patch_features != expected_features:
        raise ValueError(
            f"patch feature size must be channels*patch_volume={expected_features}; "
            f"got {patch_features}"
        )
    grid_t, grid_h, grid_w = info.grid_size
    patch_t, patch_h, patch_w = info.patch_size
    # blocked: [B,Tg,Hg,Wg,C,Pt,Ph,Pw].
    blocked = patches.reshape(
        batch,
        grid_t,
        grid_h,
        grid_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    # padded: [B,C,Tpadded,Hpadded,Wpadded].
    padded = blocked.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        batch,
        channels,
        *info.padded_size,
    )
    temporal, height, width = info.original_size
    # video: [B,C,T,H,W], with right-side patch padding removed.
    return padded[:, :, :temporal, :height, :width].contiguous()


class VideoPatchEmbed(nn.Module):
    """Learned non-overlapping Conv3D video patch projection."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        *,
        patch_size: Sequence[int] | int = (1, 2, 2),
        bias: bool = True,
        flatten: bool = True,
        norm: bool = True,
        pad_input: bool = True,
        padding_mode: str = "constant",
        padding_value: float = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        for name, value in (("in_channels", in_channels), ("embed_dim", embed_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("bias", bias),
            ("flatten", flatten),
            ("norm", norm),
            ("pad_input", pad_input),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if dtype is not None and (not isinstance(dtype, torch.dtype) or not dtype.is_floating_point):
            raise TypeError("dtype must be floating point")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = _positive_triple(patch_size, "patch_size")
        self.flatten = flatten
        self.pad_input = pad_input
        self.padding_mode = _validate_padding_mode(padding_mode)
        self.padding_value = float(padding_value)
        if not math.isfinite(self.padding_value):
            raise ValueError("padding_value must be finite")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
            **factory_kwargs,
        )
        self.norm = nn.LayerNorm(embed_dim, **factory_kwargs) if norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # flattened_weight: [embed_dim, in_channels*Pt*Ph*Pw].
        flattened_weight = self.projection.weight.flatten(1)
        nn.init.xavier_uniform_(flattened_weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        if isinstance(self.norm, nn.LayerNorm):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def output_info(self, video_size: Sequence[int] | int) -> VideoPatchInfo:
        return _patch_geometry(
            _positive_triple(video_size, "video_size"),
            self.patch_size,
            pad=self.pad_input,
        )

    def forward(
        self,
        video: torch.Tensor,
        *,
        return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, VideoPatchInfo]:
        if not isinstance(return_info, bool):
            raise TypeError("return_info must be bool")
        if not isinstance(video, torch.Tensor):
            raise TypeError("video must be a torch.Tensor")
        if video.ndim != 5:
            raise ValueError(f"video must have shape [B,C,T,H,W]; got {tuple(video.shape)}")
        batch, channels, temporal, height, width = video.shape
        if channels != self.in_channels:
            raise ValueError(f"video must have {self.in_channels} channels; got {channels}")
        if temporal <= 0 or height <= 0 or width <= 0:
            raise ValueError("video temporal and spatial extents must be positive")
        if not video.is_floating_point():
            raise TypeError(f"video must be floating point; got {video.dtype}")
        if video.device != self.projection.weight.device:
            raise ValueError(
                f"video must be on {self.projection.weight.device}; got {video.device}"
            )
        info = self.output_info((temporal, height, width))
        # padded: [B,C,Tpadded,Hpadded,Wpadded].
        padded = _pad_video(
            video,
            info,
            mode=self.padding_mode,
            value=self.padding_value,
        )
        # embedded_grid: [B,D,Tg,Hg,Wg].
        embedded_grid = self.projection(padded)
        if self.flatten:
            # tokens_before_norm/tokens: [B,Tg*Hg*Wg,D].
            tokens_before_norm = embedded_grid.flatten(2).transpose(1, 2)
            output = self.norm(tokens_before_norm)
        elif isinstance(self.norm, nn.LayerNorm):
            # channels_last: [B,Tg,Hg,Wg,D]; output: [B,D,Tg,Hg,Wg].
            channels_last = embedded_grid.permute(0, 2, 3, 4, 1)
            output = self.norm(channels_last).permute(0, 4, 1, 2, 3).contiguous()
        else:
            output = embedded_grid
        if output.shape[0] != batch:
            raise RuntimeError("patch embedding unexpectedly changed the batch dimension")
        return (output, info) if return_info else output

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, embed_dim={self.embed_dim}, "
            f"patch_size={self.patch_size}, flatten={self.flatten}, "
            f"pad_input={self.pad_input}, padding_mode={self.padding_mode!r}"
        )


PatchEmbed3D = VideoPatchEmbed
