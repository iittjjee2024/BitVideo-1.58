"""3D VAE decoder with chunked temporal inference for video generation."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class ResBlock3D(nn.Module):
    """3D residual block with GroupNorm and SiLU activation."""

    def __init__(
        self,
        channels: int,
        *,
        out_channels: int | None = None,
        num_groups: int = 32,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.out_channels = channels if out_channels is None else _positive_int(
            out_channels, "out_channels"
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        # Clamp num_groups to avoid exceeding channel count.
        effective_groups = min(num_groups, channels)
        effective_groups_out = min(num_groups, self.out_channels)
        self.norm1 = nn.GroupNorm(effective_groups, channels, **factory_kwargs)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1, **factory_kwargs)
        self.norm2 = nn.GroupNorm(effective_groups_out, self.out_channels, **factory_kwargs)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)
        self.shortcut = (
            nn.Conv3d(channels, self.out_channels, 1, **factory_kwargs)
            if channels != self.out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process residual block.

        Args:
            x: Input ``[B, C, T, H, W]``.

        Returns:
            Output ``[B, C_out, T, H, W]``.
        """

        # residual: [B, C_out, T, H, W].
        residual = self.shortcut(x)
        # h: [B, C, T, H, W] -> [B, C_out, T, H, W].
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        return h + residual


class Upsample3D(nn.Module):
    """Spatial upsampling (2x) with optional temporal upsampling."""

    def __init__(
        self,
        channels: int,
        *,
        temporal: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.channels = _positive_int(channels, "channels")
        self.temporal = bool(temporal)
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample spatially (and optionally temporally).

        Args:
            x: Input ``[B, C, T, H, W]``.

        Returns:
            Upsampled ``[B, C, T', 2H, 2W]``.
        """

        if self.temporal:
            # Upsample all axes by 2x: [B, C, 2T, 2H, 2W].
            x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        else:
            # Upsample spatial only: [B, C, T, 2H, 2W].
            batch, channels, temporal, height, width = x.shape
            # Reshape to [B*T, C, H, W] for spatial interpolation.
            x = x.permute(0, 2, 1, 3, 4).reshape(batch * temporal, channels, height, width)
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = x.reshape(batch, temporal, channels, height * 2, width * 2).permute(0, 2, 1, 3, 4)
        return self.conv(x)


class VideoVAEDecoder(nn.Module):
    """Lightweight 3D VAE decoder for reconstructing video from latent space.

    Architecture: latent conv -> residual blocks with upsampling -> output conv.
    Mirrors the decoder structure used in latent video diffusion models.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        *,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (4, 4, 2, 1),
        num_res_blocks: int = 2,
        num_groups: int = 32,
        temporal_upsample_indices: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.base_channels = _positive_int(base_channels, "base_channels")
        factory_kwargs = {"device": device, "dtype": dtype}

        channel_multipliers = tuple(channel_multipliers)
        if not channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        num_levels = len(channel_multipliers)
        temporal_ups = set(temporal_upsample_indices or [])

        # Initial convolution from latent space.
        initial_channels = base_channels * channel_multipliers[0]
        self.conv_in = nn.Conv3d(
            in_channels, initial_channels, 3, padding=1, **factory_kwargs
        )

        # Middle block.
        self.mid_block = nn.Sequential(
            ResBlock3D(initial_channels, num_groups=num_groups, **factory_kwargs),
            ResBlock3D(initial_channels, num_groups=num_groups, **factory_kwargs),
        )

        # Decoder levels (from deepest to shallowest).
        self.up_blocks = nn.ModuleList()
        current_channels = initial_channels
        for level_idx in range(num_levels):
            level_channels = base_channels * channel_multipliers[level_idx]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResBlock3D(
                        current_channels,
                        out_channels=level_channels,
                        num_groups=num_groups,
                        **factory_kwargs,
                    )
                )
                current_channels = level_channels
            # Upsample at all levels except the last.
            if level_idx < num_levels - 1:
                upsample = Upsample3D(
                    current_channels,
                    temporal=level_idx in temporal_ups,
                    **factory_kwargs,
                )
            else:
                upsample = nn.Identity()
            self.up_blocks.append(nn.ModuleDict({"blocks": blocks, "upsample": upsample}))

        # Final normalization and output convolution.
        effective_groups = min(num_groups, current_channels)
        self.norm_out = nn.GroupNorm(effective_groups, current_channels, **factory_kwargs)
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv3d(
            current_channels, out_channels, 3, padding=1, **factory_kwargs
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to pixel space.

        Args:
            latents: Latent tensor ``[B, C_latent, T, H, W]``.

        Returns:
            Decoded video ``[B, C_out, T', H', W']``.
        """

        if not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if latents.ndim != 5:
            raise ValueError(f"latents must have shape [B,C,T,H,W]; got {tuple(latents.shape)}")
        if latents.shape[1] != self.in_channels:
            raise ValueError(
                f"latents must have {self.in_channels} channels; got {latents.shape[1]}"
            )

        # h: [B, initial_channels, T, H, W].
        h = self.conv_in(latents)
        # Mid block: [B, initial_channels, T, H, W].
        h = self.mid_block(h)
        # Up blocks with residual and upsampling.
        for up_block in self.up_blocks:
            for block in up_block["blocks"]:
                h = block(h)
            h = up_block["upsample"](h)
        # Final: [B, out_channels, T', H', W'].
        h = self.activation(self.norm_out(h))
        return self.conv_out(h)


class ChunkedVideoDecoder(nn.Module):
    """Memory-efficient chunked temporal decoding for long videos.

    Splits the temporal dimension into overlapping chunks, decodes each
    independently, and blends the overlapping regions for seamless output.
    """

    def __init__(
        self,
        decoder: VideoVAEDecoder,
        *,
        temporal_chunk_size: int = 4,
        temporal_overlap: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(decoder, VideoVAEDecoder):
            raise TypeError("decoder must be a VideoVAEDecoder instance")
        self.decoder = decoder
        self.temporal_chunk_size = _positive_int(temporal_chunk_size, "temporal_chunk_size")
        if isinstance(temporal_overlap, bool) or not isinstance(temporal_overlap, int) or temporal_overlap < 0:
            raise ValueError("temporal_overlap must be a non-negative integer")
        if temporal_overlap >= temporal_chunk_size:
            raise ValueError("temporal_overlap must be less than temporal_chunk_size")
        self.temporal_overlap = temporal_overlap

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents in temporal chunks with blending.

        Args:
            latents: Latent tensor ``[B, C, T, H, W]``.

        Returns:
            Decoded video ``[B, C_out, T', H', W']``.
        """

        if not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if latents.ndim != 5:
            raise ValueError(f"latents must have shape [B,C,T,H,W]; got {tuple(latents.shape)}")

        temporal = latents.shape[2]
        chunk_size = self.temporal_chunk_size
        overlap = self.temporal_overlap
        stride = chunk_size - overlap

        # If the video fits in one chunk, decode directly.
        if temporal <= chunk_size:
            return self.decoder(latents)

        # Split into overlapping chunks along temporal axis.
        chunks: list[torch.Tensor] = []
        start = 0
        while start < temporal:
            end = min(start + chunk_size, temporal)
            # chunk_latent: [B, C, chunk_T, H, W].
            chunk_latent = latents[:, :, start:end]
            # decoded_chunk: [B, C_out, chunk_T', H', W'].
            decoded_chunk = self.decoder(chunk_latent)
            chunks.append(decoded_chunk)
            if end >= temporal:
                break
            start += stride

        # Blend overlapping regions using linear weights.
        if overlap == 0 or len(chunks) == 1:
            return torch.cat(chunks, dim=2)

        # Compute output temporal sizes from each chunk.
        # Assume each input temporal frame maps to the same output temporal frame
        # (no temporal upsampling in the simple case).
        blended_parts: list[torch.Tensor] = []
        for i, chunk in enumerate(chunks):
            chunk_t = chunk.shape[2]
            if i == 0:
                # First chunk: take all except the overlap end region.
                keep_end = chunk_t - overlap
                blended_parts.append(chunk[:, :, :keep_end])
                # Store the overlap tail for blending with next chunk.
                prev_overlap = chunk[:, :, keep_end:]
            elif i == len(chunks) - 1:
                # Last chunk: blend start with previous overlap, then take rest.
                current_overlap = chunk[:, :, :overlap]
                # Linear blend weights: [1, 1, overlap, 1, 1].
                weight = torch.linspace(0.0, 1.0, overlap, device=chunk.device, dtype=chunk.dtype)
                weight = weight.reshape(1, 1, overlap, 1, 1)
                blended = prev_overlap * (1.0 - weight) + current_overlap * weight
                blended_parts.append(blended)
                blended_parts.append(chunk[:, :, overlap:])
            else:
                # Middle chunk: blend start, keep middle, store end overlap.
                current_overlap = chunk[:, :, :overlap]
                weight = torch.linspace(0.0, 1.0, overlap, device=chunk.device, dtype=chunk.dtype)
                weight = weight.reshape(1, 1, overlap, 1, 1)
                blended = prev_overlap * (1.0 - weight) + current_overlap * weight
                blended_parts.append(blended)
                keep_end = chunk_t - overlap
                blended_parts.append(chunk[:, :, overlap:keep_end])
                prev_overlap = chunk[:, :, keep_end:]

        # output: [B, C_out, total_T, H', W'].
        return torch.cat(blended_parts, dim=2)
