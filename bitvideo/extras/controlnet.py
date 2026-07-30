"""ControlNet-style conditioning for guided video generation."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from bitvideo.models.video_dit_block import VideoDiTBlock


class ControlNetConditioner(nn.Module):
    """ControlNet conditioning module for spatial/structural guidance.

    Creates a trainable copy of the first N transformer blocks that processes
    a conditioning signal (depth maps, edges, poses) and adds residual
    outputs to the main model's intermediate features.

    Architecture follows the ControlNet paper: the conditioning is injected
    through zero-initialized linear layers for stable initialization.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        conditioning_channels: int = 3,
        depth: int = 4,
        context_dim: int | None = None,
        patch_size: Sequence[int] | int = (1, 2, 2),
        zero_init: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        **block_kwargs: Any,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Conditioning input projection (converts condition image to token space).
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        patch_volume = 1
        for p in patch_size:
            patch_volume *= p
        input_dim = conditioning_channels * patch_volume
        self.input_projection = nn.Linear(input_dim, dim, device=device, dtype=dtype)

        # Trainable copy of transformer blocks.
        self.blocks = nn.ModuleList([
            VideoDiTBlock(
                dim, num_heads,
                context_dim=context_dim or dim,
                conditioning_dim=dim,
                device=device, dtype=dtype,
                **block_kwargs,
            )
            for _ in range(depth)
        ])

        # Zero-initialized output projections for stable training.
        self.zero_convs = nn.ModuleList([
            nn.Linear(dim, dim, device=device, dtype=dtype)
            for _ in range(depth)
        ])
        if zero_init:
            for conv in self.zero_convs:
                nn.init.zeros_(conv.weight)
                nn.init.zeros_(conv.bias)

    def forward(
        self,
        conditioning: torch.Tensor,
        timestep_embedding: torch.Tensor,
        *,
        temporal_size: int,
        spatial_size: int,
        context: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Process conditioning through control blocks.

        Args:
            conditioning: Conditioning tokens ``[B, T*H*W, C*patch_volume]``.
            timestep_embedding: Diffusion step ``[B, dim]``.
            temporal_size: Number of frames.
            spatial_size: Spatial tokens per frame.
            context: Text conditioning for cross-attention.

        Returns:
            List of residual tensors to add to main model, one per block.
        """
        # Project conditioning to model dimension: [B, L, dim].
        hidden = self.input_projection(conditioning)
        residuals: list[torch.Tensor] = []

        for block, zero_conv in zip(self.blocks, self.zero_convs):
            hidden = block(
                hidden,
                timestep_embedding=timestep_embedding,
                temporal_size=temporal_size,
                spatial_size=spatial_size,
                context=context,
            )
            # Zero-conv gated residual: [B, L, dim].
            residuals.append(zero_conv(hidden))

        return residuals

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"
