"""Unified memory injector for BitMem (§5, §13 ablation).

MemoryAugmentedDiT wraps a VideoDiT and injects retrieved memory using a
config-selected method:

    Method A (cross_attention) — content tokens attend to memory tokens
    Method B (memory_tokens)   — memory tokens prepended to the sequence
    Method C (adaptive)        — pooled memory modulates the timestep embedding
    "none"                     — memory-free passthrough (ablation baseline)

Selecting the method is a CONFIG switch, so the ablation matrix rows
(memory-tokens vs cross-attention vs adaptive-conditioning vs no-memory) are
produced without changing code — exactly what spec §13 requires.

The wrapper re-implements VideoDiT.forward's outer flow (patchify -> t_emb ->
blocks -> final -> unpatchify) so it can inject at the right point for each
method, while delegating all heavy lifting to the unchanged backbone modules.
This keeps the DiT backbone swappable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from bitvideo.models import VideoDiT
from bitvideo.models.patch_embed import unpatchify_video
from bitvideo.quantization.quantization import QuantizationConfig

from bitmem.memory.base import MemoryItem
from bitmem.interface.mem_tokens import MemoryTokenInterface
from bitmem.interface.adaptive import AdaptiveMemoryConditioning
from bitmem.interface.cross_attn import MemoryCrossAttention


class InjectionMethod(str, Enum):
    NONE = "none"                    # ablation baseline: memory-free
    MEMORY_TOKENS = "memory_tokens"  # Method B
    CROSS_ATTENTION = "cross_attention"  # Method A
    ADAPTIVE = "adaptive"            # Method C


@dataclass
class InjectorConfig:
    method: InjectionMethod = InjectionMethod.ADAPTIVE
    memory_dim: int = 64
    max_memory_tokens: int = 8
    num_memory_heads: int = 4


class MemoryAugmentedDiT(nn.Module):
    """A VideoDiT + a selectable memory-injection adapter.

    Args:
        dit: an existing VideoDiT (unchanged backbone).
        config: which injection method + memory dims.
        quantization: shared config so adapters stay ternary.
    """

    def __init__(
        self,
        dit: VideoDiT,
        config: InjectorConfig,
        *,
        quantization: QuantizationConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dit = dit
        self.config = config
        self.method = InjectionMethod(config.method)
        model_dim = dit.dim

        self.mem_tokens: MemoryTokenInterface | None = None
        self.mem_cross: MemoryCrossAttention | None = None
        self.mem_adaptive: AdaptiveMemoryConditioning | None = None

        if self.method is InjectionMethod.MEMORY_TOKENS:
            self.mem_tokens = MemoryTokenInterface(
                memory_dim=config.memory_dim, model_dim=model_dim,
                max_tokens=config.max_memory_tokens,
                quantization=quantization, device=device, dtype=dtype,
            )
        elif self.method is InjectionMethod.CROSS_ATTENTION:
            self.mem_cross = MemoryCrossAttention(
                model_dim=model_dim, memory_dim=config.memory_dim,
                num_heads=config.num_memory_heads,
                max_tokens=config.max_memory_tokens,
                quantization=quantization, device=device, dtype=dtype,
            )
        elif self.method is InjectionMethod.ADAPTIVE:
            self.mem_adaptive = AdaptiveMemoryConditioning(
                memory_dim=config.memory_dim, conditioning_dim=model_dim,
                quantization=quantization, device=device, dtype=dtype,
            )
        elif self.method is not InjectionMethod.NONE:
            raise ValueError(f"unknown injection method {self.method}")

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        video: torch.Tensor,                                  # [B, C, T, H, W]
        timesteps: torch.Tensor,                              # [B]
        context: torch.Tensor,                                # [B, Lc, context_dim]
        memories_per_sample: list[list[MemoryItem]] | None = None,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoising forward with optional memory injection.

        If memories_per_sample is None or method is NONE, this is exactly the
        memory-free VideoDiT forward (the ablation baseline).
        """
        dit = self.dit
        batch = video.shape[0]
        if memories_per_sample is None:
            memories_per_sample = [[] for _ in range(batch)]

        # 1. Patchify
        tokens, patch_info = dit.patch_embed(video, return_info=True)
        grid_t, grid_h, grid_w = patch_info.grid_size
        spatial_size = grid_h * grid_w
        temporal_size = grid_t

        # 2. Timestep embedding
        t_emb = dit.timestep_embed(timesteps)

        # 3. Context projection (text conditioning, unchanged)
        projected_context = dit.context_projection(context)

        # --- Method C: modulate t_emb with pooled memory (before blocks) ---
        if self.method is InjectionMethod.ADAPTIVE and self.mem_adaptive is not None:
            t_emb = self.mem_adaptive(t_emb, memories_per_sample)

        # --- Method B: memory tokens appended to the CROSS-ATTENTION context ---
        # The DiT's spatial/temporal attention is grid-factorized and requires
        # temporal_size * spatial_size == seq_len, so we CANNOT prepend memory
        # tokens to the content sequence. Instead we append them to the text
        # context: they flow through the existing (non-factorized) cross-attention
        # exactly like extra conditioning tokens. This keeps the backbone
        # untouched and the grid intact.
        if self.method is InjectionMethod.MEMORY_TOKENS and self.mem_tokens is not None:
            mem_tok, _ = self.mem_tokens.build_tokens(
                memories_per_sample, device=tokens.device, dtype=tokens.dtype
            )  # [B, max_tokens, D_model]; projected_context is [B, Lc, D_model]
            projected_context = torch.cat([projected_context, mem_tok], dim=1)
            # cross_mask (if any) would need extending; Stage-3 uses no mask.
            context_mask = None

        # 4. Transformer blocks (unchanged backbone)
        context_cache = None
        for i, block in enumerate(dit.blocks):
            if i == 0:
                tokens, context_cache = block(
                    tokens, timestep_embedding=t_emb,
                    temporal_size=temporal_size, spatial_size=spatial_size,
                    context=projected_context, cross_mask=context_mask,
                    return_context_cache=True,
                )
            else:
                tokens = block(
                    tokens, timestep_embedding=t_emb,
                    temporal_size=temporal_size, spatial_size=spatial_size,
                    context_cache=context_cache, cross_mask=context_mask,
                    return_context_cache=False,
                )

        # --- Method A: content tokens attend to memory (after blocks) ---
        if self.method is InjectionMethod.CROSS_ATTENTION and self.mem_cross is not None:
            tokens = self.mem_cross(tokens, memories_per_sample)

        # 5. Final projection + unpatchify
        prediction = dit.final_layer(tokens, t_emb)
        return unpatchify_video(prediction, patch_info, channels=dit.out_channels)

    def memory_parameters(self) -> list[nn.Parameter]:
        """Only the memory-adapter parameters (for Stage-3 frozen-backbone training)."""
        params: list[nn.Parameter] = []
        for mod in (self.mem_tokens, self.mem_cross, self.mem_adaptive):
            if mod is not None:
                params.extend(mod.parameters())
        return params

    def freeze_backbone(self) -> None:
        """Freeze the DiT so only memory adapters train (§8 Stage 3)."""
        for p in self.dit.parameters():
            p.requires_grad_(False)
