"""Method A — Memory Cross-Attention injection (§5).

A dedicated, gated cross-attention module that lets DiT content tokens attend to
retrieved memory tokens. Reuses bitvideo's CrossAttention (which is built from
BitLinear), so the memory attention is on the ternary path exactly like the
text cross-attention already in every VideoDiTBlock.

Highest bandwidth of the three methods (tokens attend to individual memories),
highest cost (extra attention per application). Zero-init gate => memory starts
as a no-op, so training is stable and improvements are attributable.

Usage: apply after the content tokens are computed, before/after the block loop,
or inside a wrapped block. The Stage-3 unified injector applies it once on the
full token sequence (a "memory attention" layer) rather than per-block, keeping
the backbone untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bitvideo.models.cross_attention import CrossAttention
from bitvideo.quantization.bit_linear import BitLinear
from bitvideo.quantization.quantization import QuantizationConfig

from bitmem.memory.base import MemoryItem


class MemoryCrossAttention(nn.Module):
    """Gated cross-attention from content tokens to memory tokens.

    Args:
        model_dim: DiT hidden dim (query side).
        memory_dim: retrieved-embedding dim (key/value side, D_mem).
        num_heads: attention heads.
        max_tokens: max memory tokens per sample (padding/truncation).
        quantization: shared config to keep projections ternary.
    """

    def __init__(
        self,
        model_dim: int,
        memory_dim: int,
        *,
        num_heads: int = 4,
        max_tokens: int = 8,
        quantization: QuantizationConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.memory_dim = memory_dim
        self.max_tokens = max_tokens

        # Reuse bitvideo's CrossAttention: query in model space, KV in memory space.
        # gate=True gives a zero-init learnable gate -> memory starts as a no-op.
        self.attention = CrossAttention(
            model_dim,
            num_heads,
            context_dim=memory_dim,
            residual=False,          # we own the residual add
            gate=True,               # zero-init gate for stable start
            norm_type="rmsnorm",
            quantization=quantization,
            device=device,
            dtype=dtype,
        )

    def _stack_memory(
        self,
        memories_per_sample: list[list[MemoryItem]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build [B, max_tokens, D_mem] padded memory-embedding tensor."""
        batch = len(memories_per_sample)
        emb = torch.zeros(batch, self.max_tokens, self.memory_dim, device=device, dtype=dtype)
        for b, mems in enumerate(memories_per_sample):
            for i, m in enumerate(mems[: self.max_tokens]):
                emb[b, i] = m.embedding.to(device=device, dtype=dtype)
        return emb

    def forward(
        self,
        tokens: torch.Tensor,                        # [B, L, D_model]
        memories_per_sample: list[list[MemoryItem]],
    ) -> torch.Tensor:
        """Content tokens attend to memory; gated residual added.

        Args:
            tokens: DiT content tokens [B, L, D_model].
            memories_per_sample: retrieved memories per batch element.

        Returns:
            [B, L, D_model] tokens with memory information mixed in.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.model_dim:
            raise ValueError(
                f"tokens must be [B, L, {self.model_dim}]; got {tuple(tokens.shape)}"
            )
        memory_context = self._stack_memory(
            memories_per_sample, device=tokens.device, dtype=tokens.dtype
        )
        # CrossAttention with residual=False + gate=True returns gated attention
        # output; we add it as a residual so at init (gate=0) this is identity.
        attn_out = self.attention(tokens, context=memory_context)
        return tokens + attn_out

    @torch.no_grad()
    def pack_weights(self, *args, **kwargs) -> None:
        self.attention.pack_weights(*args, **kwargs)
