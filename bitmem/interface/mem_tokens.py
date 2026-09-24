"""Method B — Memory Tokens injection (§5).

Converts retrieved memories into learned memory tokens and prepends them to the
transformer sequence. This is the simplest of the three injection methods and
the one Stage 0 validates.

The projection from memory-embedding space (D_mem) to model space (D_model) is
a BitLinear, so the memory pathway stays on the ternary/packed path exactly like
the rest of the DiT. This matters for the efficiency comparison (§12): the memory
interface must not smuggle in full-precision compute.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bitvideo.quantization.bit_linear import BitLinear
from bitvideo.quantization.quantization import QuantizationConfig

from bitmem.memory.base import MemoryItem


class MemoryTokenInterface(nn.Module):
    """Projects retrieved memory embeddings into learned prepended tokens.

    Given a list of MemoryItems (per batch element), project their embeddings
    into D_model tokens, add a learned "memory type" embedding, and return them
    ready to prepend to the DiT token sequence.

    Args:
        memory_dim: dimensionality of memory embeddings (D_mem).
        model_dim: DiT hidden dim (D_model).
        max_tokens: maximum number of memory tokens per sample (padding/truncation).
        quantization: shared QuantizationConfig so the projection is ternary.
        device, dtype: placement.
    """

    def __init__(
        self,
        memory_dim: int,
        model_dim: int,
        *,
        max_tokens: int = 8,
        quantization: QuantizationConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if memory_dim <= 0 or model_dim <= 0:
            raise ValueError("memory_dim and model_dim must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.memory_dim = memory_dim
        self.model_dim = model_dim
        self.max_tokens = max_tokens

        # Ternary projection memory_dim -> model_dim (stays on packed path).
        self.projection = BitLinear(
            memory_dim,
            model_dim,
            bias=True,
            quantization=quantization,
            device=device,
            dtype=dtype,
        )
        # Learned marker so the transformer can distinguish memory from content.
        self.memory_type_embedding = nn.Parameter(
            torch.zeros(model_dim, device=device, dtype=dtype)
        )
        # Learned padding token for batch elements with fewer memories.
        self.pad_token = nn.Parameter(
            torch.zeros(model_dim, device=device, dtype=dtype)
        )

    def build_tokens(
        self,
        memories_per_sample: list[list[MemoryItem]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build padded memory-token tensor + mask.

        Args:
            memories_per_sample: for each batch element, its retrieved memories.
            device, dtype: target placement for the output tensors.

        Returns:
            tokens: [B, max_tokens, D_model] projected memory tokens.
            mask:   [B, max_tokens] bool, True where a real memory exists.
        """
        batch = len(memories_per_sample)
        mask = torch.zeros(batch, self.max_tokens, dtype=torch.bool, device=device)

        # Stack embeddings into [B, max_tokens, D_mem], zero-padded.
        emb = torch.zeros(batch, self.max_tokens, self.memory_dim, device=device, dtype=dtype)
        for b, mems in enumerate(memories_per_sample):
            for i, mem in enumerate(mems[: self.max_tokens]):
                emb[b, i] = mem.embedding.to(device=device, dtype=dtype)
                mask[b, i] = True

        # Project to model space: [B, max_tokens, D_model].
        tokens = self.projection(emb)

        # Add memory-type marker to real tokens, pad-token to empty slots.
        marker = self.memory_type_embedding.view(1, 1, -1)
        pad = self.pad_token.view(1, 1, -1)
        mask3 = mask.unsqueeze(-1)  # [B, max_tokens, 1]
        tokens = torch.where(mask3, tokens + marker, pad.expand_as(tokens))

        return tokens, mask

    def inject(
        self,
        tokens: torch.Tensor,                        # [B, L, D_model]
        memories_per_sample: list[list[MemoryItem]],
        t_emb: torch.Tensor | None = None,           # unused for Method B
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepend memory tokens to the content token sequence.

        Args:
            tokens: DiT content tokens [B, L, D_model].
            memories_per_sample: retrieved memories per batch element.
            t_emb: timestep embedding (unused here; Method C uses it).

        Returns:
            augmented: [B, max_tokens + L, D_model] with memory tokens prepended.
            mem_mask:  [B, max_tokens] bool marking valid memory positions.
        """
        if tokens.ndim != 3 or tokens.shape[-1] != self.model_dim:
            raise ValueError(
                f"tokens must be [B, L, {self.model_dim}]; got {tuple(tokens.shape)}"
            )
        if len(memories_per_sample) != tokens.shape[0]:
            raise ValueError(
                f"memories_per_sample has {len(memories_per_sample)} entries; "
                f"expected batch {tokens.shape[0]}"
            )
        mem_tokens, mem_mask = self.build_tokens(
            memories_per_sample, device=tokens.device, dtype=tokens.dtype
        )
        augmented = torch.cat([mem_tokens, tokens], dim=1)
        return augmented, mem_mask

    @staticmethod
    def strip(augmented: torch.Tensor, num_memory_tokens: int) -> torch.Tensor:
        """Remove the prepended memory tokens, returning content tokens only."""
        return augmented[:, num_memory_tokens:, :]

    @torch.no_grad()
    def pack_weights(self, *args, **kwargs) -> None:
        """Pack the ternary projection for inference."""
        self.projection.pack_weights(*args, **kwargs)
