"""Method C — Adaptive Conditioning memory injection (§5).

Pools retrieved memories into a single conditioning vector and turns it into
FiLM/AdaLN-style (gamma, beta) modulation that is ADDED to the diffusion
timestep embedding. The DiT then conditions every block on memory through its
existing adaptive-norm pathway — no change to the backbone's attention.

This is the CHEAPEST of the three injection methods: one pooled vector per
sample, one small MLP, no extra attention or sequence length. Its cost is that
it is also the lowest-bandwidth (a single vector must summarize all memory).

Built from BitLinear so the memory pathway stays on the ternary path — the
efficiency comparison (§12) stays honest.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bitvideo.quantization.bit_linear import BitLinear
from bitvideo.quantization.quantization import QuantizationConfig

from bitmem.memory.base import MemoryItem


def pool_memory_embeddings(
    memories_per_sample: list[list[MemoryItem]],
    memory_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    weight_by: str = "utility",
) -> torch.Tensor:
    """Pool each sample's retrieved memories into one [D_mem] vector.

    Args:
        memories_per_sample: retrieved memories per batch element.
        memory_dim: embedding dim.
        weight_by: 'utility' | 'importance' | 'confidence' | 'uniform' —
            how to weight the pooled mean (a memory the agent values more
            contributes more to the conditioning).

    Returns:
        [B, D_mem] pooled memory embeddings (zeros where no memory retrieved).
    """
    batch = len(memories_per_sample)
    out = torch.zeros(batch, memory_dim, device=device, dtype=dtype)
    for b, mems in enumerate(memories_per_sample):
        if not mems:
            continue
        embs = torch.stack([m.embedding.to(device=device, dtype=dtype) for m in mems])
        if weight_by == "uniform":
            weights = torch.ones(len(mems), device=device, dtype=dtype)
        else:
            weights = torch.tensor(
                [float(getattr(m, weight_by)) for m in mems],
                device=device, dtype=dtype,
            )
        weights = weights / weights.sum().clamp_min(1e-8)
        out[b] = (weights.unsqueeze(-1) * embs).sum(dim=0)
    return out


class AdaptiveMemoryConditioning(nn.Module):
    """Turns pooled memory into an additive conditioning signal for the DiT.

    Produces a vector in the model's conditioning space (same dim as the
    timestep embedding) that is ADDED to t_emb before the blocks run. Because
    the DiT already broadcasts t_emb through AdaLayerNormZero in every block,
    this modulates the entire network with one cheap pooled signal.

    Zero-initialized output projection: at init, memory has NO effect (the model
    behaves exactly like the memory-free baseline), so training starts stable
    and any improvement is attributable to learned memory use (§8 Stage 3).

    Args:
        memory_dim: retrieved-embedding dim (D_mem).
        conditioning_dim: DiT conditioning dim (== model dim, matches t_emb).
        hidden_dim: MLP width (defaults to conditioning_dim).
        quantization: shared config to keep the projection ternary.
    """

    def __init__(
        self,
        memory_dim: int,
        conditioning_dim: int,
        *,
        hidden_dim: int | None = None,
        weight_by: str = "utility",
        quantization: QuantizationConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if memory_dim <= 0 or conditioning_dim <= 0:
            raise ValueError("dims must be positive")
        self.memory_dim = memory_dim
        self.conditioning_dim = conditioning_dim
        self.weight_by = weight_by
        hidden = hidden_dim or conditioning_dim

        self.up = BitLinear(
            memory_dim, hidden, bias=True,
            quantization=quantization, device=device, dtype=dtype,
        )
        self.act = nn.SiLU()
        self.down = BitLinear(
            hidden, conditioning_dim, bias=True,
            quantization=quantization, device=device, dtype=dtype,
        )
        # Zero-init the final projection so memory starts as a no-op.
        with torch.no_grad():
            self.down.weight.zero_()
            if self.down.bias is not None:
                self.down.bias.zero_()

    def forward(
        self,
        t_emb: torch.Tensor,                          # [B, conditioning_dim]
        memories_per_sample: list[list[MemoryItem]],
    ) -> torch.Tensor:
        """Return t_emb augmented with a memory-derived conditioning signal."""
        if t_emb.ndim != 2 or t_emb.shape[-1] != self.conditioning_dim:
            raise ValueError(
                f"t_emb must be [B, {self.conditioning_dim}]; got {tuple(t_emb.shape)}"
            )
        pooled = pool_memory_embeddings(
            memories_per_sample, self.memory_dim,
            device=t_emb.device, dtype=t_emb.dtype, weight_by=self.weight_by,
        )
        delta = self.down(self.act(self.up(pooled)))   # [B, conditioning_dim]
        return t_emb + delta

    @torch.no_grad()
    def pack_weights(self, *args, **kwargs) -> None:
        self.up.pack_weights(*args, **kwargs)
        self.down.pack_weights(*args, **kwargs)
