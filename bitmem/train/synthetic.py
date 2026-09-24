"""Deterministic synthetic diffusion task for BitMem baseline evaluation.

Why synthetic? Stage 1/2's job is to validate the TRAINING HARNESS and MEASURE
QUANTIZATION DEGRADATION reproducibly — not to make pretty videos. A synthetic
task lets us:

  * run on CPU in seconds (no dataset download, no VAE)
  * get bit-reproducible results across FP16 and ternary runs
  * verify the model actually LEARNS (loss must drop), so a "degradation" number
    is meaningful rather than comparing two untrained noise generators

The task: each "video latent" is a low-rank structured signal derived from its
text embedding via a FIXED random linear map. A model that learns the diffusion
objective must implicitly learn to use the conditioning — so training loss drops
well below the noise floor, and we can measure how much ternarization hurts that.

This is a research prototype fixture (spec §16: clearly separate prototype from
production). It is NOT a claim about real video quality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticSpec:
    """Shape + difficulty configuration for the synthetic task."""

    in_channels: int = 8
    frames: int = 3
    height: int = 16
    width: int = 16
    context_len: int = 8
    context_dim: int = 64
    signal_rank: int = 4       # low-rank structure; lower = easier to learn
    noise_floor: float = 0.1   # additive structural noise on the clean signal

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return (self.in_channels, self.frames, self.height, self.width)


def _make_projection(spec: SyntheticSpec, generator: torch.Generator) -> torch.Tensor:
    """Fixed random map from context -> clean latent (the 'ground truth')."""
    latent_numel = (
        spec.in_channels * spec.frames * spec.height * spec.width
    )
    # Low-rank: context_dim -> rank -> latent_numel keeps the signal learnable.
    a = torch.randn(spec.context_dim, spec.signal_rank, generator=generator)
    b = torch.randn(spec.signal_rank, latent_numel, generator=generator)
    return (a @ b) / math.sqrt(spec.signal_rank)  # [context_dim, latent_numel]


def make_clean_latent(
    context: torch.Tensor,        # [B, L, context_dim]
    projection: torch.Tensor,     # [context_dim, latent_numel]
    spec: SyntheticSpec,
) -> torch.Tensor:
    """Deterministic clean latent from a text embedding (pooled)."""
    pooled = context.mean(dim=1)                 # [B, context_dim]
    flat = pooled @ projection                   # [B, latent_numel]
    b = context.shape[0]
    latent = flat.view(b, *spec.latent_shape)
    # Normalize to unit-ish scale so the diffusion noise levels are sensible.
    latent = latent / (latent.flatten(1).std(dim=1).view(b, 1, 1, 1, 1) + 1e-6)
    return latent


class SyntheticDiffusionDataset(Dataset):
    """A fixed-size deterministic dataset of (video_latent, text_embedding) pairs.

    Given a seed, the SAME samples are produced every run, so FP16 and ternary
    trainers see identical data — the only variable is the quantization.
    """

    def __init__(
        self,
        num_samples: int = 256,
        spec: SyntheticSpec | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.spec = spec or SyntheticSpec()
        self.num_samples = int(num_samples)
        self.seed = int(seed)

        gen = torch.Generator().manual_seed(seed)
        # Fixed ground-truth map shared by all samples.
        self.projection = _make_projection(self.spec, gen)
        # Pre-generate all context embeddings deterministically.
        self._contexts = torch.randn(
            num_samples, self.spec.context_len, self.spec.context_dim, generator=gen
        )
        # Structural noise seed (kept fixed per-sample).
        self._noise = torch.randn(
            num_samples, *self.spec.latent_shape, generator=gen
        ) * self.spec.noise_floor

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        context = self._contexts[idx]                       # [L, context_dim]
        clean = make_clean_latent(
            context.unsqueeze(0), self.projection, self.spec
        ).squeeze(0)                                        # [C, T, H, W]
        clean = clean + self._noise[idx]
        return {
            "video_latent": clean,
            "text_embedding": context,
        }


def make_synthetic_batch(
    batch_size: int = 4,
    spec: SyntheticSpec | None = None,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Build a single deterministic batch (for smoke tests / benchmarks)."""
    ds = SyntheticDiffusionDataset(num_samples=batch_size, spec=spec, seed=seed)
    videos = torch.stack([ds[i]["video_latent"] for i in range(batch_size)])
    contexts = torch.stack([ds[i]["text_embedding"] for i in range(batch_size)])
    return {
        "video_latent": videos.to(device=device, dtype=dtype),
        "text_embedding": contexts.to(device=device, dtype=dtype),
    }
