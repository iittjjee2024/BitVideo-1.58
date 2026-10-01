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


# ---------------------------------------------------------------------------
# Memory-sensitive task (Stage 5 — the hypothesis test)
# ---------------------------------------------------------------------------


@dataclass
class MemorySensitiveSpec:
    """Shape + difficulty for the memory-sensitive task (§8 Stage 5, §18).

    The clean latent has two additive parts:

        clean = shared_from_text  +  memory_gain * prototype[task_id]

    * `shared_from_text` is a low-rank function of the text embedding — every
      model (memory or not) can learn it from conditioning alone.
    * `prototype[task_id]` is an independent per-task vector that is NOT a
      function of the text. It cannot be predicted from conditioning; the best a
      memory-free model can do is regress toward the mean prototype (0), leaving
      an irreducible error proportional to `memory_gain`. A model that retrieves
      the correct prototype from memory can subtract it and drive that error to
      ~0.

    `memory_gain` is the knob that sets how much of the signal variance lives in
    memory. At gain 0 the task is memory-insensitive (Stage 1/2 regime); as gain
    grows, memory becomes the dominant source of predictable structure and the
    hypothesis (small ternary + memory beats larger memory-free) becomes testable.
    """

    in_channels: int = 8
    frames: int = 3
    height: int = 16
    width: int = 16
    context_len: int = 8
    context_dim: int = 64
    signal_rank: int = 4
    num_tasks: int = 16          # distinct prototypes to memorize
    memory_gain: float = 2.0     # weight of the memory-only component
    memory_dim: int = 64         # dim of the retrieval key + stored content
    noise_floor: float = 0.05

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return (self.in_channels, self.frames, self.height, self.width)

    @property
    def latent_numel(self) -> int:
        return self.in_channels * self.frames * self.height * self.width


class MemorySensitiveDataset(Dataset):
    """Deterministic task where part of the signal is retrievable only from memory.

    Each sample is tagged with a `task_id`. The dataset also exposes, per task:
      * `task_key[task_id]`   — the retrieval key (query embedding) [memory_dim]
      * `task_content[task_id]`— the stored memory content the model consumes
                                 (the flattened prototype, projected to memory_dim)

    A training loop is expected to (a) build a MemorySystem, writing each task's
    (key -> content) once, and (b) at each step retrieve with the sample's key so
    the memory-augmented model receives the right prototype. The memory-free
    ablation simply ignores keys/content.

    Reproducible: a fixed seed fixes projections, prototypes, keys, and per-sample
    contexts, so every ablation cell sees identical data.
    """

    def __init__(
        self,
        num_samples: int = 512,
        spec: MemorySensitiveSpec | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.spec = spec or MemorySensitiveSpec()
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        s = self.spec

        gen = torch.Generator().manual_seed(seed)

        # Text -> shared component (low-rank, learnable from conditioning).
        self.projection = _make_projection_ms(s, gen)  # [context_dim, latent_numel]

        # Per-task prototypes: independent of text, normalized to unit std
        # PER ELEMENT (so `memory_gain * prototype` is directly comparable to the
        # unit-std shared component). Shape [num_tasks, latent_numel].
        proto = torch.randn(s.num_tasks, s.latent_numel, generator=gen)
        proto = proto / (proto.std(dim=1, keepdim=True) + 1e-6)
        self.prototypes = proto

        # Retrieval keys per task (what the query embedding looks like). Unit norm.
        keys = torch.randn(s.num_tasks, s.memory_dim, generator=gen)
        self.task_key = keys / (keys.norm(dim=1, keepdim=True) + 1e-6)

        # Stored memory content per task = prototype projected to memory_dim, so the
        # injection adapters (which expect [*, memory_dim]) can consume it. This is a
        # FIXED encoding; the adapter learns to map it back into the latent space.
        enc = torch.randn(s.latent_numel, s.memory_dim, generator=gen) / math.sqrt(
            s.latent_numel
        )
        self._content_encoder = enc
        self.task_content = self.prototypes @ enc  # [num_tasks, memory_dim]

        # Per-sample data: context + task assignment + structural noise.
        self._contexts = torch.randn(
            num_samples, s.context_len, s.context_dim, generator=gen
        )
        self._task_ids = torch.randint(
            0, s.num_tasks, (num_samples,), generator=gen
        )
        self._noise = torch.randn(
            num_samples, *s.latent_shape, generator=gen
        ) * s.noise_floor

        # PRECOMPUTE all clean latents once (vectorized). __getitem__ then becomes
        # pure indexing, so training loops that sample thousands of batches are not
        # bottlenecked by a per-sample [context_dim]x[latent_numel] matmul on CPU.
        pooled = self._contexts.mean(dim=1)                 # [N, context_dim]
        shared = pooled @ self.projection                   # [N, latent_numel]
        shared = shared / (shared.std(dim=1, keepdim=True) + 1e-6)
        proto = self.prototypes[self._task_ids]             # [N, latent_numel]
        clean_flat = shared + s.memory_gain * proto
        clean = clean_flat.view(num_samples, *s.latent_shape) + self._noise
        self._clean = clean                                 # [N, C, T, H, W]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        task_id = int(self._task_ids[idx])
        return {
            "video_latent": self._clean[idx],
            "text_embedding": self._contexts[idx],
            "task_id": task_id,
            "memory_key": self.task_key[task_id],       # [memory_dim] retrieval query
            "memory_content": self.task_content[task_id],  # [memory_dim] stored content
        }

    # --- helpers for the training loop ---

    def memory_floor_mse(self) -> float:
        """Irreducible epsilon-MSE lower bound for a MEMORY-FREE model.

        A memory-free model cannot predict the per-task prototype, so on the
        clean-signal side its best constant guess for the memory component is 0.
        In epsilon-prediction the model's error on the memory component shows up
        scaled by the noise schedule; this returns the clean-space variance of the
        memory component (memory_gain^2 * E||prototype||^2 / latent_numel) as an
        interpretable proxy for "how much signal is memory-only".
        """
        s = self.spec
        proto_var = float((self.prototypes ** 2).mean().item())  # ~1.0 (unit std)
        return (s.memory_gain ** 2) * proto_var


def _make_projection_ms(spec: MemorySensitiveSpec, generator: torch.Generator) -> torch.Tensor:
    """Fixed low-rank map context -> shared latent for the memory-sensitive task."""
    a = torch.randn(spec.context_dim, spec.signal_rank, generator=generator)
    b = torch.randn(spec.signal_rank, spec.latent_numel, generator=generator)
    return (a @ b) / math.sqrt(spec.signal_rank)
