"""Stage 1 (FP16 baseline) + Stage 2 (ternary) trainer for BitMem.

One trainer, four quantization modes selected by config (§2, §6, §9):
    FP16     — weight+activation quant disabled (full precision baseline)
    INT8     — INT8 activations only, full-precision weights
    TERNARY  — W1.58 ternary weights + A8 activations (the target)
    MIXED    — ternary weights + full-precision activations

Because the ONLY thing that changes between Stage 1 and Stage 2 is the
QuantizationConfig, the comparison is clean: same data (seeded synthetic task),
same init (seeded), same optimizer, same steps. Any MSE gap is attributable to
quantization, not confounds.

This is a research prototype harness (§16). It trains a small DiT on a synthetic
task to validate the pipeline and produce the degradation table — not to make
production video.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bitvideo.models import VideoDiT
from bitvideo.quantization.quantization import (
    ActivationQuantizationConfig,
    QuantizationConfig,
    WeightQuantizationConfig,
)

from bitmem.train.synthetic import SyntheticDiffusionDataset, SyntheticSpec


# ---------------------------------------------------------------------------
# Quantization modes (§2, §9)
# ---------------------------------------------------------------------------


class QuantMode(str, Enum):
    FP16 = "fp16"        # full precision (weights + activations)
    INT8 = "int8"        # INT8 activations, full-precision weights
    TERNARY = "ternary"  # W1.58 + A8 (the target)
    MIXED = "mixed"      # ternary weights, full-precision activations


def quantization_for_mode(mode: QuantMode) -> QuantizationConfig:
    """Map a QuantMode to a QuantizationConfig.

    Uses the `enabled` flags on weight/activation configs as the escape hatch
    to full precision (confirmed behavior: fake-quant early-returns when disabled).
    """
    mode = QuantMode(mode)
    if mode is QuantMode.FP16:
        return QuantizationConfig(
            weight=WeightQuantizationConfig(enabled=False),
            activation=ActivationQuantizationConfig(enabled=False),
        )
    if mode is QuantMode.INT8:
        return QuantizationConfig(
            weight=WeightQuantizationConfig(enabled=False),
            activation=ActivationQuantizationConfig(enabled=True, bits=8),
        )
    if mode is QuantMode.TERNARY:
        # Defaults are W1.58 (threshold_factor=0.5) + A8 (bits=8) — packed compatible.
        return QuantizationConfig()
    if mode is QuantMode.MIXED:
        return QuantizationConfig(
            weight=WeightQuantizationConfig(enabled=True),
            activation=ActivationQuantizationConfig(enabled=False),
        )
    raise ValueError(f"unknown quant mode {mode}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BaselineConfig:
    """Configuration for the Stage 1/2 baseline trainer."""

    # Model (tiny by default so it runs on CPU in seconds)
    dim: int = 128
    depth: int = 2
    num_heads: int = 4

    # Quantization mode
    quant_mode: QuantMode = QuantMode.FP16

    # Task
    spec: SyntheticSpec = field(default_factory=SyntheticSpec)
    num_samples: int = 256
    data_seed: int = 0

    # Training
    max_steps: int = 300
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    num_train_timesteps: int = 1000

    # Reproducibility
    init_seed: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    # Logging
    log_every: int = 50


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class BaselineTrainer:
    """Trains a small VideoDiT on the synthetic task under one quant mode.

    The diffusion objective is epsilon-prediction with a cosine alpha schedule
    matching bitmem.eval.metrics.denoising_mse, so training and eval agree.
    """

    def __init__(self, config: BaselineConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Deterministic init so FP16 and ternary start from the SAME weights.
        torch.manual_seed(config.init_seed)

        self.quantization = quantization_for_mode(config.quant_mode)
        self.model = VideoDiT(
            in_channels=config.spec.in_channels,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            context_dim=config.spec.context_dim,
            patch_size=(1, 2, 2),
            quantization=self.quantization,
            device=self.device,
            dtype=config.dtype,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.dataset = SyntheticDiffusionDataset(
            num_samples=config.num_samples,
            spec=config.spec,
            seed=config.data_seed,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.global_step = 0
        self.history: list[dict] = []

    def _add_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Cosine schedule matching the eval metric."""
        b = clean.shape[0]
        alpha = torch.cos(
            timesteps.float() / self.config.num_train_timesteps * math.pi / 2
        ) ** 2
        alpha = alpha.view(b, 1, 1, 1, 1)
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def training_step(self, batch: dict) -> float:
        cfg = self.config
        video = batch["video_latent"].to(self.device, dtype=cfg.dtype)
        context = batch["text_embedding"].to(self.device, dtype=cfg.dtype)
        b = video.shape[0]

        timesteps = torch.randint(0, cfg.num_train_timesteps, (b,), device=self.device)
        noise = torch.randn_like(video)
        noisy = self._add_noise(video, noise, timesteps)

        pred = self.model(noisy, timesteps.float(), context)
        loss = F.mse_loss(pred, noise)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def train(self, *, verbose: bool = True) -> dict:
        """Run training; returns {'first_loss', 'final_loss', 'history'}."""
        cfg = self.config
        self.model.train()
        # Seed the training RNG stream separately so both modes see the same
        # noise/timestep sequence (data order + noise are deterministic).
        torch.manual_seed(cfg.init_seed + 1000)

        data_iter = iter(self.dataloader)
        first_loss = None
        running = 0.0
        t0 = time.perf_counter()

        while self.global_step < cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            loss = self.training_step(batch)
            if first_loss is None:
                first_loss = loss
            running += loss
            self.global_step += 1

            if self.global_step % cfg.log_every == 0:
                avg = running / cfg.log_every
                self.history.append({"step": self.global_step, "loss": avg})
                if verbose:
                    print(
                        f"  [{cfg.quant_mode.value:>7}] step {self.global_step:>4} "
                        f"| loss {avg:.5f}"
                    )
                running = 0.0

        elapsed = time.perf_counter() - t0
        final_loss = self.evaluate()
        return {
            "quant_mode": cfg.quant_mode.value,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "steps": self.global_step,
            "train_seconds": elapsed,
            "history": self.history,
            "params": self.model.parameter_count(),
        }

    @torch.no_grad()
    def evaluate(self, *, eval_batches: int = 8, eval_seed: int = 777) -> float:
        """Deterministic held-out MSE (same seed => comparable across modes)."""
        self.model.eval()
        cfg = self.config
        gen = torch.Generator(device="cpu").manual_seed(eval_seed)
        total, count = 0.0, 0

        for _ in range(eval_batches):
            idx = torch.randint(
                0, len(self.dataset), (cfg.batch_size,), generator=gen
            ).tolist()
            video = torch.stack(
                [self.dataset[i]["video_latent"] for i in idx]
            ).to(self.device, dtype=cfg.dtype)
            context = torch.stack(
                [self.dataset[i]["text_embedding"] for i in idx]
            ).to(self.device, dtype=cfg.dtype)
            b = video.shape[0]

            # Deterministic eval noise/timesteps.
            t_gen = torch.Generator(device="cpu").manual_seed(eval_seed + count)
            timesteps = torch.randint(
                0, cfg.num_train_timesteps, (b,), generator=t_gen
            ).to(self.device)
            noise = torch.randn(video.shape, generator=t_gen).to(
                self.device, dtype=cfg.dtype
            )
            noisy = self._add_noise(video, noise, timesteps)
            pred = self.model(noisy, timesteps.float(), context)
            total += float(F.mse_loss(pred, noise).item())
            count += 1

        self.model.train()
        return total / max(count, 1)
