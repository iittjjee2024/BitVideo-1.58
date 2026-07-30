"""Quantization-aware training script for BitVideo-1.58 Video DiT."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from bitvideo.models import VideoDiT
from bitvideo.pipeline.schedulers import NoiseScheduler, DDIMScheduler

from .datasets import SyntheticVideoDataset, create_dataloader
from .losses import DiffusionLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for QAT training."""

    # Model.
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    context_dim: int = 768
    in_channels: int = 4
    patch_size: tuple[int, int, int] = (1, 2, 2)

    # Data.
    num_frames: int = 16
    height: int = 32
    width: int = 32
    text_length: int = 77
    batch_size: int = 1

    # Training.
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100_000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    prediction_type: str = "epsilon"
    loss_type: str = "mse"
    snr_gamma: float | None = 5.0

    # Scheduler.
    num_train_timesteps: int = 1000
    beta_schedule: str = "linear"

    # Mixed precision.
    mixed_precision: str = "bf16"  # none, fp16, bf16

    # Checkpointing.
    output_dir: str = "outputs"
    save_every_steps: int = 5000
    log_every_steps: int = 100

    # Hardware.
    seed: int = 42
    num_workers: int = 0


class QATTrainer:
    """Quantization-aware training loop for Video DiT.

    Handles the complete training lifecycle:
    - Model instantiation with quantization config
    - Optimizer and LR scheduler setup
    - Mixed-precision training with gradient scaling
    - Noise scheduling and diffusion loss computation
    - Checkpointing and logging
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_amp()

    def _setup_model(self) -> None:
        """Initialize the Video DiT model."""
        cfg = self.config
        self.model = VideoDiT(
            in_channels=cfg.in_channels,
            dim=cfg.dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            context_dim=cfg.context_dim,
            patch_size=cfg.patch_size,
            qk_norm=True,
            device=self.device,
            dtype=torch.float32,
        )
        self.model.train()
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {param_count:,}")

    def _setup_optimizer(self) -> None:
        """Configure AdamW optimizer with weight decay."""
        cfg = self.config
        # Separate parameters that should/shouldn't have weight decay.
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "gate" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            eps=1.0e-8,
        )

    def _setup_scheduler(self) -> None:
        """Configure noise scheduler and LR scheduler."""
        cfg = self.config
        self.noise_scheduler = DDIMScheduler(
            num_train_steps=cfg.num_train_timesteps,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
        )
        # Cosine LR schedule with warmup.
        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return float(step) / max(1.0, float(cfg.warmup_steps))
            progress = float(step - cfg.warmup_steps) / max(
                1.0, float(cfg.max_steps - cfg.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def _setup_loss(self) -> None:
        """Configure diffusion loss function."""
        cfg = self.config
        self.loss_fn = DiffusionLoss(
            prediction_type=cfg.prediction_type,
            loss_type=cfg.loss_type,
            snr_gamma=cfg.snr_gamma,
        )

    def _setup_amp(self) -> None:
        """Configure automatic mixed precision."""
        cfg = self.config
        if cfg.mixed_precision == "fp16" and self.device.type == "cuda":
            self.autocast_dtype = torch.float16
            self.scaler = GradScaler()
        elif cfg.mixed_precision == "bf16" and self.device.type == "cuda":
            self.autocast_dtype = torch.bfloat16
            self.scaler = None
        else:
            self.autocast_dtype = None
            self.scaler = None

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single training step.

        Args:
            batch: Dict with 'video_latent' [B,C,T,H,W] and
                   'text_embedding' [B,L,D].

        Returns:
            Scalar loss for this step.
        """
        video = batch["video_latent"].to(self.device)
        context = batch["text_embedding"].to(self.device)
        batch_size = video.shape[0]

        # Sample random timesteps: [B].
        timesteps = torch.randint(
            0, self.config.num_train_timesteps, (batch_size,), device=self.device
        )
        # Sample noise: same shape as video.
        noise = torch.randn_like(video)
        # Add noise to video: [B, C, T, H, W].
        noisy_video = self.noise_scheduler.add_noise(video, noise, timesteps)

        # Compute target based on prediction_type.
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(video, noise, timesteps)
        else:
            target = video

        # Forward pass with optional mixed precision.
        if self.autocast_dtype is not None:
            with torch.autocast(self.device.type, dtype=self.autocast_dtype):
                model_output = self.model(
                    noisy_video, timesteps.float(), context
                )
                loss = self.loss_fn(
                    model_output, target,
                    timesteps=timesteps,
                    alphas_cumprod=self.noise_scheduler.alphas_cumprod.to(self.device),
                )
        else:
            model_output = self.model(noisy_video, timesteps.float(), context)
            loss = self.loss_fn(
                model_output, target,
                timesteps=timesteps,
                alphas_cumprod=self.noise_scheduler.alphas_cumprod.to(self.device),
            )
        return loss

    def train(self, dataloader: DataLoader) -> None:
        """Run the full training loop.

        Args:
            dataloader: DataLoader yielding video-text batch dicts.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        torch.manual_seed(cfg.seed)
        logger.info(f"Starting training for {cfg.max_steps} steps")
        start_time = time.time()

        data_iter = iter(dataloader)
        self.model.train()

        while self.global_step < cfg.max_steps:
            accumulated_loss = 0.0
            for accum_step in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                loss = self.training_step(batch)
                loss = loss / cfg.gradient_accumulation_steps

                # Backward pass.
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated_loss += loss.item()

            # Gradient clipping.
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm
            )

            # Optimizer step.
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self.global_step += 1

            # Logging.
            if self.global_step % cfg.log_every_steps == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"step={self.global_step}, loss={accumulated_loss:.4f}, "
                    f"lr={lr:.2e}, elapsed={elapsed:.1f}s"
                )

            # Checkpointing.
            if self.global_step % cfg.save_every_steps == 0:
                self.save_checkpoint()

        logger.info(f"Training complete at step {self.global_step}")
        self.save_checkpoint()

    def save_checkpoint(self) -> None:
        """Save model and optimizer state."""
        cfg = self.config
        path = Path(cfg.output_dir) / f"checkpoint-{self.global_step}.pt"
        torch.save(
            {
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model and optimizer state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")


def train_qat(config: TrainingConfig | None = None) -> None:
    """Entry point for QAT training.

    Args:
        config: Training configuration. Uses defaults if None.
    """
    if config is None:
        config = TrainingConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    trainer = QATTrainer(config)
    dataset = SyntheticVideoDataset(
        num_samples=max(config.batch_size * 10, 100),
        latent_channels=config.in_channels,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        text_length=config.text_length,
        text_dim=config.context_dim,
        seed=config.seed,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    trainer.train(dataloader)
