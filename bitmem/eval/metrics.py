"""BitMem evaluation metrics (§12).

Stage 1 focuses on GENERATION quality (denoising MSE on a synthetic task) and
EFFICIENCY (parameter memory, peak VRAM, latency, samples/s). FID / CLIP-sim and
the memory/agent metrics arrive when we have real data and the agent controller.

Everything here is deterministic given a seed so the baseline vs ternary
comparison (§8 Stage 1 vs Stage 2) is reproducible.

Distinguishes THEORETICAL 1.58-bit storage from ACTUAL memory usage (§9): a
ternary weight is log2(3) ≈ 1.585 bits in theory, but the master weight during
training is still FP32/BF16. We report both.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Total parameter count."""
    return sum(
        p.numel() for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )


def theoretical_ternary_bytes(num_ternary_params: int) -> float:
    """Theoretical storage for ternary params at log2(3) ≈ 1.585 bits each.

    This is the STORAGE lower bound, NOT the training-time memory (which holds
    FP32/BF16 master weights). Reported separately to avoid the common
    '1.58-bit' overclaim (§9).
    """
    bits_per_param = math.log2(3)  # ≈ 1.585
    return num_ternary_params * bits_per_param / 8.0


def fp16_bytes(num_params: int) -> float:
    """Actual bytes if params were stored at FP16 (2 bytes each)."""
    return num_params * 2.0


# ---------------------------------------------------------------------------
# Generation quality
# ---------------------------------------------------------------------------


def denoising_mse(
    model: nn.Module,
    batch: dict,
    *,
    device: torch.device,
    num_train_timesteps: int = 1000,
) -> float:
    """Mean squared error of epsilon prediction on a batch.

    The core diffusion training objective. Lower is better. Used to compare
    FP16 vs ternary backbones on the SAME data with the SAME noise (§8).

    Args:
        model: a VideoDiT (or memory-augmented variant).
        batch: dict with 'video_latent' [B,C,T,H,W] and 'text_embedding' [B,L,D].
        device: evaluation device.
        num_train_timesteps: diffusion horizon.

    Returns:
        Scalar MSE as a Python float.
    """
    video = batch["video_latent"].to(device)
    context = batch["text_embedding"].to(device)
    b = video.shape[0]

    # Deterministic noise/timesteps come from the caller's seeding.
    timesteps = torch.randint(0, num_train_timesteps, (b,), device=device)
    noise = torch.randn_like(video)

    # Simple cosine alpha schedule (matches the synthetic task).
    alpha = torch.cos(timesteps.float() / num_train_timesteps * math.pi / 2) ** 2
    alpha = alpha.view(b, 1, 1, 1, 1)
    noisy = alpha.sqrt() * video + (1 - alpha).sqrt() * noise

    with torch.no_grad():
        pred = model(noisy, timesteps.float(), context)
    return float(nn.functional.mse_loss(pred, noise).item())


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


@dataclass
class EfficiencyReport:
    """Efficiency measurements for one model configuration (§12 efficiency)."""

    label: str
    total_params: int
    trainable_params: int
    theoretical_ternary_mb: float   # log2(3) bits/param storage lower bound
    fp16_equivalent_mb: float       # actual FP16 storage
    peak_vram_mb: float             # measured peak allocation (0 on CPU)
    latency_ms: float               # mean forward latency
    samples_per_sec: float
    device: str

    def as_row(self) -> dict:
        return {
            "label": self.label,
            "params": self.total_params,
            "ternary_MB(theory)": round(self.theoretical_ternary_mb, 2),
            "fp16_MB": round(self.fp16_equivalent_mb, 2),
            "peak_VRAM_MB": round(self.peak_vram_mb, 1),
            "latency_ms": round(self.latency_ms, 2),
            "samples/s": round(self.samples_per_sec, 2),
        }


def measure_efficiency(
    model: nn.Module,
    example_batch: dict,
    *,
    label: str,
    device: torch.device,
    warmup: int = 2,
    iters: int = 10,
) -> EfficiencyReport:
    """Benchmark a model's forward pass and report efficiency metrics.

    Measures actual latency, throughput, and peak VRAM. Distinguishes
    theoretical ternary storage from FP16 storage (§9).

    Args:
        model: model to benchmark (already on `device`).
        example_batch: dict with 'video_latent' and 'text_embedding'.
        label: name for this configuration (e.g. 'FP16', 'ternary').
        device: benchmark device.
        warmup: warmup iterations (not timed).
        iters: timed iterations.

    Returns:
        EfficiencyReport.
    """
    model.eval()
    video = example_batch["video_latent"].to(device)
    context = example_batch["text_embedding"].to(device)
    b = video.shape[0]
    timesteps = torch.zeros(b, device=device).float()

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Warmup (not timed) — triggers lazy packing / compilation.
    with torch.no_grad():
        for _ in range(warmup):
            model(video, timesteps, context)
    if is_cuda:
        torch.cuda.synchronize(device)

    # Timed.
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            model(video, timesteps, context)
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    latency_ms = elapsed / iters * 1000.0
    samples_per_sec = (b * iters) / elapsed
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / 1e6 if is_cuda else 0.0
    )

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)

    return EfficiencyReport(
        label=label,
        total_params=total,
        trainable_params=trainable,
        theoretical_ternary_mb=theoretical_ternary_bytes(total) / 1e6,
        fp16_equivalent_mb=fp16_bytes(total) / 1e6,
        peak_vram_mb=peak_vram_mb,
        latency_ms=latency_ms,
        samples_per_sec=samples_per_sec,
        device=str(device),
    )


# ---------------------------------------------------------------------------
# Ternary degradation (Stage 1 vs Stage 2)
# ---------------------------------------------------------------------------


@dataclass
class DegradationResult:
    """Quantization-degradation comparison (§8 Stage 2, §9)."""

    fp16_mse: float
    ternary_mse: float
    relative_increase: float  # (ternary - fp16) / fp16
    absolute_increase: float

    def summary(self) -> str:
        return (
            f"FP16 MSE={self.fp16_mse:.5f} | ternary MSE={self.ternary_mse:.5f} | "
            f"Δ={self.absolute_increase:+.5f} ({self.relative_increase:+.1%})"
        )


def ternary_degradation(fp16_mse: float, ternary_mse: float) -> DegradationResult:
    """Compute how much generation quality degrades under ternarization.

    Positive relative_increase = ternary is worse (expected). This is the
    number the hypothesis (§18) must weigh against the memory/efficiency gains.
    """
    rel = (ternary_mse - fp16_mse) / max(abs(fp16_mse), 1e-8)
    return DegradationResult(
        fp16_mse=fp16_mse,
        ternary_mse=ternary_mse,
        relative_increase=rel,
        absolute_increase=ternary_mse - fp16_mse,
    )
