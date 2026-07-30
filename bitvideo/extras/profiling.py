"""Model profiling utilities for performance analysis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ProfileResult:
    """Results from a profiling run."""

    total_params: int = 0
    trainable_params: int = 0
    total_flops: int = 0
    peak_memory_mb: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    layer_times: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"Total parameters: {self.total_params:,}",
            f"Trainable parameters: {self.trainable_params:,}",
            f"Peak memory: {self.peak_memory_mb:.1f} MB",
            f"Forward time: {self.forward_time_ms:.2f} ms",
            f"Backward time: {self.backward_time_ms:.2f} ms",
            f"Throughput: {self.throughput_samples_per_sec:.1f} samples/s",
        ]
        if self.total_flops > 0:
            lines.append(f"Total FLOPs: {self.total_flops:,}")
        return "\n".join(lines)


class ModelProfiler:
    """Profile a model's computational characteristics.

    Measures parameter count, memory usage, forward/backward timing,
    and optionally per-layer breakdown.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        warmup_steps: int = 3,
        measure_steps: int = 10,
    ) -> None:
        self.model = model
        self.warmup_steps = max(1, warmup_steps)
        self.measure_steps = max(1, measure_steps)

    def count_parameters(self) -> tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def profile(
        self,
        *example_inputs: Any,
        backward: bool = True,
        batch_size: int = 1,
    ) -> ProfileResult:
        """Run a full profiling pass.

        Args:
            example_inputs: Example inputs to the model.
            backward: Whether to also measure backward pass.
            batch_size: Batch size for throughput calculation.

        Returns:
            ProfileResult with timing and memory data.
        """
        result = ProfileResult()
        result.total_params, result.trainable_params = self.count_parameters()

        device = next(self.model.parameters()).device
        is_cuda = device.type == "cuda"

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # Warmup.
        self.model.train() if backward else self.model.eval()
        for _ in range(self.warmup_steps):
            with torch.set_grad_enabled(backward):
                output = self.model(*example_inputs)
                if backward and isinstance(output, torch.Tensor):
                    output.sum().backward()
                    self.model.zero_grad(set_to_none=True)

        if is_cuda:
            torch.cuda.synchronize(device)

        # Measure forward.
        forward_times: list[float] = []
        backward_times: list[float] = []

        for _ in range(self.measure_steps):
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            with torch.set_grad_enabled(backward):
                output = self.model(*example_inputs)

            if is_cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            forward_times.append((t1 - t0) * 1000.0)

            if backward and isinstance(output, torch.Tensor):
                if is_cuda:
                    torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                output.sum().backward()
                if is_cuda:
                    torch.cuda.synchronize(device)
                t3 = time.perf_counter()
                backward_times.append((t3 - t2) * 1000.0)
                self.model.zero_grad(set_to_none=True)

        result.forward_time_ms = sum(forward_times) / len(forward_times)
        if backward_times:
            result.backward_time_ms = sum(backward_times) / len(backward_times)

        total_time_ms = result.forward_time_ms + result.backward_time_ms
        if total_time_ms > 0:
            result.throughput_samples_per_sec = batch_size * 1000.0 / total_time_ms

        if is_cuda:
            result.peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        return result


def profile_model(
    model: nn.Module,
    *example_inputs: Any,
    backward: bool = True,
    batch_size: int = 1,
    warmup_steps: int = 3,
    measure_steps: int = 10,
) -> ProfileResult:
    """Convenience function for quick model profiling.

    Args:
        model: Model to profile.
        example_inputs: Example forward inputs.
        backward: Measure backward pass too.
        batch_size: For throughput calculation.
        warmup_steps: GPU warmup iterations.
        measure_steps: Measurement iterations.

    Returns:
        ProfileResult with measurements.
    """
    profiler = ModelProfiler(model, warmup_steps=warmup_steps, measure_steps=measure_steps)
    return profiler.profile(*example_inputs, backward=backward, batch_size=batch_size)
