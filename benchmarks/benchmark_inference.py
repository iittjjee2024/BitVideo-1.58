"""Inference benchmark for BitVideo-1.58 Video DiT.

Measures throughput, latency, and memory for various model configurations.
Run with: python benchmarks/benchmark_inference.py
"""

import time
import torch
from bitvideo.models import VideoDiT
from bitvideo.extras import profile_model


def benchmark_config(
    name: str,
    dim: int,
    depth: int,
    num_heads: int,
    *,
    batch_size: int = 1,
    num_frames: int = 4,
    height: int = 8,
    width: int = 8,
    context_dim: int = 64,
    context_length: int = 5,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    warmup: int = 3,
    measure: int = 10,
) -> dict:
    """Benchmark a single model configuration."""
    model = VideoDiT(
        in_channels=4, dim=dim, depth=depth, num_heads=num_heads,
        context_dim=context_dim, patch_size=(1, 2, 2),
        ffn_expansion_ratio=2.0, qk_norm=True,
        device=device, dtype=dtype,
    ).eval()

    video = torch.randn(batch_size, 4, num_frames, height, width, device=device, dtype=dtype)
    timesteps = torch.full((batch_size,), 500.0, device=device)
    context = torch.randn(batch_size, context_length, context_dim, device=device, dtype=dtype)

    result = profile_model(
        model, video, timesteps, context,
        backward=False, batch_size=batch_size,
        warmup_steps=warmup, measure_steps=measure,
    )

    return {
        "name": name,
        "params": result.total_params,
        "forward_ms": result.forward_time_ms,
        "throughput": result.throughput_samples_per_sec,
        "peak_memory_mb": result.peak_memory_mb,
    }


def main():
    print("=" * 70)
    print("BitVideo-1.58 Inference Benchmark")
    print("=" * 70)

    configs = [
        ("Tiny (D=64, L=2)", 64, 2, 4),
        ("Small (D=128, L=4)", 128, 4, 8),
        ("Base (D=256, L=6)", 256, 6, 8),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {dtype}\n")

    print(f"{'Config':<25} {'Params':>12} {'Forward (ms)':>14} {'Throughput':>14} {'Memory (MB)':>12}")
    print("-" * 70)

    for name, dim, depth, heads in configs:
        result = benchmark_config(
            name, dim, depth, heads,
            device=device, dtype=dtype,
            warmup=2, measure=5,
        )
        print(
            f"{result['name']:<25} "
            f"{result['params']:>12,} "
            f"{result['forward_ms']:>12.2f}ms "
            f"{result['throughput']:>11.1f}/s "
            f"{result['peak_memory_mb']:>10.1f}"
        )

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
