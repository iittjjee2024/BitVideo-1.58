"""BitMem Stage 1 + Stage 2 CLI — FP16 baseline vs ternary degradation.

Trains the SAME tiny DiT on the SAME seeded synthetic task under multiple
quantization modes, then reports:
  - convergence (first vs final loss) proving the harness learns
  - the quantization-degradation table (§8 Stage 2, §9)
  - the efficiency table (params, theoretical ternary storage, FP16 storage,
    latency, samples/s) (§12)

Because init, data, and eval are all seeded, any MSE gap between modes is
attributable to quantization — not to different random draws.

Usage:
    python scripts/bitmem_stage1_baseline.py                     # all 4 modes
    python scripts/bitmem_stage1_baseline.py --modes fp16 ternary
    python scripts/bitmem_stage1_baseline.py --steps 500 --dim 192
    python scripts/bitmem_stage1_baseline.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitmem.train.baseline import BaselineConfig, BaselineTrainer, QuantMode
from bitmem.train.synthetic import SyntheticSpec, make_synthetic_batch
from bitmem.eval.metrics import (
    measure_efficiency,
    ternary_degradation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BitMem Stage 1/2 baseline vs ternary")
    p.add_argument("--modes", nargs="+",
                   default=["fp16", "int8", "ternary", "mixed"],
                   choices=["fp16", "int8", "ternary", "mixed"])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    spec = SyntheticSpec()
    print("=" * 68)
    print("BitMem Stage 1 + Stage 2 — Baseline vs Ternary Degradation")
    print("=" * 68)
    print(f"Model: dim={args.dim} depth={args.depth} heads={args.heads}")
    print(f"Task:  synthetic low-rank (rank={spec.signal_rank}, "
          f"noise_floor={spec.noise_floor}), {args.num_samples} samples")
    print(f"Train: {args.steps} steps, batch {args.batch_size}, lr {args.lr}, device {device}")
    print("=" * 68)

    results: dict[str, dict] = {}
    efficiency = []

    example = make_synthetic_batch(
        batch_size=args.batch_size, spec=spec, seed=args.seed, device=device
    )

    for mode_str in args.modes:
        mode = QuantMode(mode_str)
        print(f"\n── Training [{mode.value}] ──")
        cfg = BaselineConfig(
            dim=args.dim, depth=args.depth, num_heads=args.heads,
            quant_mode=mode, spec=spec, num_samples=args.num_samples,
            max_steps=args.steps, batch_size=args.batch_size,
            learning_rate=args.lr, data_seed=args.seed, init_seed=args.seed,
            device=str(device),
        )
        trainer = BaselineTrainer(cfg)
        result = trainer.train(verbose=not args.quiet)
        results[mode.value] = result

        eff = measure_efficiency(
            trainer.model, example, label=mode.value, device=device
        )
        efficiency.append(eff)

        print(f"   first_loss={result['first_loss']:.5f} "
              f"-> final_loss={result['final_loss']:.5f} "
              f"({result['train_seconds']:.1f}s)")

    # ---- Convergence check (Stage 1 validity) ----
    print("\n" + "=" * 68)
    print("CONVERGENCE (harness must learn: final < first)")
    print("=" * 68)
    for mode, r in results.items():
        learned = r["final_loss"] < r["first_loss"]
        print(f"  {mode:>7}: {r['first_loss']:.5f} -> {r['final_loss']:.5f}  "
              f"{'LEARNED' if learned else 'NO LEARNING'}")

    # ---- Degradation table (Stage 2) ----
    if "fp16" in results:
        print("\n" + "=" * 68)
        print("QUANTIZATION DEGRADATION (vs FP16 baseline)")
        print("=" * 68)
        fp16_mse = results["fp16"]["final_loss"]
        print(f"  {'mode':>8} | {'final MSE':>10} | {'Δ vs FP16':>12} | {'relative':>10}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
        for mode, r in results.items():
            deg = ternary_degradation(fp16_mse, r["final_loss"])
            print(f"  {mode:>8} | {r['final_loss']:>10.5f} | "
                  f"{deg.absolute_increase:>+12.5f} | {deg.relative_increase:>+9.1%}")

    # ---- Efficiency table ----
    print("\n" + "=" * 68)
    print("EFFICIENCY (theoretical ternary storage vs FP16 actual)")
    print("=" * 68)
    print(f"  {'mode':>8} | {'params':>9} | {'ternary_MB':>10} | "
          f"{'fp16_MB':>8} | {'lat_ms':>7} | {'samp/s':>7}")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}")
    for eff in efficiency:
        row = eff.as_row()
        print(f"  {row['label']:>8} | {row['params']:>9,} | "
              f"{row['ternary_MB(theory)']:>10.3f} | {row['fp16_MB']:>8.3f} | "
              f"{row['latency_ms']:>7.2f} | {row['samples/s']:>7.1f}")

    print("\n" + "=" * 68)
    print("NOTE: ternary_MB is the log2(3)-bit STORAGE lower bound. Training-time")
    print("memory still holds FP32/BF16 master weights (§9). Latency here is the")
    print("QAT fake-quant path on CPU, NOT the packed-inference kernels.")
    print("=" * 68)


if __name__ == "__main__":
    main()
