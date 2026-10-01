"""BitMem Stage 5 — joint optimization + full ablation matrix (§8 Stage 5, §13, §18).

This is the falsifiable-hypothesis harness. It trains cells of an ablation
matrix on the memory-sensitive task (where a large fraction of the signal is
retrievable ONLY from memory) and reports, per cell:

    final denoising MSE (x0 objective) | params | theoretical ternary bytes | fp16 bytes

Everything runs on the GPU automatically when one is present (device="auto").

Three experiments, each answering a distinct question:

  A. MATCHED-SIZE memory ablation (the clean test).
     Same backbone (size + quant), memory ON vs OFF. Isolates memory's effect
     with NO size confound: does adding a memory adapter lower the loss?

  B. INJECTION-METHOD ablation (§5, §13).
     Fixed small ternary backbone; none / memory_tokens / cross_attention /
     adaptive. Which injection method uses memory best?

  C. THE HYPOTHESIS (§18): small ternary + memory  vs  larger fp16 no-memory.
     Does memory let a small, cheap, ternary model rival a much larger
     full-precision one? Run at LOW task-diversity (few prototypes the big model
     can memorize in weights) and HIGH task-diversity (too many to memorize),
     because the answer depends on whether the task actually FORCES memory use.

The verdict is stated honestly: we do NOT claim a win unless the numbers show it,
and we report the conditions under which the hypothesis holds and fails.

Caveats (§16, printed at the end):
  * Synthetic task, tiny models, x0 objective — validates the MECHANISM and the
    ablation methodology, NOT real-video quality.
  * Latency is not a fair speed benchmark here: the ternary path uses fake-quant
    (QAT) ops, not packed kernels, so wall-clock is dominated by Python/op
    overhead, not the arithmetic the 1.58-bit format would actually use.

Usage:
    python scripts/bitmem_stage5_ablation.py                # default quick matrix
    python scripts/bitmem_stage5_ablation.py --steps 1500   # longer training
    python scripts/bitmem_stage5_ablation.py --experiment A  # just one experiment
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitmem.train.joint import JointTrainer, JointConfig
from bitmem.train.synthetic import MemorySensitiveSpec
from bitmem.train.baseline import QuantMode
from bitmem.interface.unified import InjectionMethod


def _run(name, *, spec, steps, num_samples, batch, lr, device, **kw) -> dict:
    cfg = JointConfig(
        num_heads=4, spec=spec, num_samples=num_samples, max_steps=steps,
        batch_size=batch, log_every=steps, predict="x0", learning_rate=lr,
        device=device, **kw,
    )
    t0 = time.perf_counter()
    tr = JointTrainer(cfg)
    r = tr.train(verbose=False)
    r["name"] = name
    r["seconds"] = time.perf_counter() - t0
    r["device"] = str(tr.device)
    return r


def _print_rows(rows: list[dict]) -> None:
    hdr = f"  {'cell':>34} | {'final MSE':>9} | {'params':>10} | {'ternary KB':>10} | {'fp16 KB':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['name']:>34} | {r['final_loss']:>9.5f} | {r['total_params']:>10,} | "
            f"{r['ternary_bytes_theory']/1e3:>10.1f} | {r['fp16_bytes']/1e3:>9.1f}"
        )


def experiment_A(args, device) -> list[dict]:
    """Matched-size memory ON vs OFF (clean, no size confound)."""
    print("\n" + "=" * 78)
    print("EXPERIMENT A — matched-size memory ablation (does memory help, size held fixed?)")
    print("=" * 78)
    spec = MemorySensitiveSpec(num_tasks=args.num_tasks, memory_gain=args.gain)
    common = dict(spec=spec, steps=args.steps, num_samples=args.num_samples,
                  batch=args.batch, lr=args.lr, device=device,
                  dim=args.small_dim, depth=args.small_depth,
                  quant_mode=QuantMode.TERNARY)
    rows = [
        _run("ternary d%dx%d  NO memory" % (args.small_dim, args.small_depth),
             injection=InjectionMethod.NONE, **common),
        _run("ternary d%dx%d  + memory(adaptive)" % (args.small_dim, args.small_depth),
             injection=InjectionMethod.ADAPTIVE, **common),
        _run("ternary d%dx%d  + memory(tokens)" % (args.small_dim, args.small_depth),
             injection=InjectionMethod.MEMORY_TOKENS, **common),
    ]
    _print_rows(rows)
    off = rows[0]["final_loss"]
    best_on = min(r["final_loss"] for r in rows[1:])
    delta = (off - best_on) / off
    print(f"\n  memory OFF = {off:.5f} | best memory ON = {best_on:.5f} "
          f"| improvement = {delta:+.1%}")
    if best_on < off:
        print("  => VERDICT A: memory LOWERS loss at fixed size (mechanism works).")
    else:
        print("  => VERDICT A: memory did NOT help at fixed size (mechanism failed here).")
    return rows


def experiment_B(args, device) -> list[dict]:
    """Injection-method ablation on a fixed small ternary backbone."""
    print("\n" + "=" * 78)
    print("EXPERIMENT B — injection method ablation (which method uses memory best?)")
    print("=" * 78)
    spec = MemorySensitiveSpec(num_tasks=args.num_tasks, memory_gain=args.gain)
    common = dict(spec=spec, steps=args.steps, num_samples=args.num_samples,
                  batch=args.batch, lr=args.lr, device=device,
                  dim=args.small_dim, depth=args.small_depth,
                  quant_mode=QuantMode.TERNARY)
    rows = [
        _run("none", injection=InjectionMethod.NONE, **common),
        _run("memory_tokens", injection=InjectionMethod.MEMORY_TOKENS, **common),
        _run("cross_attention", injection=InjectionMethod.CROSS_ATTENTION, **common),
        _run("adaptive", injection=InjectionMethod.ADAPTIVE, **common),
    ]
    _print_rows(rows)
    best = min(rows[1:], key=lambda r: r["final_loss"])
    print(f"\n  => best injection method: {best['name']} (MSE {best['final_loss']:.5f})")
    return rows


def experiment_C(args, device) -> list[dict]:
    """THE HYPOTHESIS: small ternary + memory vs larger fp16 no-memory.

    Run at both low and high task-diversity, because the big model can memorize
    a few prototypes in its weights but not many.
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT C — HYPOTHESIS: small ternary+memory vs LARGER fp16 no-memory")
    print("=" * 78)
    all_rows: list[dict] = []
    for label, num_tasks in (("low diversity", args.low_tasks), ("high diversity", args.high_tasks)):
        print(f"\n  --- task diversity: {label} ({num_tasks} prototypes) ---")
        spec = MemorySensitiveSpec(num_tasks=num_tasks, memory_gain=args.gain)
        common = dict(spec=spec, steps=args.steps, num_samples=args.num_samples,
                      batch=args.batch, lr=args.lr, device=device)
        small = _run(
            f"[{label}] small TERNARY+mem d{args.small_dim}x{args.small_depth}",
            dim=args.small_dim, depth=args.small_depth,
            quant_mode=QuantMode.TERNARY, injection=InjectionMethod.MEMORY_TOKENS,
            **common,
        )
        large = _run(
            f"[{label}] LARGE fp16 no-mem d{args.large_dim}x{args.large_depth}",
            dim=args.large_dim, depth=args.large_depth,
            quant_mode=QuantMode.FP16, injection=InjectionMethod.NONE,
            **common,
        )
        _print_rows([small, large])
        param_ratio = large["total_params"] / max(small["total_params"], 1)
        byte_ratio = large["fp16_bytes"] / max(small["ternary_bytes_theory"], 1)
        print(f"\n  small uses {param_ratio:.1f}x fewer params, "
              f"{byte_ratio:.0f}x smaller (ternary-theory vs fp16 storage)")
        if small["final_loss"] <= large["final_loss"]:
            print(f"  => VERDICT C ({label}): HYPOTHESIS SUPPORTED — small ternary+memory "
                  f"matches/beats the {param_ratio:.1f}x larger fp16 model "
                  f"({small['final_loss']:.5f} <= {large['final_loss']:.5f}).")
        else:
            gap = (small["final_loss"] - large["final_loss"]) / large["final_loss"]
            print(f"  => VERDICT C ({label}): NOT supported — larger fp16 model wins by "
                  f"{gap:+.1%} ({small['final_loss']:.5f} vs {large['final_loss']:.5f}).")
        all_rows += [small, large]
    return all_rows


def main() -> None:
    p = argparse.ArgumentParser(description="BitMem Stage 5 ablation matrix")
    p.add_argument("--experiment", choices=["A", "B", "C", "all"], default="all")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--gain", type=float, default=2.0)
    p.add_argument("--num-tasks", type=int, default=64, help="tasks for A/B")
    p.add_argument("--low-tasks", type=int, default=8, help="C low diversity")
    p.add_argument("--high-tasks", type=int, default=256, help="C high diversity")
    p.add_argument("--small-dim", type=int, default=48)
    p.add_argument("--small-depth", type=int, default=2)
    p.add_argument("--large-dim", type=int, default=128)
    p.add_argument("--large-depth", type=int, default=3)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )

    print("=" * 78)
    print("BitMem Stage 5 — joint optimization + ablation matrix")
    print("=" * 78)
    print(f"device={device} | steps={args.steps} | task=memory-sensitive x0 "
          f"| gain={args.gain}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows: list[dict] = []
    if args.experiment in ("A", "all"):
        rows += experiment_A(args, device)
    if args.experiment in ("B", "all"):
        rows += experiment_B(args, device)
    if args.experiment in ("C", "all"):
        rows += experiment_C(args, device)

    print("\n" + "=" * 78)
    print("CAVEATS (do not over-read these numbers):")
    print("  * Synthetic task + tiny models + x0 objective: validates the MECHANISM")
    print("    and the ablation methodology, NOT real-video quality.")
    print("  * Ternary path uses fake-quant (QAT) ops, not packed 1.58-bit kernels,")
    print("    so wall-clock is NOT a fair speed benchmark; storage bytes are theoretical.")
    print("  * final MSE is the x0 (clean-latent) objective on held-out samples.")
    print("=" * 78)


if __name__ == "__main__":
    main()
