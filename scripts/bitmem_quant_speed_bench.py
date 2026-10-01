"""Quantized-matmul speed benchmark — real measured numbers (§9, §12).

This closes the caveat threaded through the whole project: "ternary speed is not
benchmarked." It produces MEASURED latency on this machine for four execution
paths, so the efficiency story is grounded in numbers rather than theory:

  1. fp16 F.linear            — the dense full-precision baseline
  2. BitLinear QAT (fake-quant) — what training / eval-with-grad actually pays:
                                 fake-quantize activations + ternary weights, then
                                 a float matmul. Extra ops, so it is SLOWER than fp16.
  3. BitLinear packed inference — the no-grad packed path (3-tier dispatcher). On a
                                 box without the compiled CUDA extension / Triton
                                 this runs the portable Torch tier.
  4. torch._int_mm INT8        — a genuine low-bit matmul primitive, as a LOWER
                                 BOUND on what a real packed 1.58-bit kernel could
                                 deliver on this GPU.

It also reports numerical error (quantized output vs fp16) so speed is never read
without the accuracy it costs.

Honest scope (§16):
  * True W1.58 packed kernels require the compiled `bitvideo._C` extension (not
    built here) or Triton (not installed); without them the packed path falls to
    the portable tier and is NOT a fair representation of 1.58-bit arithmetic.
  * `torch._int_mm` is INT8xINT8->INT32, not ternary; it is the closest real
    low-bit matmul available and bounds the achievable speedup from below.
  * Storage compression (log2(3) bits/weight) is theoretical and separate from
    these latency numbers.

Usage:
    python scripts/bitmem_quant_speed_bench.py
    python scripts/bitmem_quant_speed_bench.py --sizes 1024 2048 4096 --iters 50
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitvideo.quantization.bit_linear import BitLinear
from bitvideo.quantization.quantization import QuantizationConfig
from bitvideo.ops.backends import backend_status


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, *, warmup, iters, device) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0  # ms/iter


def bench_size(dim: int, batch: int, *, device, dtype, warmup, iters) -> dict:
    torch.manual_seed(0)
    x = torch.randn(batch, dim, device=device, dtype=dtype)
    # Dense fp16 reference layer.
    lin = torch.nn.Linear(dim, dim, bias=False, device=device, dtype=dtype)

    # --- 1. fp16 F.linear ---
    fp16_ms = _time(lambda: F.linear(x, lin.weight), warmup=warmup, iters=iters, device=device)
    with torch.no_grad():
        ref = F.linear(x, lin.weight)

    # --- 2. BitLinear QAT (fake-quant) forward (grad-enabled path) ---
    bl = BitLinear(dim, dim, bias=False, quantization=QuantizationConfig(),
                   device=device, dtype=dtype)
    with torch.no_grad():
        bl.weight.copy_(lin.weight)
    bl.train()  # force the QAT fake-quant path
    def qat_call():
        return bl(x)
    qat_ms = _time(qat_call, warmup=warmup, iters=iters, device=device)
    with torch.no_grad():
        bl.eval()
        qat_out = bl._qat_forward(x)
    qat_err = float((qat_out - ref).abs().mean() / (ref.abs().mean() + 1e-8))

    # --- 3. BitLinear packed inference (no_grad; portable tier here) ---
    packed_ms = None
    packed_err = None
    try:
        bl.eval()
        with torch.inference_mode():
            bl.pack_weights()
            def packed_call():
                return bl(x)
            packed_ms = _time(packed_call, warmup=warmup, iters=iters, device=device)
            packed_out = bl(x)
        packed_err = float((packed_out.float() - ref.float()).abs().mean()
                           / (ref.float().abs().mean() + 1e-8))
    except Exception as exc:  # noqa: BLE001 — report, don't crash the sweep
        packed_ms = float("nan")
        packed_err = f"unavailable ({type(exc).__name__})"

    # --- 4. torch._int_mm INT8 (real low-bit matmul lower bound) ---
    int8_ms = None
    if callable(getattr(torch, "_int_mm", None)) and device.type == "cuda":
        xi = torch.randint(-127, 128, (batch, dim), dtype=torch.int8, device=device)
        wi = torch.randint(-127, 128, (dim, dim), dtype=torch.int8, device=device)
        # _int_mm requires contiguous int8 [M,K] x [K,N].
        int8_ms = _time(lambda: torch._int_mm(xi, wi), warmup=warmup, iters=iters, device=device)

    return {
        "dim": dim, "batch": batch,
        "fp16_ms": fp16_ms,
        "qat_ms": qat_ms, "qat_err": qat_err,
        "packed_ms": packed_ms, "packed_err": packed_err,
        "int8_ms": int8_ms,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Quantized-matmul speed benchmark")
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    st = backend_status()
    print("=" * 84)
    print("Quantized-matmul speed benchmark (real measured latency)")
    print("=" * 84)
    print(f"device={device} dtype={dtype} batch(M)={args.batch} iters={args.iters}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"backends available: cuda_ext={st.cuda_extension} triton={st.triton} "
          f"torch_int_mm={st.torch_int_mm}")
    if not st.cuda_extension and not st.triton:
        print("NOTE: no compiled 1.58-bit kernels -> packed path uses the PORTABLE tier;")
        print("      it is not a fair representation of true ternary arithmetic.")
    print("-" * 84)
    print(f"  {'dim (KxN)':>10} | {'fp16 ms':>8} | {'QAT ms':>8} | {'packed ms':>10} | "
          f"{'int8 ms':>8} | {'int8 vs fp16':>12} | {'QAT err':>8}")
    print("  " + "-" * 80)

    for dim in args.sizes:
        r = bench_size(dim, args.batch, device=device, dtype=dtype,
                       warmup=args.warmup, iters=args.iters)
        packed = f"{r['packed_ms']:.3f}" if isinstance(r['packed_ms'], float) and r['packed_ms'] == r['packed_ms'] else "n/a"
        int8 = f"{r['int8_ms']:.3f}" if r['int8_ms'] is not None else "n/a"
        speed = (f"{r['fp16_ms']/r['int8_ms']:.2f}x"
                 if r['int8_ms'] else "n/a")
        qerr = f"{r['qat_err']:.4f}" if isinstance(r['qat_err'], float) else str(r['qat_err'])
        print(f"  {dim:>10} | {r['fp16_ms']:>8.3f} | {r['qat_ms']:>8.3f} | {packed:>10} | "
              f"{int8:>8} | {speed:>12} | {qerr:>8}")

    print("-" * 84)
    print("Reading honestly:")
    print("  * QAT (fake-quant) is SLOWER than fp16 by design — it adds quant ops on top")
    print("    of a float matmul; it exists for TRAINING accuracy, not inference speed.")
    print("  * int8 vs fp16 is the real low-bit matmul speedup available on THIS GPU and")
    print("    bounds from below what a true packed 1.58-bit kernel could achieve.")
    print("  * the packed path here runs the portable tier (no compiled kernels), so its")
    print("    latency reflects PyTorch overhead, not 1.58-bit arithmetic.")
    print("  * QAT err = mean |quantized - fp16| / mean|fp16|: the accuracy cost of W1.58A8.")
    print("=" * 84)


if __name__ == "__main__":
    main()
