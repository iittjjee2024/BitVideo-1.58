"""TurboQuant benchmark — honest reconstruction + retrieval + compression numbers.

Measures, across bit-widths, what TurboQuant actually buys the BitMem memory
store (the paper's vector-DB / nearest-neighbor use case, arXiv:2504.19874):

  * reconstruction MSE of stored keys (lower = closer to fp32)
  * retrieval recall@k vs an EXACT fp32 brute-force index (the ground truth)
  * storage compression ratio (persisted code bytes vs fp32 keys)

Nothing is asserted here — the script prints MEASURED values. The verdict is
simply: at what bit-width does TurboQuant keep retrieval ~lossless while
shrinking the stored keys? We report it rather than claim it.

Usage:
    python scripts/bitmem_turboquant_bench.py
    python scripts/bitmem_turboquant_bench.py --n 2000 --dim 256 --k 10
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitmem.memory.base import MemoryItem
from bitmem.memory.turbo_store import TurboQuantMemoryStore
from bitmem.memory.turboquant import TurboQuantMSE, TurboQuantConfig


def exact_topk(keys: torch.Tensor, queries: torch.Tensor, k: int) -> torch.Tensor:
    """Ground-truth top-k by cosine similarity (fp32 brute force)."""
    kn = keys / keys.norm(dim=1, keepdim=True).clamp_min(1e-8)
    qn = queries / queries.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sims = qn @ kn.t()                       # [Q, N]
    return sims.topk(k, dim=1).indices       # [Q, k]


def recall_at_k(approx_idx: list[list[int]], exact_idx: torch.Tensor, k: int) -> float:
    """Fraction of each query's true top-k that the approximate index recovered."""
    total = 0.0
    for q, true_row in enumerate(exact_idx.tolist()):
        true_set = set(true_row[:k])
        got = set(approx_idx[q][:k])
        total += len(true_set & got) / k
    return total / len(approx_idx)


def main() -> None:
    p = argparse.ArgumentParser(description="TurboQuant store benchmark")
    p.add_argument("--n", type=int, default=1000, help="number of stored keys")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--noise", type=float, default=0.05, help="query noise vs a stored key")
    p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 8])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    keys = torch.randn(args.n, args.dim)
    keys = keys / keys.norm(dim=1, keepdim=True)

    # Queries = noisy versions of random stored keys (realistic near-duplicate recall).
    pick = torch.randint(0, args.n, (args.queries,))
    queries = keys[pick] + args.noise * torch.randn(args.queries, args.dim)

    exact = exact_topk(keys, queries, args.k)

    print("=" * 74)
    print("TurboQuant store benchmark (vector-DB compression for BitMem memory)")
    print("=" * 74)
    print(f"keys={args.n} dim={args.dim} queries={args.queries} k={args.k} "
          f"query_noise={args.noise}")
    print(f"ground truth = exact fp32 cosine top-{args.k}")
    print("-" * 74)
    print(f"  {'bits':>4} | {'recon MSE':>10} | {'recall@%d' % args.k:>9} | "
          f"{'bits/coord':>10} | {'compression':>11}")
    print("  " + "-" * 70)

    # fp32 reference row (store without quantization = plain cosine).
    fp32_recall = recall_at_k(
        [exact[i].tolist() for i in range(args.queries)], exact, args.k
    )
    print(f"  {'fp32':>4} | {0.0:>10.6f} | {fp32_recall:>9.3f} | "
          f"{32.0:>10.1f} | {1.0:>10.2f}x")

    for bits in args.bits:
        # Reconstruction MSE on the raw keys.
        codec = TurboQuantMSE(args.dim, TurboQuantConfig(bits=bits, seed=args.seed))
        recon = codec.quantize(keys)
        recon_mse = float(((keys - recon) ** 2).mean())

        # Build a TurboQuant store and measure recall vs the exact fp32 index.
        store = TurboQuantMemoryStore(args.dim, bits=bits, seed=args.seed)
        for i in range(args.n):
            store.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=1.0))
        approx = []
        for qi in range(args.queries):
            hits = store.retrieve(queries[qi], args.k)
            approx.append([h.content for h in hits])
        rec = recall_at_k(approx, exact, args.k)

        print(f"  {bits:>4} | {recon_mse:>10.6f} | {rec:>9.3f} | "
              f"{codec.bits_per_coordinate():>10.1f} | {store.compression_ratio():>10.2f}x")

    print("-" * 74)
    print("Reading: pick the smallest bit-width whose recall@k stays ~fp32.")
    print("Honest notes:")
    print("  * recon MSE is on raw stored keys; recall is end-to-end retrieval quality.")
    print("  * compression is PERSISTED code bytes vs fp32 keys (what a vector DB saves")
    print("    on disk / ships); in-RAM reconstruction footprint is not reduced here.")
    print("  * data-oblivious codec: no training/calibration, one fixed rotation.")
    print("=" * 74)


if __name__ == "__main__":
    main()
