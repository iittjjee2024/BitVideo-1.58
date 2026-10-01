"""ANN vs brute-force retrieval benchmark — honest recall + latency numbers.

Measures what the HNSW ANN backend (`AnnMemoryStore`) buys over the exact
brute-force `DictMemoryStore` as the store grows:

  * recall@k  : fraction of the exact top-k neighbors the ANN index recovers
  * query latency : mean per-query wall time for each backend

Nothing is asserted — the script prints MEASURED values so the recall/speed
tradeoff is reported honestly. HNSW is approximate, so recall < 1.0 is expected
and acceptable when it buys a large latency reduction at scale.

Usage:
    python scripts/bitmem_ann_bench.py
    python scripts/bitmem_ann_bench.py --sizes 1000 5000 20000 --k 10 --queries 300
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitmem.memory.base import MemoryItem
from bitmem.memory.ann_store import AnnMemoryStore, _hnswlib_available
from bitmem.memory.storage import DictMemoryStore


def _unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def _time_queries(store, queries, k) -> tuple[float, list[set]]:
    sets = []
    t0 = time.perf_counter()
    for q in queries:
        sets.append({m.content for m in store.retrieve(q, k)})
    elapsed = (time.perf_counter() - t0) / len(queries) * 1000.0  # ms/query
    return elapsed, sets


def main() -> None:
    p = argparse.ArgumentParser(description="ANN vs brute-force retrieval benchmark")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 76)
    print("ANN (HNSW) vs brute-force retrieval — recall + latency")
    print("=" * 76)
    if not _hnswlib_available():
        print("hnswlib NOT installed — ANN store would fall back to brute force.")
        print("Install hnswlib to benchmark the ANN path.")
        return
    print(f"dim={args.dim} k={args.k} queries={args.queries} query_noise={args.noise}")
    print("-" * 76)
    print(f"  {'N':>7} | {'recall@%d' % args.k:>9} | {'brute ms/q':>11} | "
          f"{'ann ms/q':>9} | {'speedup':>8}")
    print("  " + "-" * 72)

    for n in args.sizes:
        keys = _unit(n, args.dim, seed=args.seed)
        bf = DictMemoryStore()
        ann = AnnMemoryStore(args.dim, use_ann=True, max_items=n + 16)
        for i in range(n):
            bf.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))
            ann.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))

        g = torch.Generator().manual_seed(args.seed + 1)
        pick = torch.randint(0, n, (args.queries,), generator=g)
        queries = keys[pick] + args.noise * torch.randn(args.queries, args.dim, generator=g)

        bf_ms, bf_sets = _time_queries(bf, queries, args.k)
        ann_ms, ann_sets = _time_queries(ann, queries, args.k)

        hit = sum(len(a & b) for a, b in zip(ann_sets, bf_sets))
        tot = sum(len(b) for b in bf_sets)
        recall = hit / max(tot, 1)
        speedup = bf_ms / ann_ms if ann_ms > 0 else float("inf")

        print(f"  {n:>7} | {recall:>9.3f} | {bf_ms:>11.3f} | {ann_ms:>9.3f} | "
              f"{speedup:>7.2f}x")

    print("-" * 76)
    print("Reading: ANN keeps recall high while query time stays ~flat as N grows,")
    print("whereas brute force is O(N). The crossover favors ANN at larger stores.")
    print("Honest notes:")
    print("  * HNSW is APPROXIMATE: recall < 1.0 is expected; tune ef_query/M to trade")
    print("    recall vs speed. Reranking still uses the full RetrievalPolicy.")
    print("  * timings are CPU wall-clock, single-threaded Python loop; absolute ms")
    print("    will differ by machine — the N-scaling trend is the point.")
    print("=" * 76)


if __name__ == "__main__":
    main()
