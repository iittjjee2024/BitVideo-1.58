"""Memory-store backend comparison — recall, latency, and storage, side by side.

Compares the four store backends on identical data so the recall / latency /
compression tradeoffs are visible in one table:

  * brute-fp32   : DictMemoryStore            — exact, O(N), full fp32 keys
  * ann-fp32     : AnnMemoryStore             — HNSW, ~O(log N), full fp32 keys
  * turbo-brute  : TurboQuantMemoryStore      — exact O(N), compressed keys
  * turbo-ann    : TurboAnnMemoryStore        — HNSW + compressed keys (both wins)

Metrics (all MEASURED, nothing asserted):
  * recall@k vs the exact fp32 brute-force index (ground truth)
  * mean query latency (ms)
  * persisted key-storage compression vs fp32 (1.0x for the fp32 backends)

The point: `turbo-ann` should keep recall close to `ann-fp32` while matching the
storage compression of `turbo-brute` — scalable AND compact at once.

Usage:
    python scripts/bitmem_backend_compare.py
    python scripts/bitmem_backend_compare.py --n 5000 --dim 128 --k 10 --bits 8
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitmem.memory.base import MemoryItem
from bitmem.memory.storage import DictMemoryStore
from bitmem.memory.ann_store import AnnMemoryStore, _hnswlib_available
from bitmem.memory.turbo_store import TurboQuantMemoryStore
from bitmem.memory.turbo_ann_store import TurboAnnMemoryStore


def _unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def _fill(store, keys):
    for i in range(keys.shape[0]):
        store.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))


def _eval(store, queries, k):
    sets, t0 = [], time.perf_counter()
    for q in queries:
        sets.append({m.content for m in store.retrieve(q, k)})
    ms = (time.perf_counter() - t0) / len(queries) * 1000.0
    return sets, ms


def _compression(store):
    fn = getattr(store, "compression_ratio", None)
    return fn() if callable(fn) else 1.0


def main() -> None:
    p = argparse.ArgumentParser(description="Memory-store backend comparison")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--queries", type=int, default=150)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    keys = _unit(args.n, args.dim, seed=args.seed)
    g = torch.Generator().manual_seed(args.seed + 1)
    pick = torch.randint(0, args.n, (args.queries,), generator=g)
    queries = keys[pick] + args.noise * torch.randn(args.queries, args.dim, generator=g)

    print("=" * 80)
    print("Memory-store backend comparison (recall / latency / storage)")
    print("=" * 80)
    print(f"N={args.n} dim={args.dim} k={args.k} queries={args.queries} "
          f"noise={args.noise} turbo_bits={args.bits}")
    if not _hnswlib_available():
        print("NOTE: hnswlib not installed — ANN backends fall back to brute force.")
    print("-" * 80)

    # Ground truth = exact fp32 brute force.
    bf = DictMemoryStore()
    _fill(bf, keys)
    exact_sets, bf_ms = _eval(bf, queries, args.k)

    backends = [
        ("brute-fp32", bf, bf_ms, exact_sets),
    ]

    ann = AnnMemoryStore(args.dim, use_ann=True, max_items=args.n + 16)
    _fill(ann, keys)
    s, ms = _eval(ann, queries, args.k)
    backends.append(("ann-fp32", ann, ms, s))

    tq = TurboQuantMemoryStore(args.dim, bits=args.bits, seed=args.seed)
    _fill(tq, keys)
    s, ms = _eval(tq, queries, args.k)
    backends.append(("turbo-brute", tq, ms, s))

    ta = TurboAnnMemoryStore(args.dim, bits=args.bits, max_items=args.n + 16, seed=args.seed)
    _fill(ta, keys)
    s, ms = _eval(ta, queries, args.k)
    backends.append(("turbo-ann", ta, ms, s))

    print(f"  {'backend':>12} | {'recall@%d' % args.k:>9} | {'ms/query':>9} | "
          f"{'speedup':>8} | {'storage':>8}")
    print("  " + "-" * 60)
    for name, store, ms, sets in backends:
        hit = sum(len(a & b) for a, b in zip(sets, exact_sets))
        tot = sum(len(b) for b in exact_sets)
        recall = hit / max(tot, 1)
        speed = bf_ms / ms if ms > 0 else float("inf")
        comp = _compression(store)
        print(f"  {name:>12} | {recall:>9.3f} | {ms:>9.3f} | {speed:>7.1f}x | "
              f"{comp:>7.2f}x")

    print("-" * 80)
    print("Reading: turbo-ann aims to match ann-fp32's recall+speed AND")
    print("turbo-brute's storage compression — scalable and compact together.")
    print("Honest notes:")
    print("  * recall vs exact fp32 brute force; HNSW + quantization are both")
    print("    approximate, so recall < 1.0 is expected (tune bits / ef_query).")
    print("  * storage = persisted key bytes vs fp32 (in-RAM reconstruction is")
    print("    separate); fp32 backends are 1.0x by definition.")
    print("  * CPU single-thread timings; the N-scaling trend is the point.")
    print("=" * 80)


if __name__ == "__main__":
    main()
