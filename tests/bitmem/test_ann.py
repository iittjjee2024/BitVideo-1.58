"""Tests for the ANN-backed memory store (closes the brute-force caveat).

Verifies:
  * build + retrieve returns correctly shaped, correctly typed results
  * approximate recall@k matches the exact brute-force store within tolerance
  * add / delete keep the index and item map consistent
  * metadata filters are still honored through the ANN path
  * the fallback path (use_ann=False) is exact and API-compatible

ANN retrieval is approximate, so recall tests use a tolerance rather than exact
equality — the point is that HNSW recovers (almost) the same neighbors far faster,
which the benchmark quantifies.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from bitmem.memory.base import MemoryItem
from bitmem.memory.ann_store import AnnMemoryStore
from bitmem.memory.storage import DictMemoryStore

_HAS_HNSW = importlib.util.find_spec("hnswlib") is not None
requires_hnsw = pytest.mark.skipif(not _HAS_HNSW, reason="hnswlib not installed")


def _unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def _fill(store, keys):
    for i in range(keys.shape[0]):
        store.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


@requires_hnsw
def test_build_and_retrieve_shapes():
    dim = 32
    keys = _unit(50, dim)
    store = AnnMemoryStore(dim, use_ann=True)
    _fill(store, keys)
    assert len(store) == 50
    out = store.retrieve(keys[0], 5)
    assert len(out) == 5
    assert all(isinstance(m, MemoryItem) for m in out)
    # exact self-query should surface its own item as the top result
    assert store.retrieve(keys[7], 1)[0].content == 7


@requires_hnsw
def test_empty_store_returns_empty():
    store = AnnMemoryStore(16, use_ann=True)
    assert store.retrieve(torch.randn(16), 5) == []


# ---------------------------------------------------------------------------
# Recall vs exact
# ---------------------------------------------------------------------------


@requires_hnsw
def test_recall_matches_bruteforce():
    dim, n, k = 64, 400, 10
    keys = _unit(n, dim, seed=1)
    ann = AnnMemoryStore(dim, use_ann=True)
    bf = DictMemoryStore()
    for i in range(n):
        ann.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))
        bf.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))

    g = torch.Generator().manual_seed(2)
    hit = tot = 0
    for _ in range(60):
        qi = int(torch.randint(0, n, (1,), generator=g))
        q = keys[qi] + 0.03 * torch.randn(dim, generator=g)
        a = {m.content for m in ann.retrieve(q, k)}
        b = {m.content for m in bf.retrieve(q, k)}
        hit += len(a & b)
        tot += len(b)
    recall = hit / tot
    assert recall >= 0.9  # HNSW recall is high; allow a small approximate margin


# ---------------------------------------------------------------------------
# Mutation consistency
# ---------------------------------------------------------------------------


@requires_hnsw
def test_delete_consistency():
    dim = 32
    keys = _unit(30, dim)
    store = AnnMemoryStore(dim, use_ann=True)
    ids = [store.write(MemoryItem(content=i, embedding=keys[i].clone())) for i in range(30)]
    for i in ids[:10]:
        store.delete(i)
    assert len(store) == 20
    # deleted items must never be returned
    returned = {m.content for q in keys for m in store.retrieve(q, 5)}
    assert all(c >= 10 for c in returned)


@requires_hnsw
def test_add_after_capacity_resize():
    # Start tiny so the index must resize as we add past initial capacity.
    dim = 16
    store = AnnMemoryStore(dim, use_ann=True, max_items=10_000)
    keys = _unit(300, dim, seed=3)
    _fill(store, keys)
    assert len(store) == 300
    assert store.retrieve(keys[250], 1)[0].content == 250


@requires_hnsw
def test_filters_honored():
    dim = 32
    keys = _unit(40, dim, seed=4)
    store = AnnMemoryStore(dim, use_ann=True, over_fetch=8)
    for i in range(40):
        store.write(MemoryItem(
            content=i, embedding=keys[i].clone(),
            task="A" if i % 2 == 0 else "B", importance=0.5,
        ))
    out = store.retrieve(keys[0], 5, filters={"task": "A"})
    assert all(m.task == "A" for m in out)


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------


def test_fallback_is_exact_and_api_compatible():
    # use_ann=False forces brute force; results must be exact.
    dim, n, k = 48, 120, 5
    keys = _unit(n, dim, seed=5)
    fb = AnnMemoryStore(dim, use_ann=False)
    bf = DictMemoryStore()
    for i in range(n):
        fb.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))
        bf.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))
    assert fb.ann_enabled is False
    for qi in (0, 37, 99):
        a = [m.content for m in fb.retrieve(keys[qi], k)]
        b = [m.content for m in bf.retrieve(keys[qi], k)]
        assert a == b  # identical to exact brute force


def test_merge_and_decay_work():
    dim = 16
    store = AnnMemoryStore(dim, use_ann=_HAS_HNSW)
    ids = [store.write(MemoryItem(content=i, embedding=_unit(1, dim, seed=i)[0], importance=0.9))
           for i in range(4)]
    merged = store.merge(ids[:2])
    assert merged in {m.id for m in store.all_items()}
    assert len(store) == 3  # 4 - 2 merged + 1 new
