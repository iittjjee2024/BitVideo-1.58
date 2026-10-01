"""Tests for the combined TurboQuant + ANN memory store.

Verifies the backend is simultaneously compact (TurboQuant codes) and scalable
(HNSW retrieval), and that both wins compose without breaking the MemoryStore
contract:
  * recall@k vs an exact fp32 brute-force index within tolerance
  * storage compression > 1 (keys persisted as low-bit codes)
  * add / delete consistency through the index
  * metadata filters honored
  * exact brute-force fallback when the ANN path is disabled
  * MemorySystem.with_turbo_ann factory wiring
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from bitmem.memory.base import MemoryItem
from bitmem.memory.turbo_ann_store import TurboAnnMemoryStore
from bitmem.memory.storage import DictMemoryStore
from bitmem.memory.typed import MemorySystem

_HAS_HNSW = importlib.util.find_spec("hnswlib") is not None
requires_hnsw = pytest.mark.skipif(not _HAS_HNSW, reason="hnswlib not installed")


def _unit(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=1, keepdim=True)


def _fill(store, keys):
    for i in range(keys.shape[0]):
        store.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))


@requires_hnsw
def test_build_and_retrieve_shapes():
    dim = 64
    keys = _unit(60, dim)
    store = TurboAnnMemoryStore(dim, bits=8, seed=0)
    _fill(store, keys)
    assert len(store) == 60
    out = store.retrieve(keys[0], 5)
    assert len(out) == 5
    assert all(isinstance(m, MemoryItem) for m in out)


@requires_hnsw
def test_recall_vs_bruteforce_8bit():
    dim, n, k = 128, 500, 10
    keys = _unit(n, dim, seed=1)
    store = TurboAnnMemoryStore(dim, bits=8, seed=0)
    bf = DictMemoryStore()
    for i in range(n):
        store.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))
        bf.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5))
    g = torch.Generator().manual_seed(2)
    hit = tot = 0
    for _ in range(60):
        qi = int(torch.randint(0, n, (1,), generator=g))
        q = keys[qi] + 0.03 * torch.randn(dim, generator=g)
        a = {m.content for m in store.retrieve(q, k)}
        b = {m.content for m in bf.retrieve(q, k)}
        hit += len(a & b)
        tot += len(b)
    # 8-bit TurboQuant + HNSW should stay close to exact fp32 brute force.
    assert hit / tot >= 0.9


@requires_hnsw
def test_compression_ratio():
    dim = 128
    keys = _unit(50, dim, seed=3)
    store8 = TurboAnnMemoryStore(dim, bits=8, seed=0)
    store4 = TurboAnnMemoryStore(dim, bits=4, seed=0)
    _fill(store8, keys)
    _fill(store4, keys)
    assert store8.compression_ratio() > 3.0   # 8-bit keys vs fp32
    assert store4.compression_ratio() > store8.compression_ratio()  # fewer bits = more compression


@requires_hnsw
def test_delete_consistency():
    dim = 32
    keys = _unit(30, dim, seed=4)
    store = TurboAnnMemoryStore(dim, bits=8, seed=0)
    ids = [store.write(MemoryItem(content=i, embedding=keys[i].clone())) for i in range(30)]
    for i in ids[:10]:
        store.delete(i)
    assert len(store) == 20
    returned = {m.content for q in keys for m in store.retrieve(q, 5)}
    assert all(c >= 10 for c in returned)
    # codes pruned along with items
    assert len(store._codes) == 20


@requires_hnsw
def test_filters_honored():
    dim = 64
    keys = _unit(40, dim, seed=5)
    store = TurboAnnMemoryStore(dim, bits=8, seed=0, over_fetch=8)
    for i in range(40):
        store.write(MemoryItem(
            content=i, embedding=keys[i].clone(),
            task="A" if i % 2 == 0 else "B", importance=0.5,
        ))
    out = store.retrieve(keys[0], 5, filters={"task": "B"})
    assert all(m.task == "B" for m in out)


def test_fallback_still_compresses_and_retrieves():
    # use_ann=False -> brute force over reconstructions, still TurboQuant-compressed.
    dim, n = 48, 80
    keys = _unit(n, dim, seed=6)
    store = TurboAnnMemoryStore(dim, bits=8, use_ann=False)
    _fill(store, keys)
    assert store.ann_enabled is False
    # Still compressed vs fp32 (exact ratio depends on dim padding + scale
    # overhead; at dim=48 -> padded 64 the 8-bit codes are ~2.8x smaller).
    assert store.compression_ratio() > 1.5
    assert store.retrieve(keys[10], 1)[0].content == 10


def test_memory_system_factory():
    ms = MemorySystem.with_turbo_ann(64, bits=8, seed=0)
    assert type(ms.episodic.store).__name__ == "TurboAnnMemoryStore"
    assert type(ms.long_term.store).__name__ == "TurboAnnMemoryStore"
    torch.manual_seed(0)
    keys = _unit(25, 64, seed=7)
    for i in range(25):
        ms.write(MemoryItem(content=i, embedding=keys[i].clone(), importance=0.5),
                 memory_type="episodic")
    hits = sum(1 for i in range(25) if (r := ms.retrieve(keys[i], 1)) and r[0].content == i)
    assert hits >= 24
