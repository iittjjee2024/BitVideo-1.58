"""Stage-0 unit + integration tests for BitMem.

Validates:
  - MemoryItem construction, validation, and lifecycle
  - DictMemoryStore operations (write/read/retrieve/update/merge/consolidate/decay/delete)
  - Retrieval policies (cosine + weighted, with ablatable terms)
  - Diversity reranking
  - Method-B memory token interface (shapes, mask, ternary projection)
  - End-to-end prototype forward pass

Run:
    python -m pytest tests/bitmem/test_stage0.py -v
"""

from __future__ import annotations

import time

import pytest
import torch

from bitmem.memory.base import (
    MemoryItem,
    cosine_similarity,
    recency_weight,
)
from bitmem.memory.retrieval import (
    CosineRetrievalPolicy,
    WeightedRetrievalPolicy,
    rerank_with_diversity,
)
from bitmem.memory.storage import DictMemoryStore
from bitmem.interface.mem_tokens import MemoryTokenInterface


# ---------------------------------------------------------------------------
# MemoryItem
# ---------------------------------------------------------------------------


def _item(dim: int = 16, **kwargs) -> MemoryItem:
    emb = kwargs.pop("embedding", torch.randn(dim))
    return MemoryItem(content=kwargs.pop("content", "x"), embedding=emb, **kwargs)


def test_memory_item_valid():
    item = _item(importance=0.7, confidence=0.9)
    assert item.importance == 0.7
    assert item.confidence == 0.9
    assert item.access_count == 0
    assert len(item.id) > 0


def test_memory_item_rejects_bad_embedding():
    with pytest.raises(ValueError):
        MemoryItem(content="x", embedding=torch.randn(4, 4))  # not 1-D
    with pytest.raises(TypeError):
        MemoryItem(content="x", embedding=[1, 2, 3])  # not a tensor


def test_memory_item_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        _item(importance=1.5)
    with pytest.raises(ValueError):
        _item(confidence=-0.1)


def test_memory_item_touch():
    item = _item()
    item.recency = 0.1
    item.touch()
    assert item.access_count == 1
    assert item.recency == 1.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def test_cosine_similarity():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    c = torch.tensor([0.0, 1.0, 0.0])
    assert cosine_similarity(a, c) == pytest.approx(0.0, abs=1e-6)


def test_recency_weight_decays():
    now = 1000.0
    fresh = recency_weight(now, now, half_life=100.0)
    old = recency_weight(now - 100.0, now, half_life=100.0)
    assert fresh == pytest.approx(1.0)
    assert old == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# DictMemoryStore
# ---------------------------------------------------------------------------


def test_store_write_read():
    store = DictMemoryStore()
    item = _item(content="hello")
    item_id = store.write(item)
    assert len(store) == 1
    got = store.read(item_id)
    assert got is not None
    assert got.content == "hello"
    assert got.access_count == 1  # read touches


def test_store_retrieve_ranks_by_similarity():
    store = DictMemoryStore(policy=CosineRetrievalPolicy())
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    store.write(_item(embedding=torch.tensor([1.0, 0.0, 0.0, 0.0]), content="match"))
    store.write(_item(embedding=torch.tensor([0.0, 1.0, 0.0, 0.0]), content="orthogonal"))
    store.write(_item(embedding=torch.tensor([0.9, 0.1, 0.0, 0.0]), content="close"))
    results = store.retrieve(target, k=2)
    assert len(results) == 2
    assert results[0].content == "match"  # most similar first


def test_store_retrieve_with_filters():
    store = DictMemoryStore()
    store.write(_item(content="a", task="alpha", confidence=0.9))
    store.write(_item(content="b", task="beta", confidence=0.3))
    q = torch.randn(16)
    results = store.retrieve(q, k=5, filters={"task": "alpha"})
    assert all(r.task == "alpha" for r in results)
    results2 = store.retrieve(q, k=5, filters={"min_confidence": 0.5})
    assert all(r.confidence >= 0.5 for r in results2)


def test_store_update():
    store = DictMemoryStore()
    item_id = store.write(_item(importance=0.5))
    store.update(item_id, importance=0.9)
    assert store.read(item_id).importance == 0.9


def test_store_merge():
    store = DictMemoryStore()
    id1 = store.write(_item(embedding=torch.ones(8), importance=0.3))
    id2 = store.write(_item(embedding=torch.ones(8) * 3, importance=0.8))
    merged_id = store.merge([id1, id2])
    assert len(store) == 1
    merged = store.read(merged_id)
    assert merged.compressed is True
    assert merged.importance == 0.8  # max
    # embedding is mean of [1,1,...] and [3,3,...] = [2,2,...]
    assert torch.allclose(merged.embedding, torch.ones(8) * 2)


def test_store_consolidate_boosts_importance():
    store = DictMemoryStore()
    id1 = store.write(_item(importance=0.5, embedding=torch.randn(8)))
    id2 = store.write(_item(importance=0.5, embedding=torch.randn(8)))
    cid = store.consolidate([id1, id2])
    consolidated = store.read(cid)
    assert consolidated.importance == pytest.approx(0.7)  # 0.5 + 0.2 boost
    assert consolidated.provenance["op"] == "consolidate"


def test_store_decay_forgets_stale_low_utility():
    store = DictMemoryStore()
    old_ts = time.time() - 100_000
    store.write(_item(embedding=torch.randn(8), utility=0.1, timestamp=old_ts))
    deleted = store.decay(time.time(), half_life=100.0)
    assert deleted == 1
    assert len(store) == 0


def test_store_decay_keeps_useful():
    store = DictMemoryStore()
    old_ts = time.time() - 100_000
    store.write(_item(embedding=torch.randn(8), utility=0.9, timestamp=old_ts))
    deleted = store.decay(time.time(), half_life=100.0)
    assert deleted == 0
    assert len(store) == 1


def test_store_eviction_respects_max_items():
    store = DictMemoryStore(max_items=3)
    for i in range(5):
        store.write(_item(utility=float(i) / 10, embedding=torch.randn(8)))
    assert len(store) <= 3


# ---------------------------------------------------------------------------
# Retrieval policies
# ---------------------------------------------------------------------------


def test_weighted_policy_ablation_semantic_only():
    # Setting beta=gamma=epsilon=0 -> pure semantic (matches cosine policy)
    policy = WeightedRetrievalPolicy(alpha=1.0, beta=0.0, gamma=0.0, epsilon=0.0)
    a = torch.tensor([1.0, 0.0])
    item = MemoryItem(content="x", embedding=torch.tensor([1.0, 0.0]))
    now = time.time()
    assert policy.score(a, item, now) == pytest.approx(1.0)


def test_weighted_policy_importance_term():
    policy = WeightedRetrievalPolicy(alpha=0.0, beta=0.0, gamma=1.0, epsilon=0.0)
    item = MemoryItem(content="x", embedding=torch.randn(4), importance=0.7)
    score = policy.score(torch.randn(4), item, time.time())
    assert score == pytest.approx(0.7)


def test_diversity_rerank_reduces_redundancy():
    # Three items: two nearly identical, one different.
    e1 = torch.tensor([1.0, 0.0])
    e2 = torch.tensor([0.99, 0.01])  # near-duplicate of e1
    e3 = torch.tensor([0.0, 1.0])    # diverse
    items = [
        MemoryItem(content="1", embedding=e1),
        MemoryItem(content="2", embedding=e2),
        MemoryItem(content="3", embedding=e3),
    ]
    ranked = [(1.0, items[0]), (0.98, items[1]), (0.5, items[2])]
    # With diversity, item 3 should be preferred over near-duplicate item 2.
    result = rerank_with_diversity(ranked, top_m=2, diversity_weight=1.0)
    contents = {r.content for r in result}
    assert "1" in contents
    assert "3" in contents  # diverse item chosen over near-duplicate


# ---------------------------------------------------------------------------
# Memory token interface (Method B)
# ---------------------------------------------------------------------------


def test_memory_token_interface_shapes():
    iface = MemoryTokenInterface(memory_dim=64, model_dim=128, max_tokens=4)
    mems = [
        [MemoryItem(content="a", embedding=torch.randn(64))],
        [MemoryItem(content="b", embedding=torch.randn(64)),
         MemoryItem(content="c", embedding=torch.randn(64))],
    ]
    tokens, mask = iface.build_tokens(mems, device=torch.device("cpu"), dtype=torch.float32)
    assert tokens.shape == (2, 4, 128)
    assert mask.shape == (2, 4)
    # sample 0 has 1 memory, sample 1 has 2
    assert mask[0].sum() == 1
    assert mask[1].sum() == 2


def test_memory_token_interface_inject_prepends():
    iface = MemoryTokenInterface(memory_dim=64, model_dim=128, max_tokens=4)
    tokens = torch.randn(2, 10, 128)
    mems = [[MemoryItem(content="a", embedding=torch.randn(64))] for _ in range(2)]
    augmented, mask = iface.inject(tokens, mems)
    assert augmented.shape == (2, 14, 128)  # 4 memory + 10 content
    stripped = MemoryTokenInterface.strip(augmented, 4)
    assert stripped.shape == (2, 10, 128)


def test_memory_token_interface_empty_memories():
    iface = MemoryTokenInterface(memory_dim=64, model_dim=128, max_tokens=4)
    tokens = torch.randn(1, 5, 128)
    augmented, mask = iface.inject(tokens, [[]])  # no memories
    assert augmented.shape == (1, 9, 128)
    assert mask.sum() == 0  # all padding


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


def test_prototype_end_to_end():
    from bitmem.prototype import run_prototype
    diag = run_prototype(device="cpu", verbose=False)
    # Output shape must match input shape (denoising prediction)
    assert diag["output_shape"] == diag["input_shape"]
    assert diag["output_finite"] is True
    assert diag["memory_items"] == 20
    assert all(n <= 3 for n in diag["retrieved_per_sample"])
    assert diag["dit_params"] > 0
