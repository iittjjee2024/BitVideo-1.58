"""Approximate-nearest-neighbor memory store — closes the brute-force caveat (§15).

Stage 0's `DictMemoryStore` does O(N) brute-force retrieval, which it honestly
flags as a prototype limitation. `AnnMemoryStore` replaces the candidate-recall
step with a real ANN index (hnswlib HNSW) so retrieval scales to large stores,
while keeping everything else — metadata, filters, decay, merge, diversity
reranking, and the configurable `RetrievalPolicy` — identical.

Two-stage retrieval (the standard vector-DB pattern):
  1. ANN RECALL: the HNSW index returns the top `over_fetch * k` candidates by
     cosine similarity in ~O(log N) instead of O(N).
  2. POLICY RERANK: the existing `RetrievalPolicy.score` reranks those candidates
     with the full multi-factor score (semantic + recency + importance + ...),
     then diversity reranking selects the final k. So the ANN only accelerates
     candidate generation; the retrieval SEMANTICS are unchanged.

If hnswlib is not installed, the store transparently falls back to the exact
brute-force `DictMemoryStore` (same API, same results, just O(N)). Callers get
correctness everywhere and speed where the library is present.

Honesty (§16): HNSW is APPROXIMATE — recall is high but not guaranteed 1.0. The
benchmark (`scripts/bitmem_ann_bench.py`) measures recall@k vs the exact index so
the tradeoff is reported, not assumed. The index stores a copy of each embedding
(the win is query latency, not RAM); a disk-backed index is future work.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any

import torch

from bitmem.memory.base import MemoryItem, RetrievalPolicy
from bitmem.memory.retrieval import CosineRetrievalPolicy, rerank_with_diversity
from bitmem.memory.storage import DictMemoryStore


def _hnswlib_available() -> bool:
    return importlib.util.find_spec("hnswlib") is not None


class AnnMemoryStore:
    """HNSW-backed memory store with exact-brute-force fallback.

    Args:
        dim: embedding dimensionality (D_mem).
        policy: retrieval scoring policy applied to ANN candidates.
        max_items: capacity hint (the HNSW index grows via resize if exceeded).
        diversity_weight: zeta for diversity reranking.
        over_fetch: ANN recalls `over_fetch * k` candidates before policy rerank,
            trading a little query time for higher recall.
        ef_construction / M / ef_query: HNSW build/query knobs (recall vs speed).
        use_ann: force-disable the ANN path (fall back to brute force) for testing.
    """

    def __init__(
        self,
        dim: int,
        *,
        policy: RetrievalPolicy | None = None,
        max_items: int = 100_000,
        diversity_weight: float = 0.0,
        over_fetch: int = 8,
        ef_construction: int = 200,
        M: int = 16,
        ef_query: int = 128,
        use_ann: bool = True,
        seed: int = 100,
    ) -> None:
        self.dim = int(dim)
        self.policy: RetrievalPolicy = policy or CosineRetrievalPolicy()
        self.max_items = int(max_items)
        self.diversity_weight = float(diversity_weight)
        self.over_fetch = max(1, int(over_fetch))
        self._ef_construction = ef_construction
        self._M = M
        self._ef_query = ef_query

        self._items: dict[str, MemoryItem] = {}
        self.stats = {
            "writes": 0, "reads": 0, "retrievals": 0,
            "merges": 0, "consolidations": 0, "decayed": 0, "deleted": 0,
            "ann_queries": 0, "bruteforce_queries": 0,
        }

        self.ann_enabled = bool(use_ann) and _hnswlib_available()
        self._index = None
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._next_label = 0
        self._capacity = max(16, min(self.max_items, 1024))

        # Fallback store mirrors items for exact retrieval when ANN is off.
        self._fallback = DictMemoryStore(
            policy=self.policy, max_items=max_items, diversity_weight=diversity_weight
        )

        if self.ann_enabled:
            self._init_index(self._capacity, seed)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _init_index(self, capacity: int, seed: int = 100) -> None:
        import hnswlib

        idx = hnswlib.Index(space="cosine", dim=self.dim)
        idx.init_index(
            max_elements=capacity, ef_construction=self._ef_construction,
            M=self._M, random_seed=seed,
        )
        idx.set_ef(self._ef_query)
        self._index = idx
        self._capacity = capacity

    def _ensure_capacity(self, additional: int = 1) -> None:
        need = self._next_label + additional
        if need > self._capacity:
            new_cap = max(need, int(self._capacity * 2))
            self._index.resize_index(new_cap)
            self._capacity = new_cap

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def write(self, item: MemoryItem) -> str:
        if not isinstance(item, MemoryItem):
            raise TypeError("item must be a MemoryItem")
        if item.embedding.shape[-1] != self.dim:
            raise ValueError(
                f"embedding dim {item.embedding.shape[-1]} != store dim {self.dim}"
            )
        self._items[item.id] = item
        self.stats["writes"] += 1

        if self.ann_enabled:
            self._ensure_capacity(1)
            label = self._next_label
            self._next_label += 1
            self._label_to_id[label] = item.id
            self._id_to_label[item.id] = label
            vec = item.embedding.detach().cpu().float().numpy().reshape(1, -1)
            self._index.add_items(vec, [label])
        else:
            self._fallback.write(item)
        return item.id

    def read(self, item_id: str) -> MemoryItem | None:
        item = self._items.get(item_id)
        if item is not None:
            item.touch()
            self.stats["reads"] += 1
        return item

    def retrieve(
        self, query: torch.Tensor, k: int, filters: dict | None = None
    ) -> list[MemoryItem]:
        """Two-stage retrieval: ANN candidate recall, then policy rerank."""
        self.stats["retrievals"] += 1
        if not isinstance(query, torch.Tensor):
            raise TypeError("query must be a torch.Tensor")
        if not self._items:
            return []

        now = time.time()
        candidates = self._candidates(query, k, filters)
        if not candidates:
            return []

        scored = [(self.policy.score(query, it, now), it) for it in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        pool = scored[: max(k * 3, k)]
        result = rerank_with_diversity(pool, k, self.diversity_weight)
        for it in result:
            it.touch(now)
        return result

    def _candidates(
        self, query: torch.Tensor, k: int, filters: dict | None
    ) -> list[MemoryItem]:
        """Return a candidate set for reranking (ANN when available)."""
        if not self.ann_enabled:
            self.stats["bruteforce_queries"] += 1
            return self._apply_filters(list(self._items.values()), filters)

        self.stats["ann_queries"] += 1
        n = len(self._items)
        want = min(n, max(k * self.over_fetch, k))
        # HNSW recall depends on ef >= the number of candidates explored. A fixed
        # ef degrades recall as N grows, so scale ef with the fetch size (and keep
        # a floor). This keeps recall high at large N for a small latency cost.
        ef = max(self._ef_query, want * 2)
        self._index.set_ef(min(ef, max(n, 1)))
        q = query.detach().cpu().float().numpy().reshape(1, -1)
        try:
            labels, _ = self._index.knn_query(q, k=want)
        except RuntimeError:
            # ef still too small for requested k: widen further and retry once.
            self._index.set_ef(min(max(want * 4, 2 * ef), max(n, 1)))
            labels, _ = self._index.knn_query(q, k=min(want, n))
        cand: list[MemoryItem] = []
        for lbl in labels[0].tolist():
            item_id = self._label_to_id.get(int(lbl))
            if item_id is not None:
                it = self._items.get(item_id)
                if it is not None:
                    cand.append(it)
        return self._apply_filters(cand, filters)

    def update(self, item_id: str, **fields: Any) -> None:
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"no memory with id {item_id}")
        for key, value in fields.items():
            if not hasattr(item, key):
                raise AttributeError(f"MemoryItem has no field {key}")
            setattr(item, key, value)
        if not self.ann_enabled:
            try:
                self._fallback.update(item_id, **fields)
            except KeyError:
                pass

    def merge(self, ids: list[str]) -> str:
        items = [self._items[i] for i in ids if i in self._items]
        if not items:
            raise KeyError("no valid ids to merge")
        emb = torch.stack([it.embedding.float() for it in items]).mean(dim=0)
        merged = MemoryItem(
            content=[it.content for it in items],
            embedding=emb,
            source="merge",
            task=items[0].task,
            confidence=max(it.confidence for it in items),
            importance=max(it.importance for it in items),
            utility=max(it.utility for it in items),
            relationships=sorted({r for it in items for r in it.relationships}),
            compressed=True,
            provenance={"op": "merge", "from": [it.id for it in items]},
        )
        for i in ids:
            self.delete(i)
        self.write(merged)
        self.stats["merges"] += 1
        return merged.id

    def consolidate(self, ids: list[str]) -> str:
        merged_id = self.merge(ids)
        self._items[merged_id].importance = min(
            1.0, self._items[merged_id].importance + 0.2
        )
        self._items[merged_id].provenance["op"] = "consolidate"
        self.stats["consolidations"] += 1
        self.stats["merges"] -= 1
        return merged_id

    def decay(self, now: float, half_life: float = 3600.0) -> int:
        from bitmem.memory.base import recency_weight

        deleted = 0
        for item_id in list(self._items.keys()):
            item = self._items[item_id]
            item.recency = recency_weight(item.timestamp, now, half_life)
            if item.recency < 0.05 and item.utility < 0.2 and item.access_count < 2:
                self.delete(item_id)
                deleted += 1
        self.stats["decayed"] += deleted
        return deleted

    def delete(self, item_id: str) -> None:
        if self._items.pop(item_id, None) is not None:
            self.stats["deleted"] += 1
            if self.ann_enabled:
                label = self._id_to_label.pop(item_id, None)
                if label is not None:
                    self._label_to_id.pop(label, None)
                    try:
                        self._index.mark_deleted(label)
                    except RuntimeError:
                        pass  # already deleted
            else:
                self._fallback.delete(item_id)

    def __len__(self) -> int:
        return len(self._items)

    def all_items(self) -> list[MemoryItem]:
        return list(self._items.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_filters(
        self, items: list[MemoryItem], filters: dict | None
    ) -> list[MemoryItem]:
        if not filters:
            return items
        out = []
        for item in items:
            ok = True
            for key, value in filters.items():
                if key == "min_confidence":
                    ok = ok and item.confidence >= value
                elif key == "task":
                    ok = ok and item.task == value
                elif key == "source":
                    ok = ok and item.source == value
                elif key == "min_importance":
                    ok = ok and item.importance >= value
                else:
                    ok = ok and getattr(item, key, None) == value
            if ok:
                out.append(item)
        return out
