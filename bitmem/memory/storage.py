"""In-RAM memory store for BitMem Stage 0 (§3, §15 storage.py).

DictMemoryStore is the reference implementation of the MemoryStore protocol.
It is a research prototype (NOT production): brute-force retrieval, no ANN index,
no persistence. It exists to validate the memory operations and the retrieval
pipeline before we swap in a FAISS/HNSW backend at Stage 3+.

Known limitations (stated up front, per spec §16):
  - retrieve() is O(N) brute force — fine for thousands of items, not millions
  - no disk persistence
  - merge/consolidate use mean-pooling of embeddings as a placeholder; a real
    consolidation model (summarizer) arrives in Stage 3 (consolidation.py)
"""

from __future__ import annotations

import time
from typing import Any

import torch

from bitmem.memory.base import (
    MemoryItem,
    RetrievalPolicy,
    recency_weight,
)
from bitmem.memory.retrieval import CosineRetrievalPolicy, rerank_with_diversity


class DictMemoryStore:
    """Dictionary-backed memory store with brute-force retrieval.

    Args:
        policy: retrieval scoring policy (default cosine similarity).
        max_items: soft cap; when exceeded, decay() and eviction free space.
        diversity_weight: zeta for diversity reranking during retrieve().
    """

    def __init__(
        self,
        policy: RetrievalPolicy | None = None,
        *,
        max_items: int = 10_000,
        diversity_weight: float = 0.0,
    ) -> None:
        self._items: dict[str, MemoryItem] = {}
        self.policy: RetrievalPolicy = policy or CosineRetrievalPolicy()
        self.max_items = int(max_items)
        self.diversity_weight = float(diversity_weight)
        # Telemetry for evaluation (§12 memory metrics)
        self.stats = {
            "writes": 0, "reads": 0, "retrievals": 0,
            "merges": 0, "consolidations": 0, "decayed": 0, "deleted": 0,
        }

    # --- core operations ---

    def write(self, item: MemoryItem) -> str:
        if not isinstance(item, MemoryItem):
            raise TypeError("item must be a MemoryItem")
        self._items[item.id] = item
        self.stats["writes"] += 1
        if len(self._items) > self.max_items:
            self._evict_lowest_utility()
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
        """Brute-force scored retrieval with optional metadata filtering.

        Pipeline (§6): query -> filter -> score -> rerank(diversity) -> return.
        """
        self.stats["retrievals"] += 1
        if not isinstance(query, torch.Tensor):
            raise TypeError("query must be a torch.Tensor")
        now = time.time()

        candidates = self._apply_filters(list(self._items.values()), filters)
        if not candidates:
            return []

        scored = [(self.policy.score(query, item, now), item) for item in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Diversity-aware reranking (§6). Take a wider pool then rerank to k.
        pool = scored[: max(k * 3, k)]
        result = rerank_with_diversity(pool, k, self.diversity_weight)

        # Touch retrieved items (updates recency + access_count)
        for item in result:
            item.touch(now)
        return result

    def update(self, item_id: str, **fields: Any) -> None:
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"no memory with id {item_id}")
        for key, value in fields.items():
            if not hasattr(item, key):
                raise AttributeError(f"MemoryItem has no field {key}")
            setattr(item, key, value)

    def merge(self, ids: list[str]) -> str:
        """Merge several memories into one (mean-pooled embedding placeholder).

        Content becomes a list of the merged contents; a real summarizer replaces
        this in Stage 3. Confidence/importance/utility take the max (optimistic),
        relationships are unioned, provenance records the merge.
        """
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
            self._items.pop(i, None)
        self._items[merged.id] = merged
        self.stats["merges"] += 1
        return merged.id

    def consolidate(self, ids: list[str]) -> str:
        """Consolidate a cluster into a long-term memory (Stage-0 = merge alias).

        A dedicated consolidation model with information-loss tracking arrives in
        Stage 3. For now, consolidation == merge + importance boost.
        """
        merged_id = self.merge(ids)
        self._items[merged_id].importance = min(
            1.0, self._items[merged_id].importance + 0.2
        )
        self._items[merged_id].provenance["op"] = "consolidate"
        self.stats["consolidations"] += 1
        self.stats["merges"] -= 1  # don't double-count
        return merged_id

    def decay(self, now: float, half_life: float = 3600.0) -> int:
        """Apply recency decay to all items; delete those below utility floor.

        Returns the number of items deleted this pass.
        """
        deleted = 0
        for item_id in list(self._items.keys()):
            item = self._items[item_id]
            item.recency = recency_weight(item.timestamp, now, half_life)
            # An item that is old, low-utility, and rarely accessed is forgotten.
            if item.recency < 0.05 and item.utility < 0.2 and item.access_count < 2:
                del self._items[item_id]
                deleted += 1
        self.stats["decayed"] += deleted
        return deleted

    def delete(self, item_id: str) -> None:
        if self._items.pop(item_id, None) is not None:
            self.stats["deleted"] += 1

    def __len__(self) -> int:
        return len(self._items)

    # --- helpers ---

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
                    # unknown filter key: attribute equality
                    ok = ok and getattr(item, key, None) == value
            if ok:
                out.append(item)
        return out

    def _evict_lowest_utility(self) -> None:
        """Evict the single lowest-utility item to respect max_items."""
        if not self._items:
            return
        victim = min(self._items.values(), key=lambda it: (it.utility, it.recency))
        self.delete(victim.id)

    def all_items(self) -> list[MemoryItem]:
        """Return all stored items (for evaluation / debugging)."""
        return list(self._items.values())
