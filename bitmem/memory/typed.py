"""Typed memory stores for BitMem (§3 JEV-Mem taxonomy).

Each memory type wraps a DictMemoryStore but applies type-appropriate write and
decay policies. The types differ in *lifecycle*, not mechanism:

  EpisodicMemory   — specific experiences; short-to-medium half-life; decays fast
  SemanticMemory   — distilled facts/concepts; long half-life; high importance floor
  ProceduralMemory — how-to policies; long-lived; keyed by task
  LongTermMemory   — consolidated, high-utility; effectively permanent

A MemorySystem ties them together and routes reads/writes. Consolidation
(consolidation.py) promotes episodic clusters into semantic/long-term memory.

Design note: typed stores compose the existing DictMemoryStore rather than
subclassing it, so the storage backend can still be swapped (e.g. FAISS) without
touching the type policies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from bitmem.memory.base import MemoryItem, RetrievalPolicy
from bitmem.memory.retrieval import CosineRetrievalPolicy, WeightedRetrievalPolicy
from bitmem.memory.storage import DictMemoryStore


# ---------------------------------------------------------------------------
# Base typed store
# ---------------------------------------------------------------------------


@dataclass
class TypedMemoryConfig:
    """Lifecycle knobs for one memory type."""

    half_life: float = 3600.0        # recency decay half-life (seconds)
    importance_floor: float = 0.0    # writes below this importance are rejected
    max_items: int = 5000
    diversity_weight: float = 0.0


class TypedMemory:
    """A memory type = a DictMemoryStore + a write/decay policy.

    Subclasses set `memory_type` and override `_accept` / lifecycle defaults.
    """

    memory_type: str = "generic"

    def __init__(
        self,
        config: TypedMemoryConfig | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self.config = config or TypedMemoryConfig()
        self.store = DictMemoryStore(
            policy=policy or CosineRetrievalPolicy(),
            max_items=self.config.max_items,
            diversity_weight=self.config.diversity_weight,
        )

    def _accept(self, item: MemoryItem) -> bool:
        """Write policy: decide whether this item belongs in this store."""
        return item.importance >= self.config.importance_floor

    def write(self, item: MemoryItem) -> str | None:
        """Write if the policy accepts; tag the item with its memory type."""
        if not self._accept(item):
            return None
        item.provenance.setdefault("memory_type", self.memory_type)
        return self.store.write(item)

    def retrieve(
        self, query: torch.Tensor, k: int, filters: dict | None = None
    ) -> list[MemoryItem]:
        return self.store.retrieve(query, k, filters)

    def decay(self, now: float | None = None) -> int:
        return self.store.decay(now or time.time(), self.config.half_life)

    def __len__(self) -> int:
        return len(self.store)

    def all_items(self) -> list[MemoryItem]:
        return self.store.all_items()


class EpisodicMemory(TypedMemory):
    """Specific past experiences. Fast decay, low write bar (record freely).

    Episodic memory captures raw experiences; consolidation later distills the
    useful ones into semantic/long-term memory.
    """

    memory_type = "episodic"

    def __init__(
        self,
        config: TypedMemoryConfig | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        # Episodic: fast recency decay, no importance bar, recency-weighted retrieval.
        super().__init__(
            config or TypedMemoryConfig(half_life=1800.0, importance_floor=0.0),
            policy or WeightedRetrievalPolicy(alpha=1.0, beta=0.5, gamma=0.1),
        )


class SemanticMemory(TypedMemory):
    """Distilled facts/concepts. Long-lived, high importance bar.

    Only reasonably important items are admitted; retrieval weights importance
    and confidence more than recency (facts do not go stale as fast).
    """

    memory_type = "semantic"

    def __init__(
        self,
        config: TypedMemoryConfig | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        super().__init__(
            config or TypedMemoryConfig(half_life=86400.0, importance_floor=0.3),
            policy or WeightedRetrievalPolicy(
                alpha=1.0, beta=0.05, gamma=0.4, epsilon=0.2
            ),
        )


class ProceduralMemory(TypedMemory):
    """How-to policies keyed by task. Long-lived; retrieval is task-conditioned."""

    memory_type = "procedural"

    def __init__(
        self,
        config: TypedMemoryConfig | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        super().__init__(
            config or TypedMemoryConfig(half_life=86400.0, importance_floor=0.2),
            policy or WeightedRetrievalPolicy(alpha=0.8, beta=0.05, gamma=0.3, delta=0.5),
        )

    def retrieve(
        self, query: torch.Tensor, k: int, filters: dict | None = None
    ) -> list[MemoryItem]:
        # Procedural retrieval is usually scoped to a task; callers pass
        # filters={'task': ...} to get task-specific procedures.
        return super().retrieve(query, k, filters)


class LongTermMemory(TypedMemory):
    """Consolidated, high-utility memories. Effectively permanent.

    Populated by consolidation, not by direct experience writes. Very slow decay
    and a high importance floor keep it small and valuable.
    """

    memory_type = "long_term"

    def __init__(
        self,
        config: TypedMemoryConfig | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        super().__init__(
            config or TypedMemoryConfig(
                half_life=604800.0, importance_floor=0.5, max_items=2000
            ),
            policy or WeightedRetrievalPolicy(alpha=1.0, beta=0.0, gamma=0.5, epsilon=0.3),
        )


# ---------------------------------------------------------------------------
# Memory system (routes across types)
# ---------------------------------------------------------------------------


class MemorySystem:
    """Orchestrates the typed memories and provides a unified retrieval view.

    Reads can query one type or all types (merged + reranked). Writes route by
    the item's declared memory_type (default episodic). This is the object the
    agent controller (Stage 4) talks to.
    """

    def __init__(
        self,
        *,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        procedural: ProceduralMemory | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.procedural = procedural or ProceduralMemory()
        self.long_term = long_term or LongTermMemory()
        self._by_type = {
            "episodic": self.episodic,
            "semantic": self.semantic,
            "procedural": self.procedural,
            "long_term": self.long_term,
        }

    def write(self, item: MemoryItem, memory_type: str = "episodic") -> str | None:
        store = self._by_type.get(memory_type)
        if store is None:
            raise ValueError(f"unknown memory_type {memory_type}")
        return store.write(item)

    def retrieve(
        self,
        query: torch.Tensor,
        k: int,
        *,
        types: list[str] | None = None,
        filters: dict | None = None,
    ) -> list[MemoryItem]:
        """Retrieve across one or more memory types, merged and re-sorted.

        Args:
            query: retrieval key [D_mem].
            k: total items to return across all queried types.
            types: which memory types to search (default: all).
            filters: metadata filters passed to each store.
        """
        types = types or list(self._by_type.keys())
        now = time.time()
        pooled: list[tuple[float, MemoryItem]] = []
        for t in types:
            store = self._by_type[t]
            # Over-fetch from each type, then merge-rerank.
            for item in store.retrieve(query, k, filters):
                score = store.store.policy.score(query, item, now)
                pooled.append((score, item))
        pooled.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in pooled[:k]]

    def decay_all(self, now: float | None = None) -> dict[str, int]:
        now = now or time.time()
        return {t: store.decay(now) for t, store in self._by_type.items()}

    def counts(self) -> dict[str, int]:
        return {t: len(store) for t, store in self._by_type.items()}

    def total(self) -> int:
        return sum(len(s) for s in self._by_type.values())
