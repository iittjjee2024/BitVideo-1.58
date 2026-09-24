"""Core memory types and protocol interfaces for BitMem.

These are the contracts (§2 of the design doc). Every protocol has at least two
implementations so the ablation matrix is a config switch, not a code change.

A MemoryItem carries structured metadata (§3 of the spec) including provenance
and confidence — required for the failure-handling story (§14): hallucinated,
contradictory, stale, or poisoned memories must be traceable and down-weightable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Memory item
# ---------------------------------------------------------------------------


@dataclass
class MemoryItem:
    """A single stored experience with structured metadata.

    The embedding is the retrieval key. Everything else is metadata used by
    retrieval scoring, consolidation, decay, and failure handling.

    Fields map directly to spec §3.
    """

    content: Any
    embedding: torch.Tensor  # [D_mem] retrieval key (float32, unit-normalized recommended)

    # Identity + timing
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    # Source / task / context
    source: str = "unknown"
    task: str = "default"
    context: str = ""

    # Scores (all in [0, 1] by convention)
    confidence: float = 1.0
    importance: float = 0.5
    utility: float = 0.5
    recency: float = 1.0

    # Usage
    access_count: int = 0

    # Graph
    relationships: list[str] = field(default_factory=list)

    # Lifecycle
    compressed: bool = False

    # Failure handling / falsifiability (§14)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor")
        if self.embedding.ndim != 1:
            raise ValueError(f"embedding must be 1-D [D_mem]; got {tuple(self.embedding.shape)}")
        if not self.embedding.is_floating_point():
            raise TypeError("embedding must be floating point")
        for name in ("confidence", "importance", "utility", "recency"):
            v = getattr(self, name)
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]; got {v}")
        if self.access_count < 0:
            raise ValueError("access_count must be non-negative")

    def touch(self, now: float | None = None) -> None:
        """Record an access: increment count and refresh recency to full."""
        self.access_count += 1
        self.recency = 1.0
        if now is not None:
            self.timestamp = now


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Storage backend for memory items (§3 operations).

    The store does not decide *what* to keep — that is the write policy's job.
    It provides the mechanical operations; policies compose on top.
    """

    def write(self, item: MemoryItem) -> str: ...
    def read(self, item_id: str) -> MemoryItem | None: ...
    def retrieve(
        self, query: torch.Tensor, k: int, filters: dict | None = None
    ) -> list[MemoryItem]: ...
    def update(self, item_id: str, **fields: Any) -> None: ...
    def merge(self, ids: list[str]) -> str: ...
    def consolidate(self, ids: list[str]) -> str: ...
    def decay(self, now: float, half_life: float) -> int: ...
    def delete(self, item_id: str) -> None: ...
    def __len__(self) -> int: ...


@runtime_checkable
class RetrievalPolicy(Protocol):
    """Scores a candidate memory for a query (§6 configurable score)."""

    def score(self, query: torch.Tensor, item: MemoryItem, now: float) -> float: ...


@runtime_checkable
class MemoryInterface(Protocol):
    """Injects retrieved memories into DiT tokens (§5 Method A/B/C)."""

    def inject(
        self,
        tokens: torch.Tensor,          # [B, L, D]
        memories: list[MemoryItem],
        t_emb: torch.Tensor,           # [B, D]
    ) -> torch.Tensor: ...


@runtime_checkable
class WritePolicy(Protocol):
    """Decides whether an experience deserves storage (§3, §11)."""

    def should_write(self, item: MemoryItem, store: MemoryStore) -> bool: ...


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Cosine similarity between two 1-D tensors, returned as a Python float."""
    a = a.float().flatten()
    b = b.float().flatten()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b) / denom)


def recency_weight(timestamp: float, now: float, half_life: float) -> float:
    """Exponential recency decay r(m) = 2^(-(now - t) / half_life), in (0, 1]."""
    if half_life <= 0:
        return 1.0
    age = max(0.0, now - timestamp)
    return float(2.0 ** (-age / half_life))
