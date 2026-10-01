"""TurboQuant-backed memory store — a swappable compressed backend (§3, §15).

`TurboQuantMemoryStore` is a drop-in `MemoryStore` that compresses every stored
retrieval key with TurboQuant (the data-oblivious vector quantizer, see
`turboquant.py`). It demonstrates the paper's own use case — compressing the
vectors in a vector database / nearest-neighbor index — inside BitMem.

Design (no fork): it composes the existing `DictMemoryStore` for all the
bookkeeping (ids, filters, decay, merge, diversity reranking) and only changes
ONE thing: on write, the item's fp32 embedding is replaced by its TurboQuant
reconstruction, and the compact integer code is retained for honest storage
accounting. Retrieval, scoring, and every policy keep working unchanged because
they still see a `MemoryItem` with a (now quantized) `embedding`.

What this buys (measured by the benchmark, not asserted here):
  * storage: ~`bits` per coordinate instead of 32, i.e. up to ~10x smaller keys
    at 3-4 bits with high retrieval recall.
  * retrieval quality: near-fp32 recall because the rotation + per-coordinate
    scalar quantizer preserves inner-product geometry.

Honesty (§16): this is a research prototype. It keeps the reconstructed fp32
embedding in RAM for scoring (so in-RAM footprint is not reduced here) — the
compression win is in the PERSISTED/transmitted code size (`code_bytes()`), which
is what a real vector DB stores on disk / ships over the wire. A production
backend would score directly against codes; we keep reconstruction for clarity.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from bitmem.memory.base import MemoryItem, RetrievalPolicy
from bitmem.memory.storage import DictMemoryStore
from bitmem.memory.turboquant import (
    TurboCode,
    TurboQuantConfig,
    TurboQuantMSE,
)


class TurboQuantMemoryStore:
    """A compressed `MemoryStore` backed by TurboQuant key quantization.

    Args:
        dim: dimensionality of the retrieval keys (D_mem).
        bits: bits per coordinate for the TurboQuant MSE quantizer.
        policy: retrieval scoring policy (default: inherited DictMemoryStore one).
        max_items / diversity_weight: forwarded to the inner DictMemoryStore.
        seed: rotation/quantizer seed (data-oblivious, fixed per store).
    """

    def __init__(
        self,
        dim: int,
        *,
        bits: int = 4,
        policy: RetrievalPolicy | None = None,
        max_items: int = 10_000,
        diversity_weight: float = 0.0,
        seed: int = 0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dim = int(dim)
        self.bits = int(bits)
        self.codec = TurboQuantMSE(
            dim, TurboQuantConfig(bits=bits, seed=seed), device=device, dtype=dtype
        )
        self._inner = DictMemoryStore(
            policy=policy, max_items=max_items, diversity_weight=diversity_weight
        )
        # Retain compact codes for honest storage accounting (id -> TurboCode).
        self._codes: dict[str, TurboCode] = {}

    # --- core operations ---

    def write(self, item: MemoryItem) -> str:
        if not isinstance(item, MemoryItem):
            raise TypeError("item must be a MemoryItem")
        if item.embedding.shape[-1] != self.dim:
            raise ValueError(
                f"embedding dim {item.embedding.shape[-1]} != store dim {self.dim}"
            )
        code = self.codec.encode(item.embedding)
        # Replace the stored key with its TurboQuant reconstruction so retrieval
        # scores against the compressed representation (what a real index does).
        item.embedding = self.codec.decode(code).to(item.embedding.dtype).flatten()
        item.provenance.setdefault("turboquant_bits", self.bits)
        item_id = self._inner.write(item)
        self._codes[item_id] = code
        # Keep the code dict in sync if the inner store evicted something.
        self._prune_codes()
        return item_id

    def read(self, item_id: str) -> MemoryItem | None:
        return self._inner.read(item_id)

    def retrieve(
        self, query: torch.Tensor, k: int, filters: dict | None = None
    ) -> list[MemoryItem]:
        return self._inner.retrieve(query, k, filters)

    def update(self, item_id: str, **fields: Any) -> None:
        self._inner.update(item_id, **fields)

    def merge(self, ids: list[str]) -> str:
        new_id = self._inner.merge(ids)
        merged = self._inner.read(new_id)
        if merged is not None:
            # Re-quantize the merged key so it, too, lives on the compressed path.
            code = self.codec.encode(merged.embedding)
            merged.embedding = self.codec.decode(code).to(merged.embedding.dtype).flatten()
            self._codes[new_id] = code
        self._prune_codes()
        return new_id

    def consolidate(self, ids: list[str]) -> str:
        new_id = self._inner.consolidate(ids)
        self._prune_codes()
        return new_id

    def decay(self, now: float, half_life: float = 3600.0) -> int:
        n = self._inner.decay(now, half_life)
        self._prune_codes()
        return n

    def delete(self, item_id: str) -> None:
        self._inner.delete(item_id)
        self._codes.pop(item_id, None)

    def __len__(self) -> int:
        return len(self._inner)

    def all_items(self) -> list[MemoryItem]:
        return self._inner.all_items()

    @property
    def stats(self) -> dict:
        return self._inner.stats

    @property
    def policy(self):
        return self._inner.policy

    # --- storage accounting (honest compression numbers) ---

    def _prune_codes(self) -> None:
        live = {it.id for it in self._inner.all_items()}
        for stale in [i for i in self._codes if i not in live]:
            self._codes.pop(stale, None)

    def code_bytes(self) -> float:
        """Total PERSISTED bytes for all compressed keys (codes + scales).

        This is the real compression win: what a vector DB stores on disk / ships
        over the wire, versus fp32 keys. In-RAM reconstruction is separate.
        """
        total = 0.0
        for code in self._codes.values():
            total += code.codes.numel() * self.bits / 8.0  # packed code bits
            total += code.scale.numel() * 4.0              # fp32 scale
        return total

    def fp32_bytes(self) -> float:
        """Bytes the same keys would take as raw fp32 vectors."""
        return len(self._codes) * self.dim * 4.0

    def compression_ratio(self) -> float:
        cb = self.code_bytes()
        return (self.fp32_bytes() / cb) if cb > 0 else 0.0
