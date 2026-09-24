"""BitMem — Ternary Diffusion Transformer + Agentic Memory.

A hybrid architecture pairing the W1.58/A8 ternary DiT (bitvideo.models.VideoDiT)
with a JEV-Mem-style agentic memory system.

Built incrementally per the design doc. Stage 0 = minimal validated prototype.

This package REUSES bitvideo primitives (BitLinear, VideoDiT, QuantizationConfig)
and does not fork them. Memory, retrieval, and the agent controller are greenfield.

Research question (NOT assumed true): can a small ternary DiT + selectively
retrieved, dynamically consolidated memory match a larger memory-free DiT at
better task-level efficiency? Every component is behind an interface so it can
be ablated to test — and potentially falsify — that hypothesis.
"""

from bitmem.memory.base import MemoryItem, MemoryStore, RetrievalPolicy
from bitmem.memory.storage import DictMemoryStore
from bitmem.memory.retrieval import CosineRetrievalPolicy, WeightedRetrievalPolicy

__all__ = [
    "MemoryItem",
    "MemoryStore",
    "RetrievalPolicy",
    "DictMemoryStore",
    "CosineRetrievalPolicy",
    "WeightedRetrievalPolicy",
]

__version__ = "0.1.0-stage0"
