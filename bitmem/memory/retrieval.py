"""Retrieval policies for BitMem (§6).

Two implementations behind the RetrievalPolicy protocol:
  - CosineRetrievalPolicy: pure semantic similarity (ablation baseline)
  - WeightedRetrievalPolicy: configurable weighted score combining
      semantic · alpha + recency · beta + importance · gamma
      + task_sim · delta + confidence · epsilon - redundancy · zeta

The weighted policy is a CONFIGURABLE score, not a hard-coded heuristic — this
is required by spec §6 so we can ablate each term independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from bitmem.memory.base import MemoryItem, cosine_similarity, recency_weight


@dataclass
class CosineRetrievalPolicy:
    """Semantic-only retrieval. The ablation baseline ('semantic retrieval only')."""

    def score(self, query: torch.Tensor, item: MemoryItem, now: float) -> float:
        return cosine_similarity(query, item.embedding)


@dataclass
class WeightedRetrievalPolicy:
    """Configurable multi-factor retrieval score (§6).

    Each weight can be set to 0 to ablate that term, directly supporting the
    ablation matrix rows:
        semantic only            -> beta=gamma=delta=epsilon=zeta=0
        semantic + recency       -> gamma=delta=epsilon=zeta=0
        semantic + importance    -> beta=delta=epsilon=zeta=0
    """

    alpha: float = 1.0      # semantic similarity
    beta: float = 0.3       # recency
    gamma: float = 0.2      # importance
    delta: float = 0.0      # task similarity (needs a task embedding; 0 by default)
    epsilon: float = 0.1    # confidence
    zeta: float = 0.0       # redundancy penalty (needs diversity context; applied in reranker)
    half_life: float = 3600.0  # recency half-life in seconds
    task_embedding: torch.Tensor | None = None  # optional [D_mem] for task_sim

    def score(self, query: torch.Tensor, item: MemoryItem, now: float) -> float:
        semantic = cosine_similarity(query, item.embedding)
        recency = recency_weight(item.timestamp, now, self.half_life)
        importance = item.importance
        confidence = item.confidence

        task_sim = 0.0
        if self.delta != 0.0 and self.task_embedding is not None:
            task_sim = cosine_similarity(self.task_embedding, item.embedding)

        return (
            self.alpha * semantic
            + self.beta * recency
            + self.gamma * importance
            + self.delta * task_sim
            + self.epsilon * confidence
            # zeta (redundancy) is applied during diversity reranking, not here
        )


def rerank_with_diversity(
    ranked: list[tuple[float, MemoryItem]],
    top_m: int,
    diversity_weight: float = 0.0,
) -> list[MemoryItem]:
    """Maximal-marginal-relevance style reranking for diversity (§6).

    Greedily selects items that are high-scoring but dissimilar to already-picked
    ones. With diversity_weight=0 this is a plain top-m by score.

    Args:
        ranked: list of (score, item) sorted descending by score.
        top_m: number of items to return.
        diversity_weight: zeta; penalizes similarity to already-selected items.

    Returns:
        Up to top_m items, reranked for diversity.
    """
    if diversity_weight <= 0.0 or top_m >= len(ranked):
        return [item for _, item in ranked[:top_m]]

    selected: list[MemoryItem] = []
    remaining = list(ranked)

    while remaining and len(selected) < top_m:
        best_idx = 0
        best_adjusted = float("-inf")
        for i, (score, item) in enumerate(remaining):
            # redundancy = max similarity to any already-selected item
            redundancy = 0.0
            for sel in selected:
                redundancy = max(redundancy, cosine_similarity(item.embedding, sel.embedding))
            adjusted = score - diversity_weight * redundancy
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i
        selected.append(remaining.pop(best_idx)[1])

    return selected
