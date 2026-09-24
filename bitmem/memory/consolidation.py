"""Memory consolidation for BitMem (§7).

When many memories are highly related, consolidation:
  1. clusters them (by embedding similarity)
  2. summarizes each cluster into one consolidated memory
  3. preserves important details (importance/utility carried forward)
  4. replaces the redundant originals
  5. TRACKS INFORMATION LOSS (spec §7 requires measuring loss)

Consolidation is what lets a small memory bank stay small: episodic clusters get
distilled and promoted into semantic / long-term memory. This directly serves
the efficiency side of the hypothesis (§18) — memory must not grow unbounded.

The Stage-3 summarizer is embedding mean-pooling (a placeholder that is honest
about its limits). A learned summarizer (autoencoder / small transformer) is a
drop-in replacement behind the `Summarizer` protocol at a later stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from bitmem.memory.base import MemoryItem, cosine_similarity


# ---------------------------------------------------------------------------
# Summarizer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Summarizer(Protocol):
    """Produces a single consolidated embedding + content from a cluster."""

    def summarize(self, cluster: list[MemoryItem]) -> tuple[torch.Tensor, object]: ...


class MeanPoolSummarizer:
    """Placeholder summarizer: mean-pool embeddings, list contents.

    Honest about its limits: mean-pooling loses per-item structure. The
    information-loss metric quantifies exactly how much (see below).
    """

    def summarize(self, cluster: list[MemoryItem]) -> tuple[torch.Tensor, object]:
        emb = torch.stack([m.embedding.float() for m in cluster]).mean(dim=0)
        content = {"summary_of": [m.content for m in cluster], "n": len(cluster)}
        return emb, content


# ---------------------------------------------------------------------------
# Information-loss measurement (§7)
# ---------------------------------------------------------------------------


def reconstruction_loss(
    consolidated: torch.Tensor, cluster: list[MemoryItem]
) -> float:
    """Mean cosine DISTANCE between the consolidated embedding and its members.

    0.0 = the summary perfectly represents every member (no loss).
    Higher = the summary is far from members (more information lost).
    This is the number consolidation must keep low to be worthwhile.
    """
    if not cluster:
        return 0.0
    dists = [1.0 - cosine_similarity(consolidated, m.embedding) for m in cluster]
    return float(sum(dists) / len(dists))


def cluster_spread(cluster: list[MemoryItem]) -> float:
    """Mean pairwise cosine distance within a cluster (its internal diversity).

    A tight cluster (low spread) consolidates with low loss; a loose cluster
    (high spread) should probably NOT be merged. Used as a merge gate.
    """
    n = len(cluster)
    if n < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - cosine_similarity(cluster[i].embedding, cluster[j].embedding)
            pairs += 1
    return float(total / max(pairs, 1))


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def greedy_cluster(
    items: list[MemoryItem], similarity_threshold: float
) -> list[list[MemoryItem]]:
    """Greedy single-pass clustering by embedding similarity.

    Simple and deterministic (research prototype). Each item joins the first
    cluster whose centroid it is close enough to, else starts a new cluster.
    A real system would use HDBSCAN / k-means at scale; this is enough to
    validate the consolidation lifecycle.

    Args:
        items: memories to cluster.
        similarity_threshold: min cosine similarity to join a cluster.

    Returns:
        List of clusters (each a list of MemoryItems). Singletons included.
    """
    clusters: list[list[MemoryItem]] = []
    centroids: list[torch.Tensor] = []

    for item in items:
        placed = False
        for ci, centroid in enumerate(centroids):
            if cosine_similarity(item.embedding, centroid) >= similarity_threshold:
                clusters[ci].append(item)
                # update centroid (running mean)
                n = len(clusters[ci])
                centroids[ci] = centroid + (item.embedding.float() - centroid) / n
                placed = True
                break
        if not placed:
            clusters.append([item])
            centroids.append(item.embedding.float().clone())

    return clusters


# ---------------------------------------------------------------------------
# Consolidation engine
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationReport:
    """Outcome of one consolidation pass (§7, §12 consolidation quality)."""

    clusters_found: int
    clusters_consolidated: int
    memories_removed: int
    memories_created: int
    mean_information_loss: float
    max_information_loss: float
    per_cluster_loss: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"consolidated {self.clusters_consolidated}/{self.clusters_found} clusters | "
            f"{self.memories_removed} removed, {self.memories_created} created | "
            f"info-loss mean={self.mean_information_loss:.4f} "
            f"max={self.max_information_loss:.4f}"
        )


@dataclass
class ConsolidationConfig:
    """Knobs for a consolidation pass."""

    similarity_threshold: float = 0.85   # min sim to cluster together
    min_cluster_size: int = 3            # only consolidate clusters this big+
    max_spread: float = 0.4              # skip clusters looser than this (safety)
    max_loss: float = 0.5                # skip if summary loss exceeds this


class ConsolidationEngine:
    """Clusters + summarizes related memories, tracking information loss.

    Usage (Stage 4 controller calls this periodically):
        engine = ConsolidationEngine()
        report = engine.consolidate(memory_system.episodic,
                                    into=memory_system.long_term)
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.config = config or ConsolidationConfig()
        self.summarizer = summarizer or MeanPoolSummarizer()

    def consolidate(self, source, into=None) -> ConsolidationReport:
        """Consolidate related memories in `source`, promoting to `into`.

        Args:
            source: a TypedMemory (e.g. episodic) to consolidate.
            into: optional target TypedMemory (e.g. long_term). If None,
                  consolidated items are written back into `source`.

        Returns:
            ConsolidationReport with information-loss telemetry.
        """
        cfg = self.config
        target = into if into is not None else source
        items = source.all_items()

        clusters = greedy_cluster(items, cfg.similarity_threshold)
        report = ConsolidationReport(
            clusters_found=len(clusters),
            clusters_consolidated=0,
            memories_removed=0,
            memories_created=0,
            mean_information_loss=0.0,
            max_information_loss=0.0,
        )

        losses: list[float] = []
        for cluster in clusters:
            if len(cluster) < cfg.min_cluster_size:
                continue
            # Safety gate: do not merge loose clusters.
            if cluster_spread(cluster) > cfg.max_spread:
                continue

            emb, content = self.summarizer.summarize(cluster)
            loss = reconstruction_loss(emb, cluster)
            if loss > cfg.max_loss:
                continue  # too lossy — leave originals intact

            consolidated = MemoryItem(
                content=content,
                embedding=emb,
                source="consolidation",
                task=cluster[0].task,
                confidence=min(m.confidence for m in cluster),  # conservative
                importance=min(1.0, max(m.importance for m in cluster) + 0.1),
                utility=max(m.utility for m in cluster),
                relationships=sorted({r for m in cluster for r in m.relationships}),
                compressed=True,
                provenance={
                    "op": "consolidate",
                    "from": [m.id for m in cluster],
                    "information_loss": loss,
                    "cluster_size": len(cluster),
                },
            )

            # Remove originals from source, write consolidated into target.
            for m in cluster:
                source.store.delete(m.id)
                report.memories_removed += 1
            target.write(consolidated)
            report.memories_created += 1
            report.clusters_consolidated += 1
            losses.append(loss)

        if losses:
            report.per_cluster_loss = losses
            report.mean_information_loss = float(sum(losses) / len(losses))
            report.max_information_loss = float(max(losses))
        return report
