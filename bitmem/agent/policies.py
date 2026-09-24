"""Agent policies for BitMem (§4, §11).

Three decisions the controller delegates, each behind a Protocol so heuristic,
learned, and RL variants are swappable (§11 requires comparing approaches):

  RetrievalGate     — should we retrieve at all, and with what query?
  WritePolicy       — is this experience worth storing?
  ConsolidationPolicy — is it time to consolidate?

Design principle (§4): the controller keeps a COMPACT state; policies read that
state rather than the whole memory database, so we never inject the entire store
into any decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from bitmem.agent.evaluator import ExperienceRecord
from bitmem.memory.base import MemoryItem


# ---------------------------------------------------------------------------
# Retrieval gate
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalGate(Protocol):
    """Decides whether to retrieve and builds the query."""

    def should_retrieve(self, state: "object") -> bool: ...
    def build_query(self, state: "object") -> torch.Tensor: ...


@dataclass
class AlwaysRetrieve:
    """Ablation baseline: always retrieve. Query = the state's context embedding.

    Useful for measuring the 'unnecessary retrieval rate' — how often retrieval
    fires when it does not help. A good learned gate should beat this on
    efficiency without losing task success.
    """

    def should_retrieve(self, state) -> bool:
        return True

    def build_query(self, state) -> torch.Tensor:
        return state.query_embedding


@dataclass
class HeuristicRetrievalGate:
    """Retrieve only when the task looks like it needs memory.

    Heuristic signal: retrieve when the running memory-benefit estimate for this
    task is positive (memory has helped before) OR we haven't tried this task
    enough times to know. This avoids retrieving for tasks memory can't help.

    Args:
        min_attempts: always retrieve for the first N attempts at a task (explore).
        benefit_threshold: after that, retrieve only if observed benefit > this.
    """

    min_attempts: int = 3
    benefit_threshold: float = 0.0

    def should_retrieve(self, state) -> bool:
        attempts = state.task_attempts.get(state.task, 0)
        if attempts < self.min_attempts:
            return True  # explore
        benefit = state.task_benefit.get(state.task, 0.0)
        return benefit > self.benefit_threshold

    def build_query(self, state) -> torch.Tensor:
        return state.query_embedding


class LearnedRetrievalGate(nn.Module):
    """A small MLP that predicts P(retrieve helps) from a compact state vector.

    Trained (Stage 5) on (state -> reward) pairs collected by the controller.
    At Stage 4 it can run untrained (as a stand-in) or be trained online with the
    supplied `train_step`. Kept intentionally tiny — the gate must be far cheaper
    than the retrieval it decides on, or it defeats the purpose.

    The gate consumes the query embedding plus a few scalar state features, NOT
    the memory database (compact-state principle, §4).
    """

    def __init__(self, query_dim: int, hidden: int = 32) -> None:
        super().__init__()
        # +2 scalar features: task_attempts (norm), task_benefit.
        self.net = nn.Sequential(
            nn.Linear(query_dim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.threshold = 0.5

    def _features(self, state) -> torch.Tensor:
        q = state.query_embedding.float().flatten()
        attempts = float(state.task_attempts.get(state.task, 0))
        benefit = float(state.task_benefit.get(state.task, 0.0))
        extra = torch.tensor([min(attempts / 10.0, 1.0), benefit], dtype=q.dtype)
        return torch.cat([q, extra])

    @torch.no_grad()
    def should_retrieve(self, state) -> bool:
        p = torch.sigmoid(self.net(self._features(state))).item()
        return p >= self.threshold

    def build_query(self, state) -> torch.Tensor:
        return state.query_embedding

    def predict_logit(self, state) -> torch.Tensor:
        """Differentiable logit for online training (Stage 5)."""
        return self.net(self._features(state)).squeeze(-1)


# ---------------------------------------------------------------------------
# Write policy
# ---------------------------------------------------------------------------


@runtime_checkable
class WritePolicy(Protocol):
    """Decides whether an experience is worth storing (§3: don't append everything)."""

    def should_write(self, record: ExperienceRecord, state: "object") -> bool: ...


@dataclass
class HeuristicWritePolicy:
    """Store an experience only if it is informative and not redundant.

    Rules (§3):
      - skip if there's nothing to store
      - skip if a very similar memory already exists (redundancy)
      - otherwise store when the experience is either SURPRISING (|reward| high,
        good or bad — failures are as informative as successes, §11) or NOVEL
        (a non-redundant observation the store has not seen before)

    The novelty clause resolves the cold-start deadlock in associative-recall
    tasks: on a fresh, empty store no retrieval can beat the memory-free
    baseline, so reward is ~0 and a reward-only gate would never write anything,
    which in turn keeps the store empty forever. A first encounter of a query is
    exactly the kind of new knowledge worth recording. Redundancy still prevents
    re-writing what we already hold, so the store does not grow without bound.
    """

    min_abs_reward: float = 0.1
    redundancy_threshold: float = 0.95
    store_novel: bool = True  # write non-redundant experiences even at low reward

    def should_write(self, record: ExperienceRecord, state) -> bool:
        if record.new_knowledge is None or record.query_embedding is None:
            return False
        # Redundancy check against recently seen embeddings (compact state).
        from bitmem.memory.base import cosine_similarity
        for prev in state.recent_write_embeddings:
            if cosine_similarity(record.query_embedding, prev) >= self.redundancy_threshold:
                return False  # already stored something essentially identical
        # Surprising outcome: always worth storing.
        if abs(record.reward) >= self.min_abs_reward:
            return True
        # Novel but low-reward: store it if novelty writes are enabled. This is
        # what lets the agent bootstrap a store it can later retrieve from.
        return self.store_novel


# ---------------------------------------------------------------------------
# Consolidation policy
# ---------------------------------------------------------------------------


@runtime_checkable
class ConsolidationPolicy(Protocol):
    """Decides when to run consolidation."""

    def should_consolidate(self, state: "object") -> bool: ...


@dataclass
class PeriodicConsolidation:
    """Consolidate every N steps, or when episodic memory exceeds a size cap.

    Keeps the store from growing unbounded (§7, §14 excessive-growth guard).
    """

    every_n_steps: int = 50
    episodic_cap: int = 200

    def should_consolidate(self, state) -> bool:
        if state.episodic_size >= self.episodic_cap:
            return True
        return state.step > 0 and state.step % self.every_n_steps == 0
