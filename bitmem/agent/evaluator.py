"""Experience records and outcome evaluation for BitMem (§11).

After each task the agent creates an ExperienceRecord capturing everything
needed to (a) learn better behavior and (b) write a useful memory:

    Task, Context, Retrieved memories, Actions, Generation, Outcome,
    Reward/utility, Failure modes, New knowledge

The Evaluator turns a generation outcome into a scalar reward/utility. Reward is
what drives:
  - the memory utility update (§11): u_m <- (1-eta) u_m + eta * 1[retrieved] * reward
  - the write decision (only useful experiences are worth storing)
  - the (optional) learned retrieval gate's training signal

The default UtilityEvaluator scores by improvement over a memory-free baseline:
"did using memory make the generation better than not using it?" — which is the
quantity the whole hypothesis rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from bitmem.memory.base import MemoryItem


# ---------------------------------------------------------------------------
# Experience record (§11)
# ---------------------------------------------------------------------------


@dataclass
class ExperienceRecord:
    """A structured record of one task attempt (spec §11 fields)."""

    task: str
    context: str
    retrieved_ids: list[str] = field(default_factory=list)
    actions: dict[str, Any] = field(default_factory=dict)   # e.g. {'retrieved': True, 'method': 'adaptive'}
    generation: Any = None                                  # the produced sample / summary
    outcome: str = ""                                       # human/auto description
    reward: float = 0.0                                     # scalar utility in [-1, 1]
    failure_modes: list[str] = field(default_factory=list)  # e.g. ['stale_memory']
    new_knowledge: Any = None                               # what to store, if anything

    # The embedding under which this experience would be stored/retrieved.
    query_embedding: torch.Tensor | None = None

    def to_memory_item(self, *, source: str = "experience") -> MemoryItem | None:
        """Convert to a storable MemoryItem, or None if there's nothing to store.

        Utility is seeded from reward (clamped to [0,1]); importance from the
        magnitude of the reward (surprising outcomes, good or bad, are important).
        Provenance records the originating task + failure modes for later
        contradiction/poison auditing (§14).
        """
        if self.query_embedding is None or self.new_knowledge is None:
            return None
        reward01 = max(0.0, min(1.0, (self.reward + 1.0) / 2.0))
        return MemoryItem(
            content=self.new_knowledge,
            embedding=self.query_embedding.detach().cpu().float(),
            source=source,
            task=self.task,
            context=self.context,
            confidence=1.0 if not self.failure_modes else 0.5,
            importance=max(0.0, min(1.0, abs(self.reward))),
            utility=reward01,
            provenance={
                "op": "experience_write",
                "reward": self.reward,
                "failure_modes": list(self.failure_modes),
                "retrieved_ids": list(self.retrieved_ids),
            },
        )


# ---------------------------------------------------------------------------
# Evaluator protocol + default
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Scores a generation outcome into a reward in [-1, 1]."""

    def evaluate(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        baseline_prediction: torch.Tensor | None = None,
    ) -> float: ...


@dataclass
class UtilityEvaluator:
    """Reward = relative improvement in denoising error vs a memory-free baseline.

    reward = (baseline_err - memory_err) / baseline_err, clamped to [-1, 1].

    Positive => memory helped (lower error). Negative => memory hurt (a signal
    that the retrieval was contaminated / stale / off-task). If no baseline is
    provided, falls back to a bounded transform of the raw error so the reward
    is still finite and comparable.
    """

    error_scale: float = 1.0  # for the no-baseline fallback

    def evaluate(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        baseline_prediction: torch.Tensor | None = None,
    ) -> float:
        mem_err = float(torch.nn.functional.mse_loss(prediction, target).item())
        if baseline_prediction is not None:
            base_err = float(
                torch.nn.functional.mse_loss(baseline_prediction, target).item()
            )
            if base_err <= 1e-8:
                return 0.0
            reward = (base_err - mem_err) / base_err
            return float(max(-1.0, min(1.0, reward)))
        # No baseline: map error to a bounded reward (lower error -> higher reward).
        return float(max(-1.0, min(1.0, 1.0 - mem_err / self.error_scale)))


def update_memory_utilities(
    memories: list[MemoryItem],
    reward: float,
    *,
    eta: float = 0.1,
    now: float | None = None,
) -> None:
    """Apply the value-aware retention update (§11) to retrieved memories.

    u_m <- (1 - eta) * u_m + eta * reward01, where reward01 maps [-1,1]->[0,1].
    Memories that consistently precede good outcomes accumulate utility and
    survive decay; those that precede bad outcomes lose utility and are forgotten.
    """
    reward01 = max(0.0, min(1.0, (reward + 1.0) / 2.0))
    for m in memories:
        m.utility = (1.0 - eta) * m.utility + eta * reward01
        # A retrieval that led to a bad outcome also dents confidence (poison
        # defense: contaminated memories lose trust over time, §14).
        if reward < 0:
            m.confidence = max(0.0, m.confidence * (1.0 + reward * eta))
