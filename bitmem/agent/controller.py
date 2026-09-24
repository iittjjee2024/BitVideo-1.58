"""Agent controller for BitMem (§4).

Runs the control loop that sits between the diffusion model and memory:

    Observe -> Interpret -> Retrieve -> Generate -> Evaluate -> Consolidate -> Update

The controller maintains a COMPACT state (a few scalars + small ring buffers),
never the full memory database, so decisions stay cheap (§4). It delegates the
three decisions to swappable policies (retrieval gate / write / consolidation)
and turns each outcome into an ExperienceRecord that feeds value-aware retention.

Failure guards built in (§14):
  - retrieval loop guard: cap retrievals per step
  - excessive-growth guard: consolidation triggered by episodic cap
  - poison/stale defense: negative-reward retrievals dent memory confidence
  - contradiction audit: provenance records which memories preceded each outcome
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from bitmem.memory.base import MemoryItem
from bitmem.memory.typed import MemorySystem
from bitmem.memory.consolidation import ConsolidationEngine
from bitmem.interface.unified import InjectionMethod, MemoryAugmentedDiT
from bitmem.agent.evaluator import (
    Evaluator,
    ExperienceRecord,
    UtilityEvaluator,
    update_memory_utilities,
)
from bitmem.agent.policies import (
    AlwaysRetrieve,
    ConsolidationPolicy,
    HeuristicWritePolicy,
    PeriodicConsolidation,
    RetrievalGate,
    WritePolicy,
)


# ---------------------------------------------------------------------------
# Compact controller state (§4)
# ---------------------------------------------------------------------------


@dataclass
class ControllerState:
    """Everything a policy is allowed to see — deliberately small.

    NOT the memory database. Just enough summary to make cheap decisions.
    """

    step: int = 0
    task: str = "default"
    query_embedding: torch.Tensor | None = None

    # Per-task running summaries (small dicts, one entry per task).
    task_attempts: dict[str, int] = field(default_factory=dict)
    task_benefit: dict[str, float] = field(default_factory=dict)  # EMA of reward

    # Recent write embeddings for redundancy checks (bounded ring).
    recent_write_embeddings: deque = field(default_factory=lambda: deque(maxlen=32))

    # Size summary for the consolidation policy.
    episodic_size: int = 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ControllerConfig:
    retrieval_k: int = 3
    utility_eta: float = 0.15         # value-aware retention rate
    benefit_ema: float = 0.2          # per-task benefit smoothing
    max_retrievals_per_step: int = 1  # retrieval-loop guard (§14)
    write_memory_type: str = "episodic"


# ---------------------------------------------------------------------------
# Agent controller
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Telemetry for one control-loop step (feeds §12 agent metrics)."""

    retrieved: bool
    num_retrieved: int
    wrote: bool
    consolidated: bool
    reward: float
    failure_modes: list[str] = field(default_factory=list)


class AgentController:
    """Coordinates memory + a memory-augmented DiT via the control loop.

    Args:
        model: a MemoryAugmentedDiT (backbone + injection adapter).
        memory: the typed MemorySystem.
        retrieval_gate: decides whether/what to retrieve.
        write_policy: decides whether to store an experience.
        consolidation_policy: decides when to consolidate.
        evaluator: scores outcomes into rewards.
        consolidation_engine: performs consolidation when triggered.
        config: loop hyperparameters + failure guards.
    """

    def __init__(
        self,
        model: MemoryAugmentedDiT,
        memory: MemorySystem,
        *,
        retrieval_gate: RetrievalGate | None = None,
        write_policy: WritePolicy | None = None,
        consolidation_policy: ConsolidationPolicy | None = None,
        evaluator: Evaluator | None = None,
        consolidation_engine: ConsolidationEngine | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.model = model
        self.memory = memory
        self.retrieval_gate = retrieval_gate or AlwaysRetrieve()
        self.write_policy = write_policy or HeuristicWritePolicy()
        self.consolidation_policy = consolidation_policy or PeriodicConsolidation()
        self.evaluator = evaluator or UtilityEvaluator()
        self.consolidation_engine = consolidation_engine or ConsolidationEngine()
        self.config = config or ControllerConfig()
        self.state = ControllerState()

        # Aggregate telemetry for evaluation (§12 agent metrics).
        self.telemetry = {
            "steps": 0, "retrievals": 0, "writes": 0, "consolidations": 0,
            "unnecessary_retrievals": 0,   # retrieved but reward <= 0
            "useful_writes": 0,            # wrote something that later helped
            "recoveries": 0,               # negative then positive on same task
        }

    # ------------------------------------------------------------------
    # The control loop
    # ------------------------------------------------------------------

    def step(
        self,
        *,
        task: str,
        query_embedding: torch.Tensor,
        generate_fn: Callable[[list[list[MemoryItem]] | None], torch.Tensor],
        target: torch.Tensor,
        new_knowledge: Any = None,
        inject_failure: str | None = None,
    ) -> StepResult:
        """Run one Observe->...->Update cycle.

        Args:
            task: task id (for per-task summaries).
            query_embedding: [D_mem] retrieval key for this observation.
            generate_fn: called with retrieved memories (or None) -> prediction.
                         This lets the caller own the DiT forward (batching, noise)
                         while the controller owns the memory decisions.
            target: ground-truth for reward computation.
            new_knowledge: what to store if the write policy accepts.
            inject_failure: for testing failure handling — a label to attach.

        Returns:
            StepResult telemetry.
        """
        cfg = self.config
        st = self.state
        st.step += 1
        st.task = task
        st.query_embedding = query_embedding
        st.episodic_size = len(self.memory.episodic)
        st.task_attempts[task] = st.task_attempts.get(task, 0) + 1

        failure_modes: list[str] = []
        if inject_failure:
            failure_modes.append(inject_failure)

        # --- Observe / Interpret / Retrieve ---
        retrieved: list[MemoryItem] = []
        did_retrieve = False
        if self.retrieval_gate.should_retrieve(st):
            did_retrieve = True
            n_calls = 0
            q = self.retrieval_gate.build_query(st)
            # retrieval-loop guard (§14)
            while n_calls < cfg.max_retrievals_per_step:
                retrieved = self.memory.retrieve(q, cfg.retrieval_k)
                n_calls += 1
                break
            self.telemetry["retrievals"] += 1

        memories_arg = [retrieved] if did_retrieve else None

        # --- Generate (memory-augmented) ---
        prediction = generate_fn(memories_arg)

        # --- Generate baseline (memory-free) for reward attribution ---
        baseline = generate_fn(None) if did_retrieve else None

        # --- Evaluate ---
        reward = self.evaluator.evaluate(
            prediction=prediction, target=target, baseline_prediction=baseline
        )

        # unnecessary retrieval = retrieved but it didn't help
        if did_retrieve and reward <= 0:
            self.telemetry["unnecessary_retrievals"] += 1

        # recovery detection: this task previously had negative benefit, now positive
        prev_benefit = st.task_benefit.get(task, 0.0)
        if prev_benefit < 0 and reward > 0:
            self.telemetry["recoveries"] += 1

        # --- Update: value-aware retention on retrieved memories (§11) ---
        if retrieved:
            update_memory_utilities(retrieved, reward, eta=cfg.utility_eta)

        # update per-task benefit EMA (compact state)
        st.task_benefit[task] = (
            (1 - cfg.benefit_ema) * prev_benefit + cfg.benefit_ema * reward
        )

        # --- Consolidate (Update) ---
        record = ExperienceRecord(
            task=task,
            context="",
            retrieved_ids=[m.id for m in retrieved],
            actions={"retrieved": did_retrieve},
            generation=None,
            reward=reward,
            failure_modes=failure_modes,
            new_knowledge=new_knowledge,
            query_embedding=query_embedding,
        )

        wrote = False
        if self.write_policy.should_write(record, st):
            item = record.to_memory_item()
            if item is not None:
                self.memory.write(item, memory_type=cfg.write_memory_type)
                st.recent_write_embeddings.append(query_embedding.detach().cpu())
                wrote = True
                self.telemetry["writes"] += 1
                if reward > 0:
                    self.telemetry["useful_writes"] += 1

        consolidated = False
        st.episodic_size = len(self.memory.episodic)
        if self.consolidation_policy.should_consolidate(st):
            report = self.consolidation_engine.consolidate(
                self.memory.episodic, into=self.memory.long_term
            )
            consolidated = report.clusters_consolidated > 0
            if consolidated:
                self.telemetry["consolidations"] += 1

        self.telemetry["steps"] += 1
        return StepResult(
            retrieved=did_retrieve,
            num_retrieved=len(retrieved),
            wrote=wrote,
            consolidated=consolidated,
            reward=reward,
            failure_modes=failure_modes,
        )

    # ------------------------------------------------------------------
    # Metrics (§12 agent)
    # ------------------------------------------------------------------

    def agent_metrics(self) -> dict[str, float]:
        """Derived agent metrics: rates, not just counts."""
        t = self.telemetry
        steps = max(t["steps"], 1)
        retr = max(t["retrievals"], 1)
        writes = max(t["writes"], 1)
        return {
            "steps": t["steps"],
            "retrieval_rate": t["retrievals"] / steps,
            "unnecessary_retrieval_rate": t["unnecessary_retrievals"] / retr,
            "useful_write_rate": t["useful_writes"] / writes,
            "writes": t["writes"],
            "consolidations": t["consolidations"],
            "recoveries": t["recoveries"],
            "memory_total": self.memory.total(),
        }
