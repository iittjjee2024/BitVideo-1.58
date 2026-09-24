"""Stage-4 tests: agent controller, policies, evaluator, failure handling.

Validates:
  - UtilityEvaluator reward sign (memory helps -> positive, hurts -> negative)
  - ExperienceRecord -> MemoryItem conversion + provenance
  - value-aware utility update (§11) + poison confidence dent (§14)
  - each policy (AlwaysRetrieve, Heuristic gate, Learned gate, Write, Consolidation)
  - the full control loop over a stub generator (no DiT needed for logic tests)
  - failure handling: retrieval-loop guard, unnecessary-retrieval accounting,
    recovery detection, redundant-write suppression

Uses a stub generate_fn so the control-loop LOGIC is tested fast and
deterministically, independent of the (separately tested) DiT.

Run:
    python -m pytest tests/bitmem/test_stage4.py -v
"""

from __future__ import annotations

import torch

from bitmem.memory.base import MemoryItem
from bitmem.memory.typed import MemorySystem
from bitmem.agent.evaluator import (
    ExperienceRecord,
    UtilityEvaluator,
    update_memory_utilities,
)
from bitmem.agent.policies import (
    AlwaysRetrieve,
    HeuristicRetrievalGate,
    HeuristicWritePolicy,
    LearnedRetrievalGate,
    PeriodicConsolidation,
)
from bitmem.agent.controller import (
    AgentController,
    ControllerConfig,
    ControllerState,
)


def _mem(dim=8, **kw):
    return MemoryItem(content=kw.pop("content", "x"),
                      embedding=kw.pop("embedding", torch.randn(dim)), **kw)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_evaluator_positive_when_memory_beats_baseline():
    ev = UtilityEvaluator()
    target = torch.zeros(4)
    memory_pred = torch.full((4,), 0.1)     # closer to target
    baseline_pred = torch.full((4,), 0.5)   # farther
    reward = ev.evaluate(prediction=memory_pred, target=target,
                         baseline_prediction=baseline_pred)
    assert reward > 0


def test_evaluator_negative_when_memory_worse():
    ev = UtilityEvaluator()
    target = torch.zeros(4)
    memory_pred = torch.full((4,), 0.9)     # farther
    baseline_pred = torch.full((4,), 0.2)   # closer
    reward = ev.evaluate(prediction=memory_pred, target=target,
                         baseline_prediction=baseline_pred)
    assert reward < 0


def test_experience_to_memory_item():
    rec = ExperienceRecord(
        task="t", context="c", reward=0.6,
        new_knowledge="learned", query_embedding=torch.randn(8),
    )
    item = rec.to_memory_item()
    assert item is not None
    assert item.task == "t"
    assert item.provenance["reward"] == 0.6
    assert 0.0 <= item.utility <= 1.0


def test_experience_no_knowledge_returns_none():
    rec = ExperienceRecord(task="t", context="c", reward=0.6,
                           query_embedding=torch.randn(8))  # no new_knowledge
    assert rec.to_memory_item() is None


# ---------------------------------------------------------------------------
# Utility update (§11) + poison defense (§14)
# ---------------------------------------------------------------------------


def test_utility_update_positive_reward_raises_utility():
    m = _mem(utility=0.5)
    update_memory_utilities([m], reward=1.0, eta=0.5)
    assert m.utility > 0.5


def test_utility_update_negative_reward_lowers_utility():
    m = _mem(utility=0.5)
    update_memory_utilities([m], reward=-1.0, eta=0.5)
    assert m.utility < 0.5


def test_negative_reward_dents_confidence():
    m = _mem(utility=0.5, confidence=1.0)
    update_memory_utilities([m], reward=-0.8, eta=0.5)
    assert m.confidence < 1.0  # poison defense


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_always_retrieve():
    gate = AlwaysRetrieve()
    st = ControllerState(query_embedding=torch.randn(8))
    assert gate.should_retrieve(st) is True


def test_heuristic_gate_explores_then_gates():
    gate = HeuristicRetrievalGate(min_attempts=2, benefit_threshold=0.0)
    st = ControllerState(task="t", query_embedding=torch.randn(8))
    # First attempts -> explore (retrieve).
    st.task_attempts["t"] = 1
    assert gate.should_retrieve(st) is True
    # After min_attempts with negative benefit -> don't retrieve.
    st.task_attempts["t"] = 5
    st.task_benefit["t"] = -0.2
    assert gate.should_retrieve(st) is False
    # Positive benefit -> retrieve.
    st.task_benefit["t"] = 0.3
    assert gate.should_retrieve(st) is True


def test_learned_gate_runs():
    gate = LearnedRetrievalGate(query_dim=8)
    st = ControllerState(task="t", query_embedding=torch.randn(8))
    st.task_attempts["t"] = 1
    decision = gate.should_retrieve(st)
    assert isinstance(decision, bool)
    logit = gate.predict_logit(st)
    assert logit.requires_grad  # differentiable for Stage-5 training


def test_write_policy_rejects_low_reward_when_novelty_off():
    # With novelty writes disabled the policy is reward-only: a low-reward,
    # unsurprising outcome carries no signal and must be dropped.
    pol = HeuristicWritePolicy(min_abs_reward=0.1, store_novel=False)
    st = ControllerState()
    rec = ExperienceRecord(task="t", context="", reward=0.01,
                           new_knowledge="x", query_embedding=torch.randn(8))
    assert pol.should_write(rec, st) is False


def test_write_policy_accepts_novel_low_reward():
    # Cold-start fix: a NON-redundant experience is worth storing even at
    # near-zero reward, because on an empty store retrieval cannot yet beat the
    # memory-free baseline (reward ~ 0). Without this, the store would stay empty
    # forever and the agent could never learn to retrieve.
    pol = HeuristicWritePolicy(min_abs_reward=0.1, store_novel=True)
    st = ControllerState()
    rec = ExperienceRecord(task="t", context="", reward=0.01,
                           new_knowledge="x", query_embedding=torch.randn(8))
    assert pol.should_write(rec, st) is True


def test_write_policy_accepts_surprising():
    pol = HeuristicWritePolicy(min_abs_reward=0.1)
    st = ControllerState()
    rec = ExperienceRecord(task="t", context="", reward=-0.9,  # surprising failure
                           new_knowledge="x", query_embedding=torch.randn(8))
    assert pol.should_write(rec, st) is True


def test_write_policy_rejects_redundant():
    pol = HeuristicWritePolicy(redundancy_threshold=0.9)
    st = ControllerState()
    emb = torch.tensor([1.0, 0.0, 0.0])
    st.recent_write_embeddings.append(emb.clone())
    rec = ExperienceRecord(task="t", context="", reward=0.5,
                           new_knowledge="x", query_embedding=emb.clone())
    assert pol.should_write(rec, st) is False  # too similar to a recent write


def test_consolidation_policy_triggers_on_cap():
    pol = PeriodicConsolidation(every_n_steps=1000, episodic_cap=10)
    st = ControllerState(episodic_size=10)
    assert pol.should_consolidate(st) is True


# ---------------------------------------------------------------------------
# Control loop (with a stub generator)
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal stand-in with a .dit.dim attribute the controller doesn't use."""
    pass


def _make_controller(gate=None):
    memory = MemorySystem()
    # model is unused by step() (generate_fn is supplied), pass a stub.
    ctrl = AgentController(
        model=_StubModel(),
        memory=memory,
        retrieval_gate=gate or AlwaysRetrieve(),
        config=ControllerConfig(retrieval_k=3),
    )
    return ctrl


def test_control_loop_single_step():
    ctrl = _make_controller()
    target = torch.zeros(4)

    def generate_fn(mems):
        # memory present -> better prediction; None -> worse
        return torch.full((4,), 0.1) if mems else torch.full((4,), 0.5)

    q = torch.randn(8)
    result = ctrl.step(
        task="t", query_embedding=q,
        generate_fn=generate_fn, target=target,
        new_knowledge="fact",
    )
    assert result.retrieved is True
    assert result.reward > 0  # memory helped
    assert ctrl.telemetry["retrievals"] == 1


def test_control_loop_unnecessary_retrieval_tracked():
    ctrl = _make_controller()
    target = torch.zeros(4)

    def generate_fn(mems):
        # memory HURTS: memory pred worse than baseline
        return torch.full((4,), 0.9) if mems else torch.full((4,), 0.2)

    result = ctrl.step(
        task="t", query_embedding=torch.randn(8),
        generate_fn=generate_fn, target=target, new_knowledge="fact",
    )
    assert result.reward < 0
    assert ctrl.telemetry["unnecessary_retrievals"] == 1


def test_control_loop_writes_useful_experience():
    ctrl = _make_controller()
    target = torch.zeros(4)

    def generate_fn(mems):
        return torch.full((4,), 0.1) if mems else torch.full((4,), 0.5)

    ctrl.step(task="t", query_embedding=torch.randn(8),
              generate_fn=generate_fn, target=target, new_knowledge="fact")
    assert ctrl.telemetry["writes"] >= 1
    assert ctrl.memory.total() >= 1


def test_control_loop_recovery_detection():
    ctrl = _make_controller()
    target = torch.zeros(4)
    q = torch.randn(8)

    # Step 1: memory hurts (negative benefit recorded for task).
    def bad(mems):
        return torch.full((4,), 0.9) if mems else torch.full((4,), 0.2)
    ctrl.step(task="t", query_embedding=q, generate_fn=bad, target=target,
              new_knowledge=None)

    # Step 2: memory now helps -> recovery.
    def good(mems):
        return torch.full((4,), 0.1) if mems else torch.full((4,), 0.5)
    ctrl.step(task="t", query_embedding=q, generate_fn=good, target=target,
              new_knowledge=None)

    assert ctrl.telemetry["recoveries"] >= 1


def test_agent_metrics_shape():
    ctrl = _make_controller()
    target = torch.zeros(4)

    def generate_fn(mems):
        return torch.full((4,), 0.1) if mems else torch.full((4,), 0.5)

    for _ in range(3):
        ctrl.step(task="t", query_embedding=torch.randn(8),
                  generate_fn=generate_fn, target=target, new_knowledge="fact")

    m = ctrl.agent_metrics()
    assert m["steps"] == 3
    assert 0.0 <= m["retrieval_rate"] <= 1.0
    assert 0.0 <= m["unnecessary_retrieval_rate"] <= 1.0
    assert 0.0 <= m["useful_write_rate"] <= 1.0
