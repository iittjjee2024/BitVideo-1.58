"""BitMem agent subsystem (§4, §11).

The agent controller sits between the diffusion model and the memory system.
It decides whether to retrieve, what to query, whether a generation produced a
useful experience, and whether to write it back — the control loop:

    Observe -> Interpret -> Retrieve -> Generate -> Evaluate -> Consolidate -> Update

Policies (retrieval / write / consolidation) are modular so heuristic, learned,
and RL variants can be compared (§11).
"""

from bitmem.agent.evaluator import (
    ExperienceRecord,
    Evaluator,
    UtilityEvaluator,
)
from bitmem.agent.policies import (
    AlwaysRetrieve,
    ConsolidationPolicy,
    HeuristicRetrievalGate,
    HeuristicWritePolicy,
    LearnedRetrievalGate,
    PeriodicConsolidation,
    RetrievalGate,
    WritePolicy,
)
from bitmem.agent.controller import AgentController, ControllerConfig, ControllerState

__all__ = [
    "AgentController",
    "AlwaysRetrieve",
    "ConsolidationPolicy",
    "ControllerConfig",
    "ControllerState",
    "Evaluator",
    "ExperienceRecord",
    "HeuristicRetrievalGate",
    "HeuristicWritePolicy",
    "LearnedRetrievalGate",
    "PeriodicConsolidation",
    "RetrievalGate",
    "UtilityEvaluator",
    "WritePolicy",
]
