"""BitMem Stage 4 — Agentic memory with REAL retrieval (no oracle) (§8 Stage 4).

Stage 3 used oracle retrieval (the correct memory was always handed to the model)
to isolate "can the DiT use memory". Stage 4 removes that crutch: the agent must

  1. WRITE its own experiences into the store, keyed by a query embedding it
     controls, and
  2. RETRIEVE from the real store at test time, where retrieval can fail (return
     the wrong memory) if keys collide or the store is noisy.

Now retrieval QUALITY matters, and the agent metrics (§12) become meaningful:
useful-write rate, unnecessary-retrieval rate, recovery, memory growth.

Task (deterministic, learnable, memory-sensitive):
  There are K tasks. Each task k has a fixed "answer vector" a_k and a query key
  q_k. The agent's job across a stream of episodes is to associate q_k -> a_k in
  memory, so that when it next sees task k it can retrieve a_k. We measure
  reward as cosine similarity between the retrieved memory's stored answer and
  the true a_k — i.e. did real retrieval surface the RIGHT past experience?

This is an associative-recall probe of the memory+agent system as a whole,
independent of the DiT (the DiT's ability to *use* correct memory was already
shown in Stage 3). It directly stress-tests retrieval, writing, and the
value-aware utility loop.

Falsifiable: if real retrieval cannot beat a no-memory / random-memory baseline
here, the store + policies are not doing their job.

Usage:
    python scripts/bitmem_stage4_agent.py
    python scripts/bitmem_stage4_agent.py --episodes 400 --num-tasks 8 --gate heuristic
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitmem.memory.base import MemoryItem, cosine_similarity
from bitmem.memory.typed import MemorySystem
from bitmem.agent.controller import AgentController, ControllerConfig
from bitmem.agent.policies import (
    AlwaysRetrieve,
    HeuristicRetrievalGate,
    HeuristicWritePolicy,
)


# ---------------------------------------------------------------------------
# Associative-recall environment (no oracle)
# ---------------------------------------------------------------------------


class RecallEnv:
    """K tasks, each with a query key q_k and a hidden answer a_k.

    Each episode samples a task, gives the agent a noisy version of q_k, and the
    agent must recall a_k from memories it wrote on earlier episodes.
    """

    def __init__(self, *, num_tasks: int = 6, dim: int = 32, key_noise: float = 0.05,
                 seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.num_tasks = num_tasks
        self.dim = dim
        self.key_noise = key_noise
        self.keys = torch.randn(num_tasks, dim, generator=g)
        self.keys = self.keys / self.keys.norm(dim=1, keepdim=True)
        self.answers = torch.randn(num_tasks, dim, generator=g)
        self.answers = self.answers / self.answers.norm(dim=1, keepdim=True)
        self._g = g

    def sample(self):
        k = int(torch.randint(0, self.num_tasks, (1,), generator=self._g))
        noise = torch.randn(self.dim, generator=self._g) * self.key_noise
        query = self.keys[k] + noise
        query = query / query.norm()
        return k, query, self.answers[k]


# ---------------------------------------------------------------------------
# Run one agent configuration over an episode stream
# ---------------------------------------------------------------------------


def run_agent(
    env: RecallEnv,
    *,
    gate_name: str,
    episodes: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    memory = MemorySystem()

    if gate_name == "always":
        gate = AlwaysRetrieve()
    elif gate_name == "heuristic":
        gate = HeuristicRetrievalGate(min_attempts=2, benefit_threshold=-0.5)
    else:
        raise ValueError(gate_name)

    ctrl = AgentController(
        model=object(),  # unused: we supply generate_fn
        memory=memory,
        retrieval_gate=gate,
        write_policy=HeuristicWritePolicy(min_abs_reward=0.05, redundancy_threshold=0.98),
        config=ControllerConfig(retrieval_k=1, utility_eta=0.2),
    )

    # Track recall accuracy over time (does retrieval surface the right answer?)
    # We separate a cold-start "warmup" phase (store has not yet seen every task,
    # so some queries have no correct memory to find) from the "steady-state"
    # phase (all tasks written at least once). Reporting only the overall number
    # would understate retrieval quality by blaming it for episodes where the
    # right answer simply had not been written yet — an honest split avoids that.
    recall_scores: list[float] = []
    correct_recalls = 0
    total_with_memory = 0
    steady_correct = 0
    steady_total = 0
    tasks_seen: set[int] = set()

    for ep in range(episodes):
        task_id, query, true_answer = env.sample()

        # The "generation" here is simply the recalled answer: if memory is
        # provided, use the stored answer of the top retrieved item; else zeros.
        def generate_fn(mems, _true=true_answer, _dim=env.dim):
            if mems and mems[0]:
                stored = mems[0][0].content  # content holds the answer vector
                if isinstance(stored, torch.Tensor):
                    return stored
            return torch.zeros(_dim)

        # target is the true answer; reward = improvement of recalled vs zeros.
        result = ctrl.step(
            task=f"task_{task_id}",
            query_embedding=query,
            generate_fn=generate_fn,
            target=true_answer,
            new_knowledge=true_answer,  # store the answer under this query key
        )

        # Is the store in steady state (has it written a memory for this task)?
        steady = task_id in tasks_seen
        tasks_seen.add(task_id)

        # Independent recall-accuracy measurement (not used for reward).
        recalled = memory.retrieve(query, k=1)
        if recalled:
            total_with_memory += 1
            stored = recalled[0].content
            if isinstance(stored, torch.Tensor):
                sim = cosine_similarity(stored, true_answer)
                recall_scores.append(sim)
                hit = sim > 0.9
                if hit:
                    correct_recalls += 1
                if steady:
                    steady_total += 1
                    if hit:
                        steady_correct += 1

    metrics = ctrl.agent_metrics()
    metrics["mean_recall_similarity"] = (
        sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    )
    metrics["recall_accuracy@0.9"] = (
        correct_recalls / max(total_with_memory, 1)
    )
    metrics["recall_accuracy@0.9_steady"] = (
        steady_correct / max(steady_total, 1)
    )
    metrics["gate"] = gate_name
    return metrics


def run_random_baseline(env: RecallEnv, *, episodes: int, seed: int) -> dict:
    """Control: memory filled with RANDOM answers (retrieval can't help)."""
    torch.manual_seed(seed + 1)
    memory = MemorySystem()
    # Seed with random noise memories.
    for _ in range(env.num_tasks * 4):
        memory.write(
            MemoryItem(content=torch.randn(env.dim), embedding=torch.randn(env.dim),
                       importance=0.5),
            memory_type="episodic",
        )
    correct, total = 0, 0
    scores = []
    for _ in range(episodes):
        task_id, query, true_answer = env.sample()
        recalled = memory.retrieve(query, k=1)
        if recalled and isinstance(recalled[0].content, torch.Tensor):
            total += 1
            sim = cosine_similarity(recalled[0].content, true_answer)
            scores.append(sim)
            if sim > 0.9:
                correct += 1
    acc = correct / max(total, 1)
    return {
        "gate": "random_memory",
        "mean_recall_similarity": sum(scores) / len(scores) if scores else 0.0,
        "recall_accuracy@0.9": acc,
        "recall_accuracy@0.9_steady": acc,  # random has no warmup to exclude
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="BitMem Stage 4 agentic real-retrieval eval")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--num-tasks", type=int, default=6)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--key-noise", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = RecallEnv(num_tasks=args.num_tasks, dim=args.dim,
                    key_noise=args.key_noise, seed=args.seed)

    print("=" * 68)
    print("BitMem Stage 4 — Agentic Memory with REAL Retrieval (no oracle)")
    print("=" * 68)
    print(f"Env: {args.num_tasks} tasks, dim={args.dim}, key_noise={args.key_noise}")
    print(f"Episodes: {args.episodes}")
    print("Task: agent must WRITE experiences and RETRIEVE the right one later.")
    print("=" * 68)

    results = []

    # Random-memory control (retrieval cannot surface the right answer).
    rnd = run_random_baseline(env, episodes=args.episodes, seed=args.seed)
    results.append(rnd)
    print(f"\n[random_memory] recall_acc@0.9={rnd['recall_accuracy@0.9']:.3f} "
          f"mean_sim={rnd['mean_recall_similarity']:.3f}")

    # Agent with always-retrieve and heuristic gate.
    for gate in ("always", "heuristic"):
        env2 = RecallEnv(num_tasks=args.num_tasks, dim=args.dim,
                         key_noise=args.key_noise, seed=args.seed)
        m = run_agent(env2, gate_name=gate, episodes=args.episodes, seed=args.seed)
        results.append(m)
        print(f"\n[{gate}] recall_acc@0.9={m['recall_accuracy@0.9']:.3f} "
              f"(steady={m['recall_accuracy@0.9_steady']:.3f}) "
              f"mean_sim={m['mean_recall_similarity']:.3f}")
        print(f"    retrieval_rate={m['retrieval_rate']:.3f} "
              f"unnecessary_retrieval_rate={m['unnecessary_retrieval_rate']:.3f}")
        print(f"    useful_write_rate={m['useful_write_rate']:.3f} "
              f"writes={m['writes']} memory_total={m['memory_total']} "
              f"recoveries={m['recoveries']}")

    # Verdict.
    print("\n" + "=" * 68)
    print("AGENT METRICS SUMMARY")
    print("=" * 68)
    print(f"  {'config':>14} | {'recall@0.9':>10} | {'steady':>7} | {'mean_sim':>9}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*7}-+-{'-'*9}")
    for r in results:
        print(f"  {r['gate']:>14} | {r['recall_accuracy@0.9']:>10.3f} | "
              f"{r['recall_accuracy@0.9_steady']:>7.3f} | "
              f"{r['mean_recall_similarity']:>9.3f}")

    agent_best = max(
        (r["recall_accuracy@0.9_steady"] for r in results if r["gate"] in ("always", "heuristic")),
        default=0.0,
    )
    rnd_acc = rnd["recall_accuracy@0.9_steady"]
    print("\n" + "=" * 68)
    if agent_best > rnd_acc + 0.1:
        print(f"VERDICT: real retrieval WORKS (steady-state agent recall "
              f"{agent_best:.3f} >> random {rnd_acc:.3f}).")
        print("The agent writes useful experiences and retrieves the right one")
        print("without an oracle. Memory + retrieval + write policy validated.")
    else:
        print(f"VERDICT: real retrieval did NOT beat random ({agent_best:.3f} vs "
              f"{rnd_acc:.3f}). Store/policies need investigation.")
    print("=" * 68)
    print("NOTE: associative-recall probe of the memory+agent system (no DiT).")
    print("Stage 3 already showed the DiT can USE correct memory; this shows the")
    print("agent can FIND it via real retrieval.")
    print("=" * 68)


if __name__ == "__main__":
    main()
