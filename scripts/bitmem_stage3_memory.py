"""BitMem Stage 3 — Memory-benefit evaluation (§8 Stage 3, §18 hypothesis probe).

Question: does giving a ternary DiT access to retrieved memory reduce its
denoising error versus the SAME ternary DiT with no memory?

To make this answerable, the task is designed so memory CAN help. Each sample
belongs to one of K "tasks", and its clean latent is:

    clean = shared_lowrank(text)  +  task_prototype[task_id]

The task_prototype component is NOT predictable from the text alone — but it IS
stored in memory (keyed by a per-task embedding). A model that retrieves the
right prototype can subtract it out; a memory-free model cannot, so it carries
irreducible error on that component.

This is a DIAGNOSTIC designed to FALSIFY as readily as confirm (§18): if memory
injection does NOT reduce MSE here, the injection interface is broken or
uninformative. If it does, we have a controlled demonstration that retrieved
memory is usable by the ternary DiT — the prerequisite for the full hypothesis.

Honest scope: synthetic, tiny model, oracle retrieval (correct prototype always
retrievable). It shows the MECHANISM works, not that it helps on real video.

Usage:
    python scripts/bitmem_stage3_memory.py
    python scripts/bitmem_stage3_memory.py --methods none adaptive cross_attention memory_tokens
    python scripts/bitmem_stage3_memory.py --steps 400 --num-tasks 8
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitvideo.models import VideoDiT
from bitvideo.quantization.quantization import QuantizationConfig

from bitmem.memory.base import MemoryItem
from bitmem.interface.unified import (
    InjectionMethod,
    InjectorConfig,
    MemoryAugmentedDiT,
)


# ---------------------------------------------------------------------------
# Memory-benefit synthetic task
# ---------------------------------------------------------------------------


class MemoryBenefitTask:
    """A task where the answer has a memory-retrievable component.

    clean[i] = shared_map(text[i]) + prototype[task_id[i]]

    - shared_map: fixed low-rank map (learnable from text, no memory needed)
    - prototype[k]: a fixed per-task latent, retrievable from memory by task key
    """

    def __init__(
        self,
        *,
        num_tasks: int = 4,
        in_channels: int = 8,
        frames: int = 3,
        height: int = 16,
        width: int = 16,
        context_dim: int = 64,
        memory_dim: int = 64,
        seed: int = 0,
    ) -> None:
        self.num_tasks = num_tasks
        self.latent_shape = (in_channels, frames, height, width)
        self.context_dim = context_dim
        self.memory_dim = memory_dim
        g = torch.Generator().manual_seed(seed)

        latent_numel = in_channels * frames * height * width
        # Shared low-rank text -> latent map (the "easy" component).
        a = torch.randn(context_dim, 4, generator=g)
        b = torch.randn(4, latent_numel, generator=g) / 2.0
        self.shared = (a @ b) / math.sqrt(4)

        # Per-task prototype latents (the "memory-only" component).
        self.prototypes = torch.randn(num_tasks, *self.latent_shape, generator=g)
        # normalize prototype scale
        self.prototypes = self.prototypes / self.prototypes.flatten(1).std(
            dim=1
        ).view(num_tasks, 1, 1, 1, 1)

        # Per-task memory keys (what retrieval matches on).
        self.task_keys = torch.randn(num_tasks, memory_dim, generator=g)
        self.task_keys = self.task_keys / self.task_keys.norm(dim=1, keepdim=True)

        self._gen = g

    def sample_batch(self, batch_size: int, *, device, dtype):
        """Return (video_latent, text_embedding, task_ids, memory_query)."""
        g = self._gen
        task_ids = torch.randint(0, self.num_tasks, (batch_size,), generator=g)
        context = torch.randn(batch_size, 8, self.context_dim, generator=g)

        pooled = context.mean(dim=1)                        # [B, ctx]
        shared_flat = pooled @ self.shared                  # [B, numel]
        shared = shared_flat.view(batch_size, *self.latent_shape)
        shared = shared / (shared.flatten(1).std(dim=1).view(batch_size, 1, 1, 1, 1) + 1e-6)

        proto = self.prototypes[task_ids]                   # [B, C,T,H,W]
        clean = shared + proto                              # answer has both parts

        queries = self.task_keys[task_ids]                  # [B, mem]
        return (
            clean.to(device=device, dtype=dtype),
            context.to(device=device, dtype=dtype),
            task_ids,
            queries.to(device=device, dtype=dtype),
        )

    def build_memories(self, task_ids, queries):
        """Oracle retrieval: return the correct prototype memory per sample.

        The memory embedding IS the task key; the content is unused by the
        current injection methods (they consume the embedding). This isolates
        'can the model USE a correct retrieved memory' from 'can retrieval find
        it' — retrieval quality is stress-tested separately in Stage 4.
        """
        mems = []
        for b in range(len(task_ids)):
            k = int(task_ids[b])
            item = MemoryItem(
                content=f"prototype_{k}",
                embedding=queries[b].detach().cpu().float(),
                task=f"task_{k}",
                importance=0.9, confidence=1.0, utility=1.0,
            )
            mems.append([item])
        return mems


# ---------------------------------------------------------------------------
# Train one configuration
# ---------------------------------------------------------------------------


def train_config(
    method: InjectionMethod,
    task: MemoryBenefitTask,
    *,
    dim: int,
    depth: int,
    heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    dit = VideoDiT(
        in_channels=task.latent_shape[0], dim=dim, depth=depth, num_heads=heads,
        context_dim=task.context_dim, patch_size=(1, 2, 2),
        quantization=QuantizationConfig(),  # ternary
        device=device,
    )
    model = MemoryAugmentedDiT(
        dit,
        InjectorConfig(method=method, memory_dim=task.memory_dim, max_memory_tokens=4),
        quantization=QuantizationConfig(),
        device=device,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    num_train_timesteps = 1000

    torch.manual_seed(seed + 1)
    model.train()
    first_loss = None
    for step in range(steps):
        clean, context, task_ids, queries = task.sample_batch(
            batch_size, device=device, dtype=torch.float32
        )
        mems = None
        if method is not InjectionMethod.NONE:
            mems = task.build_memories(task_ids, queries)

        b = clean.shape[0]
        t = torch.randint(0, num_train_timesteps, (b,), device=device)
        noise = torch.randn_like(clean)
        alpha = torch.cos(t.float() / num_train_timesteps * math.pi / 2) ** 2
        alpha = alpha.view(b, 1, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

        pred = model(noisy, t.float(), context, mems)
        loss = F.mse_loss(pred, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first_loss is None:
            first_loss = float(loss.item())

    # Deterministic eval.
    final = evaluate(model, task, method, device=device, seed=seed + 999)
    return {
        "method": method.value,
        "first_loss": first_loss,
        "final_loss": final,
        "params": sum(p.numel() for p in model.parameters()),
    }


@torch.no_grad()
def evaluate(model, task, method, *, device, seed, batches=8, batch_size=16) -> float:
    model.eval()
    num_train_timesteps = 1000
    total, count = 0.0, 0
    for i in range(batches):
        torch.manual_seed(seed + i)
        clean, context, task_ids, queries = task.sample_batch(
            batch_size, device=device, dtype=torch.float32
        )
        mems = None
        if method is not InjectionMethod.NONE:
            mems = task.build_memories(task_ids, queries)
        b = clean.shape[0]
        t = torch.randint(0, num_train_timesteps, (b,), device=device)
        noise = torch.randn_like(clean)
        alpha = torch.cos(t.float() / num_train_timesteps * math.pi / 2) ** 2
        alpha = alpha.view(b, 1, 1, 1, 1)
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        pred = model(noisy, t.float(), context, mems)
        total += float(F.mse_loss(pred, noise).item())
        count += 1
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="BitMem Stage 3 memory-benefit eval")
    p.add_argument("--methods", nargs="+",
                   default=["none", "adaptive", "cross_attention", "memory_tokens"])
    p.add_argument("--num-tasks", type=int, default=4)
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    task = MemoryBenefitTask(num_tasks=args.num_tasks, seed=args.seed)

    print("=" * 68)
    print("BitMem Stage 3 — Memory-Benefit Evaluation")
    print("=" * 68)
    print(f"Task: {args.num_tasks} tasks, clean = shared(text) + prototype[task]")
    print(f"      prototype is memory-retrievable but NOT text-predictable")
    print(f"Model: ternary DiT dim={args.dim} depth={args.depth} (all modes identical)")
    print(f"Train: {args.steps} steps, batch {args.batch_size}, device {device}")
    print("=" * 68)

    results = {}
    for m in args.methods:
        method = InjectionMethod(m)
        print(f"\n── Training [{method.value}] ──")
        r = train_config(
            method, task,
            dim=args.dim, depth=args.depth, heads=args.heads,
            steps=args.steps, batch_size=args.batch_size, lr=args.lr,
            device=device, seed=args.seed,
        )
        results[method.value] = r
        print(f"   first={r['first_loss']:.5f} -> final={r['final_loss']:.5f}")

    # ---- Memory-benefit table ----
    print("\n" + "=" * 68)
    print("MEMORY BENEFIT (final MSE; lower = better)")
    print("=" * 68)
    baseline = results.get("none", {}).get("final_loss")
    print(f"  {'method':>16} | {'final MSE':>10} | {'vs no-memory':>14}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*14}")
    for method, r in results.items():
        if baseline and method != "none":
            delta = r["final_loss"] - baseline
            rel = delta / baseline
            note = f"{rel:+.1%}"
        else:
            note = "(baseline)"
        print(f"  {method:>16} | {r['final_loss']:>10.5f} | {note:>14}")

    # ---- Verdict ----
    print("\n" + "=" * 68)
    if baseline:
        best_mem = min(
            (r["final_loss"] for m, r in results.items() if m != "none"),
            default=None,
        )
        if best_mem is not None:
            if best_mem < baseline:
                print(f"VERDICT: memory HELPS on this diagnostic "
                      f"(best {best_mem:.5f} < no-memory {baseline:.5f}).")
                print("The ternary DiT can use retrieved memory. Hypothesis")
                print("prerequisite satisfied — proceed to agentic control (Stage 4).")
            else:
                print(f"VERDICT: memory did NOT help (best {best_mem:.5f} >= "
                      f"baseline {baseline:.5f}).")
                print("Injection interface may be uninformative — investigate before Stage 4.")
    print("=" * 68)
    print("NOTE: synthetic diagnostic with oracle retrieval. Shows the injection")
    print("MECHANISM works, not that memory helps on real video.")
    print("=" * 68)


if __name__ == "__main__":
    main()
