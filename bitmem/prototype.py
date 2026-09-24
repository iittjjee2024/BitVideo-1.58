"""Stage-0 minimal working prototype (§10 build order, step 1).

Wires together:
  - a tiny ternary VideoDiT (random init)
  - a DictMemoryStore with cosine retrieval
  - Method-B memory-token injection

and runs a single end-to-end forward pass, proving the plumbing and tensor
shapes are correct. This validates INTEGRATION, not generation quality.

Known limitations (stated up front, spec §16):
  - The DiT is randomly initialized; output is noise, not meaningful.
  - The memory store is in-RAM with brute-force retrieval.
  - Tasks are synthetic random tensors.
  - No training happens here. This is a smoke test.

Run:
    python -m bitmem.prototype
"""

from __future__ import annotations

import time

import torch

from bitvideo.models import VideoDiT
from bitvideo.quantization.quantization import QuantizationConfig

from bitmem.memory.base import MemoryItem
from bitmem.memory.retrieval import CosineRetrievalPolicy, WeightedRetrievalPolicy
from bitmem.memory.storage import DictMemoryStore
from bitmem.interface.mem_tokens import MemoryTokenInterface


def build_tiny_dit(
    *,
    memory_as_context: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> VideoDiT:
    """Build a small ternary VideoDiT for smoke testing.

    Uses the ternary QuantizationConfig defaults (W1.58 + A8).
    """
    return VideoDiT(
        in_channels=8,
        dim=128,
        depth=2,
        num_heads=4,
        context_dim=128,   # memory tokens will live in model space (128)
        patch_size=(1, 2, 2),
        quantization=QuantizationConfig(),  # ternary defaults
        device=device,
        dtype=dtype,
    )


def seed_memory(store: DictMemoryStore, memory_dim: int, n: int = 20) -> None:
    """Populate the store with n synthetic memories."""
    torch.manual_seed(0)
    for i in range(n):
        emb = torch.randn(memory_dim)
        emb = emb / emb.norm()
        store.write(
            MemoryItem(
                content=f"synthetic memory {i}",
                embedding=emb,
                task="smoke",
                importance=float(torch.rand(1)),
                confidence=float(0.5 + 0.5 * torch.rand(1)),
                source="prototype",
                provenance={"index": i},
            )
        )


def run_prototype(device: str = "cpu", verbose: bool = True) -> dict:
    """Run the Stage-0 end-to-end forward pass and return diagnostics."""
    torch.manual_seed(42)
    dev = torch.device(device)
    dtype = torch.float32

    model_dim = 128
    memory_dim = 64
    batch = 2

    # --- 1. Build components ---
    dit = build_tiny_dit(device=dev, dtype=dtype).eval()
    store = DictMemoryStore(policy=CosineRetrievalPolicy())
    mem_interface = MemoryTokenInterface(
        memory_dim=memory_dim,
        model_dim=model_dim,
        max_tokens=4,
        quantization=QuantizationConfig(),
        device=dev,
        dtype=dtype,
    ).eval()

    # --- 2. Seed memory ---
    seed_memory(store, memory_dim, n=20)

    # --- 3. Build a synthetic diffusion batch ---
    video = torch.randn(batch, 8, 3, 16, 16, device=dev, dtype=dtype)  # [B,C,T,H,W]
    timesteps = torch.randint(0, 1000, (batch,), device=dev).float()

    # --- 4. Controller decides + retrieves (Stage 0: always retrieve) ---
    queries = [torch.randn(memory_dim, device=dev) for _ in range(batch)]
    queries = [q / q.norm() for q in queries]
    memories_per_sample = [store.retrieve(q, k=3) for q in queries]

    # --- 5. Build memory tokens, project to model space ---
    mem_tokens, mem_mask = mem_interface.build_tokens(
        memories_per_sample, device=dev, dtype=dtype
    )

    # --- 6. Run DiT with memory tokens as cross-attention context ---
    # Method B here injects memory as the cross-attention *context* (context_dim
    # == model_dim), the cleanest way to feed it into the existing VideoDiT
    # without modifying the backbone. (Method B-as-prepend and Methods A/C are
    # exercised in the interface unit tests and arrive fully at Stage 3.)
    # NOTE (Stage 0): we pass context_mask=None. Empty memory slots carry a
    # learned pad embedding rather than garbage, so attending to them is safe.
    # A proper key-padding mask (broadcast to the batch x frames attention
    # layout used by the spatial cross-attention) is wired in at Stage 3 when
    # Method A gets its own dedicated memory cross-attention path.
    t0 = time.time()
    with torch.no_grad():
        output = dit(video, timesteps, context=mem_tokens, context_mask=None)
    elapsed = time.time() - t0

    diagnostics = {
        "dit_params": dit.parameter_count(),
        "memory_items": len(store),
        "retrieved_per_sample": [len(m) for m in memories_per_sample],
        "mem_tokens_shape": tuple(mem_tokens.shape),
        "output_shape": tuple(output.shape),
        "output_finite": bool(torch.isfinite(output).all()),
        "input_shape": tuple(video.shape),
        "forward_seconds": elapsed,
        "store_stats": dict(store.stats),
    }

    if verbose:
        print("=" * 60)
        print("BitMem Stage-0 Prototype — Smoke Test")
        print("=" * 60)
        print(f"DiT parameters:        {diagnostics['dit_params']:,}")
        print(f"Memory items:          {diagnostics['memory_items']}")
        print(f"Retrieved per sample:  {diagnostics['retrieved_per_sample']}")
        print(f"Memory token shape:    {diagnostics['mem_tokens_shape']}")
        print(f"Input video shape:     {diagnostics['input_shape']}")
        print(f"Output shape:          {diagnostics['output_shape']}")
        print(f"Output all finite:     {diagnostics['output_finite']}")
        print(f"Forward pass:          {diagnostics['forward_seconds']*1000:.1f} ms")
        print(f"Store stats:           {diagnostics['store_stats']}")
        print("=" * 60)
        ok = (
            diagnostics["output_shape"] == diagnostics["input_shape"]
            and diagnostics["output_finite"]
        )
        print("RESULT:", "PASS — plumbing validated" if ok else "FAIL")
        print("=" * 60)

    return diagnostics


if __name__ == "__main__":
    run_prototype()
