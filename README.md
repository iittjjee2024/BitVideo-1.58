# BitVideo-1.58 + BitMem

**Production-grade W1.58A8 Video Diffusion Transformer** — ternary quantized video generation using 1.58-bit weights and 8-bit activations — **now augmented with BitMem, a JEV-Mem-style agentic memory system**.

Two subsystems in one repository:

| Package | Role |
|---------|------|
| `bitvideo/` | The ternary Video Diffusion Transformer (backbone, quantization, kernels, training, pipeline) |
| `bitmem/` | The agentic memory layer (memory store, retrieval, consolidation, agent controller, DiT injection) |

## Overview

BitVideo-1.58 implements the BitNet b1.58 quantization scheme (ternary weights {-1, 0, +1} with per-tensor absmean scaling) applied to a Video Diffusion Transformer architecture. Every linear projection uses quantization-aware training (QAT) with straight-through estimators, enabling massive memory and compute savings while maintaining generation quality.

**BitMem** extends this with an externalized, agentic memory system inspired by JEV-Mem. Instead of treating memory as a static external database, the diffusion model can query memory *during* generation, and an agent controller learns which memories are useful, when to retrieve them, and when to create new ones. The central research question — deliberately designed to be falsifiable — is whether a **small ternary DiT + selectively retrieved, dynamically consolidated memory** can match or beat a **substantially larger memory-free model** at task-level efficiency.

## Key Features

- **W1.58A8 Quantization**: All linear layers use ternary weights (1.58 bits) with INT8 activations
- **Custom CUDA Kernels**: Optimized GEMV, DP4A GEMM, and MMA tensor-core kernels for ternary computation
- **3-Tier Dispatch**: Automatic backend selection (CUDA extension > Triton > PyTorch fallback)
- **Video DiT Architecture**: Spatial + temporal factorized attention with adaLN-Zero conditioning
- **6 Noise Schedulers**: DDIM, Euler, Euler Ancestral, DPM-Solver++, PNDM, UniPC
- **Training Stack**: QAT trainer with min-SNR weighting, knowledge distillation, LoRA fine-tuning
- **Production Ready**: ControlNet, ONNX/TorchScript export, profiling, structured logging

## Architecture

```
Video [B,C,T,H,W]
    │
    ▼ Conv3D Patch Embedding
Tokens [B, T*H*W, D]
    │
    ▼ N × VideoDiTBlock
    │   ├── adaLN-Zero → Spatial Self-Attention → gate → residual
    │   ├── adaLN-Zero → Temporal Self-Attention → gate → residual
    │   ├── adaLN → Cross-Attention (text) → residual
    │   └── adaLN-Zero → SwiGLU FFN → gate → residual
    │
    ▼ Final Layer (adaLN + Linear)
Prediction [B, T*H*W, C*patch_vol]
    │
    ▼ Unpatchify
Output [B,C,T,H,W]
```

---

# BitMem — JEV-Mem Agentic Memory Integration

BitMem couples the ternary DiT above with a **JEV-Mem-style agentic memory system**.
Where a standard diffusion model is a pure function of `(noise, timestep, text)`,
a BitMem-augmented model is a function of `(noise, timestep, text, retrieved_memory)`
where the retrieved memory is chosen *dynamically by an agent* and injected *during*
generation.

## What "JEV-Mem-style" means here

JEV-Mem (Joint Experience & Value Memory) is treated as an **engineering pattern**,
not a fixed implementation. We adopt its core ideas:

1. **Typed memory** — not one flat store, but working / episodic / semantic /
   procedural / long-term consolidated memories, each with different lifecycles.
2. **Experience-centric writes** — the system records structured *experiences*
   (task, context, retrieved memories, actions, generation, outcome, reward,
   failure modes, new knowledge), not just raw content.
3. **Value-aware retention** — memories carry a learned *utility* that is updated
   from downstream reward, so the store keeps what proves useful and forgets what
   does not.
4. **Agentic control** — a controller decides *whether* to retrieve, *what* to
   query, *whether* the result was useful, and *whether* to write a new memory —
   rather than blindly appending everything.
5. **Consolidation** — related memories are clustered, summarized, and promoted to
   long-term memory, tracking information loss.

## Combined Architecture

```
                          ┌─────────────────────────────────────────┐
                          │            AGENT CONTROLLER               │
                          │  Observe → Interpret → Retrieve →         │
                          │  Generate → Evaluate → Consolidate →      │
                          │  Update   (compact state, not full DB)    │
                          └────────┬─────────────────────┬───────────┘
             retrieval query       │                     │  write-back experience
                          ┌────────▼────────┐    ┌────────▼───────────┐
                          │   RETRIEVAL      │    │   MEMORY STORE      │
                          │ query → ANN →    │◄───┤ working / episodic /│
                          │ filter → rerank →│    │ semantic / proc /   │
                          │ compress         │    │ long-term (LTM)     │
                          └────────┬────────┘    └────────▲───────────┘
             memory tokens /       │                      │ WRITE / READ / RETRIEVE
             conditioning          │                      │ UPDATE / MERGE /
                          ┌────────▼──────────────────────┴──────────┐
                          │  TERNARY DiT  (bitvideo.models.VideoDiT)  │
                          │  + Memory-Injection Interface             │
                          │    Method A: memory cross-attention       │
                          │    Method B: memory tokens (prepend)      │
                          │    Method C: adaptive conditioning (FiLM)  │
                          └───────────────────────────────────────────┘
```

The DiT backbone is **unchanged** — memory enters through a thin, swappable
adapter, so the backbone, quantization method, memory implementation, and agent
controller can each be replaced independently.

## The Agentic Control Loop

```
obs → controller.should_retrieve? ──no──► DiT.generate (memory-free path)
                    │ yes
                    ▼
  controller.build_query(obs) ─────────► q [D_mem]
                    ▼
  store.retrieve(q, k, filters) ───────► candidates
                    ▼
  policy.score + rerank + diversity ───► top-m memories
                    ▼
  consolidation.compress(top-m) ───────► conditioning payload
                    ▼
  interface.inject(tokens, payload) ───► DiT blocks run with memory
                    ▼
  DiT.generate ────────────────────────► sample
                    ▼
  evaluator.utility(sample, task) ─────► reward
                    ▼
  controller.should_write? ────────────► store.write(experience)   [+provenance +confidence]
                    ▼
  periodic maintenance ────────────────► store.decay(now); cluster → consolidate
```

## Memory Types (JEV-Mem taxonomy)

| Type | Lifespan | Contents | Example |
|------|----------|----------|---------|
| **Working** | current task only | active retrieval context | memories injected into the current generation |
| **Episodic** | medium | specific past experiences | "prompt X produced sample Y with reward R" |
| **Semantic** | long | distilled facts / concepts | "aerial drone shots benefit from slow motion" |
| **Procedural** | long | how-to policies | "for smooth FPS, condition on 121-frame clips" |
| **Long-term (LTM)** | permanent | consolidated, high-utility | merged clusters promoted from episodic |

## Memory Item (structured metadata)

Every memory carries the metadata required for retrieval scoring, consolidation,
decay, and — critically — **failure handling** (provenance + confidence let us
trace and down-weight hallucinated, contradictory, stale, or poisoned memories):

```python
MemoryItem(
    content,          # raw payload (latent, caption, experience record)
    embedding,        # [D_mem] retrieval key
    timestamp, source, task, context,
    confidence, importance, utility, recency, access_count,
    relationships,    # ids of linked memories (a memory graph)
    compressed,       # consolidation status
    provenance,       # where it came from — falsifiability + poison defense
)
```

## Memory Operations

`WRITE · READ · RETRIEVE · UPDATE · MERGE · CONSOLIDATE · DECAY · DELETE`

The store provides the mechanics; **policies** decide what deserves storage, what
to ignore, what to summarize, what to merge, what to promote to LTM, and what to
let decay. The store never blindly appends everything.

## Retrieval Pipeline (configurable, not hard-coded)

```
query → vector retrieval → metadata filtering → rerank → compression → conditioning
```

The retrieval score is a **configurable weighted sum**, so each term can be
ablated independently (this directly powers the ablation matrix below):

```
score(q, m) = α·cos(q, e_m)          # semantic similarity
            + β·recency(m)            # exponential recency decay
            + γ·importance(m)         # curated / learned importance
            + δ·task_similarity(m)    # task-conditioned relevance
            + ε·confidence(m)         # provenance confidence
            − ζ·redundancy(m)         # diversity penalty (MMR reranking)
```

Setting `β=γ=δ=ε=ζ=0` recovers pure semantic retrieval (the ablation baseline).

## Memory-to-Diffusion Interface — three methods

| Method | Mechanism | Relative cost | Integration point |
|--------|-----------|---------------|-------------------|
| **A. Cross-Attention** | a second, gated `CrossAttention(context_dim=D_mem)` per block | high (per-block KV) | new `memory_attention` in `VideoDiTBlock` |
| **B. Memory Tokens** | project memories → learned tokens, prepend to sequence | medium (longer seq) | before the block loop in `VideoDiT` |
| **C. Adaptive Conditioning** | pool memory → FiLM / AdaLN gates modulating each block | low | extra modulation added to `t_emb` |

All three are built from `bitvideo.quantization.BitLinear`, so the memory pathway
stays on the **ternary / packed** path — the memory interface never smuggles in
full-precision compute, which keeps the efficiency comparison honest.

## Quantization Modes (shared with BitVideo, for controlled ablation)

| Mode | `QuantizationConfig` |
|------|----------------------|
| FP16 / BF16 baseline | `weight.enabled=False, activation.enabled=False` |
| INT8 activations only | `weight.enabled=False, activation.enabled=True` |
| Ternary W1.58 + A8 (packed) | defaults (`threshold_factor=0.5, bits=8`) |
| Mixed | different `QuantizationConfig` per module |

## Agentic Learning

After each task, an **experience record** is created:

```
Task · Context · Retrieved memories · Actions · Generation ·
Outcome · Reward/utility · Failure modes · New knowledge
```

These records improve future retrieval and behavior. The memory-selection /
write policy is modular so several approaches can be compared: heuristic utility
scoring, supervised memory selection, reinforcement learning, preference
optimization, and learned retrieval policies.

Utility is updated from downstream reward:

```
u_m ← (1 − η)·u_m + η · 1[m retrieved] · reward
```

Low-utility, stale, rarely-accessed memories are decayed and eventually forgotten.

## Failure Handling (designed in, not bolted on)

Every retrieved memory carries **provenance and confidence**. The system is
designed to handle: hallucinated memories, contradictory memories, stale
memories, retrieval contamination, memory poisoning, excessive memory growth,
retrieval loops, controller instability, quantization-induced degradation, and
numerical instability during diffusion.

## Training Stages

| Stage | Description | Trains | Status |
|-------|-------------|--------|--------|
| **0** | Minimal prototype: tiny ternary DiT + dict memory + Method-B tokens | nothing (smoke test) | ✅ **done — 22 tests pass** |
| 1 | FP16 DiT baseline train/eval harness (no memory) | DiT | planned |
| 2 | Ternary DiT, measure quantization degradation | DiT (QAT) | planned |
| 3 | Memory-augmented DiT, frozen policy; episodic+semantic memory, Methods A & C | memory adapters | planned |
| 4 | Agentic memory — learn *when* to retrieve / write / consolidate | controller | planned |
| 5 | Joint optimization: DiT + scales + adapters + retrieval + write policy | all | planned |

## Ablation Matrix

Controlled experiments removing one component at a time, to isolate exactly what
contributes:

`no memory · random memory · semantic-only · semantic+recency · semantic+importance ·
no consolidation · no controller · memory-tokens · cross-attention ·
adaptive-conditioning · FP16 backbone · ternary backbone`

## Evaluation Protocol

- **Generation:** FID, reconstruction MSE, CLIP-sim (where captions exist).
- **Memory:** retrieval precision/recall, utility, stale-rate, consolidation loss,
  growth, forgetting rate.
- **Agent:** task success, #retrieval calls, unnecessary-retrieval rate,
  useful-write rate, failure recovery, long-horizon consistency.
- **Efficiency (5-way):** `FP16 DiT vs INT8 DiT vs 1.58-bit DiT vs 1.58-bit+memory
  vs 1.58-bit+agentic-memory` — param memory, peak VRAM, samples/s, latency,
  retrieval overhead.

## The Central (Falsifiable) Hypothesis

> A smaller ternary diffusion transformer augmented with selectively retrieved,
> dynamically consolidated memory may achieve better task-level efficiency than a
> substantially larger model without persistent agentic memory.

**This is not assumed true.** The ablation matrix and the 5-way efficiency table
are designed so the experiments can *falsify* it — e.g. if `1.58-bit + agentic
memory` fails to beat a larger memory-free FP16 model at equal compute budget.

## BitMem Package Layout

```
bitmem/
├── memory/
│   ├── base.py         MemoryItem + all Protocol interfaces
│   ├── storage.py      DictMemoryStore (reference MemoryStore implementation)
│   └── retrieval.py    Cosine + Weighted policies, diversity reranking
├── interface/
│   └── mem_tokens.py   Method B: memory-token injection (ternary BitLinear projection)
└── prototype.py        Stage-0 end-to-end smoke test
tests/bitmem/
└── test_stage0.py      22 unit + integration tests (all passing)
```

Planned additions per stage: `memory/{episodic,semantic,procedural,consolidation}.py`,
`agent/{controller,policies,planner,evaluator}.py`,
`interface/{cross_attn,adaptive}.py`, `eval/{metrics,ablation}.py`.

## BitMem Quick Start

```python
import torch
from bitvideo.models import VideoDiT
from bitvideo.quantization.quantization import QuantizationConfig
from bitmem.memory import DictMemoryStore, MemoryItem, CosineRetrievalPolicy
from bitmem.interface import MemoryTokenInterface

# 1. Ternary DiT (memory-free backbone, unchanged)
dit = VideoDiT(in_channels=8, dim=128, depth=2, num_heads=4,
               context_dim=128, quantization=QuantizationConfig()).eval()

# 2. Agentic memory store + retrieval policy
store = DictMemoryStore(policy=CosineRetrievalPolicy())
store.write(MemoryItem(content="a prior experience",
                       embedding=torch.randn(64), task="demo"))

# 3. Method-B memory-token injection (ternary projection)
mem_iface = MemoryTokenInterface(memory_dim=64, model_dim=128, max_tokens=4,
                                 quantization=QuantizationConfig()).eval()

# 4. Retrieve, project, inject, generate
query = torch.randn(64)
memories = [store.retrieve(query, k=3)]           # per-sample retrieval
mem_tokens, mask = mem_iface.build_tokens(memories, device=torch.device("cpu"),
                                          dtype=torch.float32)
video = torch.randn(1, 8, 3, 16, 16)
t = torch.randint(0, 1000, (1,)).float()
with torch.no_grad():
    out = dit(video, t, context=mem_tokens)        # memory as conditioning
```

## Run BitMem

```bash
# Stage-0 end-to-end smoke test
python -m bitmem.prototype
# -> RESULT: PASS — plumbing validated

# Full BitMem test suite
python -m pytest tests/bitmem/test_stage0.py -v
# -> 22 passed
```

---

## Installation

```bash
# Basic installation (PyTorch fallback)
pip install -e .

# With CUDA extension (requires nvcc + CUDA 12+)
pip install -e .
```

## Quick Start

```python
import torch
from bitvideo.models import VideoDiT
from bitvideo.pipeline import BitVideoPipeline, DDIMScheduler

# Create model
model = VideoDiT(
    in_channels=4, dim=768, depth=12, num_heads=12,
    context_dim=768, patch_size=(1, 2, 2), qk_norm=True,
)

# Setup pipeline
scheduler = DDIMScheduler(num_train_steps=1000)
pipeline = BitVideoPipeline(model, scheduler)

# Generate (with pre-computed text embeddings)
context = torch.randn(1, 77, 768)  # From text encoder
video = pipeline(context, num_frames=16, height=32, width=32,
                 num_inference_steps=50)
```

## Training

```python
from bitvideo.training.train_qat_video import TrainingConfig, train_qat

config = TrainingConfig(
    dim=768, depth=12, num_heads=12,
    learning_rate=1e-4, max_steps=100_000,
    mixed_precision="bf16",
)
train_qat(config)
```

## Project Structure

```
bitvideo/           # ── Ternary Video Diffusion Transformer ──
├── cuda/           # CUDA extension (GEMV, DP4A, MMA kernels)
├── ops/            # Python operator layer (3-tier dispatch)
├── quantization/   # STE, quantizers, BitLinear
├── models/         # RoPE, patch embed, attention, Video DiT
├── pipeline/       # Schedulers, VAE decoder, inference pipeline
├── training/       # Losses, distillation, datasets, QAT trainer, streaming
├── triton/         # Optional Triton kernel backends
└── extras/         # LoRA, ControlNet, export, profiling, config

bitmem/             # ── JEV-Mem Agentic Memory (reuses bitvideo, no fork) ──
├── memory/         # MemoryItem, DictMemoryStore, retrieval policies
├── interface/      # DiT injection: memory tokens (A/C planned)
└── prototype.py    # Stage-0 end-to-end smoke test

tests/              # BitVideo integration tests
tests/bitmem/       # BitMem unit + integration tests (Stage 0: 22 passing)
benchmarks/         # Inference throughput benchmarks
notebooks/          # Kaggle / Colab / Lightning training notebooks
ltx-finetune/       # LTX-2.3 LoRA fine-tuning + distillation pipeline
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (2.8+ recommended for full torch.compile support)
- CUDA 12+ (optional, for native kernels)

## Benchmarks (RTX 5050 Laptop GPU, FP16)

| Config | Parameters | Forward (ms) | Throughput |
|--------|-----------|-------------|-----------|
| Tiny (D=64, L=2) | 272K | 75ms | 13.3/s |
| Small (D=128, L=4) | 2.0M | 145ms | 6.9/s |
| Base (D=256, L=6) | 11.7M | 198ms | 5.1/s |

## References

- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (2024)
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, 2023)
- BitNet: Scaling 1-bit Transformers for Large Language Models (2023)
- JEV-Mem: agentic memory pattern (typed memory, experience-centric writes,
  value-aware retention, consolidation) — adopted as an engineering pattern for
  the BitMem subsystem, not a fixed implementation.
- Retrieval-Augmented Generation and Maximal Marginal Relevance (diversity
  reranking) inform the BitMem retrieval pipeline.

## Research Note

BitMem is a **research prototype**, clearly separated from the production-ready
BitVideo backbone. Stage 0 validates plumbing and tensor shapes only — the DiT is
randomly initialized, the memory store is in-RAM brute-force, and tasks are
synthetic. Generation-quality and hypothesis-level claims come only after Stages
1–5 on real data with the full ablation matrix. Experimental assumptions are
stated explicitly and never hidden.

## License

Research use. See LICENSE for details.
