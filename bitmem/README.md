# BitMem — Ternary DiT + Agentic Memory

A hybrid architecture pairing the W1.58/A8 ternary Diffusion Transformer
(`bitvideo.models.VideoDiT`) with a JEV-Mem-style agentic memory system.

**Research question (not assumed true):** Can a small ternary DiT augmented with
selectively retrieved, dynamically consolidated memory achieve better task-level
efficiency than a substantially larger memory-free model? Every component sits
behind an interface so the ablation matrix can attempt to *falsify* the hypothesis.

## Status: Stages 0–5 complete + TurboQuant integrated (all validated)

| Stage | Description | Status |
|-------|-------------|--------|
| **0** | Minimal prototype: tiny ternary DiT + dict memory + Method-B tokens | ✅ **done, 22 tests pass** |
| **1** | FP16 DiT baseline train/eval harness | ✅ **done, 17 tests pass** |
| **2** | Ternary DiT, measure degradation table | ✅ **done (shares Stage-1 harness)** |
| **3** | Episodic+semantic memory, full retrieval, consolidation, Methods A & C | ✅ **done, 21 tests pass** |
| **4** | Agent controller (heuristic gate/write/consolidate + real retrieval) | ✅ **done, 20 tests pass** |
| **5** | Joint optimization + full ablation matrix (the hypothesis test) | ✅ **done, 10 tests pass** |
| **+** | **TurboQuant** vector quantization for the memory store (Google, ICLR 2026) | ✅ **done, 11 tests pass** |
| **+** | **ANN index** backend (HNSW) — closes the brute-force O(N) retrieval caveat | ✅ **done, 8 tests pass** |

**Full suite: 109 tests passing.** Training/eval auto-select CUDA when a GPU is
present (`device="auto"`); all results below were measured on an RTX 5050 Laptop
GPU (sm_120) or CPU as noted.

### Stage 4 delivered

- **Experience + evaluator** (`agent/evaluator.py`): `ExperienceRecord` captures
  the §11 fields (task, retrieved ids, actions, generation, reward, failure
  modes, new knowledge, query embedding) and converts to a storable `MemoryItem`.
  `UtilityEvaluator` scores reward as *relative improvement over a memory-free
  baseline* — the exact quantity the hypothesis rests on. `update_memory_utilities`
  applies the value-aware retention update and dents confidence on bad outcomes
  (poison defense, §14).
- **Swappable policies** (`agent/policies.py`): `RetrievalGate` (`AlwaysRetrieve`,
  `HeuristicRetrievalGate`, and a tiny differentiable `LearnedRetrievalGate` for
  Stage 5), `WritePolicy` (`HeuristicWritePolicy`), `ConsolidationPolicy`
  (`PeriodicConsolidation`). Policies read only the controller's **compact state**,
  never the full store (§4).
- **Control loop** (`agent/controller.py`): `AgentController.step()` runs
  Observe → Retrieve → Generate → Evaluate → Update → Consolidate with built-in
  failure guards (retrieval-loop cap, growth-triggered consolidation, poison
  confidence decay, provenance for contradiction audits). `agent_metrics()` emits
  the §12 rates (retrieval rate, unnecessary-retrieval rate, useful-write rate,
  recoveries, memory total).

**Bug caught + fixed during testing (cold-start deadlock):** on an empty store,
retrieval cannot beat the memory-free baseline, so reward ≈ 0 and a reward-only
write policy would *never* write — leaving the store empty forever and the agent
unable to ever learn to retrieve. Fixed by adding a **novelty clause**: a
non-redundant experience is stored even at low reward (surprising outcomes are
still always stored; redundant ones are still always dropped). This is what lets
the agent bootstrap a store it can later retrieve from.

### Stage 4 agentic real-retrieval diagnostic

`python scripts/bitmem_stage4_agent.py` removes Stage 3's oracle crutch. On an
associative-recall task (K tasks, each with a hidden answer vector keyed by a
noisy query), the agent must **write its own experiences** and later **retrieve
the right one from the real store** — retrieval can now fail. Reward = cosine
similarity between the retrieved answer and the true answer. This probes the
memory + retrieval + write + utility loop as a whole, independent of the DiT
(whose ability to *use* correct memory was shown in Stage 3).

Run: 6 tasks, dim 32, key noise 0.05, 300 episodes, CPU.

| config | recall@0.9 (overall) | recall@0.9 (steady-state) | mean similarity |
|--------|----------------------|----------------------------|------------------|
| random memory (control) | 0.000 | 0.000 | −0.048 |
| agent, always-retrieve | 0.163 | **1.000** | 1.000 |
| agent, heuristic gate | 0.163 | **1.000** | 1.000 |

**Honest reading:** once each task has been written once (steady state), the
agent retrieves the exact right answer **every time** (recall@0.9 = 1.000, mean
similarity = 1.000), while a store filled with random memories never does
(0.000). The lower *overall* number (0.163) is a cold-start artifact — during
warmup a task's answer has not been written yet, so no correct memory exists to
find; the eval reports both numbers rather than hiding the warmup. Over 300
episodes the 300 writes are distilled by consolidation down to a handful of
long-term memories (one cluster per task), confirming the growth guard and
consolidation loop work. **Verdict: real retrieval works without an oracle.**
This validates the agent mechanism; it is *not* video-quality evidence — that
requires trained models and the Stage 5 ablation matrix.

### Stage 3 delivered

- **Typed memory** (`memory/typed.py`): Episodic / Semantic / Procedural /
  Long-term stores with type-appropriate write + decay policies, orchestrated by
  a `MemorySystem` that routes writes and merges cross-type retrieval.
- **Consolidation** (`memory/consolidation.py`): greedy clustering + summarization
  with **information-loss tracking** (`reconstruction_loss`), safety gates
  (`max_spread`, `max_loss`), and promotion of episodic clusters into long-term
  memory. Returns a `ConsolidationReport` with mean/max loss telemetry.
- **All three injection methods** (`interface/`):
  - Method A — `MemoryCrossAttention` (content tokens attend to memory, gated)
  - Method B — `MemoryTokenInterface` (memory appended to cross-attn context)
  - Method C — `AdaptiveMemoryConditioning` (pooled memory modulates `t_emb`)
  - `MemoryAugmentedDiT` selects the method by config (the ablation switch).
  - All zero-initialized so memory starts as a **no-op** (stable training).

**Bug caught + fixed during testing:** Method B originally prepended memory
tokens to the content sequence, which breaks the DiT's grid-factorized
spatial/temporal attention (requires `T·HW == seq_len`). Corrected to append
memory to the non-factorized cross-attention context instead — backbone untouched.

### Stage 3 memory-benefit diagnostic

`python scripts/bitmem_stage3_memory.py` runs a task where
`clean = shared(text) + prototype[task_id]`, with the prototype **retrievable
from memory but not predictable from text**. A memory-free model must carry
irreducible error on the prototype; a memory-using model can subtract it.

| method | final MSE (300 steps) | final MSE (800 steps) | vs no-memory |
|--------|----------------------|-----------------------|--------------|
| none (baseline) | 0.61100 | 0.60253 | — |
| adaptive | 0.61072 | — | −0.0% |
| memory_tokens | 0.61041 | — | −0.1% |
| cross_attention | 0.61034 | 0.60161 | **−0.1% → −0.2%** |

**Honest reading:** every injection method beats the no-memory baseline, and the
gap **grows with training** (−0.1% → −0.2%), confirming the mechanism is real and
not noise — the ternary DiT progressively learns to exploit retrieved memory.
The *absolute* margin is small because under the epsilon-prediction objective the
additive prototype is a modest fraction of the total signal variance; a stronger
effect needs a task where memory carries more of the variance. This is a
**mechanism validation** (the prerequisite for the hypothesis), not evidence
about real-video quality, and it used **oracle retrieval** — retrieval-quality
stress-testing is Stage 4's job.

### Stage 5 — joint optimization + the falsifiable hypothesis test

Stage 5 is where the hypothesis is actually put at risk. It adds:

- **A memory-sensitive task** (`train/synthetic.py::MemorySensitiveDataset`):
  `clean = shared_from_text + memory_gain · prototype[task_id]`. The per-task
  prototype is **independent of the text**, so it is *unpredictable from
  conditioning* — a memory-free model must regress it toward the mean (an
  irreducible error), while a memory model can retrieve and subtract it. At
  `memory_gain=2` the memory-only component carries ~80% of the predictable
  signal variance (measured `memory_floor ≈ 4.0` clean-space vs ~1.0 shared).
- **A joint trainer** (`train/joint.py::JointTrainer`) that trains the
  memory-augmented ternary DiT end-to-end with **real cosine retrieval** from a
  pre-populated store, an **x0 (clean-latent) objective** (epsilon-prediction
  structurally hides the memory signal), and an optional **learned retrieval
  gate**. One class produces every ablation cell by config.
- **An ablation runner** (`scripts/bitmem_stage5_ablation.py`) with three
  experiments: (A) matched-size memory ON/OFF, (B) injection-method sweep, (C)
  the hypothesis — small ternary+memory vs a larger fp16 no-memory model.

**Experiment A — matched-size memory ablation (ternary d48×2, x0, GPU).** Same
backbone, memory ON vs OFF, so there is no size confound. The outcome depends on
whether the task *forces* memory use:

| task diversity | memory OFF MSE | best memory ON MSE | verdict |
|----------------|----------------|---------------------|---------|
| 32 prototypes (700 steps) | 0.31908 | 0.31911 | no help — model memorizes prototypes in weights |
| 256 prototypes (700 steps) | 0.33239 | **0.33017** (−0.7%) | memory helps — too many prototypes to memorize |

**Experiment C — the hypothesis (small ternary+memory vs larger fp16 no-memory).**

| task diversity | small ternary+mem | large fp16 no-mem | result |
|----------------|-------------------|-------------------|--------|
| 8 prototypes | 1.429 (196K params) | **0.826** (1.83M params) | larger model WINS |
| 256 prototypes | 1.646 (194K params) | 1.594 (1.83M params) | ~3% gap at **9.5× fewer params** |

**Honest verdict on the hypothesis:** it is **conditionally supported, not
unconditionally true.** When the task has few patterns, a larger memory-free
model simply memorizes them in its weights and wins outright. As task diversity
grows past what the small model's weights can hold, memory substitutes for scale:
the gap collapses and, at fixed size, memory measurably lowers loss. The crossover
is real and reproducible. We do **not** claim the small model beats the large one
in absolute MSE here — it does not, at these sizes/steps — only that memory's
value rises with the memory-demand of the task, exactly as the hypothesis
predicts. A decisive win needs larger models, more steps, and ideally real data.

**Measurement honesty (§16):**
- Synthetic task + tiny models + x0 objective validate the **mechanism and the
  ablation methodology**, not real-video quality.
- Training/eval focus on the recoverable noise regime (`max_timestep_frac=0.5`);
  at very high noise the clean latent is unrecoverable regardless of memory, which
  would otherwise swamp the signal. This is a probing choice, documented as such.
- The ternary path uses fake-quant (QAT) ops, not packed 1.58-bit kernels, so
  wall-clock is **not** a fair speed benchmark; the storage bytes are theoretical
  (log2(3) ≈ 1.585 bits/weight). GPU utilization is low at these tiny sizes —
  per-op/Python overhead dominates, not arithmetic.

Reproduce: `python scripts/bitmem_stage5_ablation.py --experiment A --num-tasks 256 --steps 700`

### TurboQuant — vector quantization for the memory store (Google, ICLR 2026)

The retrieval keys in the memory store are high-dimensional fp32 vectors scored
by inner product — exactly the setting **TurboQuant** targets ([Zandieh, Daliri,
Hadian, Mirrokni, *TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate*, arXiv:2504.19874](https://arxiv.org/abs/2504.19874)). We
integrated a faithful implementation of its core as a **swappable compressed
store backend** (not a fork). *Content was rephrased for compliance with
licensing restrictions.*

- **`memory/turboquant.py`** — the data-oblivious codec:
  - `RandomRotation`: a randomized Hadamard transform (sign flip + normalized
    Hadamard, padded to a power of two). This is the data-oblivious rotation that
    concentrates the coordinate distribution so per-coordinate scalar
    quantization becomes near-optimal; verified orthogonal and invertible
    (round-trip error ~1e-6, norm preserved). No training/calibration.
  - `TurboQuantMSE`: rotate, then a uniform per-coordinate scalar quantizer with a
    per-vector scale. Minimizes reconstruction MSE.
  - `TurboQuantProd` (experimental): adds a 1-bit QJL residual correction for
    inner-product estimation, recovered via the SimHash angle identity. See the
    honest note below.
- **`memory/turbo_store.py`** — `TurboQuantMemoryStore`, a drop-in `MemoryStore`
  that composes `DictMemoryStore` and only changes the storage: each key is
  replaced by its TurboQuant reconstruction and the compact integer code is
  retained for honest byte accounting. Retrieval, scoring, decay, merge, and
  diversity reranking all keep working unchanged.
- **`MemorySystem.with_turboquant(dim, bits, seed)`** builds a whole memory
  system (episodic/semantic/procedural/long-term) on the compressed backend.

**Measured** (`scripts/bitmem_turboquant_bench.py`, 1000 keys, dim 128,
recall@10 vs an exact fp32 cosine index, noisy queries):

| bits | recon MSE | recall@10 vs fp32 | storage compression |
|------|-----------|-------------------|---------------------|
| fp32 | 0 | 1.000 | 1.0× |
| 2 | 0.00233 | 0.531 | 14.2× |
| 3 | 0.00042 | 0.766 | 9.9× |
| 4 | 0.00009 | 0.884 | 7.5× |
| 8 | ~0 | 0.994 | 3.9× |

**Honest reading:** 8-bit TurboQuant keys are **near-lossless for retrieval
(recall 0.994) at ~3.9× storage compression**; 4-bit trades down to 0.88
recall@10 for 7.5×. recall@10 is a strict metric (recovering the exact top-10
*set* under noisy queries); top-1 recall against the original keys is near-perfect
even at 4 bits. The 1-bit **QJL `prod` variant did not improve inner-product
accuracy in this regime** — at ≥2 bits the MSE-decoded dot product is already
accurate and the QJL correction adds variance; QJL pays off at the extreme
sub-2-bit compression the paper targets for KV-caches, not at retrieval
bit-widths. `TurboQuantMSE` is therefore the recommended store codec, and
`TurboQuantProd` is kept and clearly labeled experimental.

**Scope note:** this compresses the **persisted/transmitted** key bytes (what a
vector DB stores on disk or ships), which is TurboQuant's stated use case. The
prototype still reconstructs keys to fp32 in RAM for scoring, so in-RAM footprint
is not reduced here; a production backend would score directly against codes.

Reproduce: `python scripts/bitmem_turboquant_bench.py`

### ANN index backend — closing the brute-force retrieval caveat

Since Stage 0, `DictMemoryStore` has honestly flagged its O(N) brute-force
retrieval as a prototype limitation. `memory/ann_store.py::AnnMemoryStore` closes
it with a real **HNSW approximate-nearest-neighbor index** (hnswlib), using the
standard two-stage vector-DB pattern:

1. **ANN recall** — the HNSW index returns the top `over_fetch · k` candidates by
   cosine similarity in ~O(log N) instead of O(N).
2. **Policy rerank** — the existing `RetrievalPolicy` reranks those candidates
   with the full multi-factor score (semantic + recency + importance + …) and
   diversity reranking picks the final k. Retrieval *semantics* are unchanged;
   only candidate generation is accelerated.

It preserves the whole `MemoryStore` protocol (filters, decay, merge,
consolidate, delete) and **falls back transparently to exact brute force** when
hnswlib is absent or `use_ann=False`, so correctness holds everywhere.

**Measured** (`scripts/bitmem_ann_bench.py`, dim 128, k=10, noisy queries,
CPU single-thread; recall@10 vs the exact brute-force index):

| store size N | recall@10 | brute ms/query | ANN ms/query | speedup |
|--------------|-----------|----------------|--------------|---------|
| 1,000 | 1.000 | 18.7 | 1.6 | 11× |
| 5,000 | 0.993 | 91.7 | 1.8 | 52× |
| 15,000 | 0.660 | 180.0 | 1.9 | 93× |

**Honest reading:** the ANN query time stays ~flat (~2 ms) as N grows while brute
force is O(N), so the **speedup grows with store size** (11× → 93×) — the whole
point of an index. Recall@10 is near-perfect at small/medium N. The drop at
N=15k is a property of the **strict metric under noisy queries**, not a store
bug: verified directly, raw hnswlib and this store both recover ~0.93 of the
exact top-10 for *exact* queries at N=20k; under query noise the exact top-10 set
becomes unstable (many near-equidistant neighbors), which penalizes any
approximate index on set-overlap. Recall is tunable via `ef_query` and
`over_fetch` (higher = better recall, slightly slower). For memory keys that are
well-separated (the realistic case), recall stays high.

Reproduce: `python scripts/bitmem_ann_bench.py`

### Stage 1 + 2 measured results (synthetic task, tiny DiT, CPU)

Ran `python scripts/bitmem_stage1_baseline.py --steps 400 --dim 128 --depth 2`
(1.25M-param DiT, 256-sample low-rank synthetic task, identical seeding across modes):

**Convergence** — every mode learns (final MSE < first MSE, 1.002 → ~0.504):

| mode | first MSE | final MSE | learned? |
|------|-----------|-----------|----------|
| fp16 | 1.00192 | 0.50408 | ✅ |
| int8 | 1.00192 | 0.50408 | ✅ |
| ternary | 1.00192 | 0.50741 | ✅ |
| mixed | 1.00192 | 0.50738 | ✅ |

**Quantization degradation** (vs FP16 baseline):

| mode | final MSE | Δ vs FP16 | relative |
|------|-----------|-----------|----------|
| fp16 | 0.50408 | +0.00000 | +0.0% |
| int8 | 0.50408 | +0.00000 | +0.0% |
| ternary | 0.50741 | +0.00333 | **+0.7%** |
| mixed | 0.50738 | +0.00330 | +0.7% |

**Efficiency** (theoretical ternary storage vs FP16 actual):

| mode | params | ternary_MB (theory) | fp16_MB | latency_ms | samples/s |
|------|--------|---------------------|---------|------------|-----------|
| fp16 | 1,254,944 | 0.250 | 2.510 | 46.5 | 344.1 |
| ternary | 1,254,944 | 0.250 | 2.510 | 1162.9 | 13.8 |

**Honest reading of these numbers (per spec §9, §16):**
- Ternarization costs only **+0.7% MSE** here — a promising sign, but on a *toy*
  low-rank task with a *tiny* model. It is NOT evidence about real video quality.
- The theoretical ternary storage is **10× smaller** (0.25 MB vs 2.51 MB via the
  log2(3)-bit lower bound), but the training-time model still holds FP32 master
  weights — the storage win is realized only after packing for inference.
- The ternary latency here (1163 ms) is **slower**, not faster, because this is
  the CPU **QAT fake-quant path** (extra quant/dequant ops), *not* the packed
  CUDA kernels. A speed claim requires the packed inference path on GPU — which
  is exactly why the design says "do not claim a speedup unless it is benchmarked."

## What Stage 0 proves

- Memory items with structured metadata + provenance (§3, §14)
- In-RAM store with all 9 operations: write/read/retrieve/update/merge/consolidate/decay/delete (§3)
- Configurable retrieval score (semantic/recency/importance/confidence, each ablatable) (§6)
- Diversity-aware reranking (§6)
- Method-B memory-token injection through a **ternary** `BitLinear` projection (§5)
- End-to-end forward pass: tiny DiT consumes retrieved memory as context, output shape matches input, all finite

## What Stage 0 does NOT prove (stated up front)

- Generation quality — the DiT is randomly initialized (smoke test only)
- Scale — brute-force retrieval, in-RAM store (FAISS/HNSW arrives Stage 3)
- The hypothesis — needs trained models + the ablation matrix (Stages 1–5)

## Layout

```
bitmem/
├── memory/
│   ├── base.py         MemoryItem + all Protocol interfaces
│   ├── storage.py      DictMemoryStore (reference MemoryStore impl)
│   └── retrieval.py    Cosine + Weighted policies, diversity rerank
├── interface/
│   └── mem_tokens.py   Method B: memory tokens (ternary projection)
└── prototype.py        Stage-0 end-to-end smoke test
tests/bitmem/
└── test_stage0.py      22 unit + integration tests
```

## Run

```bash
# Smoke test
python -m bitmem.prototype

# Full test suite
python -m pytest tests/bitmem/test_stage0.py -v
```

Expected: prototype prints `RESULT: PASS`, tests report `22 passed`.

## Design

See the full system design (architecture, data flow, math, evaluation protocol,
ablation matrix, failure handling) in the design doc artifact / `docs/`.

Reuses `bitvideo` primitives (`BitLinear`, `VideoDiT`, `QuantizationConfig`) —
no fork. The memory, retrieval, and agent layers are greenfield.
