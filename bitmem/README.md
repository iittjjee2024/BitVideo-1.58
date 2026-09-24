# BitMem — Ternary DiT + Agentic Memory

A hybrid architecture pairing the W1.58/A8 ternary Diffusion Transformer
(`bitvideo.models.VideoDiT`) with a JEV-Mem-style agentic memory system.

**Research question (not assumed true):** Can a small ternary DiT augmented with
selectively retrieved, dynamically consolidated memory achieve better task-level
efficiency than a substantially larger memory-free model? Every component sits
behind an interface so the ablation matrix can attempt to *falsify* the hypothesis.

## Status: Stage 0 complete (validated)

| Stage | Description | Status |
|-------|-------------|--------|
| **0** | Minimal prototype: tiny ternary DiT + dict memory + Method-B tokens | ✅ **done, 22 tests pass** |
| **1** | FP16 DiT baseline train/eval harness | ✅ **done, 17 tests pass** |
| **2** | Ternary DiT, measure degradation table | ✅ **done (shares Stage-1 harness)** |
| 3 | Episodic+semantic memory, full retrieval, consolidation, Methods A & C | planned |
| 4 | Agent controller (heuristic → learned) | planned |
| 5 | Joint optimization + full ablation matrix | planned |

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
